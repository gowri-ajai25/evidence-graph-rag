"""
Trains the TreatmentRanker GNN using PATIENT-level supervision.

Positive pairs: (patient, drug) where the patient has a mutated gene with
an evidence-backed path to that drug. Label = that path's evidence_score.

Hard negatives: (patient, drug) where the drug exists in the graph but is
NOT reachable from any of the patient's mutated genes — forces the model
to learn mutation-specific relevance, not just general drug popularity.

Train/test split is done by PATIENT, so we evaluate generalization to
held-out patients, not just held-out edges.
"""

from dotenv import load_dotenv
import torch
import torch.nn.functional as F
import pickle
import random
import os
import sys

sys.path.append(os.path.dirname(__file__))
from gnn_model import TreatmentRanker

load_dotenv()

torch.manual_seed(42)
random.seed(42)


def load_graph():
    with open("data/processed/hetero_graph.pkl", "rb") as f:
        saved = pickle.load(f)
    return saved["data"], saved["patient_idx"], saved["gene_idx"], saved["drug_idx"]


def build_gene_drug_evidence(data, gene_idx, drug_idx):
    """
    Maps gene_index -> {drug_index: evidence_score}, SUMMING scores when a
    gene-drug pair appears via both edge types (e.g. DGIdb support AND CIViC
    resistance for the same pair) -- this preserves conflicting evidence as
    a net signal instead of letting positive evidence silently mask
    documented resistance, matching the retrieval module's logic.
    """
    gene_to_drugs = {}
    for edge_type in [("gene", "targeted_by", "drug"), ("gene", "has_clinical_evidence", "drug")]:
        edge_index = data[edge_type].edge_index
        edge_attr = data[edge_type].edge_attr.squeeze(-1)
        for i in range(edge_index.shape[1]):
            g = edge_index[0, i].item()
            d = edge_index[1, i].item()
            score = edge_attr[i].item()
            gene_to_drugs.setdefault(g, {})
            gene_to_drugs[g][d] = gene_to_drugs[g].get(d, 0.0) + score

    # Clamp to [-1, 1] in case multiple edges push the sum outside the
    # valid evidence score range
    for g in gene_to_drugs:
        for d in gene_to_drugs[g]:
            gene_to_drugs[g][d] = max(-1.0, min(1.0, gene_to_drugs[g][d]))

    return gene_to_drugs


def build_patient_gene_map(data):
    """Maps patient_index -> list of mutated gene_indices."""
    edge_index = data["patient", "has_mutation", "gene"].edge_index
    patient_to_genes = {}
    for i in range(edge_index.shape[1]):
        p = edge_index[0, i].item()
        g = edge_index[1, i].item()
        patient_to_genes.setdefault(p, []).append(g)
    return patient_to_genes


def build_patient_drug_pairs(patient_to_genes, gene_to_drugs, num_drugs, num_patients):
    """
    For each patient, find candidate drugs via their mutated genes (positives,
    labeled with the best evidence score among their genes), and hard-negative
    drugs that exist in the graph but aren't reachable from this patient's genes.
    """
    positives = []  # (patient_idx, drug_idx, score)
    negatives = []  # (patient_idx, drug_idx, 0.0)

    all_drug_indices = set(range(num_drugs))

    for p, genes in patient_to_genes.items():
        reachable_drugs = {}
        for g in genes:
            for d, score in gene_to_drugs.get(g, {}).items():
                reachable_drugs[d] = reachable_drugs.get(d, 0.0) + score

        # Clamp again at the patient level for the same reason
        for d in reachable_drugs:
            reachable_drugs[d] = max(-1.0, min(1.0, reachable_drugs[d]))

        for d, score in reachable_drugs.items():
            positives.append((p, d, score))

        # Hard negatives: drugs NOT reachable by this patient's genes
        unreachable = list(all_drug_indices - set(reachable_drugs.keys()))
        random.shuffle(unreachable)
        num_neg = min(len(reachable_drugs), len(unreachable))  # balance pos/neg per patient
        for d in unreachable[:max(num_neg, 1)]:
            negatives.append((p, d, 0.0))

    return positives, negatives


def main():
    print("Training Evidence-Aware GNN Treatment Ranker (patient-level supervision)")
    print("=" * 70)

    print("\n[1/5] Loading graph...")
    data, patient_idx, gene_idx, drug_idx = load_graph()
    num_patients, num_genes, num_drugs = len(patient_idx), len(gene_idx), len(drug_idx)
    print(f"   Patients: {num_patients}, Genes: {num_genes}, Drugs: {num_drugs}")

    print("\n[2/5] Building gene->drug evidence map and patient->gene map...")
    gene_to_drugs = build_gene_drug_evidence(data, gene_idx, drug_idx)
    patient_to_genes = build_patient_gene_map(data)
    print(f"   Genes with at least one drug link: {len(gene_to_drugs)}")
    print(f"   Patients with at least one mutated gene in KG: {len(patient_to_genes)}")

    print("\n[3/5] Building (patient, drug) training pairs...")
    positives, negatives = build_patient_drug_pairs(patient_to_genes, gene_to_drugs, num_drugs, num_patients)
    print(f"   Positive pairs: {len(positives)}")
    print(f"   Hard negative pairs: {len(negatives)}")

    # Split by PATIENT so we test generalization to unseen patients
    all_patients_with_data = list(patient_to_genes.keys())
    random.shuffle(all_patients_with_data)
    split = int(len(all_patients_with_data) * 0.8)
    train_patients = set(all_patients_with_data[:split])
    val_patients = set(all_patients_with_data[split:])
    print(f"\n   Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")

    all_pairs = positives + negatives

    def to_tensors(pairs):
        p = torch.tensor([x[0] for x in pairs], dtype=torch.long)
        d = torch.tensor([x[1] for x in pairs], dtype=torch.long)
        s = torch.tensor([x[2] for x in pairs], dtype=torch.float)
        return p, d, s

    train_pairs = [x for x in all_pairs if x[0] in train_patients]
    val_pairs = [x for x in all_pairs if x[0] in val_patients]

    train_p, train_d, train_s = to_tensors(train_pairs)
    val_p, val_d, val_s = to_tensors(val_pairs)

    print(f"   Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    print("\n[4/5] Building model...")
    model = TreatmentRanker(hidden_channels=32, out_channels=16, metadata=data.metadata())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    print("\n[5/5] Training...")
    print("-" * 70)

    x_dict = data.x_dict
    edge_index_dict = data.edge_index_dict

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()

        embeddings = model(x_dict, edge_index_dict)
        patient_emb = embeddings["patient"]
        drug_emb = embeddings["drug"]

        preds = model.score_pairs(patient_emb, drug_emb, train_p, train_d)
        loss = F.mse_loss(torch.tanh(preds), train_s)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings = model(x_dict, edge_index_dict)
            patient_emb = embeddings["patient"]
            drug_emb = embeddings["drug"]
            val_preds = model.score_pairs(patient_emb, drug_emb, val_p, val_d)
            val_loss = F.mse_loss(torch.tanh(val_preds), val_s)

        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "train_patients": train_patients,
                "val_patients": val_patients
            }, "data/processed/best_gnn_model.pt")
        else:
            patience += 1
            if patience > 30:
                print(f"\n   Early stopping at epoch {epoch}")
                break

    print("-" * 70)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print("Model saved to data/processed/best_gnn_model.pt")


if __name__ == "__main__":
    main()