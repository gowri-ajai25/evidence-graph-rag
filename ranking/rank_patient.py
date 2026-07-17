"""
Loads the trained GNN and produces a ranked treatment list for a given patient,
using their learned embedding propagated through the graph.
"""

from dotenv import load_dotenv
import torch
import pickle
import os
import sys

sys.path.append(os.path.dirname(__file__))
from gnn_model import TreatmentRanker

load_dotenv()


def load_graph():
    with open("data/processed/hetero_graph.pkl", "rb") as f:
        saved = pickle.load(f)
    return saved["data"], saved["patient_idx"], saved["gene_idx"], saved["drug_idx"]


def main():
    print("Ranking treatments using trained GNN...")
    print("=" * 60)

    data, patient_idx, gene_idx, drug_idx = load_graph()
    idx_to_drug = {v: k for k, v in drug_idx.items()}
    idx_to_gene = {v: k for k, v in gene_idx.items()}

    checkpoint = torch.load("data/processed/best_gnn_model.pt", weights_only=False)
    model = TreatmentRanker(hidden_channels=32, out_channels=16, metadata=data.metadata())
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    val_patients = checkpoint["val_patients"]
    print(f"\nModel was validated on {len(val_patients)} held-out patients (never seen during training)")

    # Find which patient has the most mutated genes among the held-out (validation) set
    mut_edge_index = data["patient", "has_mutation", "gene"].edge_index
    patient_mut_counts = {}
    for i in range(mut_edge_index.shape[1]):
        p = mut_edge_index[0, i].item()
        if p in val_patients:
            patient_mut_counts[p] = patient_mut_counts.get(p, 0) + 1

    if not patient_mut_counts:
        print("No held-out patients with mutations found.")
        return

    target_patient_idx = max(patient_mut_counts, key=patient_mut_counts.get)
    idx_to_patient = {v: k for k, v in patient_idx.items()}
    target_patient_id = idx_to_patient[target_patient_idx]

    # Show this patient's mutated genes
    mutated_genes = [
        idx_to_gene[mut_edge_index[1, i].item()]
        for i in range(mut_edge_index.shape[1])
        if mut_edge_index[0, i].item() == target_patient_idx
    ]
    print(f"\nHeld-out patient: {target_patient_id}")
    print(f"Mutated genes (never used in training for this patient): {mutated_genes}")

    # Run inference - get embeddings for ALL nodes
    with torch.no_grad():
        embeddings = model(data.x_dict, data.edge_index_dict)
        patient_emb = embeddings["patient"]
        drug_emb = embeddings["drug"]

        # Score this patient against every drug in the graph
        num_drugs = len(drug_idx)
        patient_indices = torch.full((num_drugs,), target_patient_idx, dtype=torch.long)
        drug_indices = torch.arange(num_drugs, dtype=torch.long)

        scores = model.score_pairs(patient_emb, drug_emb, patient_indices, drug_indices)
        probs = torch.tanh(scores)

    # Rank drugs by predicted score
    ranked = sorted(
        [(idx_to_drug[i], probs[i].item()) for i in range(num_drugs)],
        key=lambda x: x[1], reverse=True
    )

    print(f"\nTop 10 GNN-ranked treatments for {target_patient_id}:")
    print("-" * 60)
    for rank, (drug, score) in enumerate(ranked[:10], 1):
        print(f"   {rank:2d}. {drug:25s}  score: {score:.4f}")


if __name__ == "__main__":
    main()