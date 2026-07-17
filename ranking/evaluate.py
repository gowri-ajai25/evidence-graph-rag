"""
Evaluates the GNN treatment ranker using ranking metrics:
Precision@K, Recall@K, NDCG@K, MRR, and AUC-ROC.

Ground truth: drugs with net-positive evidence scores for each patient
(from the subgraph retriever) are treated as relevant.
The GNN's ranked output is compared against this ground truth.
"""

from dotenv import load_dotenv
import torch
import pickle
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gnn_model import TreatmentRanker
from retrieval.subgraph_retriever import SubgraphRetriever

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def precision_at_k(ranked_drugs, relevant_drugs, k):
    top_k = ranked_drugs[:k]
    hits = sum(1 for d in top_k if d in relevant_drugs)
    return hits / k


def recall_at_k(ranked_drugs, relevant_drugs, k):
    if not relevant_drugs:
        return 0.0
    top_k = ranked_drugs[:k]
    hits = sum(1 for d in top_k if d in relevant_drugs)
    return hits / len(relevant_drugs)


def ndcg_at_k(ranked_drugs, relevant_drugs, k):
    top_k = ranked_drugs[:k]
    dcg = sum(
        1 / np.log2(i + 2)
        for i, drug in enumerate(top_k)
        if drug in relevant_drugs
    )
    ideal_hits = min(len(relevant_drugs), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(ranked_drugs, relevant_drugs):
    for i, drug in enumerate(ranked_drugs):
        if drug in relevant_drugs:
            return 1 / (i + 1)
    return 0.0


def main():
    print("Evaluating GNN Treatment Ranker")
    print("=" * 60)

    # Load graph and model
    print("\n[1/4] Loading model and graph...")
    with open("data/processed/hetero_graph.pkl", "rb") as f:
        saved = pickle.load(f)
    data = saved["data"]
    patient_idx = saved["patient_idx"]
    drug_idx = saved["drug_idx"]
    idx_to_drug = {v: k for k, v in drug_idx.items()}

    checkpoint = torch.load(
        "data/processed/best_gnn_model.pt", weights_only=False
    )
    model = TreatmentRanker(
        hidden_channels=32, out_channels=16, metadata=data.metadata()
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    val_patients = checkpoint["val_patients"]
    print(f"   Evaluating on {len(val_patients)} held-out validation patients")

    # Get ground truth from retriever
    print("\n[2/4] Building ground truth from knowledge graph...")
    retriever = SubgraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    idx_to_patient = {v: k for k, v in patient_idx.items()}

    ground_truth = {}
    for p_idx in val_patients:
        patient_id = idx_to_patient[p_idx]
        summary = retriever.get_patient_subgraph_summary(patient_id)
        if summary and summary["candidates"]:
            relevant = set(c["drug"] for c in summary["candidates"])
            ground_truth[p_idx] = relevant

    retriever.close()
    print(f"   Patients with at least one relevant drug: {len(ground_truth)}")

    if not ground_truth:
        print("   No ground truth found. Cannot evaluate.")
        return

    # Run GNN inference on all validation patients
    print("\n[3/4] Running GNN inference on validation patients...")
    with torch.no_grad():
        embeddings = model(data.x_dict, data.edge_index_dict)
        patient_emb = embeddings["patient"]
        drug_emb = embeddings["drug"]

    results = {}
    for p_idx in ground_truth:
        num_drugs = len(drug_idx)
        p_indices = torch.full((num_drugs,), p_idx, dtype=torch.long)
        d_indices = torch.arange(num_drugs, dtype=torch.long)

        with torch.no_grad():
            scores = torch.tanh(
                model.score_pairs(patient_emb, drug_emb, p_indices, d_indices)
            )

        ranked = sorted(
            [(idx_to_drug[i], float(scores[i])) for i in range(num_drugs)],
            key=lambda x: x[1], reverse=True
        )
        results[p_idx] = {
            "ranked_drugs": [d for d, _ in ranked],
            "scores": {d: s for d, s in ranked}
        }

    # Compute metrics
    print("\n[4/4] Computing evaluation metrics...")
    print("\n" + "=" * 70)
    print("NDCG DIAGNOSTIC — per-patient breakdown")
    print("=" * 70)

    K_VALUES_DEBUG = [3, 5, 10]

    for p_idx, gt_drugs in ground_truth.items():
        ranked_drugs = results[p_idx]["ranked_drugs"]
        print(f"\nPatient idx={p_idx}  |  num_relevant_drugs={len(gt_drugs)}")

        hit_positions = [i for i, d in enumerate(ranked_drugs) if d in gt_drugs]
        print(f"   Relevant drug positions in full ranking (0-indexed): {hit_positions}")

        for k in K_VALUES_DEBUG:
            top_k = ranked_drugs[:k]
            dcg = sum(
                1 / np.log2(i + 2)
                for i, drug in enumerate(top_k)
                if drug in gt_drugs
            )
            ideal_hits = min(len(gt_drugs), k)
            idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            print(f"   K={k:2d}: DCG={dcg:.4f}  IDCG={idcg:.4f}  ideal_hits={ideal_hits}  NDCG={ndcg:.6f}")

    print("\n" + "=" * 70)

    K_VALUES = [3, 5, 10]
    metrics = {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}
    mrr_scores = []
    auc_scores = []

    for p_idx, gt_drugs in ground_truth.items():
        ranked_drugs = results[p_idx]["ranked_drugs"]
        score_map = results[p_idx]["scores"]

        for k in K_VALUES:
            metrics[k]["precision"].append(
                precision_at_k(ranked_drugs, gt_drugs, k)
            )
            metrics[k]["recall"].append(
                recall_at_k(ranked_drugs, gt_drugs, k)
            )
            metrics[k]["ndcg"].append(
                ndcg_at_k(ranked_drugs, gt_drugs, k)
            )

        mrr_scores.append(mean_reciprocal_rank(ranked_drugs, gt_drugs))

        # AUC-ROC: binary label per drug for this patient
        all_drugs = list(drug_idx.keys())
        y_true = [1 if d in gt_drugs else 0 for d in all_drugs]
        y_score = [score_map.get(d, 0.0) for d in all_drugs]

        if len(set(y_true)) > 1:
            auc_scores.append(roc_auc_score(y_true, y_score))

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"   Evaluated on: {len(ground_truth)} held-out patients\n")

    for k in K_VALUES:
        p = np.mean(metrics[k]["precision"])
        r = np.mean(metrics[k]["recall"])
        n = np.mean(metrics[k]["ndcg"])
        print(f"   @K={k}:")
        print(f"      Precision@{k}:  {p:.4f}")
        print(f"      Recall@{k}:     {r:.4f}")
        print(f"      NDCG@{k}:       {n:.4f}")
        print()

    mrr = np.mean(mrr_scores)
    print(f"   MRR (Mean Reciprocal Rank): {mrr:.4f}")

    if auc_scores:
        auc = np.mean(auc_scores)
        print(f"   AUC-ROC:                    {auc:.4f}")
    else:
        print("   AUC-ROC: not computable (patients with only one class)")

    print("\n" + "=" * 60)
    print("HOW TO INTERPRET THESE NUMBERS")
    print("=" * 60)
    print("   Precision@K: fraction of top-K drugs that are truly relevant")
    print("   Recall@K:    fraction of all relevant drugs found in top-K")
    print("   NDCG@K:      are the most relevant drugs ranked highest? (0-1)")
    print("   MRR:         on average, rank of the first correct drug (0-1)")
    print("   AUC-ROC:     overall separation ability (0.5=random, 1=perfect)")
    print("\n   Note: ground truth = drugs with net-positive evidence scores")
    print("   for each patient's specific mutations in the knowledge graph.")


if __name__ == "__main__":
    main()