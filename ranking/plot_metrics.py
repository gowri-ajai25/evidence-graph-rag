"""
Generates evaluation plots for the GNN treatment ranker:
1. Training loss curve (train vs validation)
2. Precision, Recall, NDCG at different K values (bar chart)
3. ROC curve (AUC)
4. Baseline comparison bar chart
5. Ranking score distribution
"""

from dotenv import load_dotenv
import torch
import pickle
import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gnn_model import TreatmentRanker
from retrieval.subgraph_retriever import SubgraphRetriever

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

os.makedirs("data/plots", exist_ok=True)
random.seed(42)
np.random.seed(42)

# ── Helper metric functions ───────────────────────────────────

def precision_at_k(ranked, relevant, k):
    return sum(1 for d in ranked[:k] if d in relevant) / k

def recall_at_k(ranked, relevant, k):
    if not relevant: return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)

def ndcg_at_k(ranked, relevant, k):
    dcg = sum(1/np.log2(i+2) for i, d in enumerate(ranked[:k]) if d in relevant)
    idcg = sum(1/np.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def mrr(ranked, relevant):
    for i, d in enumerate(ranked):
        if d in relevant: return 1/(i+1)
    return 0.0


def load_everything():
    with open("data/processed/hetero_graph.pkl", "rb") as f:
        saved = pickle.load(f)
    data = saved["data"]
    patient_idx = saved["patient_idx"]
    drug_idx = saved["drug_idx"]
    idx_to_drug = {v: k for k, v in drug_idx.items()}
    idx_to_patient = {v: k for k, v in patient_idx.items()}
    drug_list = list(drug_idx.keys())

    checkpoint = torch.load("data/processed/best_gnn_model.pt", weights_only=False)
    model = TreatmentRanker(hidden_channels=32, out_channels=16, metadata=data.metadata())
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    val_patients = checkpoint["val_patients"]

    return data, patient_idx, drug_idx, idx_to_drug, idx_to_patient, drug_list, model, val_patients, checkpoint


def get_ground_truth(val_patients, idx_to_patient):
    retriever = SubgraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    ground_truth = {}
    evidence_per_patient = {}

    for p_idx in val_patients:
        patient_id = idx_to_patient[p_idx]
        summary = retriever.get_patient_subgraph_summary(patient_id)
        if summary and summary["candidates"]:
            ground_truth[p_idx] = set(c["drug"] for c in summary["candidates"])
            ev = {}
            for c in summary["candidates"]:
                # use .get() with fallback to handle varying key names
                score = c.get("max_evidence_score") or c.get("evidence_score") or 0.3
                ev[c["drug"]] = score
            for c in summary.get("contraindications", []):
                score = c.get("min_evidence_score") or c.get("evidence_score") or -0.3
                ev[c["drug"]] = score
            evidence_per_patient[p_idx] = ev

    retriever.close()
    return ground_truth, evidence_per_patient


def get_gnn_scores(model, data, patient_idx_map, drug_idx_map, idx_to_drug, ground_truth):
    drug_list = list(drug_idx_map.keys())
    with torch.no_grad():
        embeddings = model(data.x_dict, data.edge_index_dict)
        patient_emb = embeddings["patient"]
        drug_emb = embeddings["drug"]

    results = {}
    for p_idx in ground_truth:
        num_drugs = len(drug_idx_map)
        p_indices = torch.full((num_drugs,), p_idx, dtype=torch.long)
        d_indices = torch.arange(num_drugs, dtype=torch.long)
        with torch.no_grad():
            scores = torch.tanh(model.score_pairs(patient_emb, drug_emb, p_indices, d_indices))
        score_map = {idx_to_drug[i]: float(scores[i]) for i in range(num_drugs)}
        ranked = sorted(drug_list, key=lambda d: score_map[d], reverse=True)
        results[p_idx] = {"ranked": ranked, "scores": score_map}
    return results


# ── PLOT 1: Training loss curve ───────────────────────────────

def plot_training_curve():
    """
    Re-runs a short training loop recording loss at each epoch
    so we can plot the actual learning curve.
    """
    print("   Re-running training to capture loss curve...")

    with open("data/processed/hetero_graph.pkl", "rb") as f:
        saved = pickle.load(f)
    data = saved["data"]
    patient_idx = saved["patient_idx"]
    drug_idx = saved["drug_idx"]

    checkpoint = torch.load("data/processed/best_gnn_model.pt", weights_only=False)
    val_patients = checkpoint["val_patients"]
    idx_to_patient = {v: k for k, v in patient_idx.items()}
    idx_to_drug = {v: k for k, v in drug_idx.items()}

    # Build training pairs (same logic as train.py)
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
    for g in gene_to_drugs:
        for d in gene_to_drugs[g]:
            gene_to_drugs[g][d] = max(-1.0, min(1.0, gene_to_drugs[g][d]))

    mut_edge = data["patient", "has_mutation", "gene"].edge_index
    patient_to_genes = {}
    for i in range(mut_edge.shape[1]):
        p = mut_edge[0, i].item()
        g = mut_edge[1, i].item()
        patient_to_genes.setdefault(p, []).append(g)

    all_patients = list(patient_to_genes.keys())
    random.shuffle(all_patients)
    split = int(len(all_patients) * 0.8)
    train_patients = set(all_patients[:split])
    val_patients_plot = set(all_patients[split:])

    num_drugs = len(drug_idx)
    all_drug_indices = set(range(num_drugs))

    def make_pairs(patients):
        pairs = []
        for p, genes in patient_to_genes.items():
            if p not in patients: continue
            reachable = {}
            for g in genes:
                for d, score in gene_to_drugs.get(g, {}).items():
                    reachable[d] = reachable.get(d, 0.0) + score
            for d in reachable:
                reachable[d] = max(-1.0, min(1.0, reachable[d]))
            for d, score in reachable.items():
                pairs.append((p, d, score))
            unreachable = list(all_drug_indices - set(reachable.keys()))
            random.shuffle(unreachable)
            for d in unreachable[:max(len(reachable), 1)]:
                pairs.append((p, d, 0.0))
        return pairs

    train_pairs = make_pairs(train_patients)
    val_pairs_list = make_pairs(val_patients_plot)

    def to_tensors(pairs):
        return (
            torch.tensor([x[0] for x in pairs], dtype=torch.long),
            torch.tensor([x[1] for x in pairs], dtype=torch.long),
            torch.tensor([x[2] for x in pairs], dtype=torch.float)
        )

    train_p, train_d, train_s = to_tensors(train_pairs)
    val_p, val_d, val_s = to_tensors(val_pairs_list)

    model = TreatmentRanker(hidden_channels=32, out_channels=16, metadata=data.metadata())
    torch.manual_seed(42)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    train_losses, val_losses = [], []

    for epoch in range(1, 151):
        model.train()
        optimizer.zero_grad()
        emb = model(data.x_dict, data.edge_index_dict)
        preds = model.score_pairs(emb["patient"], emb["drug"], train_p, train_d)
        loss = torch.nn.functional.mse_loss(torch.tanh(preds), train_s)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            emb = model(data.x_dict, data.edge_index_dict)
            val_preds = model.score_pairs(emb["patient"], emb["drug"], val_p, val_d)
            val_loss = torch.nn.functional.mse_loss(torch.tanh(val_preds), val_s)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        # Early stop
        if epoch > 30 and val_losses[-1] > min(val_losses[:-1]) and \
           all(val_losses[-i] > val_losses[-30] for i in range(1, 10)):
            break

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, label="Train loss", color="#2196F3", linewidth=2)
    ax.plot(epochs, val_losses, label="Validation loss", color="#E91E63", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("GNN Training Loss Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))
    plt.tight_layout()
    plt.savefig("data/plots/1_training_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: data/plots/1_training_loss_curve.png")


# ── PLOT 2: Precision, Recall, NDCG bar chart ─────────────────

def plot_ranking_metrics(ground_truth, gnn_results, drug_list):
    K_VALUES = [3, 5, 10]
    metrics = {"Precision": [], "Recall": [], "NDCG": []}

    for k in K_VALUES:
        p_scores, r_scores, n_scores = [], [], []
        for p_idx, gt in ground_truth.items():
            ranked = gnn_results[p_idx]["ranked"]
            p_scores.append(precision_at_k(ranked, gt, k))
            r_scores.append(recall_at_k(ranked, gt, k))
            n_scores.append(ndcg_at_k(ranked, gt, k))
        metrics["Precision"].append(np.mean(p_scores))
        metrics["Recall"].append(np.mean(r_scores))
        metrics["NDCG"].append(np.mean(n_scores))

    x = np.arange(len(K_VALUES))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision, Recall, NDCG at K=3, 5, 10", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"K={k}" for k in K_VALUES])
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/plots/2_precision_recall_ndcg.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: data/plots/2_precision_recall_ndcg.png")


# ── PLOT 3: ROC Curve ─────────────────────────────────────────

def plot_roc_curve(ground_truth, gnn_results, drug_list):
    all_y_true, all_y_score = [], []

    for p_idx, gt in ground_truth.items():
        score_map = gnn_results[p_idx]["scores"]
        y_true = [1 if d in gt else 0 for d in drug_list]
        y_score = [score_map.get(d, 0.0) for d in drug_list]
        all_y_true.extend(y_true)
        all_y_score.extend(y_score)

    fpr, tpr, _ = roc_curve(all_y_true, all_y_score)
    auc = roc_auc_score(all_y_true, all_y_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2.5, label=f"GNN ranker (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#999", linestyle="--", linewidth=1.5, label="Random baseline (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — GNN Treatment Ranker", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("data/plots/3_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: data/plots/3_roc_curve.png")


# ── PLOT 4: Baseline comparison ───────────────────────────────

def plot_baseline_comparison(ground_truth, gnn_results, evidence_per_patient, drug_list):
    K_VALUES = [3, 5, 10]

    def score_method(method):
        results = {}
        for p_idx, gt in ground_truth.items():
            if method == "random":
                shuffled = drug_list.copy()
                random.shuffle(shuffled)
                score_map = {d: random.random() for d in drug_list}
                results[p_idx] = {"ranked": shuffled, "scores": score_map}
            elif method == "evidence":
                ev = evidence_per_patient.get(p_idx, {})
                score_map = {d: ev.get(d, 0.0) for d in drug_list}
                ranked = sorted(drug_list, key=lambda d: score_map[d], reverse=True)
                results[p_idx] = {"ranked": ranked, "scores": score_map}
        return results

    random_results = score_method("random")
    evidence_results = score_method("evidence")

    def avg_metric(results, metric_fn, k):
        return np.mean([metric_fn(results[p]["ranked"], gt, k)
                        for p, gt in ground_truth.items()])

    methods = ["Random", "Evidence Score", "GNN (ours)"]
    colors = ["#9E9E9E", "#FF9800", "#2196F3"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metric_fns = [precision_at_k, recall_at_k, ndcg_at_k]
    metric_names = ["Precision@K", "Recall@K", "NDCG@K"]

    for ax, fn, name in zip(axes, metric_fns, metric_names):
        for k_i, k in enumerate(K_VALUES):
            vals = [
                avg_metric(random_results, fn, k),
                avg_metric(evidence_results, fn, k),
                avg_metric(gnn_results, fn, k)
            ]
            x = np.arange(len(methods))
            bars = ax.bar(x + k_i * 0.25, vals, 0.22,
                          color=colors, alpha=0.8 - k_i * 0.15)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(len(methods)) + 0.25)
        ax.set_xticklabels(methods, fontsize=9, rotation=10)
        ax.set_ylim(0, 0.65)
        ax.grid(True, axis="y", alpha=0.3)

        patches = [mpatches.Patch(color=c, label=f"K={k}")
                   for c, k in zip(["#333", "#666", "#999"], K_VALUES)]
        ax.legend(handles=patches, fontsize=8)

    fig.suptitle("Baseline Comparison: Random vs Evidence Score vs GNN",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/plots/4_baseline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: data/plots/4_baseline_comparison.png")


# ── PLOT 5: Score distribution ────────────────────────────────

def plot_score_distribution(ground_truth, gnn_results):
    positive_scores = []
    negative_scores = []

    for p_idx, gt in ground_truth.items():
        score_map = gnn_results[p_idx]["scores"]
        for drug, score in score_map.items():
            if drug in gt:
                positive_scores.append(score)
            else:
                negative_scores.append(score)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(negative_scores, bins=25, alpha=0.7, color="#E53935",
            label="Non-relevant drugs", density=True)
    ax.hist(positive_scores, bins=15, alpha=0.8, color="#43A047",
            label="Relevant drugs (ground truth)", density=True)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.6, label="Score = 0")
    ax.set_xlabel("GNN Predicted Score (tanh)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("GNN Score Distribution: Relevant vs Non-relevant Drugs",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/plots/5_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: data/plots/5_score_distribution.png")


# ── MAIN ──────────────────────────────────────────────────────

def main():
    print("Generating evaluation plots...")
    print("=" * 60)

    print("\n[1/6] Loading model and graph...")
    (data, patient_idx, drug_idx, idx_to_drug,
     idx_to_patient, drug_list, model, val_patients, checkpoint) = load_everything()

    print("\n[2/6] Building ground truth from Neo4j...")
    ground_truth, evidence_per_patient = get_ground_truth(val_patients, idx_to_patient)
    print(f"   Patients with ground truth: {len(ground_truth)}")

    print("\n[3/6] Running GNN inference...")
    gnn_results = get_gnn_scores(model, data, patient_idx, drug_idx, idx_to_drug, ground_truth)

    print("\n[4/6] Plot 1 — Training loss curve...")
    plot_training_curve()

    print("\n[5/6] Plot 2-5 — Metric plots...")
    plot_ranking_metrics(ground_truth, gnn_results, drug_list)
    plot_roc_curve(ground_truth, gnn_results, drug_list)
    plot_baseline_comparison(ground_truth, gnn_results, evidence_per_patient, drug_list)
    plot_score_distribution(ground_truth, gnn_results)

    print("\n[6/6] Done!")
    print("=" * 60)
    print("All plots saved to data/plots/:")
    print("   1_training_loss_curve.png")
    print("   2_precision_recall_ndcg.png")
    print("   3_roc_curve.png")
    print("   4_baseline_comparison.png")
    print("   5_score_distribution.png")


if __name__ == "__main__":
    main()