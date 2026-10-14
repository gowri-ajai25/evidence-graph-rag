# evaluation/compare_rankers.py
"""
Objective 4: compare treatment rankers on UNSEEN patients, against a ground
truth that is independent of the knowledge-graph scoring.

Rankers
  Random              chance level
  Popularity          patient-agnostic: drugs ordered by how many genes target them
  Rule-based          sum of evidence scores over the patient's mutated genes
                      (the evidence-aggregation rule used by the retriever)
  XGBoost-LTR         XGBRanker (rank:pairwise) on hand-built patient-drug features
  GraphSAGE (patient IDs)
                      the TreatmentRanker from ranking/gnn_model.py with the exported
                      one-hot patient IDs (current ranking/train.py design)
  GraphSAGE (mutation features)
                      same model, but each patient node carries its mutation vector,
                      which is what lets it transfer to unseen patients
  GraphSAGE (gene-path)
                      the TreatmentRanker scoring gene-drug pairs (its documented design);
                      a patient's score is the sum over their mutated genes

Protocol
  - 5-fold cross-validation BY PATIENT: every patient is scored by a model that
    never trained on that patient. Rule-based / popularity / random need no
    training. Learned models (XGBoost-LTR, GraphSAGE) are trained on the KG
    evidence target (the same supervision ranking/train.py uses).
  - Ground truth: data/raw/evaluation/guideline_ground_truth.csv
      standard_of_care   relevant for every patient
      biomarker_directed relevant only if the patient carries a mutation in the
                         trigger gene(s) (e.g. PARP inhibitors for BRCA1/BRCA2)
    Only drugs that have at least one gene link in the graph can be ranked; the
    others are reported and excluded.
  - Metrics: NDCG@K, Precision@K, Recall@K, MRR, with 95% bootstrap CIs over
    patients, and paired bootstrap differences against the rule-based ranker.

Run from the repo root:
  python evaluation/compare_rankers.py
  python evaluation/compare_rankers.py --no-gnn        # skip GraphSAGE (faster)
"""

import os
import sys
import json
import copy
import pickle
import random
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "ranking"))

PROCESSED = os.path.join(ROOT, "data", "processed")
GT_PATH = os.path.join(ROOT, "data", "raw", "evaluation", "guideline_ground_truth.csv")
KS = (3, 5, 10)
N_FOLDS = 5
SEED = 42
N_BOOT = 2000


# ── 1. graph -> plain numpy structures ────────────────────────────────

def _np(t):
    return t.numpy() if hasattr(t, "numpy") else np.asarray(t)


def extract_structures(data, patient_idx, gene_idx, drug_idx):
    """Dense matrices from the exported HeteroData, so every ranker sees the
    same inputs."""
    N, G, D = len(patient_idx), len(gene_idx), len(drug_idx)
    P = len(data["pathway"].node_id)

    M = np.zeros((N, G), dtype=np.float32)
    ei = _np(data["patient", "has_mutation", "gene"].edge_index)
    M[ei[0], ei[1]] = 1.0

    def weights(edge_type):
        W = np.zeros((G, D), dtype=np.float32)
        e = data[edge_type]
        idx, attr = _np(e.edge_index), _np(e.edge_attr).reshape(-1)
        for i in range(idx.shape[1]):
            W[idx[0, i], idx[1, i]] += attr[i]
        return W

    W_dg = weights(("gene", "targeted_by", "drug"))
    W_cv = weights(("gene", "has_clinical_evidence", "drug"))

    GP = np.zeros((G, P), dtype=np.float32)
    e = _np(data["gene", "involved_in", "pathway"].edge_index)
    GP[e[0], e[1]] = 1.0
    PD = np.zeros((P, D), dtype=np.float32)
    e = _np(data["pathway", "associated_with_drug", "drug"].edge_index)
    if e.size:
        PD[e[0], e[1]] = 1.0

    return {
        "patients": [k for k, _ in sorted(patient_idx.items(), key=lambda x: x[1])],
        "genes": [k for k, _ in sorted(gene_idx.items(), key=lambda x: x[1])],
        "drugs": [k for k, _ in sorted(drug_idx.items(), key=lambda x: x[1])],
        "M": M, "W_dg": W_dg, "W_cv": W_cv, "GP": GP, "PD": PD,
    }


# ── 2. features, rule-based score, KG target ──────────────────────────

def combined_weights(S):
    """Gene->drug net evidence, summed over both edge types and clamped, exactly
    as ranking/train.py builds it."""
    return np.clip(S["W_dg"] + S["W_cv"], -1.0, 1.0)


def rule_scores(S):
    """Rule-based ranker: sum of net gene->drug evidence over mutated genes."""
    return np.clip(S["M"] @ combined_weights(S), -1.0, 1.0)


def kg_grades(rule, levels=30):
    """Graded relevance for XGBoost from the KG evidence score:
    0 = no / negative evidence, 1..levels = quantile bins of the positive scores.
    Fine bins matter: after prevalence weighting many positive scores are small
    and close together, and coarse grades (e.g. 5) cannot reproduce their order."""
    positive = rule[rule > 0]
    if positive.size < levels:
        return (rule > 0).astype(int)
    edges = np.unique(np.quantile(positive, np.linspace(0, 1, levels + 1)[1:-1]))
    return np.where(rule > 0, 1 + np.digitize(rule, edges), 0).astype(int)


def build_features(S):
    """Feature tensor [patients, drugs, F] for the learned-to-rank model."""
    M, Wc = S["M"], combined_weights(S)
    N, D = M.shape[0], Wc.shape[1]
    linked, pos, neg = (Wc != 0).astype(np.float32), (Wc > 0).astype(np.float32), (Wc < 0).astype(np.float32)

    sum_dg, sum_cv = M @ S["W_dg"], M @ S["W_cv"]
    n_linked, n_pos, n_neg = M @ linked, M @ pos, M @ neg
    path_hits = (M @ S["GP"]) @ S["PD"]

    max_pos = np.zeros((N, D), dtype=np.float32)
    min_neg = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        genes = np.flatnonzero(M[i])
        if genes.size:
            max_pos[i] = Wc[genes].max(axis=0).clip(min=0)
            min_neg[i] = Wc[genes].min(axis=0).clip(max=0)

    degree = linked.sum(0)
    drug_mean = Wc.sum(0) / np.maximum(degree, 1)
    n_mut = M.sum(1, keepdims=True)

    feats = [
        sum_dg, sum_cv, n_linked, n_pos, n_neg, path_hits, max_pos, min_neg,
        np.broadcast_to(degree, (N, D)), np.broadcast_to(drug_mean, (N, D)),
        np.broadcast_to(n_mut, (N, D)),
    ]
    return np.stack(feats, axis=-1).astype(np.float32)


# ── 3. ground truth ───────────────────────────────────────────────────

def build_ground_truth(S):
    """Relevance matrix [patients, drugs] (bool) from the guideline CSV."""
    try:
        gt = pd.read_csv(GT_PATH)
    except (OSError, pd.errors.ParserError) as e:
        sys.exit(f"Could not read ground truth {GT_PATH}: {e}")

    drug_pos = {d: i for i, d in enumerate(S["drugs"])}
    gene_pos = {g: i for i, g in enumerate(S["genes"])}
    linked = (combined_weights(S) != 0).any(axis=0)

    rel = np.zeros((len(S["patients"]), len(S["drugs"])), dtype=bool)
    used, dropped = [], []
    for r in gt.itertuples():
        d = drug_pos.get(r.drug)
        if d is None or not linked[d]:
            dropped.append(r.drug)
            continue
        used.append(r.drug)
        trig = [] if pd.isna(r.gene_trigger) else [g.strip() for g in str(r.gene_trigger).split(";") if g.strip()]
        if not trig:
            rel[:, d] = True
        else:
            cols = [gene_pos[g] for g in trig if g in gene_pos]
            if cols:
                rel[:, d] = S["M"][:, cols].sum(axis=1) > 0
    return rel, used, dropped


# ── 4. metrics ────────────────────────────────────────────────────────

def rank_matrix(scores, rng):
    """Rank (0 = best) with a seeded random tie-break; scores rounded so only
    genuinely equal scores are treated as ties."""
    noisy = np.round(scores, 6) + rng.uniform(0, 1e-7, scores.shape)
    order = np.argsort(-noisy, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(scores.shape[1])[None, :]
    return ranks


def per_patient_metrics(scores, rel, rng):
    """Dict metric -> array over patients that have at least one relevant drug."""
    ranks = rank_matrix(scores, rng)
    keep = rel.any(axis=1)
    out = {}
    n_rel = rel.sum(axis=1)
    first = np.where(rel, ranks, 10 ** 9).min(axis=1)
    out["MRR"] = np.where(keep, 1.0 / (first + 1), np.nan)
    for k in KS:
        hit = rel & (ranks < k)
        dcg = (hit / np.log2(ranks + 2)).sum(axis=1)
        ideal = np.array([(1 / np.log2(np.arange(2, min(n, k) + 2))).sum() if n else 0.0 for n in n_rel])
        out[f"NDCG@{k}"] = np.where(keep, dcg / np.maximum(ideal, 1e-12), np.nan)
        out[f"P@{k}"] = np.where(keep, hit.sum(axis=1) / k, np.nan)
        out[f"R@{k}"] = np.where(keep, hit.sum(axis=1) / np.maximum(n_rel, 1), np.nan)
    return out


def bootstrap_ci(values, rng, n_boot=N_BOOT):
    v = values[~np.isnan(values)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    means = v[rng.integers(0, v.size, (n_boot, v.size))].mean(axis=1)
    return (float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_diff_ci(a, b, rng, n_boot=N_BOOT):
    ok = ~np.isnan(a) & ~np.isnan(b)
    d = (a - b)[ok]
    if d.size == 0:
        return (np.nan, np.nan, np.nan)
    means = d[rng.integers(0, d.size, (n_boot, d.size))].mean(axis=1)
    return (float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def parp_percentile(scores, S, rel_names=("Olaparib", "Niraparib", "Rucaparib"), rng=None):
    """Mean percentile rank (1 = top) of the PARP inhibitors, per patient."""
    ranks = rank_matrix(scores, rng)
    cols = [S["drugs"].index(d) for d in rel_names if d in S["drugs"]]
    if not cols:
        return np.full(scores.shape[0], np.nan)
    return 1.0 - ranks[:, cols].mean(axis=1) / scores.shape[1]


# ── 5. learned rankers (out-of-fold) ──────────────────────────────────

def make_folds(n, seed=SEED, k=N_FOLDS):
    order = np.random.RandomState(seed).permutation(n)
    return [order[i::k] for i in range(k)]


def xgb_oof(X, labels, folds, label_name):
    """Out-of-fold XGBoost LTR scores. labels: int array [patients, drugs]."""
    try:
        from xgboost import XGBRanker
    except ImportError:
        print("   xgboost is not installed (pip install xgboost); skipping XGBoost-LTR")
        return None
    N, D, F = X.shape
    scores = np.zeros((N, D), dtype=np.float32)
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(N), test)
        model = XGBRanker(objective="rank:pairwise", n_estimators=200, max_depth=4,
                          learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                          random_state=SEED, n_jobs=1)
        model.fit(X[train].reshape(-1, F), labels[train].reshape(-1), group=[D] * len(train))
        scores[test] = model.predict(X[test].reshape(-1, F)).reshape(len(test), D)
        print(f"   {label_name}: fold {f + 1}/{len(folds)} done")
    return scores


def gnn_oof(data, patient_idx, gene_idx, drug_idx, folds, patient_x=None, label="GraphSAGE"):
    """Out-of-fold GraphSAGE scores. Uses the same model, pair construction and
    loss as ranking/train.py; early stopping uses an inner 10% of the training
    patients, never the held-out fold.

    patient_x=None keeps the exported one-hot patient IDs (the design in
    ranking/export_graph.py). A one-hot ID of a held-out patient has no trained
    weights, so that variant cannot tell unseen patients apart. Passing a
    [patients x genes] mutation matrix as patient_x gives the model features
    that transfer to unseen patients."""
    try:
        import torch
        import torch.nn.functional as F
        from gnn_model import TreatmentRanker
        import train as T
    except ImportError as e:
        print(f"   GraphSAGE skipped ({e})")
        return None

    random.seed(SEED)
    torch.manual_seed(SEED)
    N, D = len(patient_idx), len(drug_idx)

    gene_to_drugs = T.build_gene_drug_evidence(data, gene_idx, drug_idx)
    patient_to_genes = T.build_patient_gene_map(data)
    positives, negatives = T.build_patient_drug_pairs(patient_to_genes, gene_to_drugs, D, N)
    all_pairs = positives + negatives

    def tensors(pairs):
        return (torch.tensor([x[0] for x in pairs], dtype=torch.long),
                torch.tensor([x[1] for x in pairs], dtype=torch.long),
                torch.tensor([x[2] for x in pairs], dtype=torch.float))

    scores = np.zeros((N, D), dtype=np.float32)
    x_dict, ei_dict = dict(data.x_dict), data.edge_index_dict
    if patient_x is not None:
        x_dict["patient"] = torch.tensor(patient_x, dtype=torch.float)

    for f, test in enumerate(folds):
        train_all = [p for p in np.setdiff1d(np.arange(N), test) if p in patient_to_genes]
        rs = np.random.RandomState(SEED + f)
        rs.shuffle(train_all)
        n_inner = max(1, int(0.1 * len(train_all)))
        inner_val, train_set = set(train_all[:n_inner]), set(train_all[n_inner:])

        tp, td, ts = tensors([x for x in all_pairs if x[0] in train_set])
        vp, vd, vs = tensors([x for x in all_pairs if x[0] in inner_val])

        model = TreatmentRanker(hidden_channels=32, out_channels=16, metadata=data.metadata())
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        best, best_state, wait = float("inf"), None, 0

        for epoch in range(1, 201):
            model.train()
            opt.zero_grad()
            emb = model(x_dict, ei_dict)
            loss = F.mse_loss(torch.tanh(model.score_pairs(emb["patient"], emb["drug"], tp, td)), ts)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                emb = model(x_dict, ei_dict)
                vloss = F.mse_loss(torch.tanh(model.score_pairs(emb["patient"], emb["drug"], vp, vd)), vs).item()
            if vloss < best:
                best, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait > 30:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            emb = model(x_dict, ei_dict)
            for p in test:
                pi = torch.full((D,), int(p), dtype=torch.long)
                di = torch.arange(D, dtype=torch.long)
                scores[p] = torch.tanh(model.score_pairs(emb["patient"], emb["drug"], pi, di)).numpy()
        print(f"   {label}: fold {f + 1}/{len(folds)} done (epochs {epoch}, inner val {best:.4f})")
    return scores


def gnn_gene_path_scores(data, S, epochs=300, patience=30):
    """GraphSAGE scored through the patient's mutated genes.

    The patient-ID and mutation-feature variants score (patient, drug) from a
    patient embedding, which collapses to the same ordering for every patient.
    Here the TreatmentRanker scores GENE-drug pairs (its documented design),
    trained to predict the net gene->drug evidence from the graph embeddings,
    and a patient's score is the clamped sum over their mutated genes. It is
    patient-specific by construction and uses no patient labels, so held-out
    patients are unseen by definition. On edges seen in training it approximates
    the rule-based sum; what it adds is the smoothing across pathway / GO
    structure learned by message passing."""
    try:
        import torch
        import torch.nn.functional as F
        from gnn_model import TreatmentRanker
    except ImportError as e:
        print(f"   GraphSAGE (gene-path) skipped ({e})")
        return None

    torch.manual_seed(SEED)
    rs = np.random.RandomState(SEED)
    Wc = combined_weights(S)
    G, D = Wc.shape
    gi, di = np.nonzero(Wc)
    ni = np.nonzero(Wc == 0)
    pick = rs.choice(len(ni[0]), size=len(gi), replace=False)
    pairs = np.concatenate([
        np.stack([gi, di, Wc[gi, di]], axis=1),
        np.stack([ni[0][pick], ni[1][pick], np.zeros(len(gi))], axis=1),
    ])
    rs.shuffle(pairs)
    n_val = max(1, int(0.1 * len(pairs)))
    val, train = pairs[:n_val], pairs[n_val:]

    def t(a):
        return (torch.tensor(a[:, 0], dtype=torch.long), torch.tensor(a[:, 1], dtype=torch.long),
                torch.tensor(a[:, 2], dtype=torch.float))

    tg, td, ts = t(train)
    vg, vd, vs = t(val)
    model = TreatmentRanker(hidden_channels=32, out_channels=16, metadata=data.metadata())
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    x_dict, ei_dict = data.x_dict, data.edge_index_dict
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        emb = model(x_dict, ei_dict)
        loss = F.mse_loss(torch.tanh(model.score_pairs(emb["gene"], emb["drug"], tg, td)), ts)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            emb = model(x_dict, ei_dict)
            vloss = F.mse_loss(torch.tanh(model.score_pairs(emb["gene"], emb["drug"], vg, vd)), vs).item()
        if vloss < best:
            best, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait > patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb = model(x_dict, ei_dict)
        gg = torch.arange(G).repeat_interleave(D)
        dd = torch.arange(D).repeat(G)
        gene_scores = torch.tanh(model.score_pairs(emb["gene"], emb["drug"], gg, dd)).reshape(G, D).numpy()
    print(f"   GraphSAGE (gene-path): trained {epoch} epochs (val loss {best:.4f})")
    return np.clip(S["M"] @ gene_scores, -1.0, 1.0)


# ── 6. reporting ──────────────────────────────────────────────────────

def summarise(name_to_scores, rel, subset, rng, title):
    """Prints and returns mean [95% CI] per ranker and paired diff vs Rule-based."""
    sub_rel = rel[subset]
    n = int(sub_rel.any(axis=1).sum())
    print(f"\n{title}  (n = {n} patients)")
    metrics = ["NDCG@10", "P@5", "R@10", "MRR"]
    header = f"   {'Ranker':<26}" + "".join(f"{m:>26}" for m in metrics)
    print(header)
    print("   " + "-" * (len(header) - 3))

    per = {name: per_patient_metrics(s[subset], sub_rel, np.random.default_rng(SEED))
           for name, s in name_to_scores.items()}
    result = {"n_patients": n, "rankers": {}}
    for name, m in per.items():
        row, cells = {}, []
        for k in metrics:
            mean, lo, hi = bootstrap_ci(m[k], rng)
            row[k] = {"mean": mean, "ci_low": lo, "ci_high": hi}
            cells.append(f"{mean:6.3f} [{lo:5.3f},{hi:5.3f}]".rjust(26))
        if "Rule-based" in per and name != "Rule-based":
            row["NDCG@10_diff_vs_rule"] = dict(zip(("mean", "ci_low", "ci_high"),
                                                   paired_diff_ci(m["NDCG@10"], per["Rule-based"]["NDCG@10"], rng)))
        result["rankers"][name] = row
        print(f"   {name:<26}" + "".join(cells))

    if "Rule-based" in per:
        print("\n   Paired NDCG@10 difference vs Rule-based (positive = better than rule-based):")
        for name, row in result["rankers"].items():
            d = row.get("NDCG@10_diff_vs_rule")
            if d:
                sig = "CI excludes 0" if (d["ci_low"] > 0 or d["ci_high"] < 0) else "no clear difference"
                print(f"   {name:<26}{d['mean']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]  {sig}")
    return result


def biomarker_specificity(name_to_scores, S):
    """PARP-inhibitor percentile rank in BRCA1/2-mutated vs other patients."""
    cols = [S["genes"].index(g) for g in ("BRCA1", "BRCA2") if g in S["genes"]]
    mut = S["M"][:, cols].sum(axis=1) > 0
    print(f"\nBiomarker specificity: PARP-inhibitor percentile rank "
          f"(1.00 = top), BRCA1/2-mutated n={int(mut.sum())} vs others n={int((~mut).sum())}")
    print(f"   {'Ranker':<26}{'BRCA-mut':>10}{'others':>10}{'gap':>9}")
    out = {}
    for name, s in name_to_scores.items():
        pct = parp_percentile(s, S, rng=np.random.default_rng(SEED))
        a, b = float(np.nanmean(pct[mut])) if mut.any() else float("nan"), float(np.nanmean(pct[~mut]))
        out[name] = {"brca_mut": a, "others": b, "gap": a - b}
        print(f"   {name:<26}{a:>10.3f}{b:>10.3f}{a - b:>+9.3f}")
    return out


def plot_results(result, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = list(result["rankers"])
    means = [result["rankers"][n]["NDCG@10"]["mean"] for n in names]
    err = [[m - result["rankers"][n]["NDCG@10"]["ci_low"] for n, m in zip(names, means)],
           [result["rankers"][n]["NDCG@10"]["ci_high"] - m for n, m in zip(names, means)]]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(names[::-1], means[::-1], xerr=[e[::-1] for e in err], color="#3b6ea5", capsize=3)
    ax.set_xlabel("NDCG@10 vs guideline ground truth (mean, 95% CI)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ── 7. main ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gnn", action="store_true", help="skip the GraphSAGE ranker")
    args = ap.parse_args()

    print("Objective 4: ranker comparison on unseen patients (5-fold CV by patient)")
    print("=" * 74)

    print("\n[1/5] Loading graph...")
    try:
        with open(os.path.join(PROCESSED, "hetero_graph.pkl"), "rb") as fh:
            saved = pickle.load(fh)
    except (OSError, pickle.UnpicklingError, ModuleNotFoundError) as e:
        sys.exit(f"Could not load hetero_graph.pkl ({e}). Run ranking/export_graph.py first.")
    data, patient_idx, gene_idx, drug_idx = saved["data"], saved["patient_idx"], saved["gene_idx"], saved["drug_idx"]
    S = extract_structures(data, patient_idx, gene_idx, drug_idx)
    print(f"   {len(S['patients'])} patients, {len(S['genes'])} genes, {len(S['drugs'])} drugs")

    print("\n[2/5] Building ground truth...")
    rel, used, dropped = build_ground_truth(S)
    print(f"   Rankable ground-truth drugs ({len(used)}): {', '.join(used)}")
    if dropped:
        print(f"   Excluded (not in graph or no gene link): {', '.join(dropped)}")
    print(f"   Patients with at least one relevant drug: {int(rel.any(axis=1).sum())}")

    print("\n[3/5] Rule-based, popularity, random...")
    rng = np.random.default_rng(SEED)
    rule = rule_scores(S)
    popularity = np.broadcast_to((combined_weights(S) != 0).sum(0).astype(np.float32), rule.shape).copy()
    scores = {
        "Random": rng.uniform(size=rule.shape),
        "Popularity (patient-agnostic)": popularity,
        "Rule-based": rule,
    }

    folds = make_folds(len(S["patients"]))

    print("\n[4/5] Learned rankers (out-of-fold)...")
    X = build_features(S)
    xgb = xgb_oof(X, kg_grades(rule), folds, "XGBoost-LTR")
    if xgb is not None:
        scores["XGBoost-LTR"] = xgb
    if not args.no_gnn:
        gnn = gnn_oof(data, patient_idx, gene_idx, drug_idx, folds,
                      label="GraphSAGE (patient IDs)")
        if gnn is not None:
            scores["GraphSAGE (patient IDs)"] = gnn
            scores["GraphSAGE (mutation features)"] = gnn_oof(
                data, patient_idx, gene_idx, drug_idx, folds,
                patient_x=S["M"], label="GraphSAGE (mutation features)")
            path = gnn_gene_path_scores(data, S)
            if path is not None:
                scores["GraphSAGE (gene-path)"] = path

    print("\n[5/5] Evaluating...")
    everyone = np.arange(len(S["patients"]))
    cols = [S["genes"].index(g) for g in ("BRCA1", "BRCA2") if g in S["genes"]]
    brca = np.flatnonzero(S["M"][:, cols].sum(axis=1) > 0)

    boot_rng = np.random.default_rng(SEED)
    res_all = summarise(scores, rel, everyone, boot_rng, "ALL PATIENTS")
    res_brca = summarise(scores, rel, brca, boot_rng, "BRCA1/BRCA2-MUTATED PATIENTS") if brca.size else None
    spec = biomarker_specificity(scores, S)

    out = {"folds": N_FOLDS, "seed": SEED, "ground_truth_drugs": used, "excluded": dropped,
           "all_patients": res_all, "brca_mutated": res_brca, "parp_specificity": spec}
    os.makedirs(PROCESSED, exist_ok=True)
    out_path = os.path.join(PROCESSED, "objective4_results.json")
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nSaved {out_path}")
    except OSError as e:
        print(f"\nCould not save results: {e}")
    plot_results(res_all, os.path.join(PROCESSED, "objective4_ndcg.png"))

    print("\nNotes")
    print("   - The standard-of-care ground truth is patient-agnostic, so 'Popularity' is a strong")
    print("     reference on 'ALL PATIENTS'. Patient-specific behaviour is what the BRCA1/BRCA2")
    print("     subgroup and the PARP-inhibitor specificity table measure.")
    print("   - Learned models are trained on the KG evidence target; the rule-based ranker is that")
    print("     target. Differences therefore show what learning adds beyond the rule.")


if __name__ == "__main__":
    main()