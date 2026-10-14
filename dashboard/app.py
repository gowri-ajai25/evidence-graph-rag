# dashboard/app.py
"""
Streamlit front end for the Evidence-Aware Graph-RAG ovarian cancer treatment
ranking pipeline.

This connects directly to the project's real components -- no mock data:
- retrieval/subgraph_retriever.py  -> live Neo4j queries
- ranking/gnn_model.py + data/processed/best_gnn_model.pt -> trained GraphSAGE ranker
- explanation/llm_explainer.py     -> Groq / Llama-3.3-70B grounded explanations
(falls back to the pre-generated data/processed/explanations_*.json
files when a live Groq call isn't wanted, and generates fresh ones
on demand via a button)

Gene Ontology (GO) function terms retrieved from the knowledge graph are
shown alongside genes as functional context.

Run from the repo root (not from inside dashboard/):
    streamlit run dashboard/app.py
"""

import io
import os
import sys
import json
import glob
import pickle
import datetime

import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup -- import the real pipeline modules, don't reimplement them
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "retrieval"), os.path.join(ROOT, "ranking"), os.path.join(ROOT, "explanation"), os.path.join(ROOT, "report")):
    if p not in sys.path:
        sys.path.append(p)

from subgraph_retriever import SubgraphRetriever          # noqa: E402
from gnn_model import TreatmentRanker                      # noqa: E402
from llm_explainer import explain_drug                     # noqa: E402
from evidence_links import EvidenceLinker                  # noqa: E402

# PDF export needs reportlab; the rest of the dashboard works without it
try:
    from build_report import build_pdf                     # noqa: E402
    REPORT_ERROR = None
except Exception as _e:  # noqa: BLE001
    build_pdf, REPORT_ERROR = None, str(_e)

load_dotenv(os.path.join(ROOT, ".env"))

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GRAPH_PATH = os.path.join(ROOT, "data", "processed", "hetero_graph.pkl")
MODEL_PATH = os.path.join(ROOT, "data", "processed", "best_gnn_model.pt")
PLOTS_DIR = os.path.join(ROOT, "data", "plots")

# ---------------------------------------------------------------------------
# Palette -- fixed categorical order (Patient/Gene/Drug) + reserved status colors
# ---------------------------------------------------------------------------
COLOR_PATIENT = "#2a78d6"
COLOR_GENE = "#eb6834"
COLOR_DRUG = "#1baf7a"
COLOR_GOOD = "#0ca30c"       # candidate / supporting evidence
COLOR_CRITICAL = "#d03b3b"   # contraindication / opposing evidence
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"

st.set_page_config(
    page_title="Evidence-Aware Graph-RAG — Ovarian Cancer",
    layout="wide",
)

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #fcfcfb;
        border: 1px solid #e1e0d9;
        border-radius: 8px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached backend connections
# ---------------------------------------------------------------------------


@st.cache_resource
def get_retriever():
    if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
        return None, "NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD not set in .env"
    try:
        r = SubgraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        r.driver.verify_connectivity()
        return r, None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


@st.cache_resource
def load_gnn_and_graph():
    with open(GRAPH_PATH, "rb") as f:
        saved = pickle.load(f)
    data = saved["data"]
    patient_idx, gene_idx, drug_idx = saved["patient_idx"], saved["gene_idx"], saved["drug_idx"]

    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location="cpu")
    model = TreatmentRanker(hidden_channels=32, out_channels=16, metadata=data.metadata())
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    val_patients = set(checkpoint.get("val_patients", []))
    train_patients = set(checkpoint.get("train_patients", []))
    return model, data, patient_idx, drug_idx, val_patients, train_patients


@st.cache_resource
def load_vocab():
    """Gene and drug vocabularies of the knowledge graph, used by the faithfulness verifier."""
    try:
        with open(GRAPH_PATH, "rb") as f:
            saved = pickle.load(f)
        return list(saved["gene_idx"]), list(saved["drug_idx"])
    except Exception:  # noqa: BLE001
        return [], []


@st.cache_data(ttl=300)
def get_patient_list(_retriever, limit=40):
    return _retriever.list_available_patients(limit=limit)


def gnn_rank_all_drugs(model, data, patient_idx, drug_idx, patient_id):
    """Mirrors ranking/rank_patient.py's inference exactly."""
    if patient_id not in patient_idx:
        return None
    idx_to_drug = {v: k for k, v in drug_idx.items()}
    p_idx = patient_idx[patient_id]
    with torch.no_grad():
        embeddings = model(data.x_dict, data.edge_index_dict)
        patient_emb = embeddings["patient"]
        drug_emb = embeddings["drug"]
        n = len(drug_idx)
        pi = torch.full((n,), p_idx, dtype=torch.long)
        di = torch.arange(n, dtype=torch.long)
        scores = model.score_pairs(patient_emb, drug_emb, pi, di)
        probs = torch.tanh(scores)
    ranked = sorted(
        [(idx_to_drug[i], probs[i].item()) for i in range(n)],
        key=lambda x: x[1], reverse=True,
    )
    return ranked


def load_cached_explanations(patient_id):
    """
    Reads any data/processed/explanations_{patient_id}*.json already on disk.
    Returns {drug: entry_dict}. Entries written after faithfulness verification
    was added carry a "verification" block; if the same drug appears in several
    files, the verified entry wins over an older unverified one.
    """
    cache = {}

    def put(entry):
        drug = entry["drug"]
        if drug not in cache or ("verification" in entry and "verification" not in cache[drug]):
            cache[drug] = entry

    pattern = os.path.join(ROOT, "data", "processed", f"explanations_{patient_id}*.json")
    for path in glob.glob(pattern):
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        if "candidates" in payload or "contraindications" in payload:
            for entry in payload.get("candidates", []) + payload.get("contraindications", []):
                if "drug" in entry and "explanation" in entry:
                    put(entry)
        elif "drug" in payload and "explanation" in payload:
            put(payload)
    return cache


def render_explanation(exp):
    """
    Renders one explanation. A str or an entry without a "verification" block is a
    legacy (unverified) explanation. Verified entries show every claim with its
    status, the evidence IDs it cites, and the flags the verifier raised.
    """
    if isinstance(exp, str):
        st.write(exp)
        st.caption("Not verified (generated before faithfulness verification was added)")
        return
    ver = exp.get("verification")
    if ver is None:
        st.write(exp.get("explanation", ""))
        st.caption("Not verified (generated before faithfulness verification was added)")
        return
    if ver.get("faithfulness") is None:
        st.write(exp.get("explanation", ""))
        st.warning(ver.get("error", "This explanation could not be verified."))
        return

    colors = {"supported": "green", "partial": "orange", "unsupported": "red"}
    index = exp.get("evidence_index", {})
    for c in ver["claims"]:
        st.markdown(f"**:{colors[c['status']]}[{c['status'].upper()}]** {c['text']}")
        if c["cites"]:
            parts = []
            for eid in c["cites"]:
                e = index.get(eid)
                parts.append(f"{eid}: {e['gene']} ({e['side'].lower()}, {e['evidence_score']:+.3f})" if e else f"{eid}: not found")
            st.caption("Cites " + " | ".join(parts))
        else:
            st.caption("Cites nothing")
        for issue in c["issues"]:
            st.caption(f"Flag: {issue}")

    st.caption(
        f"Faithfulness {ver['faithfulness']:.0%} - {ver['n_supported']} supported, "
        f"{ver['n_partial']} partial, {ver['n_unsupported']} unsupported."
        + (f" Evidence strength stated: {exp['evidence_strength']}." if exp.get("evidence_strength") else "")
    )


@st.cache_resource
def get_linker():
    """Looks up the real CIViC / PubMed / DGIdb / ChEMBL records behind each graph edge."""
    return EvidenceLinker()


def render_source_links(entries, drug):
    """Clickable source records for each gene's evidence on one drug."""
    lines = []
    for entry, links in get_linker().links_for_entries(entries, drug):
        if not links:
            lines.append(f"- **{entry['gene']}**: no source record matched")
            continue
        parts = []
        for l in links:
            text = f"[{l['label']}]({l['url']})"
            if l.get("detail") and l["kind"] in ("CIViC", "PubMed", "DGIdb"):
                text += f" ({l['detail']})"
            parts.append(text)
        lines.append(f"- **{entry['gene']}**: " + " · ".join(parts))
    if lines:
        st.markdown("**Source records**")
        st.markdown("\n".join(lines))


def evidence_table(entries):
    """Evidence table with a GO function column (top 3 GO terms per gene)."""
    df = pd.DataFrame(entries)
    if "go_terms" in df.columns:
        df["GO function"] = df["go_terms"].apply(
            lambda t: "; ".join(t[:3]) if isinstance(t, list) else ""
        )
    else:
        df["GO function"] = ""
    return df[["gene", "mutation_type", "evidence_score", "evidence_source", "GO function"]]


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def render_subgraph(patient_id, summary, max_drugs=10):
    G = nx.Graph()
    G.add_node(patient_id, kind="patient")

    drug_entries = summary["candidates"][: max_drugs // 2] + summary["contraindications"][: max_drugs // 2]
    candidate_drugs = {c["drug"] for c in summary["candidates"]}

    for entry in drug_entries:
        drug = entry["drug"]
        G.add_node(drug, kind="drug")
        for g in entry.get("supporting_genes", []):
            gene = g["gene"]
            G.add_node(gene, kind="gene")
            G.add_edge(patient_id, gene, kind="mutation")
            G.add_edge(gene, drug, kind="support")
        for g in entry.get("opposing_genes", []):
            gene = g["gene"]
            G.add_node(gene, kind="gene")
            G.add_edge(patient_id, gene, kind="mutation")
            G.add_edge(gene, drug, kind="oppose")

    pos = nx.spring_layout(G, seed=7, k=0.85)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    kind_color = {"patient": COLOR_PATIENT, "gene": COLOR_GENE, "drug": COLOR_DRUG}
    kind_size = {"patient": 900, "gene": 450, "drug": 500}
    node_colors = [kind_color[G.nodes[n]["kind"]] for n in G.nodes]
    node_sizes = [kind_size[G.nodes[n]["kind"]] for n in G.nodes]

    mutation_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "mutation"]
    support_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "support"]
    oppose_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "oppose"]

    nx.draw_networkx_edges(G, pos, edgelist=mutation_edges, edge_color=INK_MUTED, width=1.1, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=support_edges, edge_color=COLOR_GOOD, width=1.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=oppose_edges, edge_color=COLOR_CRITICAL, width=1.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                            edgecolors=SURFACE, linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color=INK, ax=ax)

    ax.axis("off")
    legend_elems = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PATIENT, markersize=10, label="Patient"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_GENE, markersize=10, label="Gene"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_DRUG, markersize=10, label="Drug"),
        plt.Line2D([0], [0], color=COLOR_GOOD, lw=2, label="Supporting evidence"),
        plt.Line2D([0], [0], color=COLOR_CRITICAL, lw=2, label="Opposing evidence"),
    ]
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, frameon=False, fontsize=7, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    return fig


def render_ranking_chart(ranking_df, n_each=5):
    """
    Diverging bar chart of GNN scores: the top candidates (bars to the right)
    and the strongest contraindications (bars to the left), so opposing
    evidence is visible instead of being pushed out of a plain top-N list.
    """
    scored = ranking_df.dropna(subset=["GNN score"])
    candidates = scored[scored["Status"] == "Candidate"].sort_values("GNN score", ascending=False).head(n_each)
    contras = scored[scored["Status"] == "Contraindication"].sort_values("GNN score", ascending=True).head(n_each)
    df = pd.concat([candidates, contras])
    if df.empty:
        return None
    df = df.iloc[::-1]   # barh draws bottom-up; keeps candidates on top

    fig, ax = plt.subplots(figsize=(7, max(2.8, 0.42 * len(df) + 1.0)))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    colors = [COLOR_GOOD if s == "Candidate" else COLOR_CRITICAL for s in df["Status"]]
    bars = ax.barh(df["Drug"], df["GNN score"], color=colors, height=0.6)
    ax.axvline(0, color=INK_MUTED, linewidth=1)
    ax.set_xlabel("GNN-predicted relevance score", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.margins(x=0.18)   # room for the value labels on both sides

    span = float(df["GNN score"].abs().max()) or 1.0
    for bar, val in zip(bars, df["GNN score"]):
        ax.text(
            val + (0.015 * span if val >= 0 else -0.015 * span),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center", ha="left" if val >= 0 else "right",
            fontsize=8, color=INK,
        )

    legend_elems = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_GOOD, label="Candidate (supporting evidence)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_CRITICAL, label="Contraindication (opposing evidence)"),
    ]
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=2, frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    return fig


COLOR_NOT_APPLIED = "#9a9a9a"


def render_evidence_balance_chart(summary, n_candidates=7, n_contra=4, n_literature=4):
    """
    Evidence FOR (green, right) and AGAINST (red, left) each drug, from the
    retrieved knowledge-graph edges. A drug with mixed evidence shows both bars,
    so opposing evidence stays visible even when the drug's net score is positive.
    Grey hatched bars show literature evidence AGAINST a drug that is not applied to
    the score because the gene is mutated in most patients (e.g. TP53 -> platinum).
    Returns (figure, n_opposed, n_not_applied); figure is None if there is no
    evidence to show.
    """
    by_drug = {}
    for key in ("candidates", "contraindications"):
        for entry in summary.get(key, []):
            drug = entry["drug"]
            if drug in by_drug:
                continue
            by_drug[drug] = {
                "drug": drug,
                "support": sum(g["evidence_score"] for g in entry.get("supporting_genes", [])),
                "oppose": sum(g["evidence_score"] for g in entry.get("opposing_genes", [])),
                "net": entry["net_evidence_score"],
                "not_applied": sum(g.get("raw_score", 0) for g in entry.get("non_discriminating_genes", [])
                                   if g.get("raw_score", 0) < 0),
            }
    if not by_drug:
        return None, 0, 0

    rows = sorted(by_drug.values(), key=lambda r: r["net"], reverse=True)
    n_opposed = sum(1 for r in rows if r["oppose"] < 0)
    top = rows[:n_candidates]
    worst = [r for r in sorted(rows, key=lambda r: r["net"]) if r["oppose"] < 0][:n_contra]
    chosen = top + [r for r in worst if r not in top]
    lit = [r for r in rows if r["not_applied"] < 0]
    n_not_applied = len(lit)
    lit = sorted(lit, key=lambda r: r["not_applied"])[:n_literature]
    chosen += [r for r in lit if r not in chosen]
    chosen.sort(key=lambda r: r["net"], reverse=True)
    chosen = chosen[::-1]   # barh draws bottom-up; best drug on top

    fig, ax = plt.subplots(figsize=(7, max(2.8, 0.42 * len(chosen) + 1.2)))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    names = [r["drug"] for r in chosen]
    ax.barh(names, [r["support"] for r in chosen], color=COLOR_GOOD, height=0.6)
    ax.barh(names, [r["oppose"] for r in chosen], color=COLOR_CRITICAL, height=0.6)
    ax.barh(names, [r["not_applied"] for r in chosen], color="none", edgecolor=COLOR_NOT_APPLIED,
            hatch="////", linewidth=0.8, height=0.6)
    ax.axvline(0, color=INK_MUTED, linewidth=1)
    ax.set_xlabel("Evidence score from the knowledge graph (sum over the patient's mutated genes)",
                  color=INK_SECONDARY, fontsize=8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.margins(x=0.2)

    span = max([abs(r["support"]) for r in chosen] + [abs(r["oppose"]) for r in chosen]
               + [abs(r["not_applied"]) for r in chosen] + [1e-6])
    for i, r in enumerate(chosen):
        if r["support"] > 0:
            ax.text(r["support"] + 0.015 * span, i, f"{r['support']:+.3f}", va="center", ha="left",
                    fontsize=8, color=INK)
        if r["oppose"] < 0:
            ax.text(r["oppose"] - 0.015 * span, i, f"{r['oppose']:+.3f}", va="center", ha="right",
                    fontsize=8, color=INK)
        if r["not_applied"] < 0:
            ax.text(r["not_applied"] / 2, i, f"{r['not_applied']:+.3f} (not applied)",
                    va="center", ha="center", fontsize=8, color=INK_SECONDARY,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))

    legend_elems = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_GOOD, label="Supporting evidence"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_CRITICAL, label="Opposing evidence (applied)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=COLOR_NOT_APPLIED, hatch="////",
                      label="Literature evidence against, not applied"),
    ]
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    return fig, n_opposed, n_not_applied


# ---------------------------------------------------------------------------
# Sidebar -- connection status + patient selection
# ---------------------------------------------------------------------------

st.sidebar.title("Evidence-Aware Graph-RAG")
st.sidebar.caption("Ovarian cancer treatment ranking — decision-support prototype")

retriever, neo4j_error = get_retriever()

try:
    model, hetero_data, patient_idx, drug_idx, val_patients, train_patients = load_gnn_and_graph()
    gnn_error = None
except Exception as e:  # noqa: BLE001
    model = hetero_data = patient_idx = drug_idx = val_patients = train_patients = None
    gnn_error = str(e)

st.sidebar.markdown("**Backend status**")
st.sidebar.markdown(f"- Neo4j: {'connected' if retriever else 'not connected'}")
st.sidebar.markdown(f"- GNN ranker: {'loaded (' + str(len(drug_idx)) + ' drugs)' if drug_idx else 'not loaded'}")
st.sidebar.markdown(f"- Groq LLM: {'key set' if GROQ_API_KEY else 'no GROQ_API_KEY'}")

if retriever is None:
    st.error(
        "Could not connect to Neo4j.\n\n"
        f"Error: {neo4j_error}\n\n"
        "Check that Neo4j Desktop is running and that NEO4J_URI / NEO4J_USER / "
        "NEO4J_PASSWORD in your .env are correct."
    )
    st.stop()

if gnn_error:
    st.error(f"Could not load the trained GNN model / graph export.\n\nError: {gnn_error}")
    st.stop()

patients = get_patient_list(retriever)
if not patients:
    st.error("Neo4j is connected, but no patients with mutations were found in the graph.")
    st.stop()

patient_labels = {f"{p['patient_id']}  ({p['num_mutated_genes']} mutated genes)": p["patient_id"] for p in patients}
choice = st.sidebar.selectbox("Patient", list(patient_labels.keys()))
patient_id = patient_labels[choice]

if patient_id in val_patients:
    coverage_note = "held-out validation patient (never trained on)"
elif patient_id in train_patients:
    coverage_note = "used in GNN training"
elif patient_id in patient_idx:
    coverage_note = "in graph snapshot"
else:
    coverage_note = "not in last exported graph snapshot — evidence ranking only"
st.sidebar.markdown(f"**GNN coverage:** {coverage_note}")

if st.sidebar.button("Refresh patient list"):
    get_patient_list.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Evidence-Aware Graph-RAG")
st.caption("Explainable ovarian cancer treatment ranking — biomedical knowledge graph + GraphSAGE + LLM reasoning")
st.info("Research decision-support prototype. Not a certified clinical tool and not intended to inform real treatment decisions.")

tab_ranking, tab_eval = st.tabs(["Patient Ranking", "Model Evaluation"])

with tab_ranking:
    try:
        with st.spinner(f"Retrieving evidence subgraph for {patient_id} from Neo4j..."):
            summary = retriever.get_patient_subgraph_summary(patient_id)
            raw_rows = retriever.get_patient_subgraph(patient_id)
    except Exception as e:  # noqa: BLE001
        st.error(f"Subgraph retrieval failed: {e}")
        st.stop()

    if not summary:
        st.warning(f"No mutation-linked evidence subgraph found for {patient_id}.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mutated genes (KG)", summary["num_mutated_genes"])
    c2.metric("Candidate drugs", summary["num_candidate_drugs"])
    c3.metric("Contraindications", summary["num_contraindications"])
    c4.metric("GNN drug space", len(drug_idx) if patient_id in patient_idx else "n/a")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("Patient-specific evidence subgraph")
        fig = render_subgraph(patient_id, summary)
        st.pyplot(fig, use_container_width=True)

    with col_right:
        st.subheader("Mutated genes")
        genes_df = (
            pd.DataFrame(raw_rows)[["gene", "mutation_type", "protein_change"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        go_lookup = retriever.get_go_terms_for_genes(genes_df["gene"].unique())
        genes_df["GO function"] = genes_df["gene"].map(
            lambda g: "; ".join(go_lookup.get(g, [])[:3])
        )
        st.dataframe(genes_df, use_container_width=True, hide_index=True)

    st.subheader("Ranked treatments")
    st.caption(
        "GNN score = structural relevance from the trained GraphSAGE ranker (evidence-blind by design). "
        "Status / net evidence score = deterministic evidence-based classification from the knowledge graph."
    )

    gnn_ranked = gnn_rank_all_drugs(model, hetero_data, patient_idx, drug_idx, patient_id)
    candidate_lookup = {c["drug"]: c for c in summary["candidates"]}
    contra_lookup = {c["drug"]: c for c in summary["contraindications"]}

    rows = []
    if gnn_ranked:
        for rank, (drug, score) in enumerate(gnn_ranked, start=1):
            if drug in candidate_lookup:
                rows.append({
                    "GNN rank": rank, "Drug": drug, "GNN score": score,
                    "Status": "Candidate", "Net evidence score": candidate_lookup[drug]["net_evidence_score"],
                })
            elif drug in contra_lookup:
                rows.append({
                    "GNN rank": rank, "Drug": drug, "GNN score": score,
                    "Status": "Contraindication", "Net evidence score": contra_lookup[drug]["net_evidence_score"],
                })
    else:
        for c in summary["candidates"]:
            rows.append({"GNN rank": None, "Drug": c["drug"], "GNN score": None,
                         "Status": "Candidate", "Net evidence score": c["net_evidence_score"]})
        for c in summary["contraindications"]:
            rows.append({"GNN rank": None, "Drug": c["drug"], "GNN score": None,
                         "Status": "Contraindication", "Net evidence score": c["net_evidence_score"]})

    ranking_df = pd.DataFrame(rows)

    # Evidence for vs against: the primary chart. Shows opposing evidence for every drug that has it.
    balance_fig, n_opposed, n_not_applied = render_evidence_balance_chart(summary)
    if balance_fig is not None:
        st.markdown("**Evidence for and against each drug**")
        st.pyplot(balance_fig, use_container_width=True)
        if n_opposed:
            st.caption(f"{n_opposed} drug(s) have applied opposing evidence for this patient (red). "
                       "Mixed evidence is shown with both bars.")
        if n_not_applied:
            st.caption(f"Grey hatched bars: {n_not_applied} drug(s) have literature evidence against them "
                       "from a gene mutated in most patients (e.g. TP53 and platinum resistance). It is "
                       "shown for transparency but excluded from the score, because it cannot single "
                       "out an individual patient.")
        if not n_opposed and not n_not_applied:
            st.info("No opposing evidence was found for this patient in the knowledge graph.")

    # GNN score chart, kept for reference
    if not ranking_df.empty and ranking_df["GNN score"].notna().any():
        with st.expander("GNN score chart"):
            chart_fig = render_ranking_chart(ranking_df)
            if chart_fig is not None:
                st.pyplot(chart_fig, use_container_width=True)
                st.caption("Top candidates and strongest contraindications by GNN score. The GNN "
                           "score is a structural relevance estimate and can be nearly identical "
                           "across drugs; the evidence chart above is the clinical signal.")

    display_df = ranking_df.copy()
    n_opp = {d: len(e.get("opposing_genes", [])) for d, e in {**candidate_lookup, **contra_lookup}.items()}
    display_df["Opposing evidence"] = display_df["Drug"].map(lambda d: n_opp.get(d, 0))
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Evidence-grounded explanations")
    st.caption(
        "Generated by walking the GNN ranking top-down and prompting an LLM (via Groq) "
        "strictly on the retrieved evidence subgraph -- never on the GNN's raw score. "
        "Each claim cites the evidence edges it relies on and is checked against them: "
        "supported claims trace to a cited edge; flagged claims could not be traced."
    )

    groq_client = None
    if GROQ_API_KEY:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)

    cached_explanations = load_cached_explanations(patient_id)

    top_candidates, top_contras = [], []
    if gnn_ranked:
        for drug, score in gnn_ranked:
            if drug in candidate_lookup and len(top_candidates) < 3:
                top_candidates.append((drug, score, candidate_lookup[drug]))
            if drug in contra_lookup and len(top_contras) < 3:
                top_contras.append((drug, score, contra_lookup[drug]))
            if len(top_candidates) >= 3 and len(top_contras) >= 3:
                break
    else:
        top_candidates = [(c["drug"], None, c) for c in summary["candidates"][:3]]
        top_contras = [(c["drug"], None, c) for c in summary["contraindications"][:3]]

    report_sections = []   # filled by explanation_block, used by the export below

    def explanation_block(drug, gnn_score, evidence, is_candidate):
        status_label = "CANDIDATE" if is_candidate else "CONTRAINDICATION"
        color = COLOR_GOOD if is_candidate else COLOR_CRITICAL
        score_label = f"{gnn_score:+.3f}" if gnn_score is not None else "n/a"
        with st.expander(f"{drug} — {status_label}  (GNN score {score_label})"):
            st.markdown(
                f"<span style='color:{color}; font-weight:600'>{status_label}</span>",
                unsafe_allow_html=True,
            )
            supporting = evidence.get("supporting_genes", [])
            opposing = evidence.get("opposing_genes", [])
            if supporting:
                st.markdown("**Supporting evidence**")
                st.table(evidence_table(supporting))
            if opposing:
                st.markdown("**Opposing evidence**")
                st.table(evidence_table(opposing))
            not_applied = evidence.get("non_discriminating_genes", [])
            if not_applied:
                names = ", ".join(sorted({g["gene"] for g in not_applied}))
                st.caption(
                    f"Not applied: {names} has literature evidence against this drug, but the gene "
                    "is mutated in most patients, so it cannot single out this patient and is "
                    "excluded from the score."
                )
            render_source_links(supporting + opposing, drug)

            st.markdown("**Explanation**")
            state_key = f"explanation_{patient_id}_{drug}"
            exp = st.session_state.get(state_key) or cached_explanations.get(drug)
            report_sections.append({"drug": drug, "status": status_label, "gnn_score": gnn_score,
                                    "evidence": evidence, "explanation": exp})

            if exp is not None:
                render_explanation(exp)
                st.caption("Generated live via Groq" if state_key in st.session_state else "Pre-generated")

            # Generate if nothing exists yet, or offer to redo an unverified (legacy) explanation
            unverified = isinstance(exp, str) or (isinstance(exp, dict) and "verification" not in exp)
            if exp is None or unverified:
                if groq_client is None:
                    if exp is None:
                        st.caption("Set GROQ_API_KEY in .env to generate a live explanation.")
                else:
                    label = (f"Generate explanation for {drug}" if exp is None
                             else "Re-generate with faithfulness verification")
                    if st.button(label, key=f"gen_{drug}"):
                        with st.spinner("Calling the LLM via Groq and verifying claims..."):
                            try:
                                known_genes, known_drugs = load_vocab()
                                st.session_state[state_key] = explain_drug(
                                    groq_client, patient_id, drug, supporting, opposing,
                                    gnn_score if gnn_score is not None else 0.0, is_candidate,
                                    known_genes, known_drugs,
                                )
                                st.rerun()
                            except Exception as e:  # noqa: BLE001
                                st.error(f"Groq call failed: {e}")

    if top_candidates:
        st.markdown("#### Top candidates")
        for drug, score, evidence in top_candidates:
            explanation_block(drug, score, evidence, is_candidate=True)

    if top_contras:
        st.markdown("#### Top contraindications")
        for drug, score, evidence in top_contras:
            explanation_block(drug, score, evidence, is_candidate=False)

    # -- Export -------------------------------------------------------------
    st.subheader("Export report")
    st.caption("Exports the drugs shown above with their evidence, source links and claim-level "
               "verification. Generate any missing explanations first.")
    ex_pdf, ex_json = st.columns(2)

    if build_pdf is None:
        ex_pdf.info(f"PDF export needs reportlab. Run: pip install reportlab  ({REPORT_ERROR})")
    else:
        try:
            png = None
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                png = buf.getvalue()
            except Exception:  # noqa: BLE001
                png = None   # report is still useful without the figure
            pdf_bytes = build_pdf(
                patient_id, summary, ranking_df.to_dict("records"), report_sections,
                linker=get_linker(), subgraph_png=png,
                genes_rows=genes_df.to_dict("records"), coverage_note=coverage_note,
            )
            ex_pdf.download_button("Download PDF report", data=pdf_bytes,
                                   file_name=f"{patient_id}_evidence_report.pdf", mime="application/pdf")
        except Exception as e:  # noqa: BLE001
            ex_pdf.error(f"Could not build the PDF report: {e}")

    try:
        payload = {
            "patient_id": patient_id,
            "generated": datetime.date.today().isoformat(),
            "disclaimer": "Research prototype. Not for clinical use.",
            "summary": summary,
            "drugs": [
                {**sec,
                 "source_links": {
                     e["gene"]: links for e, links in get_linker().links_for_entries(
                         sec["evidence"].get("supporting_genes", []) + sec["evidence"].get("opposing_genes", []),
                         sec["drug"])}}
                for sec in report_sections
            ],
        }
        ex_json.download_button("Download data (JSON)", data=json.dumps(payload, indent=2, default=str),
                                file_name=f"{patient_id}_evidence_report.json", mime="application/json")
    except Exception as e:  # noqa: BLE001
        ex_json.error(f"Could not build the JSON export: {e}")

with tab_eval:
    st.subheader("Model evaluation")
    st.caption("Precomputed by ranking/evaluate.py and ranking/plot_metrics.py on a held-out patient split.")
    plot_paths = sorted(glob.glob(os.path.join(PLOTS_DIR, "*.png")))
    if not plot_paths:
        st.info("No evaluation plots found in data/plots/. Run ranking/evaluate.py and ranking/plot_metrics.py first.")
    else:
        cols = st.columns(2)
        for i, path in enumerate(plot_paths):
            label = os.path.splitext(os.path.basename(path))[0].split("_", 1)[-1].replace("_", " ").title()
            cols[i % 2].image(path, caption=label, use_container_width=True)