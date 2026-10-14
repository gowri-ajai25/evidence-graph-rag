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

import os
import sys
import json
import glob
import pickle

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
for p in (ROOT, os.path.join(ROOT, "retrieval"), os.path.join(ROOT, "ranking"), os.path.join(ROOT, "explanation")):
    if p not in sys.path:
        sys.path.append(p)

from subgraph_retriever import SubgraphRetriever          # noqa: E402
from gnn_model import TreatmentRanker                      # noqa: E402
from llm_explainer import build_evidence_prompt, generate_explanation  # noqa: E402

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
    """Reads any data/processed/explanations_{patient_id}*.json already on disk."""
    cache = {}
    pattern = os.path.join(ROOT, "data", "processed", f"explanations_{patient_id}*.json")
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                payload = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        if "candidates" in payload or "contraindications" in payload:
            for entry in payload.get("candidates", []):
                cache[entry["drug"]] = entry["explanation"]
            for entry in payload.get("contraindications", []):
                cache[entry["drug"]] = entry["explanation"]
        elif "drug" in payload and "explanation" in payload:
            cache[payload["drug"]] = payload["explanation"]
    return cache


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


def render_ranking_chart(ranking_df, top_n=10):
    df = ranking_df.dropna(subset=["GNN score"]).head(top_n).iloc[::-1]
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.4 * len(df))))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    colors = [COLOR_GOOD if s == "Candidate" else COLOR_CRITICAL for s in df["Status"]]
    bars = ax.barh(df["Drug"], df["GNN score"], color=colors, height=0.6)
    ax.axvline(0, color=GRIDLINE, linewidth=1)
    ax.set_xlabel("GNN-predicted relevance score", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)

    for bar, val in zip(bars, df["GNN score"]):
        ax.text(
            val + (0.004 if val >= 0 else -0.004),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left" if val >= 0 else "right",
            fontsize=8, color=INK,
        )

    legend_elems = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_GOOD, label="Candidate"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_CRITICAL, label="Contraindication"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    return fig


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

    if not ranking_df.empty and ranking_df["GNN score"].notna().any():
        chart_fig = render_ranking_chart(ranking_df)
        if chart_fig is not None:
            st.pyplot(chart_fig, use_container_width=True)
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    st.subheader("Evidence-grounded explanations")
    st.caption(
        "Generated by walking the GNN ranking top-down and prompting Llama-3.3-70B (via Groq) "
        "strictly on the retrieved evidence subgraph -- never on the GNN's raw score."
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

            st.markdown("**Explanation**")
            state_key = f"explanation_{patient_id}_{drug}"
            if drug in cached_explanations and state_key not in st.session_state:
                st.write(cached_explanations[drug])
                st.caption("Pre-generated")
            elif state_key in st.session_state:
                st.write(st.session_state[state_key])
                st.caption("Generated live via Groq")
            elif groq_client is None:
                st.caption("Set GROQ_API_KEY in .env to generate a live explanation.")
            else:
                if st.button(f"Generate explanation for {drug}", key=f"gen_{drug}"):
                    with st.spinner("Calling Llama-3.3-70B via Groq..."):
                        prompt = build_evidence_prompt(
                            patient_id, drug, supporting, opposing,
                            gnn_score if gnn_score is not None else 0.0, is_candidate,
                        )
                        try:
                            text = generate_explanation(groq_client, prompt)
                            st.session_state[state_key] = text
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