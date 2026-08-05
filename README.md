# evidence-graph-rag

An evidence-aware Graph-RAG framework for ovarian cancer treatment recommendation. The system integrates heterogeneous biomedical data into a single queryable knowledge graph, ranks candidate treatments with a heterogeneous graph neural network, and generates clinician-readable, evidence-grounded explanations using an LLM.

This is a final-year B.Tech project. The accompanying research contribution was submitted to ICCIS 2026 (Springer LNNS).

## How it works

```
Data sources (TCGA-OV, CIViC, DGIdb, KEGG)
        │
        ▼
Knowledge graph construction (Neo4j)
  Patient / Gene / Pathway / Drug nodes
  Edge-level composite evidence scoring
        │
        ▼
Single-hop retrieval  (Patient → Gene → Drug)
        │
        ▼
GraphSAGE ranker  ─────────┐
  (heterogeneous, 2-layer  │
   HeteroConv/SAGEConv)    │
        │                  │
        ▼                  ▼
   Drug ranking     LLM explanation (Groq, Llama-3.3-70B)
        │                  │
        └────────┬─────────┘
                  ▼
       Candidate / contraindication
         evidence-grounded output
```

- **Evidence scoring** is computed once, at graph-build time, and attached as an edge property. It combines evidence level (50%), independent source count (30%), and FDA-approval status (20%), then multiplies the result by a direction factor (+1.0 supportive, −1.0 resistance, 0.5 ambiguous, 0.3 unknown) — see `kg/evidence_score.py`.
- **Retrieval** is single-hop only (`Patient → Gene → Drug`). `Pathway` nodes and the KEGG-derived `Gene → Pathway → Drug` edges exist in the graph but are **not currently traversed** by retrieval — pathway-hop retrieval is a documented future-work direction.
- **The GraphSAGE ranker** is trained with the composite evidence score as its regression *target*, using patient-level train/test splitting and hard negatives (drugs unreachable from the patient's mutated genes). Node features are identity encodings, so the ranker does not see the evidence score as an input at inference — it's evidence-blind by design, and evidence-groundedness is instead enforced downstream, at the explanation stage.
- **Explanation generation** takes the GNN's full drug ranking, walks it top-down to select the top candidates and top contraindications that have retrievable graph evidence, and prompts Llama-3.3-70B (via Groq) to explain each one strictly from the retrieved subgraph facts — never from the GNN's raw score, and never from outside knowledge.

## Repository structure

```
evidence-graph-rag/
├── kg/                     Knowledge graph construction (Neo4j)
│   ├── build_graph.py          Gene/Drug nodes, TARGETED_BY & HAS_CLINICAL_EVIDENCE edges
│   ├── load_patients.py        Patient nodes, HAS_MUTATION edges (from TCGA MAF data)
│   ├── load_pathways.py        Pathway nodes, INVOLVED_IN & ASSOCIATED_WITH_DRUG edges (KEGG)
│   ├── evidence_score.py       Composite, direction-aware evidence scoring
│   ├── reset_pathways.py       Rebuild/reset pathway data in the graph
│   ├── fix_pathway_names.py    Data-cleaning utility
│   ├── verify_direction.py     Sanity checks on evidence direction
│   ├── test_contraindication.py, debug_retrieval.py, inspect_*.py   Debug/inspection utilities
│
├── ranking/                GraphSAGE treatment ranker
│   ├── gnn_model.py             Heterogeneous 2-layer HeteroConv/SAGEConv encoder + MLP scoring head
│   ├── train.py                 Patient-level supervised training with hard negatives
│   ├── evaluate.py              Precision@K, Recall@K, NDCG@K, MRR, AUC-ROC
│   ├── export_graph.py          Exports the graph for training/inference
│   ├── rank_patient.py          Ranks all drugs for a single patient
│   ├── plot_metrics.py          Generates evaluation plots (data/plots/)
│   ├── check_positive_distribution.py, debug_training_pairs.py   Debug utilities
│
├── retrieval/
│   └── subgraph_retriever.py    Single-hop Patient→Gene→Drug retrieval; splits results
│                                 into candidates (net-positive evidence) and
│                                 contraindications (net-negative evidence)
│
├── explanation/
│   └── llm_explainer.py         Combines GNN ranking + retrieved evidence to generate
│                                 grounded clinical explanations via Groq/Llama-3.3-70B
│
├── download_tcga.py             TCGA-OV clinical + mutation data (GDC API)
├── download_more_mutations.py   Additional patient MAF files (GDC API)
├── download_dgidb.py            Gene-drug interactions (DGIdb)
├── download_drug_data.py        CIViC clinical evidence + gene summaries
├── download_pathways.py         KEGG gene-pathway and pathway-drug associations
│
├── data/
│   ├── raw/                     Downloaded source data
│   ├── processed/               Trained model, exported graph, saved explanations
│   └── plots/                   Evaluation plots
│
├── test_neo4j.py                Connection smoke test
├── test_gemini.py                Exploratory script, not part of the active pipeline
└── requirements.txt
```

## Setup

**Requirements:** Python 3.10+, a running Neo4j instance, a Groq API key.

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
GROQ_API_KEY=your_groq_key
```

## Running the pipeline

1. **Download source data**
   ```bash
   python download_tcga.py
   python download_more_mutations.py
   python download_dgidb.py
   python download_drug_data.py
   python download_pathways.py
   ```
2. **Build the knowledge graph**
   ```bash
   python kg/build_graph.py
   python kg/load_patients.py
   python kg/load_pathways.py
   ```
3. **Train and evaluate the ranker**
   ```bash
   python ranking/export_graph.py
   python ranking/train.py
   python ranking/evaluate.py
   python ranking/plot_metrics.py
   ```
4. **Retrieve and explain**
   ```bash
   python retrieval/subgraph_retriever.py     # inspect a patient's evidence subgraph
   python explanation/llm_explainer.py        # generate grounded explanations
   ```

## Evaluation

`ranking/evaluate.py` reports Precision@K, Recall@K, NDCG@K, MRR, and AUC-ROC on a held-out patient split, against ground truth defined as drugs with net-positive summed evidence for that patient. Plots are written to `data/plots/`.

## Current limitations

- Retrieval is single-hop; pathway-mediated (`Patient → Gene → Pathway → Drug`) reasoning is not yet active despite the underlying edges existing in the graph.
- The ranker does not take the evidence score as an input feature — it only sees graph structure via identity-encoded node features, so its predictions should be read as a structural/topological signal, not an evidence-weighted one, until the explanation and retrieval layers add that context back in.
- No clinician-facing web application or API layer exists yet in this repository. `fastapi` and `uvicorn` are listed in `requirements.txt` for this purpose but are not yet wired up to any endpoint code.
- Evaluation is conducted on a single held-out validation split; it has not been externally validated against real clinical outcomes.

## Disclaimer

This is a research and educational prototype. It is not a certified clinical decision-support tool and should not be used to inform real treatment decisions.
