"""
Exports the Neo4j knowledge graph into a PyTorch Geometric HeteroData object.

Node types: Patient, Gene, Drug, Pathway, GOTerm (optional)
Edge types: 
  - (Patient, HAS_MUTATION, Gene)
  - (Gene, TARGETED_BY, Drug)
  - (Gene, HAS_CLINICAL_EVIDENCE, Drug)
  - (Gene, INVOLVED_IN, Pathway)
  - (Pathway, ASSOCIATED_WITH_DRUG, Drug)
  - (Gene, HAS_GO_ANNOTATION, GOTerm)   <- new, skipped with --no-go

Each edge carries its evidence_score as an edge attribute where available.

Usage:
  python ranking/export_graph.py            # with GO terms
  python ranking/export_graph.py --no-go    # baseline graph (no GO)
"""

from dotenv import load_dotenv
from neo4j import GraphDatabase
import torch
from torch_geometric.data import HeteroData
import os
import sys
import pickle

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

USE_GO = "--no-go" not in sys.argv


def fetch_all_nodes_and_edges(driver):
    with driver.session() as session:
        patients = [r["id"] for r in session.run("MATCH (p:Patient) RETURN p.id AS id")]
        genes = [r["name"] for r in session.run("MATCH (g:Gene) RETURN g.name AS name")]
        drugs = [r["name"] for r in session.run("MATCH (d:Drug) RETURN d.name AS name")]
        pathways = [r["id"] for r in session.run("MATCH (pw:Pathway) RETURN pw.id AS id")]

        mutation_edges = list(session.run(
            """
            MATCH (p:Patient)-[r:HAS_MUTATION]->(g:Gene)
            RETURN p.id AS patient, g.name AS gene, r.mutation_type AS mutation_type
            """
        ))

        targeted_edges = list(session.run(
            """
            MATCH (g:Gene)-[r:TARGETED_BY]->(d:Drug)
            RETURN g.name AS gene, d.name AS drug, r.evidence_score AS score
            """
        ))

        clinical_edges = list(session.run(
            """
            MATCH (g:Gene)-[r:HAS_CLINICAL_EVIDENCE]->(d:Drug)
            RETURN g.name AS gene, d.name AS drug, r.evidence_score AS score
            """
        ))

        gene_pathway_edges = list(session.run(
            """
            MATCH (g:Gene)-[r:INVOLVED_IN]->(pw:Pathway)
            RETURN g.name AS gene, pw.id AS pathway
            """
        ))

        pathway_drug_edges = list(session.run(
            """
            MATCH (pw:Pathway)-[r:ASSOCIATED_WITH_DRUG]->(d:Drug)
            RETURN pw.id AS pathway, d.name AS drug
            """
        ))

        # GO terms (only if requested)
        goterms, gene_go_edges = [], []
        if USE_GO:
            goterms = [r["id"] for r in session.run("MATCH (t:GOTerm) RETURN t.id AS id")]
            gene_go_edges = list(session.run(
                """
                MATCH (g:Gene)-[r:HAS_GO_ANNOTATION]->(t:GOTerm)
                RETURN g.name AS gene, t.id AS go_id
                """
            ))

    return (patients, genes, drugs, pathways, goterms,
            mutation_edges, targeted_edges, clinical_edges,
            gene_pathway_edges, pathway_drug_edges, gene_go_edges)


def build_hetero_data(patients, genes, drugs, pathways, goterms,
                      mutation_edges, targeted_edges, clinical_edges,
                      gene_pathway_edges, pathway_drug_edges, gene_go_edges):

    data = HeteroData()

    # Index mappings
    patient_idx = {pid: i for i, pid in enumerate(patients)}
    gene_idx = {g: i for i, g in enumerate(genes)}
    drug_idx = {d: i for i, d in enumerate(drugs)}
    pathway_idx = {pw: i for i, pw in enumerate(pathways)}
    go_idx = {t: i for i, t in enumerate(goterms)}

    # Node features (identity embeddings as placeholder)
    data["patient"].x = torch.eye(len(patients))
    data["gene"].x = torch.eye(len(genes))
    data["drug"].x = torch.eye(len(drugs))
    data["pathway"].x = torch.eye(len(pathways))

    data["patient"].node_id = patients
    data["gene"].node_id = genes
    data["drug"].node_id = drugs
    data["pathway"].node_id = pathways

    if USE_GO:
        data["goterm"].x = torch.eye(len(goterms))
        data["goterm"].node_id = goterms

    # Patient -> Gene (HAS_MUTATION)
    src, dst = [], []
    for e in mutation_edges:
        if e["patient"] in patient_idx and e["gene"] in gene_idx:
            src.append(patient_idx[e["patient"]])
            dst.append(gene_idx[e["gene"]])
    data["patient", "has_mutation", "gene"].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Gene -> Drug (TARGETED_BY)
    src, dst, weights = [], [], []
    for e in targeted_edges:
        if e["gene"] in gene_idx and e["drug"] in drug_idx:
            src.append(gene_idx[e["gene"]])
            dst.append(drug_idx[e["drug"]])
            weights.append(e["score"] if e["score"] is not None else 0.3)
    data["gene", "targeted_by", "drug"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    data["gene", "targeted_by", "drug"].edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)

    # Gene -> Drug (HAS_CLINICAL_EVIDENCE)
    src, dst, weights = [], [], []
    for e in clinical_edges:
        if e["gene"] in gene_idx and e["drug"] in drug_idx:
            src.append(gene_idx[e["gene"]])
            dst.append(drug_idx[e["drug"]])
            weights.append(e["score"] if e["score"] is not None else 0.3)
    data["gene", "has_clinical_evidence", "drug"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    data["gene", "has_clinical_evidence", "drug"].edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)

    # Gene -> Pathway (INVOLVED_IN)
    src, dst = [], []
    for e in gene_pathway_edges:
        if e["gene"] in gene_idx and e["pathway"] in pathway_idx:
            src.append(gene_idx[e["gene"]])
            dst.append(pathway_idx[e["pathway"]])
    data["gene", "involved_in", "pathway"].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Pathway -> Drug (ASSOCIATED_WITH_DRUG)
    src, dst = [], []
    for e in pathway_drug_edges:
        if e["pathway"] in pathway_idx and e["drug"] in drug_idx:
            src.append(pathway_idx[e["pathway"]])
            dst.append(drug_idx[e["drug"]])
    data["pathway", "associated_with_drug", "drug"].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Gene -> GOTerm (HAS_GO_ANNOTATION)
    if USE_GO:
        src, dst = [], []
        for e in gene_go_edges:
            if e["gene"] in gene_idx and e["go_id"] in go_idx:
                src.append(gene_idx[e["gene"]])
                dst.append(go_idx[e["go_id"]])
        data["gene", "has_go_annotation", "goterm"].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Reverse edges for bidirectional message passing
    data["gene", "rev_has_mutation", "patient"].edge_index = data["patient", "has_mutation", "gene"].edge_index.flip(0)
    data["drug", "rev_targeted_by", "gene"].edge_index = data["gene", "targeted_by", "drug"].edge_index.flip(0)
    data["drug", "rev_has_clinical_evidence", "gene"].edge_index = data["gene", "has_clinical_evidence", "drug"].edge_index.flip(0)
    data["pathway", "rev_involved_in", "gene"].edge_index = data["gene", "involved_in", "pathway"].edge_index.flip(0)
    data["drug", "rev_associated_with_drug", "pathway"].edge_index = data["pathway", "associated_with_drug", "drug"].edge_index.flip(0)
    if USE_GO:
        data["goterm", "rev_has_go_annotation", "gene"].edge_index = data["gene", "has_go_annotation", "goterm"].edge_index.flip(0)

    return data, patient_idx, gene_idx, drug_idx, pathway_idx


def main():
    print("Exporting Neo4j knowledge graph to PyTorch Geometric format...")
    print(f"   GO terms: {'ON' if USE_GO else 'OFF (--no-go)'}")
    print("=" * 60)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print("\n[1/2] Fetching nodes and edges from Neo4j...")
    (patients, genes, drugs, pathways, goterms,
     mutation_edges, targeted_edges, clinical_edges,
     gene_pathway_edges, pathway_drug_edges, gene_go_edges) = fetch_all_nodes_and_edges(driver)
    driver.close()

    print(f"   Patients: {len(patients)}")
    print(f"   Genes: {len(genes)}")
    print(f"   Pathways: {len(pathways)}")
    print(f"   Drugs: {len(drugs)}")
    if USE_GO:
        print(f"   GO terms: {len(goterms)}")
    print(f"   HAS_MUTATION edges: {len(mutation_edges)}")
    print(f"   TARGETED_BY edges: {len(targeted_edges)}")
    print(f"   HAS_CLINICAL_EVIDENCE edges: {len(clinical_edges)}")
    print(f"   INVOLVED_IN edges: {len(gene_pathway_edges)}")
    print(f"   ASSOCIATED_WITH_DRUG edges: {len(pathway_drug_edges)}")
    if USE_GO:
        print(f"   HAS_GO_ANNOTATION edges: {len(gene_go_edges)}")

    print("\n[2/2] Building HeteroData object...")
    data, patient_idx, gene_idx, drug_idx, pathway_idx = build_hetero_data(
        patients, genes, drugs, pathways, goterms,
        mutation_edges, targeted_edges, clinical_edges,
        gene_pathway_edges, pathway_drug_edges, gene_go_edges
    )

    print("\nGraph structure:")
    print(data)

    # Save everything for the training script
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/hetero_graph.pkl", "wb") as f:
        pickle.dump({
            "data": data,
            "patient_idx": patient_idx,
            "gene_idx": gene_idx,
            "drug_idx": drug_idx,
            "pathway_idx": pathway_idx
        }, f)

    print("\nSaved to data/processed/hetero_graph.pkl")


if __name__ == "__main__":
    main()