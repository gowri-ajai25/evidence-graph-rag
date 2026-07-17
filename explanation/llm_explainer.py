"""
LLM-based explanation generation.

Uses the GNN's ranked output to select top candidates AND top contraindications,
then retrieves the actual evidence subgraph for each from Neo4j, and prompts
the LLM to generate a clinician-readable reasoning chain grounded ONLY in
that retrieved evidence -- never in the GNN's raw score, which is
uninterpretable to a clinician on its own.
"""

from dotenv import load_dotenv
from neo4j import GraphDatabase
from groq import Groq
import torch
import pickle
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ranking"))
from subgraph_retriever import SubgraphRetriever
from gnn_model import TreatmentRanker

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def load_gnn_and_graph():
    with open("data/processed/hetero_graph.pkl", "rb") as f:
        saved = pickle.load(f)
    data, patient_idx, gene_idx, drug_idx = saved["data"], saved["patient_idx"], saved["gene_idx"], saved["drug_idx"]

    checkpoint = torch.load("data/processed/best_gnn_model.pt", weights_only=False)
    model = TreatmentRanker(hidden_channels=32, out_channels=16, metadata=data.metadata())
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, data, patient_idx, drug_idx


def get_gnn_ranked_drugs(model, data, patient_idx, drug_idx, patient_id):
    """Returns ALL drugs ranked by GNN score (descending) for this patient."""
    idx_to_drug = {v: k for k, v in drug_idx.items()}
    p_idx = patient_idx[patient_id]

    with torch.no_grad():
        embeddings = model(data.x_dict, data.edge_index_dict)
        patient_emb = embeddings["patient"]
        drug_emb = embeddings["drug"]

        num_drugs = len(drug_idx)
        patient_indices = torch.full((num_drugs,), p_idx, dtype=torch.long)
        drug_indices = torch.arange(num_drugs, dtype=torch.long)

        scores = model.score_pairs(patient_emb, drug_emb, patient_indices, drug_indices)
        probs = torch.tanh(scores)

    ranked = sorted(
        [(idx_to_drug[i], probs[i].item()) for i in range(num_drugs)],
        key=lambda x: x[1], reverse=True
    )
    return ranked


def build_evidence_prompt(patient_id, drug_name, supporting_genes, opposing_genes, gnn_score, is_candidate):
    """
    Builds a strict, evidence-grounded prompt. The LLM reasons ONLY over the
    supplied subgraph facts -- never the GNN's raw numeric score, which is
    not independently interpretable -- and is told explicitly whether this
    is being presented as a candidate or a contraindication.
    """
    evidence_lines = []

    for g in supporting_genes:
        line = (
            f"- SUPPORTING: Gene {g['gene']} | Mutation: {g['mutation_type']} "
            f"({g['protein_change'] or 'protein change not recorded'}) "
            f"-> {g['relationship_type']} -> {drug_name} "
            f"| Evidence score: {g['evidence_score']:.3f} | Source: {g['evidence_source']}"
        )
        if g.get("civic_level") and g["civic_level"] not in (None, "nan", "None"):
            line += f" | CIViC level: {g['civic_level']}"
        if g.get("clinical_significance") and g["clinical_significance"] not in (None, "nan", "None"):
            line += f" | Clinical significance: {g['clinical_significance']}"
        evidence_lines.append(line)

    for g in opposing_genes:
        line = (
            f"- OPPOSING: Gene {g['gene']} | Mutation: {g['mutation_type']} "
            f"({g['protein_change'] or 'protein change not recorded'}) "
            f"-> {g['relationship_type']} -> {drug_name} "
            f"| Evidence score: {g['evidence_score']:.3f} | Source: {g['evidence_source']}"
        )
        if g.get("clinical_significance") and g["clinical_significance"] not in (None, "nan", "None"):
            line += f" | Clinical significance: {g['clinical_significance']}"
        evidence_lines.append(line)

    evidence_block = "\n".join(evidence_lines)

    if is_candidate:
        instruction = (
            f"Write a concise (3-4 sentence) clinical reasoning explanation for why {drug_name} "
            f"is being suggested as a CANDIDATE treatment worth clinician review for this patient, "
            f"based strictly on the SUPPORTING evidence above. If any OPPOSING evidence is also listed, "
            f"acknowledge it honestly and note it tempers confidence. End by stating the overall evidence "
            f"strength (strong/moderate/limited)."
        )
    else:
        instruction = (
            f"Write a concise (3-4 sentence) clinical reasoning explanation for why {drug_name} "
            f"is flagged as a potential CONTRAINDICATION (NOT recommended) for this patient, "
            f"based strictly on the OPPOSING evidence above. If any SUPPORTING evidence is also listed, "
            f"acknowledge it but explain why the opposing evidence outweighs it. Be clear this is a caution "
            f"flag for clinician review, not a guaranteed treatment failure."
        )

    prompt = f"""You are assisting a clinical decision-support system for ovarian cancer treatment recommendation. You must reason ONLY from the structured evidence provided below, retrieved from a biomedical knowledge graph. Do NOT use outside knowledge, do NOT invent citations, and do NOT mention drugs or genes not listed below.

PATIENT: {patient_id}
DRUG: {drug_name}

RETRIEVED EVIDENCE PATHS:
{evidence_block}

{instruction}"""

    return prompt


def generate_explanation(client, prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content


def main():
    print("Generating LLM explanations using GNN-ranked treatments...")
    print("=" * 60)

    print("\n[1/4] Loading GNN model and graph...")
    model, data, patient_idx, drug_idx = load_gnn_and_graph()

    print("\n[2/4] Selecting patient...")
    retriever = SubgraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    patients = retriever.list_available_patients(limit=10)
    target_patient = patients[0]["patient_id"]
    print(f"   Patient: {target_patient}")

    print("\n[3/4] Getting GNN-ranked drugs and evidence subgraph...")
    gnn_ranked = get_gnn_ranked_drugs(model, data, patient_idx, drug_idx, target_patient)
    summary = retriever.get_patient_subgraph_summary(target_patient)
    retriever.close()

    if not summary:
        print("No subgraph found for this patient.")
        return

    candidate_drugs = {c["drug"]: c for c in summary["candidates"]}
    contraindication_drugs = {c["drug"]: c for c in summary["contraindications"]}

    # Walk GNN ranking top-down, pick top 2 candidates and top 2 contraindications
    # that actually have retrievable evidence (GNN ranks ALL 39 drugs, most won't
    # have evidence for this specific patient's genes)
    top_candidates = []
    top_contraindications = []

    for drug, gnn_score in gnn_ranked:
        if drug in candidate_drugs and len(top_candidates) < 2:
            top_candidates.append((drug, gnn_score, candidate_drugs[drug]))
        if drug in contraindication_drugs and len(top_contraindications) < 2:
            top_contraindications.append((drug, gnn_score, contraindication_drugs[drug]))
        if len(top_candidates) >= 2 and len(top_contraindications) >= 2:
            break

    client = Groq(api_key=GROQ_API_KEY)
    results = {"patient_id": target_patient, "candidates": [], "contraindications": []}

    print(f"\n[4/4] Generating explanations...")
    print("-" * 60)

    for drug, gnn_score, evidence in top_candidates:
        print(f"\nCANDIDATE: {drug} (GNN score: {gnn_score:.4f})")
        prompt = build_evidence_prompt(
            target_patient, drug, evidence["supporting_genes"], evidence.get("opposing_genes", []),
            gnn_score, is_candidate=True
        )
        explanation = generate_explanation(client, prompt)
        print(explanation)
        results["candidates"].append({
            "drug": drug, "gnn_score": gnn_score, "explanation": explanation
        })

    for drug, gnn_score, evidence in top_contraindications:
        print(f"\nCONTRAINDICATION: {drug} (GNN score: {gnn_score:.4f})")
        prompt = build_evidence_prompt(
            target_patient, drug, evidence.get("supporting_genes", []), evidence["opposing_genes"],
            gnn_score, is_candidate=False
        )
        explanation = generate_explanation(client, prompt)
        print(explanation)
        results["contraindications"].append({
            "drug": drug, "gnn_score": gnn_score, "explanation": explanation
        })

    out_path = f"data/processed/explanations_{target_patient}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nSaved to {out_path}")


if __name__ == "__main__":
    main()