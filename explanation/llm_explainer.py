"""
LLM-based explanation generation.

Uses the GNN's ranked output to select top candidates AND top contraindications,
then retrieves the actual evidence subgraph for each from Neo4j, and prompts
the LLM to generate a clinician-readable reasoning chain grounded ONLY in
that retrieved evidence -- never in the GNN's raw score, which is
uninterpretable to a clinician on its own.

Each gene's Gene Ontology (GO) function terms, retrieved from the knowledge
graph, are included in the evidence as functional context.

Faithfulness verification: every retrieved evidence line gets an ID (E1, E2, ...)
and the LLM must return structured claims that cite those IDs. Each claim is then
checked against the cited graph edges by faithfulness.py, and unsupported claims
are flagged rather than silently shown.
"""

from dotenv import load_dotenv
from neo4j import GraphDatabase
from groq import Groq
import torch
import pickle
import os
import sys
import json
import re
import unicodedata

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ranking"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from subgraph_retriever import SubgraphRetriever
from gnn_model import TreatmentRanker
from faithfulness import verify_explanation

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# llama-3.3-70b-versatile was retired by Groq on 2026-08-16; override with GROQ_MODEL in .env
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def load_gnn_and_graph():
    with open("data/processed/hetero_graph.pkl", "rb") as f:
        saved = pickle.load(f)
    data, patient_idx, gene_idx, drug_idx = saved["data"], saved["patient_idx"], saved["gene_idx"], saved["drug_idx"]

    checkpoint = torch.load("data/processed/best_gnn_model.pt", weights_only=False)
    model = TreatmentRanker(hidden_channels=32, out_channels=16, metadata=data.metadata())
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, data, patient_idx, drug_idx, gene_idx


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

    Every evidence line gets an ID (E1, E2, ...). The LLM must cite these IDs
    in structured JSON claims so each claim can be verified against the graph.

    Returns (prompt, evidence_index) where evidence_index maps each ID to
    {"side", "gene", "drug", "evidence_score"} for the verifier.
    """
    evidence_lines = []
    evidence_index = {}

    def new_id(side, g):
        eid = f"E{len(evidence_index) + 1}"
        evidence_index[eid] = {
            "side": side,
            "gene": g["gene"],
            "drug": drug_name,
            "evidence_score": float(g["evidence_score"]),
            "source": str(g.get("evidence_source") or ""),
            "protein_change": (str(g["protein_change"])
                               if g.get("protein_change") not in (None, "nan", "None", "") else None),
        }
        return eid

    for g in supporting_genes:
        eid = new_id("SUPPORTING", g)
        line = (
            f"- [{eid}] SUPPORTING: Gene {g['gene']} | Mutation: {g['mutation_type']} "
            f"({g['protein_change'] or 'protein change not recorded'}) "
            f"-> {g['relationship_type']} -> {drug_name} "
            f"| Evidence score: {g['evidence_score']:.3f} | Source: {g['evidence_source']}"
        )
        if g.get("civic_level") and g["civic_level"] not in (None, "nan", "None"):
            line += f" | CIViC level: {g['civic_level']}"
        if g.get("clinical_significance") and g["clinical_significance"] not in (None, "nan", "None"):
            line += f" | Clinical significance: {g['clinical_significance']}"
        if g.get("go_terms"):
            line += f" | GO function: {'; '.join(g['go_terms'][:3])}"
        evidence_lines.append(line)

    for g in opposing_genes:
        eid = new_id("OPPOSING", g)
        line = (
            f"- [{eid}] OPPOSING: Gene {g['gene']} | Mutation: {g['mutation_type']} "
            f"({g['protein_change'] or 'protein change not recorded'}) "
            f"-> {g['relationship_type']} -> {drug_name} "
            f"| Evidence score: {g['evidence_score']:.3f} | Source: {g['evidence_source']}"
        )
        if g.get("clinical_significance") and g["clinical_significance"] not in (None, "nan", "None"):
            line += f" | Clinical significance: {g['clinical_significance']}"
        if g.get("go_terms"):
            line += f" | GO function: {'; '.join(g['go_terms'][:3])}"
        evidence_lines.append(line)

    evidence_block = "\n".join(evidence_lines)

    if is_candidate:
        instruction = (
            f"Write a concise (3-4 sentence) clinical reasoning explanation for why {drug_name} "
            f"is being suggested as a CANDIDATE treatment worth clinician review for this patient, "
            f"based strictly on the SUPPORTING evidence above. If any OPPOSING evidence is also listed, "
            f"acknowledge it honestly and note it tempers confidence. Give the overall evidence "
            f"strength (strong/moderate/limited) in the evidence_strength field."
        )
    else:
        instruction = (
            f"Write a concise (3-4 sentence) clinical reasoning explanation for why {drug_name} "
            f"is flagged as a potential CONTRAINDICATION (NOT recommended) for this patient, "
            f"based strictly on the OPPOSING evidence above. If any SUPPORTING evidence is also listed, "
            f"acknowledge it but explain why the opposing evidence outweighs it. Be clear this is a caution "
            f"flag for clinician review, not a guaranteed treatment failure."
        )

    prompt = f"""You are assisting a clinical decision-support system for ovarian cancer treatment recommendation. You must reason ONLY from the structured evidence provided below, retrieved from a biomedical knowledge graph. Do NOT use outside knowledge, do NOT invent citations, and do NOT mention drugs or genes not listed below. Where a "GO function" is listed for a gene, you may use it to briefly describe the gene's biological role.

PATIENT: {patient_id}
DRUG: {drug_name}

RETRIEVED EVIDENCE PATHS:
{evidence_block}

{instruction}"""

    # Output contract (plain concatenation, so the JSON braces need no escaping)
    prompt += (
        "\n\nReturn ONLY valid JSON, with no other text, in exactly this form:\n"
        '{"claims": [{"text": "<one sentence>", "cites": ["E1"], '
        '"stance": "supports|opposes|neutral"}], '
        '"evidence_strength": "strong|moderate|limited"}\n'
        "Rules: write 3-4 claims, one sentence each; every claim must cite at least one "
        "evidence ID from the list above and only IDs that exist; use stance \"supports\" or "
        "\"opposes\" to say which way the cited evidence points for this drug (\"neutral\" only "
        "for caveats); quote evidence scores exactly as given; name no gene or drug that is "
        "not listed above."
    )

    return prompt, evidence_index


def generate_explanation(client, prompt):
    """Returns the raw JSON string produced by the LLM."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2000,   # reasoning models spend part of this budget on hidden reasoning
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _clean_text(t):
    """
    Models sometimes emit invisible space characters (narrow no-break space, zero-width
    space, ...) that render as missing spaces. Turn any such separator into a plain
    space, drop soft hyphens, and make non-breaking hyphens ordinary hyphens.
    """
    out = []
    for ch in str(t):
        cat = unicodedata.category(ch)
        if ch == "\u00ad":
            continue
        if ch == "\u2011":
            out.append("-")
        elif ch in "\t\r\n" or (ch != " " and cat in ("Zs", "Cf")):
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r" {2,}", " ", "".join(out)).strip()


def explain_drug(client, patient_id, drug, supporting, opposing, gnn_score,
                 is_candidate, known_genes, known_drugs):
    """
    Generates a structured explanation for one drug and verifies every claim
    against the retrieved evidence. Used by both main() and the dashboard.

    Returns a dict with:
      explanation       readable text (claims joined)
      evidence_strength strong / moderate / limited, as stated by the LLM
      evidence_index    {"E1": {...}} the edges the LLM was allowed to cite
      verification      per-claim status + overall faithfulness score
    If the LLM output is not valid JSON, the raw text is returned with
    verification["faithfulness"] = None and an error message.
    """
    prompt, evidence_index = build_evidence_prompt(
        patient_id, drug, supporting, opposing, gnn_score, is_candidate
    )
    raw = generate_explanation(client, prompt)

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed.get("claims"), list) or not parsed["claims"]:
            raise ValueError("no claims in LLM output")
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        return {
            "explanation": raw,
            "evidence_index": evidence_index,
            "verification": {"claims": [], "faithfulness": None,
                             "error": f"LLM output could not be verified: {e}"},
        }

    for c in parsed["claims"]:
        if isinstance(c, dict) and "text" in c:
            c["text"] = _clean_text(c["text"])

    verification = verify_explanation(parsed, evidence_index, drug, known_genes, known_drugs)
    return {
        "explanation": " ".join(str(c.get("text", "")) for c in parsed["claims"] if isinstance(c, dict)),
        "evidence_strength": parsed.get("evidence_strength"),
        "evidence_index": evidence_index,
        "verification": verification,
    }


DEFAULT_PATIENTS = ["TCGA-24-2289", "TCGA-24-1466", "TCGA-13-1497"]


def explain_patient(patient_id, retriever, client, model, data, patient_idx, drug_idx,
                    known_genes, known_drugs):
    """Generates and verifies explanations for one patient. Returns the result dict, or None."""
    if patient_id not in patient_idx:
        print(f"   {patient_id} is not in the exported graph snapshot -- skipping")
        return None

    gnn_ranked = get_gnn_ranked_drugs(model, data, patient_idx, drug_idx, patient_id)
    summary = retriever.get_patient_subgraph_summary(patient_id)
    if not summary:
        print(f"   No subgraph found for {patient_id} -- skipping")
        return None

    candidate_drugs = {c["drug"]: c for c in summary["candidates"]}
    contraindication_drugs = {c["drug"]: c for c in summary["contraindications"]}

    # Walk GNN ranking top-down, pick top 2 candidates and top 2 contraindications
    # that actually have retrievable evidence (GNN ranks ALL drugs, most won't
    # have evidence for this specific patient's genes)
    top_candidates, top_contraindications = [], []
    for drug, gnn_score in gnn_ranked:
        if drug in candidate_drugs and len(top_candidates) < 2:
            top_candidates.append((drug, gnn_score, candidate_drugs[drug]))
        if drug in contraindication_drugs and len(top_contraindications) < 2:
            top_contraindications.append((drug, gnn_score, contraindication_drugs[drug]))
        if len(top_candidates) >= 2 and len(top_contraindications) >= 2:
            break

    results = {"patient_id": patient_id, "candidates": [], "contraindications": []}

    for label, bucket, is_candidate in (
        ("CANDIDATE", top_candidates, True),
        ("CONTRAINDICATION", top_contraindications, False),
    ):
        for drug, gnn_score, evidence in bucket:
            print(f"\n{label}: {drug} (GNN score: {gnn_score:.4f})")
            try:
                result = explain_drug(
                    client, patient_id, drug,
                    evidence.get("supporting_genes", []), evidence.get("opposing_genes", []),
                    gnn_score, is_candidate, known_genes, known_drugs,
                )
            except Exception as e:  # noqa: BLE001  (network / API errors)
                print(f"   Explanation failed ({type(e).__name__}): {e}")
                continue

            ver = result["verification"]
            if ver.get("faithfulness") is None:
                print(f"   NOT VERIFIED: {ver.get('error')}")
            else:
                print(f"   faithfulness {ver['faithfulness']:.0%} "
                      f"({ver['n_supported']} supported, {ver['n_partial']} partial, "
                      f"{ver['n_unsupported']} unsupported)")
                for c in ver["claims"]:
                    flag = "" if c["status"] == "supported" else f"  <-- {c['status'].upper()}: {'; '.join(c['issues'])}"
                    print(f"   [{c['status']}] {c['text']}{flag}")

            key = "candidates" if is_candidate else "contraindications"
            results[key].append({"drug": drug, "gnn_score": gnn_score, **result})

    return results


def main():
    patients = sys.argv[1:] or DEFAULT_PATIENTS

    print("Generating verified LLM explanations using GNN-ranked treatments...")
    print("=" * 60)

    print("\n[1/3] Loading GNN model and graph...")
    model, data, patient_idx, drug_idx, gene_idx = load_gnn_and_graph()
    known_genes, known_drugs = list(gene_idx), list(drug_idx)

    print("\n[2/3] Connecting to Neo4j and Groq...")
    if not GROQ_API_KEY:
        print("GROQ_API_KEY is not set in .env")
        return
    retriever = SubgraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    client = Groq(api_key=GROQ_API_KEY)

    print(f"\n[3/3] Generating explanations for {len(patients)} patient(s)...")
    os.makedirs("data/processed", exist_ok=True)
    try:
        for patient_id in patients:
            print("\n" + "-" * 60)
            print(f"Patient: {patient_id}")
            results = explain_patient(patient_id, retriever, client, model, data,
                                      patient_idx, drug_idx, known_genes, known_drugs)
            if results is None:
                continue
            if not (results["candidates"] or results["contraindications"]):
                print(f"\n   No explanations were generated for {patient_id}; "
                      f"keeping any existing file untouched")
                continue
            out_path = f"data/processed/explanations_{patient_id}.json"
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                print(f"\n   Saved to {out_path}")
            except OSError as e:
                print(f"\n   Could not save {out_path}: {e}")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()