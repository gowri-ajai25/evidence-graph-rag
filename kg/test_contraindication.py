from dotenv import load_dotenv
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))
from subgraph_retriever import SubgraphRetriever

load_dotenv()

retriever = SubgraphRetriever(
    os.getenv("NEO4J_URI"), os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")
)

summary = retriever.get_patient_subgraph_summary("TCGA-13-1497")

print(f"Patient: {summary['patient_id']}")
print(f"\nCandidates ({summary['num_candidate_drugs']}):")
for c in summary["candidates"]:
    print(f"   {c['drug']}: max_score={c['max_evidence_score']}, genes={[g['gene'] for g in c['supporting_genes']]}")

print(f"\nContraindications ({summary['num_contraindications']}):")
for c in summary["contraindications"]:
    print(f"   {c['drug']}: min_score={c['min_evidence_score']}, genes={[g['gene'] for g in c['opposing_genes']]}")

retriever.close()