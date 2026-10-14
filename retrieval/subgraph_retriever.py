"""
Patient-specific subgraph retrieval engine.

Given a patient ID, retrieves the evidence-weighted neighborhood connecting
their mutated genes to candidate drugs, ready for ranking and explanation.
"""

from dotenv import load_dotenv
from neo4j import GraphDatabase
import os
import json

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


class SubgraphRetriever:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def list_available_patients(self, limit=20):
        """Returns patient IDs that have at least one mutation in the graph."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Patient)-[:HAS_MUTATION]->(g:Gene)
                RETURN p.id AS patient_id, count(g) AS num_mutated_genes
                ORDER BY num_mutated_genes DESC
                LIMIT $limit
                """,
                limit=limit
            )
            return [{"patient_id": r["patient_id"], "num_mutated_genes": r["num_mutated_genes"]} for r in result]

    def get_go_terms_for_genes(self, genes, per_gene=5):
        """
        Returns {gene: [GO term names]} for the given genes. Biological-process
        terms come first, and within each category more specific terms (shared
        by fewer genes) come before generic ones like "protein binding".
        """
        genes = list(genes)
        if not genes:
            return {}
        with self.driver.session() as session:
            counts = {
                r["id"]: r["n"]
                for r in session.run(
                    "MATCH (:Gene)-[:HAS_GO_ANNOTATION]->(t:GOTerm) "
                    "RETURN t.id AS id, count(*) AS n"
                )
            }
            rows = session.run(
                """
                MATCH (g:Gene)-[:HAS_GO_ANNOTATION]->(t:GOTerm)
                WHERE g.name IN $genes
                RETURN g.name AS gene, t.id AS id, t.name AS name, t.category AS cat
                """,
                genes=genes
            )
            per = {}
            for r in rows:
                per.setdefault(r["gene"], []).append(
                    (r["cat"] != "BP", counts.get(r["id"], 0), r["name"])
                )
        return {
            gene: [name for _, _, name in sorted(terms)[:per_gene]]
            for gene, terms in per.items()
        }

    def get_patient_subgraph(self, patient_id, min_evidence_score=-1.0):
        """
        Retrieves the full evidence-weighted neighborhood for one patient:
        their mutations -> genes -> candidate drugs, with all evidence
        properties attached, sorted by evidence strength.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Patient {id: $patient_id})-[m:HAS_MUTATION]->(g:Gene)
                MATCH (g)-[r]->(d:Drug)
                WHERE r.evidence_score >= $min_score
                RETURN
                    p.id AS patient_id,
                    g.name AS gene,
                    m.mutation_type AS mutation_type,
                    m.protein_change AS protein_change,
                    d.name AS drug,
                    type(r) AS relationship_type,
                    r.evidence_score AS evidence_score,
                    r.source AS evidence_source,
                    r.evidence_level AS civic_level,
                    r.clinical_significance AS clinical_significance,
                    r.raw_score AS raw_score,
                    coalesce(r.non_discriminating, false) AS non_discriminating
                ORDER BY r.evidence_score DESC
                """,
                patient_id=patient_id, min_score=min_evidence_score
            )

            rows = [dict(record) for record in result]
            return rows

    def get_patient_subgraph_summary(self, patient_id):
        """
        Aggregates the raw subgraph into per-drug summaries, split into:
          - candidates: drugs with net-positive summed evidence across ALL
            supporting and opposing genes (matches train.py's aggregation
            exactly, so ground truth reflects the same signal the model
            was trained on)
          - contraindications: drugs with net-negative summed evidence,
            shown separately rather than hidden, so a clinician can see
            the full picture.

        NOTE: net_score is now the SUM of every gene's evidence_score for
        this drug (clamped to [-1, 1]), not max(support) + min(opposition).
        This matches build_gene_drug_evidence() / build_patient_drug_pairs()
        in train.py, which sum every gene's contribution per drug. Using a
        different formula here previously caused evaluate.py's ground truth
        to diverge from what the model was actually trained to predict.

        Each gene entry also carries its top GO terms ("go_terms") as
        functional context for explanation; GO does not affect scoring.
        """
        rows = self.get_patient_subgraph(patient_id)

        if not rows:
            return None

        go_map = self.get_go_terms_for_genes({r["gene"] for r in rows})

        drug_data = {}
        for row in rows:
            drug = row["drug"]
            score = row["evidence_score"] or 0.0

            if drug not in drug_data:
                drug_data[drug] = {
                    "drug": drug,
                    "supporting_genes": [],
                    "opposing_genes": [],
                    "non_discriminating_genes": [],
                    "net_evidence_score": 0.0,  # summed across ALL genes for this drug
                    "evidence_sources": set()
                }

            # Resistance evidence from a gene mutated in most patients (set to 0 by
            # kg/apply_prevalence_weights.py) is reported separately: it is neither
            # support nor opposition for this patient, and is not in the net score.
            if row.get("non_discriminating"):
                drug_data[drug]["non_discriminating_genes"].append({
                    "gene": row["gene"],
                    "relationship_type": row["relationship_type"],
                    "evidence_source": row["evidence_source"],
                    "civic_level": row["civic_level"],
                    "clinical_significance": row["clinical_significance"],
                    "raw_score": row.get("raw_score"),
                })
                continue

            gene_entry = {
                "gene": row["gene"],
                "mutation_type": row["mutation_type"],
                "protein_change": row["protein_change"],
                "relationship_type": row["relationship_type"],
                "evidence_score": score,
                "evidence_source": row["evidence_source"],
                "civic_level": row["civic_level"],
                "clinical_significance": row["clinical_significance"],
                "go_terms": go_map.get(row["gene"], [])
            }

            if score > 0:
                drug_data[drug]["supporting_genes"].append(gene_entry)
            elif score < 0:
                drug_data[drug]["opposing_genes"].append(gene_entry)

            # Sum every gene's contribution -- matches train.py exactly
            drug_data[drug]["net_evidence_score"] += score
            drug_data[drug]["evidence_sources"].add(row["evidence_source"])

        candidates = []
        contraindications = []

        for drug, data in drug_data.items():
            data["evidence_sources"] = list(data["evidence_sources"])

            # Clamp to [-1, 1], same as train.py does after summing
            data["net_evidence_score"] = max(-1.0, min(1.0, data["net_evidence_score"]))

            has_support = len(data["supporting_genes"]) > 0
            has_opposition = len(data["opposing_genes"]) > 0

            if has_support and data["net_evidence_score"] >= 0:
                candidates.append(data)
            if has_opposition:
                contraindications.append(data)

        candidates.sort(key=lambda x: x["net_evidence_score"], reverse=True)
        contraindications.sort(key=lambda x: x["net_evidence_score"])  # most negative first

        return {
            "patient_id": patient_id,
            "num_mutated_genes": len(set(r["gene"] for r in rows)),
            "num_candidate_drugs": len(candidates),
            "num_contraindications": len(contraindications),
            "candidates": candidates,
            "contraindications": contraindications
        }


def main():
    retriever = SubgraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    print("Patients available for retrieval (sorted by mutation count):")
    print("=" * 60)
    patients = retriever.list_available_patients(limit=10)
    for p in patients:
        print(f"   {p['patient_id']}  —  {p['num_mutated_genes']} mutated gene(s) in KG")

    if not patients:
        print("No patients found in graph!")
        retriever.close()
        return

    # Demo: retrieve subgraph for the patient with the most mutated genes
    demo_patient = patients[0]["patient_id"]
    print(f"\n\nRetrieving subgraph for patient: {demo_patient}")
    print("=" * 60)

    summary = retriever.get_patient_subgraph_summary(demo_patient)

    print(f"\nMutated genes in KG: {summary['num_mutated_genes']}")
    print(f"Candidate drugs found: {summary['num_candidate_drugs']}")

    print("\nTop 5 ranked candidates by evidence score:")
    for c in summary["candidates"][:5]:
        print(f"\n   Drug: {c['drug']}")
        print(f"   Net evidence score: {c['net_evidence_score']}")
        print(f"   Evidence sources: {c['evidence_sources']}")
        print(f"   Supported by genes: {[g['gene'] for g in c['supporting_genes']]}")

    print(f"\n\nContraindications found: {summary['num_contraindications']}")
    for c in summary["contraindications"][:5]:
        print(f"\n   Drug: {c['drug']} (NOT RECOMMENDED)")
        print(f"   Net evidence score: {c['net_evidence_score']}")
        print(f"   Opposed by genes: {[g['gene'] for g in c['opposing_genes']]}")

    # Save full result as JSON for the next pipeline stages
    os.makedirs("data/processed", exist_ok=True)
    out_path = f"data/processed/subgraph_{demo_patient}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n\nFull subgraph saved to {out_path}")

    retriever.close()


if __name__ == "__main__":
    main()