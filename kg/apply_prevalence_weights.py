# kg/apply_prevalence_weights.py
"""
Down-weights evidence for biomarkers that are common in the cohort.

A gene mutated in almost every patient (TP53: ~97% of TCGA-OV) cannot separate
one patient's best treatment from another's, so its evidence should carry
little weight in a patient-specific ranking. Each gene gets

    prevalence      = patients with a mutation in the gene / all patients
    informativeness = 1 - prevalence

and every Gene->Drug edge (TARGETED_BY, HAS_CLINICAL_EVIDENCE) is rescaled:

    evidence_score = raw_score * informativeness

Resistance evidence is treated more strictly. A biomarker present in at least
PREVALENCE_CUTOFF of patients (default 0.5) cannot single out the patients for
whom a drug should be avoided, so its NEGATIVE edges are set to 0 and flagged
non_discriminating = true instead of being scaled. Positive evidence from such a
gene is only scaled. (Only TP53 is above 0.5 in TCGA-OV; the cutoff is not tuned
to any metric.)

The unscaled value is kept as raw_score, so the script can be re-run safely
(it always rescales from raw_score) and the original evidence stays inspectable.
The sign of the score (supportive vs resistant) is unchanged.

Run AFTER kg/load_patients.py (prevalence needs the Patient nodes) and BEFORE
ranking/export_graph.py:
    python kg/apply_prevalence_weights.py
"""

import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

PREVALENCE_CUTOFF = 0.5


def main():
    print("Applying prevalence weights to gene->drug evidence")
    print("=" * 60)
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        session = driver.session()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not connect to Neo4j: {e}")

    with session:
        n_patients = session.run("MATCH (p:Patient) RETURN count(p) AS n").single()["n"]
        if n_patients == 0:
            sys.exit("No Patient nodes found. Run kg/load_patients.py first.")

        session.run(
            """
            MATCH (g:Gene)
            OPTIONAL MATCH (p:Patient)-[:HAS_MUTATION]->(g)
            WITH g, count(DISTINCT p) AS k
            SET g.n_mutated = k,
                g.prevalence = toFloat(k) / $n,
                g.informativeness = 1.0 - toFloat(k) / $n
            """,
            n=n_patients,
        )

        for rel in ("TARGETED_BY", "HAS_CLINICAL_EVIDENCE"):
            session.run(
                f"""
                MATCH (g:Gene)-[r:{rel}]->(:Drug)
                SET r.raw_score = coalesce(r.raw_score, r.evidence_score)
                WITH g, r
                SET r.non_discriminating = (r.raw_score < 0 AND g.prevalence >= $cutoff),
                    r.evidence_score = CASE
                        WHEN r.raw_score < 0 AND g.prevalence >= $cutoff THEN 0.0
                        ELSE round(r.raw_score * g.informativeness * 10000) / 10000.0
                    END
                """,
                cutoff=PREVALENCE_CUTOFF,
            )

        print(f"   Patients: {n_patients}")
        print("\n   Gene prevalence (mutated genes with drug evidence):")
        rows = session.run(
            """
            MATCH (g:Gene)-[r]->(:Drug)
            WHERE type(r) IN ['TARGETED_BY', 'HAS_CLINICAL_EVIDENCE'] AND g.n_mutated > 0
            RETURN g.name AS gene, g.n_mutated AS k, g.prevalence AS prev,
                   g.informativeness AS w, count(r) AS edges
            ORDER BY k DESC LIMIT 12
            """
        )
        print(f"   {'gene':<10}{'patients':>9}{'prevalence':>12}{'weight':>9}{'edges':>7}")
        for r in rows:
            print(f"   {r['gene']:<10}{r['k']:>9}{r['prev']:>12.3f}{r['w']:>9.3f}{r['edges']:>7}")

        print("\n   Platinum check (CIViC edges, raw -> weighted):")
        for r in session.run(
            """
            MATCH (g:Gene)-[r:HAS_CLINICAL_EVIDENCE]->(d:Drug)
            WHERE d.name IN ['Cisplatin', 'Carboplatin']
            RETURN g.name AS gene, d.name AS drug, r.raw_score AS raw, r.evidence_score AS w,
                   r.non_discriminating AS nd
            ORDER BY raw
            """
        ):
            flag = "  (non-discriminating)" if r["nd"] else ""
            print(f"   {r['gene']:<8}{r['drug']:<12}{r['raw']:+.3f} -> {r['w']:+.4f}{flag}")
    driver.close()


if __name__ == "__main__":
    main()