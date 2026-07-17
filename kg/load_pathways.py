"""
Loads pathway data into the knowledge graph, adding a middle layer:
  Patient -> Gene -> Pathway -> Drug

This extends the existing 2-hop chain into a proper 3-hop reasoning chain,
which is a stronger demonstration of graph-based retrieval.

New nodes: Pathway
New relationships:
  Gene -[:INVOLVED_IN]-> Pathway   (from KEGG)
  Pathway -[:ASSOCIATED_WITH_DRUG]-> Drug  (from curated data)
"""

from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd
import os

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def load_pathways_into_graph(driver):

    gene_pathway_df = pd.read_csv("data/raw/pathways/gene_pathways.csv")
    pathway_drug_df = pd.read_csv("data/raw/pathways/pathway_drugs.csv")

    print(f"   Gene-pathway associations: {len(gene_pathway_df)}")
    print(f"   Pathway-drug associations: {len(pathway_drug_df)}")

    with driver.session() as session:

        # Create uniqueness constraint for Pathway nodes
        session.run(
            "CREATE CONSTRAINT pathway_id IF NOT EXISTS "
            "FOR (p:Pathway) REQUIRE p.id IS UNIQUE"
        )

        # ── Create Pathway nodes ──────────────────────────────
        pathways_created = 0
        unique_pathways = gene_pathway_df[["pathway_id", "pathway_name"]].drop_duplicates()

        for _, row in unique_pathways.iterrows():
            session.run(
                "MERGE (p:Pathway {id: $id}) SET p.name = $name",
                id=row["pathway_id"], name=row["pathway_name"]
            )
            pathways_created += 1

        print(f"   Created {pathways_created} Pathway nodes")

        # ── Gene -[:INVOLVED_IN]-> Pathway ───────────────────
        gene_pathway_count = 0
        skipped_genes = 0

        for _, row in gene_pathway_df.iterrows():
            result = session.run(
                """
                MATCH (g:Gene {name: $gene})
                MATCH (p:Pathway {id: $pathway_id})
                MERGE (g)-[r:INVOLVED_IN]->(p)
                SET r.source = $source
                RETURN r
                """,
                gene=row["gene"],
                pathway_id=row["pathway_id"],
                source=row.get("source", "KEGG")
            )
            if result.single():
                gene_pathway_count += 1
            else:
                skipped_genes += 1

        print(f"   Created {gene_pathway_count} GENE -> PATHWAY relationships")
        if skipped_genes > 0:
            print(f"   Skipped {skipped_genes} (gene not in our KG)")

        # ── Pathway -[:ASSOCIATED_WITH_DRUG]-> Drug ──────────
        pathway_drug_count = 0
        skipped_drugs = 0

        for _, row in pathway_drug_df.iterrows():
            # Match pathway by name since curated data uses names not IDs
            result = session.run(
                """
                MATCH (p:Pathway)
                WHERE toLower(p.name) CONTAINS toLower($pathway_name)
                   OR toLower($pathway_name) CONTAINS toLower(p.name)
                MATCH (d:Drug {name: $drug_name})
                MERGE (p)-[r:ASSOCIATED_WITH_DRUG]->(d)
                SET r.source = $source
                RETURN r
                """,
                pathway_name=row["pathway_name"],
                drug_name=row["drug_name"],
                source=row.get("source", "curated")
            )
            if result.single():
                pathway_drug_count += 1
            else:
                skipped_drugs += 1

        print(f"   Created {pathway_drug_count} PATHWAY -> DRUG relationships")
        if skipped_drugs > 0:
            print(f"   Skipped {skipped_drugs} (pathway or drug not found in KG)")


def print_summary(driver):
    with driver.session() as session:
        counts = {
            "Patient": session.run("MATCH (n:Patient) RETURN count(n) AS c").single()["c"],
            "Gene": session.run("MATCH (n:Gene) RETURN count(n) AS c").single()["c"],
            "Pathway": session.run("MATCH (n:Pathway) RETURN count(n) AS c").single()["c"],
            "Drug": session.run("MATCH (n:Drug) RETURN count(n) AS c").single()["c"],
        }
        total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

        print("\n" + "=" * 60)
        print("UPDATED KNOWLEDGE GRAPH SUMMARY")
        print("=" * 60)
        for label, count in counts.items():
            print(f"   {label} nodes:     {count}")
        print(f"   Total relationships: {total_rels}")

        # Show the new 3-hop chain works
        print("\n   Verifying 3-hop chain (Patient -> Gene -> Pathway -> Drug)...")
        result = session.run(
            """
            MATCH (p:Patient)-[:HAS_MUTATION]->(g:Gene)-[:INVOLVED_IN]
                  ->(pw:Pathway)-[:ASSOCIATED_WITH_DRUG]->(d:Drug)
            RETURN p.id AS patient, g.name AS gene,
                   pw.name AS pathway, d.name AS drug
            LIMIT 5
            """
        )
        rows = result.data()
        if rows:
            print(f"   Found {len(rows)} example 3-hop paths:")
            for row in rows:
                print(f"      {row['patient']} -> {row['gene']} -> {row['pathway']} -> {row['drug']}")
        else:
            print("   No 3-hop paths found yet")


def main():
    print("Loading pathway data into knowledge graph...")
    print("=" * 60)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print("\n[1/2] Loading pathways and relationships...")
    load_pathways_into_graph(driver)

    print("\n[2/2] Summary...")
    print_summary(driver)

    driver.close()


if __name__ == "__main__":
    main()