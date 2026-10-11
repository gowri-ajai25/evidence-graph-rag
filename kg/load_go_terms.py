from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd
import os

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def load_go_terms(driver):
    df = pd.read_csv("data/raw/go/gene_go_terms.csv").dropna(subset=["go_id"])
    df = df.drop_duplicates(["gene", "go_id"])
    print(f"   Raw annotations: {len(df)}, GO terms: {df.go_id.nunique()}")

    # keep only GO terms shared by >= 2 genes (single-gene terms add nothing
    # for message passing)
    counts = df.groupby("go_id").gene.nunique()
    df = df[df.go_id.isin(counts[counts >= 2].index)]
    print(f"   After shared-term filter: {len(df)} annotations, "
          f"{df.go_id.nunique()} GO terms")

    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT go_id IF NOT EXISTS "
            "FOR (t:GOTerm) REQUIRE t.id IS UNIQUE"
        )

        # GOTerm nodes
        for _, row in df[["go_id", "term", "category"]].drop_duplicates("go_id").iterrows():
            session.run(
                "MERGE (t:GOTerm {id: $id}) SET t.name = $name, t.category = $cat",
                id=row["go_id"], name=row["term"], cat=row["category"]
            )

        # Gene -[:HAS_GO_ANNOTATION]-> GOTerm
        created, skipped = 0, 0
        for _, row in df.iterrows():
            result = session.run(
                """
                MATCH (g:Gene {name: $gene})
                MATCH (t:GOTerm {id: $go_id})
                MERGE (g)-[r:HAS_GO_ANNOTATION]->(t)
                SET r.evidence = $evidence
                RETURN r
                """,
                gene=row["gene"], go_id=row["go_id"],
                evidence=row.get("evidence")
            )
            if result.single():
                created += 1
            else:
                skipped += 1

        print(f"   Created {created} GENE -> GOTERM relationships")
        if skipped:
            print(f"   Skipped {skipped} (gene not in our KG)")


def print_summary(driver):
    with driver.session() as session:
        n_terms = session.run("MATCH (t:GOTerm) RETURN count(t) AS c").single()["c"]
        n_rels = session.run(
            "MATCH ()-[r:HAS_GO_ANNOTATION]->() RETURN count(r) AS c"
        ).single()["c"]
        print(f"\n   GOTerm nodes: {n_terms}")
        print(f"   HAS_GO_ANNOTATION relationships: {n_rels}")

        rows = session.run(
            """
            MATCH (g:Gene {name: 'BRCA1'})-[:HAS_GO_ANNOTATION]->(t:GOTerm)
            RETURN t.name AS term LIMIT 5
            """
        ).data()
        print("\n   Sample (BRCA1):")
        for r in rows:
            print(f"      {r['term']}")


def main():
    print("Loading GO terms into knowledge graph...")
    print("=" * 60)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    load_go_terms(driver)
    print_summary(driver)
    driver.close()


if __name__ == "__main__":
    main()