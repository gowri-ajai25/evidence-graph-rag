"""
Loads patient mutation data from TCGA-OV MAF files into Neo4j.
Creates Patient nodes connected to Gene nodes via HAS_MUTATION relationships.

Only loads mutations in genes already in our knowledge graph (the ovarian
cancer gene panel) to keep the graph focused and relevant, plus any
clinically significant mutation types.
"""

from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd
import os
import glob

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

MUTATION_DIR = "data/raw/mutations"

# Mutation types that are clinically meaningful (skip silent mutations)
CLINICALLY_RELEVANT_TYPES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
    "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation"
}


def get_kg_genes(driver):
    """Get the list of genes already in our knowledge graph."""
    with driver.session() as session:
        result = session.run("MATCH (g:Gene) RETURN g.name AS name")
        return set(record["name"] for record in result)


def load_all_mutations(kg_genes):
    """Reads all MAF files and extracts patient -> gene mutation rows."""
    maf_files = glob.glob(os.path.join(MUTATION_DIR, "*.maf.gz"))
    print(f"Found {len(maf_files)} MAF files to process\n")

    all_mutations = []

    for i, maf_path in enumerate(maf_files, 1):
        try:
            df = pd.read_csv(
                maf_path, sep="\t", comment="#", compression="gzip",
                low_memory=False,
                usecols=["Hugo_Symbol", "Tumor_Sample_Barcode",
                         "Variant_Classification", "Variant_Type", "HGVSp_Short"]
            )
        except Exception as e:
            print(f"   [{i}] Could not read {os.path.basename(maf_path)}: {e}")
            continue

        # Keep only clinically relevant mutations in genes already in our graph
        df = df[
            df["Hugo_Symbol"].isin(kg_genes) &
            df["Variant_Classification"].isin(CLINICALLY_RELEVANT_TYPES)
        ]

        for _, row in df.iterrows():
            all_mutations.append({
                "patient_id": row["Tumor_Sample_Barcode"][:12],  # e.g. TCGA-09-1675
                "gene": row["Hugo_Symbol"],
                "mutation_type": row["Variant_Classification"],
                "variant_type": row["Variant_Type"],
                "protein_change": row.get("HGVSp_Short", "")
            })

        if i % 10 == 0 or i == len(maf_files):
            print(f"   Processed {i}/{len(maf_files)} files...")

    return pd.DataFrame(all_mutations)


def load_into_neo4j(driver, mutations_df):
    """Creates Patient nodes and HAS_MUTATION relationships."""
    with driver.session() as session:
        session.run("CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE")

        patients = mutations_df["patient_id"].unique()
        for patient_id in patients:
            session.run("MERGE (p:Patient {id: $id})", id=patient_id)
        print(f"   Created {len(patients)} Patient nodes")

        count = 0
        for _, row in mutations_df.iterrows():
            session.run(
                """
                MATCH (p:Patient {id: $patient_id})
                MATCH (g:Gene {name: $gene})
                MERGE (p)-[r:HAS_MUTATION]->(g)
                SET r.mutation_type = $mutation_type,
                    r.variant_type = $variant_type,
                    r.protein_change = $protein_change
                """,
                patient_id=row["patient_id"], gene=row["gene"],
                mutation_type=row["mutation_type"], variant_type=row["variant_type"],
                protein_change=row["protein_change"]
            )
            count += 1
        print(f"   Created {count} HAS_MUTATION relationships")


def main():
    print("Loading patient mutation data into knowledge graph...")
    print("=" * 50)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    # Clear old patient data before reloading with the larger cohort
    with driver.session() as session:
        session.run("MATCH (p:Patient) DETACH DELETE p")
    print("   Cleared old patient nodes\n")

    print("\n[1/3] Getting genes already in knowledge graph...")
    kg_genes = get_kg_genes(driver)
    print(f"   {len(kg_genes)} genes in graph: {sorted(kg_genes)[:10]}...")

    print("\n[2/3] Reading and filtering mutation data from MAF files...")
    mutations_df = load_all_mutations(kg_genes)
    print(f"\n   Total relevant mutations found: {len(mutations_df)}")

    if len(mutations_df) == 0:
        print("\n   WARNING: No mutations found in our gene panel!")
        print("   This means none of the 50 patients had mutations in our 37 KG genes.")
        print("   Saving raw mutation summary for inspection instead.")
        driver.close()
        return

    print("\n   Mutation type breakdown:")
    print(mutations_df["mutation_type"].value_counts().to_string())

    print("\n   Genes most frequently mutated:")
    print(mutations_df["gene"].value_counts().head(10).to_string())

    # Save a copy for later use (subgraph retrieval will need this too)
    os.makedirs("data/processed", exist_ok=True)
    mutations_df.to_csv("data/processed/patient_mutations.csv", index=False)
    print(f"\n   Saved processed mutation data to data/processed/patient_mutations.csv")

    print("\n[3/3] Loading into Neo4j...")
    load_into_neo4j(driver, mutations_df)

    # Final summary
    with driver.session() as session:
        patient_count = session.run("MATCH (p:Patient) RETURN count(p) AS c").single()["c"]
        total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

        print("\n" + "=" * 50)
        print("UPDATED KNOWLEDGE GRAPH SUMMARY")
        print("=" * 50)
        print(f"   Patient nodes:     {patient_count}")
        print(f"   Total nodes:       {total_nodes}")
        print(f"   Total relationships: {total_rels}")

    driver.close()


if __name__ == "__main__":
    main()