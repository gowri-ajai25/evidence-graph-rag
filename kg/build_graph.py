"""
Builds the Evidence-Aware Knowledge Graph in Neo4j.

Loads:
  - Genes (from CIViC + our key ovarian cancer gene list)
  - Drugs (from DGIdb)
  - GENE -> DRUG relationships, weighted by composite evidence score
  - CIViC evidence items linking genes/variants to drugs with clinical evidence levels
"""

from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd
import os
import sys

# Allow importing evidence_score.py from the same kg/ folder
sys.path.append(os.path.dirname(__file__))
from evidence_score import composite_score

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


class KnowledgeGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        """Wipes the graph clean before rebuilding. Use with caution."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("   Cleared existing graph data")

    def create_constraints(self):
        """Ensures no duplicate nodes for the same gene/drug name."""
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT gene_name IF NOT EXISTS FOR (g:Gene) REQUIRE g.name IS UNIQUE")
            session.run("CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE")
        print("   Created uniqueness constraints")

    def load_genes_from_dgidb(self, dgidb_df):
        """Creates Gene nodes from our DGIdb dataset."""
        with self.driver.session() as session:
            genes = dgidb_df["gene_name"].unique()
            for gene in genes:
                session.run(
                    "MERGE (g:Gene {name: $name})",
                    name=gene
                )
        print(f"   Created {len(genes)} Gene nodes")

    def load_drugs_from_dgidb(self, dgidb_df):
        """Creates Drug nodes from our DGIdb dataset."""
        with self.driver.session() as session:
            drugs = dgidb_df["drug_name"].unique()
            for drug in drugs:
                session.run(
                    "MERGE (d:Drug {name: $name})",
                    name=drug
                )
        print(f"   Created {len(drugs)} Drug nodes")

    def load_gene_drug_relationships(self, dgidb_df):
        """
        Creates GENE -[:TARGETED_BY {evidence_score}]-> DRUG relationships
        using our composite evidence scoring function.
        """
        with self.driver.session() as session:
            count = 0
            for _, row in dgidb_df.iterrows():
                gene = row["gene_name"]
                drug = row["drug_name"]
                interaction_type = row.get("interaction_types", "")
                sources = str(row.get("sources", "")).split(",")
                num_sources = len([s for s in sources if s.strip()])

                # FDA-approved drugs in our manual dataset (known approvals for ovarian cancer)
                fda_approved_drugs = {
                    "Olaparib", "Niraparib", "Rucaparib",
                    "Bevacizumab", "Trastuzumab", "Cisplatin", "Carboplatin"
                }
                has_fda = drug in fda_approved_drugs

                score = composite_score(
                    evidence_level="B",  # default since DGIdb doesn't give letter grades
                    num_sources=max(num_sources, 1),
                    has_fda_approval=has_fda
                )

                session.run(
                    """
                    MATCH (g:Gene {name: $gene})
                    MATCH (d:Drug {name: $drug})
                    MERGE (g)-[r:TARGETED_BY]->(d)
                    SET r.evidence_score = $score,
                        r.interaction_type = $interaction_type,
                        r.source = 'DGIdb'
                    """,
                    gene=gene, drug=drug, score=score, interaction_type=interaction_type
                )
                count += 1
        print(f"   Created {count} GENE -> DRUG relationships")

    def load_civic_evidence(self, civic_df):
        """
        Creates evidence relationships from CIViC clinical evidence data.
        Each row becomes a GENE -[:HAS_CLINICAL_EVIDENCE]-> DRUG relationship
        with the real CIViC evidence level.
        """
        with self.driver.session() as session:
            count = 0
            for _, row in civic_df.iterrows():
                molecular_profile = row.get("molecular_profile")
                drugs_raw = row.get("therapies")
                level = row.get("evidence_level")
                disease = row.get("disease")
                significance = row.get("significance")

                if pd.isna(molecular_profile) or pd.isna(drugs_raw):
                    continue

                # molecular_profile looks like "BRCA1 MUT" or "TP53 R273H" - extract just the gene symbol
                gene = str(molecular_profile).split()[0].strip()

                # therapies are comma or 'and' separated drug names
                drug_list = [
                    d.strip() for d in str(drugs_raw).replace(" and ", ",").split(",")
                    if d.strip()
                ]

                if not drug_list:
                    continue

                # Make sure the gene node exists
                session.run("MERGE (g:Gene {name: $gene})", gene=gene)

                for drug in drug_list:
                    session.run("MERGE (d:Drug {name: $drug})", drug=drug)

                    score = composite_score(
                        evidence_level=level,
                        num_sources=1,
                        has_fda_approval=False,
                        clinical_significance=significance
                    )

                    # Tag the relationship direction explicitly for downstream filtering
                    direction = "resistant" if score < 0 else ("supportive" if score > 0 else "neutral")

                    session.run(
                        """
                        MATCH (g:Gene {name: $gene})
                        MATCH (d:Drug {name: $drug})
                        MERGE (g)-[r:HAS_CLINICAL_EVIDENCE]->(d)
                        SET r.evidence_score = $score,
                            r.evidence_level = $level,
                            r.disease = $disease,
                            r.clinical_significance = $significance,
                            r.source = 'CIViC',
                            r.direction = $direction
                        """,
                        gene=gene, drug=drug, score=score,
                        level=str(level), disease=str(disease),
                        significance=str(significance), direction=direction
                    )
                    count += 1
        print(f"   Created {count} CIViC clinical evidence relationships")

    def print_summary(self):
        """Prints basic stats about what's now in the graph."""
        with self.driver.session() as session:
            gene_count = session.run("MATCH (g:Gene) RETURN count(g) AS c").single()["c"]
            drug_count = session.run("MATCH (d:Drug) RETURN count(d) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

            print("\n" + "=" * 50)
            print("KNOWLEDGE GRAPH SUMMARY")
            print("=" * 50)
            print(f"   Gene nodes:        {gene_count}")
            print(f"   Drug nodes:        {drug_count}")
            print(f"   Total relationships: {rel_count}")


def main():
    print("Building Evidence-Aware Knowledge Graph...")
    print("=" * 50)

    # Load our collected data
    print("\n[1/5] Loading data files...")
    dgidb_df = pd.read_csv("data/raw/drugbank/dgidb_interactions.csv")
    civic_df = pd.read_csv("data/raw/civic/civic_ovarian_evidence.csv")
    print(f"   DGIdb interactions: {len(dgidb_df)} rows")
    print(f"   CIViC evidence: {len(civic_df)} rows")

    # Connect to Neo4j
    print("\n[2/5] Connecting to Neo4j...")
    kg = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # Clear and set up
    print("\n[3/5] Preparing graph...")
    kg.clear_database()
    kg.create_constraints()

    # Load nodes and relationships
    print("\n[4/5] Loading nodes and relationships...")
    kg.load_genes_from_dgidb(dgidb_df)
    kg.load_drugs_from_dgidb(dgidb_df)
    kg.load_gene_drug_relationships(dgidb_df)
    kg.load_civic_evidence(civic_df)

    # Summary
    print("\n[5/5] Done!")
    kg.print_summary()

    kg.close()


if __name__ == "__main__":
    main()