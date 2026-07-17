"""
Resets pathway data using the biologically accurate curated dataset,
bypassing KEGG API issues entirely.
"""

from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

# ── Curated gene-pathway associations (biologically verified) ──
gene_pathway_data = [
    # BRCA1/2 and HR genes -> Homologous Recombination
    {"gene": "BRCA1", "pathway_id": "hsa03440", "pathway_name": "Homologous Recombination"},
    {"gene": "BRCA2", "pathway_id": "hsa03440", "pathway_name": "Homologous Recombination"},
    {"gene": "RAD51", "pathway_id": "hsa03440", "pathway_name": "Homologous Recombination"},
    {"gene": "PALB2", "pathway_id": "hsa03440", "pathway_name": "Homologous Recombination"},
    {"gene": "RAD51C","pathway_id": "hsa03440", "pathway_name": "Homologous Recombination"},
    {"gene": "ATM",   "pathway_id": "hsa03440", "pathway_name": "Homologous Recombination"},
    {"gene": "CDK12", "pathway_id": "hsa03440", "pathway_name": "Homologous Recombination"},
    # Fanconi Anemia pathway
    {"gene": "BRCA1", "pathway_id": "hsa03460", "pathway_name": "Fanconi Anemia Pathway"},
    {"gene": "BRCA2", "pathway_id": "hsa03460", "pathway_name": "Fanconi Anemia Pathway"},
    {"gene": "BRIP1", "pathway_id": "hsa03460", "pathway_name": "Fanconi Anemia Pathway"},
    {"gene": "PALB2", "pathway_id": "hsa03460", "pathway_name": "Fanconi Anemia Pathway"},
    # p53 signaling
    {"gene": "TP53",  "pathway_id": "hsa04115", "pathway_name": "p53 Signaling Pathway"},
    {"gene": "ATM",   "pathway_id": "hsa04115", "pathway_name": "p53 Signaling Pathway"},
    {"gene": "BRCA1", "pathway_id": "hsa04115", "pathway_name": "p53 Signaling Pathway"},
    # Cell cycle
    {"gene": "TP53",  "pathway_id": "hsa04110", "pathway_name": "Cell Cycle"},
    {"gene": "CDK4",  "pathway_id": "hsa04110", "pathway_name": "Cell Cycle"},
    {"gene": "CDK6",  "pathway_id": "hsa04110", "pathway_name": "Cell Cycle"},
    {"gene": "CDK12", "pathway_id": "hsa04110", "pathway_name": "Cell Cycle"},
    # PI3K-AKT signaling
    {"gene": "PTEN",  "pathway_id": "hsa04151", "pathway_name": "PI3K-AKT Signaling Pathway"},
    {"gene": "PIK3CA","pathway_id": "hsa04151", "pathway_name": "PI3K-AKT Signaling Pathway"},
    {"gene": "KRAS",  "pathway_id": "hsa04151", "pathway_name": "PI3K-AKT Signaling Pathway"},
    # VEGF signaling
    {"gene": "VEGFA", "pathway_id": "hsa04370", "pathway_name": "VEGF Signaling Pathway"},
    # ErbB/EGFR signaling
    {"gene": "EGFR",  "pathway_id": "hsa04012", "pathway_name": "ErbB Signaling Pathway"},
    {"gene": "ERBB2", "pathway_id": "hsa04012", "pathway_name": "ErbB Signaling Pathway"},
    {"gene": "KRAS",  "pathway_id": "hsa04012", "pathway_name": "ErbB Signaling Pathway"},
    # MAPK signaling
    {"gene": "KRAS",  "pathway_id": "hsa04010", "pathway_name": "MAPK Signaling Pathway"},
    {"gene": "EGFR",  "pathway_id": "hsa04010", "pathway_name": "MAPK Signaling Pathway"},
    {"gene": "BRAF",  "pathway_id": "hsa04010", "pathway_name": "MAPK Signaling Pathway"},
    # Apoptosis
    {"gene": "TP53",  "pathway_id": "hsa04210", "pathway_name": "Apoptosis"},
    {"gene": "PTEN",  "pathway_id": "hsa04210", "pathway_name": "Apoptosis"},
    {"gene": "EGFR",  "pathway_id": "hsa04210", "pathway_name": "Apoptosis"},
]

# ── Curated pathway-drug associations ─────────────────────────
pathway_drug_data = [
    {"pathway_id": "hsa03440", "pathway_name": "Homologous Recombination", "drug_name": "Olaparib"},
    {"pathway_id": "hsa03440", "pathway_name": "Homologous Recombination", "drug_name": "Niraparib"},
    {"pathway_id": "hsa03440", "pathway_name": "Homologous Recombination", "drug_name": "Rucaparib"},
    {"pathway_id": "hsa03440", "pathway_name": "Homologous Recombination", "drug_name": "Talazoparib"},
    {"pathway_id": "hsa03460", "pathway_name": "Fanconi Anemia Pathway",   "drug_name": "Olaparib"},
    {"pathway_id": "hsa03460", "pathway_name": "Fanconi Anemia Pathway",   "drug_name": "Cisplatin"},
    {"pathway_id": "hsa03460", "pathway_name": "Fanconi Anemia Pathway",   "drug_name": "Carboplatin"},
    {"pathway_id": "hsa04115", "pathway_name": "p53 Signaling Pathway",    "drug_name": "Cisplatin"},
    {"pathway_id": "hsa04115", "pathway_name": "p53 Signaling Pathway",    "drug_name": "Carboplatin"},
    {"pathway_id": "hsa04115", "pathway_name": "p53 Signaling Pathway",    "drug_name": "Paclitaxel"},
    {"pathway_id": "hsa04110", "pathway_name": "Cell Cycle",               "drug_name": "Palbociclib"},
    {"pathway_id": "hsa04110", "pathway_name": "Cell Cycle",               "drug_name": "Ribociclib"},
    {"pathway_id": "hsa04110", "pathway_name": "Cell Cycle",               "drug_name": "Paclitaxel"},
    {"pathway_id": "hsa04151", "pathway_name": "PI3K-AKT Signaling Pathway","drug_name": "Alpelisib"},
    {"pathway_id": "hsa04151", "pathway_name": "PI3K-AKT Signaling Pathway","drug_name": "Everolimus"},
    {"pathway_id": "hsa04151", "pathway_name": "PI3K-AKT Signaling Pathway","drug_name": "Idelalisib"},
    {"pathway_id": "hsa04370", "pathway_name": "VEGF Signaling Pathway",   "drug_name": "Bevacizumab"},
    {"pathway_id": "hsa04370", "pathway_name": "VEGF Signaling Pathway",   "drug_name": "Cediranib"},
    {"pathway_id": "hsa04012", "pathway_name": "ErbB Signaling Pathway",   "drug_name": "Erlotinib"},
    {"pathway_id": "hsa04012", "pathway_name": "ErbB Signaling Pathway",   "drug_name": "Gefitinib"},
    {"pathway_id": "hsa04012", "pathway_name": "ErbB Signaling Pathway",   "drug_name": "Trastuzumab"},
    {"pathway_id": "hsa04010", "pathway_name": "MAPK Signaling Pathway",   "drug_name": "Selumetinib"},
    {"pathway_id": "hsa04210", "pathway_name": "Apoptosis",                "drug_name": "Cisplatin"},
    {"pathway_id": "hsa04210", "pathway_name": "Apoptosis",                "drug_name": "Paclitaxel"},
]

print("Resetting pathway data in Neo4j...")
print("=" * 60)

with driver.session() as session:
    # Remove old pathway nodes and relationships cleanly
    print("\n[1/4] Clearing old pathway data...")
    session.run("MATCH (p:Pathway) DETACH DELETE p")
    print("   Done")

    # Create constraint
    session.run(
        "CREATE CONSTRAINT pathway_id IF NOT EXISTS "
        "FOR (p:Pathway) REQUIRE p.id IS UNIQUE"
    )

    # Create Pathway nodes with correct names
    print("\n[2/4] Creating Pathway nodes...")
    unique_pathways = {r["pathway_id"]: r["pathway_name"] for r in gene_pathway_data}
    for pid, pname in unique_pathways.items():
        session.run(
            "MERGE (p:Pathway {id: $id}) SET p.name = $name",
            id=pid, name=pname
        )
    print(f"   Created {len(unique_pathways)} Pathway nodes")

    # Gene -> Pathway
    print("\n[3/4] Creating Gene -> Pathway relationships...")
    count = 0
    for row in gene_pathway_data:
        result = session.run(
            """
            MATCH (g:Gene {name: $gene})
            MATCH (p:Pathway {id: $pid})
            MERGE (g)-[r:INVOLVED_IN]->(p)
            SET r.source = 'curated'
            RETURN r
            """,
            gene=row["gene"], pid=row["pathway_id"]
        )
        if result.single():
            count += 1
    print(f"   Created {count} Gene -> Pathway relationships")

    # Pathway -> Drug
    print("\n[4/4] Creating Pathway -> Drug relationships...")
    count = 0
    skipped = 0
    for row in pathway_drug_data:
        result = session.run(
            """
            MATCH (p:Pathway {id: $pid})
            MATCH (d:Drug {name: $drug})
            MERGE (p)-[r:ASSOCIATED_WITH_DRUG]->(d)
            SET r.source = 'curated'
            RETURN r
            """,
            pid=row["pathway_id"], drug=row["drug_name"]
        )
        if result.single():
            count += 1
        else:
            skipped += 1
            print(f"   Skipped: {row['pathway_name']} -> {row['drug_name']} (drug not in KG)")
    print(f"   Created {count} Pathway -> Drug relationships")

    # Verify 3-hop chain
    print("\n   Verifying 3-hop chain...")
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
        print("   No 3-hop paths found")

    # Final counts
    total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    pathway_count = session.run("MATCH (p:Pathway) RETURN count(p) AS c").single()["c"]

print(f"\n{'='*60}")
print(f"Final graph: {total_nodes} nodes, {total_rels} relationships")
print(f"Pathway nodes: {pathway_count}")

driver.close()