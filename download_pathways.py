"""
Downloads Gene -> Pathway and Pathway -> Drug associations.
Uses KEGG REST API (corrected endpoint) with curated fallback.
"""

import requests
import pandas as pd
import os
import time

os.makedirs("data/raw/pathways", exist_ok=True)

OVARIAN_GENES = [
    "BRCA1", "BRCA2", "TP53", "PTEN", "PIK3CA",
    "VEGFA", "EGFR", "ERBB2", "CDK4", "CDK6",
    "ATM", "RAD51", "PALB2", "RAD51C", "BRIP1",
    "CDK12", "KRAS", "ASS1", "SYK"
]

print("Downloading pathway data...")
print("=" * 60)

# ── 1. KEGG: Gene -> Pathway (corrected approach) ────────────
print("\n[1/2] Downloading Gene -> Pathway from KEGG...")
print("      Using KEGG gene search (hsa: human gene IDs)...")

gene_pathway_rows = []

# KEGG uses its own gene IDs - first get the KEGG ID for each gene symbol
for gene in OVARIAN_GENES:
    try:
        # Step 1: find the KEGG human gene entry
        search_url = f"https://rest.kegg.jp/find/hsa/{gene}"
        r = requests.get(search_url, timeout=15)
        time.sleep(0.3)  # be polite to KEGG's servers

        if r.status_code != 200 or not r.text.strip():
            print(f"   {gene}: not found in KEGG")
            continue

        # Take the first matching human gene entry
        first_line = r.text.strip().split("\n")[0]
        kegg_id = first_line.split("\t")[0].strip()  # e.g. "hsa:672"

        if not kegg_id.startswith("hsa:"):
            print(f"   {gene}: unexpected format {kegg_id}")
            continue

        # Step 2: get pathways for this gene
        pathway_url = f"https://rest.kegg.jp/link/pathway/{kegg_id}"
        r2 = requests.get(pathway_url, timeout=15)
        time.sleep(0.3)

        if r2.status_code != 200 or not r2.text.strip():
            print(f"   {gene} ({kegg_id}): no pathways")
            continue

        count = 0
        for line in r2.text.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) == 2:
                pathway_id = parts[1].strip().replace("path:", "")
                if pathway_id.startswith("hsa"):
                    gene_pathway_rows.append({
                        "gene": gene,
                        "pathway_id": pathway_id,
                        "source": "KEGG"
                    })
                    count += 1

        print(f"   {gene} ({kegg_id}): {count} pathways")

    except Exception as e:
        print(f"   {gene}: error — {e}")

# Step 3: get pathway names
print("\n   Getting pathway names...")
pathway_ids = list(set(r["pathway_id"] for r in gene_pathway_rows))
pathway_names = {}

for pid in pathway_ids:
    try:
        r = requests.get(f"https://rest.kegg.jp/list/{pid}", timeout=15)
        time.sleep(0.2)
        if r.status_code == 200 and r.text.strip():
            name = r.text.strip().split("\t")[1] if "\t" in r.text else pid
            pathway_names[pid] = name
    except:
        pathway_names[pid] = pid

for row in gene_pathway_rows:
    row["pathway_name"] = pathway_names.get(row["pathway_id"], row["pathway_id"])

gene_pathway_df = pd.DataFrame(gene_pathway_rows) if gene_pathway_rows else pd.DataFrame(
    columns=["gene", "pathway_id", "pathway_name", "source"])

print(f"\n   Total gene-pathway associations: {len(gene_pathway_df)}")

# If KEGG returned nothing, use a curated set based on known biology
if len(gene_pathway_df) == 0:
    print("   KEGG returned no results. Using curated gene-pathway data...")
    gene_pathway_rows = [
        {"gene": "BRCA1", "pathway_id": "hsa03440", "pathway_name": "Homologous recombination", "source": "curated"},
        {"gene": "BRCA1", "pathway_id": "hsa04110", "pathway_name": "Cell cycle", "source": "curated"},
        {"gene": "BRCA1", "pathway_id": "hsa03460", "pathway_name": "Fanconi anemia pathway", "source": "curated"},
        {"gene": "BRCA2", "pathway_id": "hsa03440", "pathway_name": "Homologous recombination", "source": "curated"},
        {"gene": "BRCA2", "pathway_id": "hsa03460", "pathway_name": "Fanconi anemia pathway", "source": "curated"},
        {"gene": "TP53",  "pathway_id": "hsa04115", "pathway_name": "p53 signaling pathway", "source": "curated"},
        {"gene": "TP53",  "pathway_id": "hsa04110", "pathway_name": "Cell cycle", "source": "curated"},
        {"gene": "TP53",  "pathway_id": "hsa04210", "pathway_name": "Apoptosis", "source": "curated"},
        {"gene": "PTEN",  "pathway_id": "hsa04151", "pathway_name": "PI3K-Akt signaling pathway", "source": "curated"},
        {"gene": "PIK3CA","pathway_id": "hsa04151", "pathway_name": "PI3K-Akt signaling pathway", "source": "curated"},
        {"gene": "VEGFA", "pathway_id": "hsa04370", "pathway_name": "VEGF signaling pathway", "source": "curated"},
        {"gene": "EGFR",  "pathway_id": "hsa04012", "pathway_name": "ErbB signaling pathway", "source": "curated"},
        {"gene": "ERBB2", "pathway_id": "hsa04012", "pathway_name": "ErbB signaling pathway", "source": "curated"},
        {"gene": "CDK4",  "pathway_id": "hsa04110", "pathway_name": "Cell cycle", "source": "curated"},
        {"gene": "CDK6",  "pathway_id": "hsa04110", "pathway_name": "Cell cycle", "source": "curated"},
        {"gene": "ATM",   "pathway_id": "hsa03440", "pathway_name": "Homologous recombination", "source": "curated"},
        {"gene": "ATM",   "pathway_id": "hsa04115", "pathway_name": "p53 signaling pathway", "source": "curated"},
        {"gene": "RAD51", "pathway_id": "hsa03440", "pathway_name": "Homologous recombination", "source": "curated"},
        {"gene": "PALB2", "pathway_id": "hsa03440", "pathway_name": "Homologous recombination", "source": "curated"},
        {"gene": "RAD51C","pathway_id": "hsa03440", "pathway_name": "Homologous recombination", "source": "curated"},
        {"gene": "BRIP1", "pathway_id": "hsa03460", "pathway_name": "Fanconi anemia pathway", "source": "curated"},
        {"gene": "CDK12", "pathway_id": "hsa04110", "pathway_name": "Cell cycle", "source": "curated"},
        {"gene": "KRAS",  "pathway_id": "hsa04010", "pathway_name": "MAPK signaling pathway", "source": "curated"},
        {"gene": "KRAS",  "pathway_id": "hsa04151", "pathway_name": "PI3K-Akt signaling pathway", "source": "curated"},
    ]
    gene_pathway_df = pd.DataFrame(gene_pathway_rows)
    print(f"   Using {len(gene_pathway_df)} curated gene-pathway associations")

gene_pathway_df.to_csv("data/raw/pathways/gene_pathways.csv", index=False)
print(f"   Saved to data/raw/pathways/gene_pathways.csv")

# ── 2. Pathway -> Drug (curated, biologically accurate) ──────
print("\n[2/2] Building Pathway -> Drug associations...")
print("      Using curated data based on published pathway-drug biology...")

pathway_drug_rows = [
    # Homologous Recombination -> PARP inhibitors
    {"pathway_id": "hsa03440", "pathway_name": "Homologous recombination", "drug_name": "Olaparib", "source": "curated"},
    {"pathway_id": "hsa03440", "pathway_name": "Homologous recombination", "drug_name": "Niraparib", "source": "curated"},
    {"pathway_id": "hsa03440", "pathway_name": "Homologous recombination", "drug_name": "Rucaparib", "source": "curated"},
    {"pathway_id": "hsa03440", "pathway_name": "Homologous recombination", "drug_name": "Talazoparib", "source": "curated"},
    # Fanconi anemia -> PARP inhibitors + platinum
    {"pathway_id": "hsa03460", "pathway_name": "Fanconi anemia pathway", "drug_name": "Olaparib", "source": "curated"},
    {"pathway_id": "hsa03460", "pathway_name": "Fanconi anemia pathway", "drug_name": "Cisplatin", "source": "curated"},
    {"pathway_id": "hsa03460", "pathway_name": "Fanconi anemia pathway", "drug_name": "Carboplatin", "source": "curated"},
    # p53 signaling -> platinum, MDM2 inhibitors
    {"pathway_id": "hsa04115", "pathway_name": "p53 signaling pathway", "drug_name": "Cisplatin", "source": "curated"},
    {"pathway_id": "hsa04115", "pathway_name": "p53 signaling pathway", "drug_name": "Carboplatin", "source": "curated"},
    {"pathway_id": "hsa04115", "pathway_name": "p53 signaling pathway", "drug_name": "Paclitaxel", "source": "curated"},
    # Cell cycle -> CDK inhibitors
    {"pathway_id": "hsa04110", "pathway_name": "Cell cycle", "drug_name": "Palbociclib", "source": "curated"},
    {"pathway_id": "hsa04110", "pathway_name": "Cell cycle", "drug_name": "Ribociclib", "source": "curated"},
    {"pathway_id": "hsa04110", "pathway_name": "Cell cycle", "drug_name": "Paclitaxel", "source": "curated"},
    # PI3K-AKT -> PI3K inhibitors, mTOR inhibitors
    {"pathway_id": "hsa04151", "pathway_name": "PI3K-Akt signaling pathway", "drug_name": "Alpelisib", "source": "curated"},
    {"pathway_id": "hsa04151", "pathway_name": "PI3K-Akt signaling pathway", "drug_name": "Everolimus", "source": "curated"},
    {"pathway_id": "hsa04151", "pathway_name": "PI3K-Akt signaling pathway", "drug_name": "Idelalisib", "source": "curated"},
    # VEGF -> anti-angiogenics
    {"pathway_id": "hsa04370", "pathway_name": "VEGF signaling pathway", "drug_name": "Bevacizumab", "source": "curated"},
    {"pathway_id": "hsa04370", "pathway_name": "VEGF signaling pathway", "drug_name": "Cediranib", "source": "curated"},
    # ErbB/EGFR -> EGFR inhibitors
    {"pathway_id": "hsa04012", "pathway_name": "ErbB signaling pathway", "drug_name": "Erlotinib", "source": "curated"},
    {"pathway_id": "hsa04012", "pathway_name": "ErbB signaling pathway", "drug_name": "Gefitinib", "source": "curated"},
    {"pathway_id": "hsa04012", "pathway_name": "ErbB signaling pathway", "drug_name": "Trastuzumab", "source": "curated"},
    # Apoptosis -> general cytotoxics
    {"pathway_id": "hsa04210", "pathway_name": "Apoptosis", "drug_name": "Cisplatin", "source": "curated"},
    {"pathway_id": "hsa04210", "pathway_name": "Apoptosis", "drug_name": "Paclitaxel", "source": "curated"},
    # MAPK -> MEK/BRAF inhibitors
    {"pathway_id": "hsa04010", "pathway_name": "MAPK signaling pathway", "drug_name": "Selumetinib", "source": "curated"},
]

pathway_drug_df = pd.DataFrame(pathway_drug_rows)
pathway_drug_df.to_csv("data/raw/pathways/pathway_drugs.csv", index=False)
print(f"   {len(pathway_drug_df)} pathway-drug associations saved")
print(f"   Saved to data/raw/pathways/pathway_drugs.csv")

print("\n" + "=" * 60)
print("Summary:")
print(f"   Gene-Pathway associations: {len(gene_pathway_df)}")
print(f"   Unique pathways: {gene_pathway_df['pathway_name'].nunique()}")
print(f"   Pathway-Drug associations: {len(pathway_drug_df)}")
print(f"   Unique drugs in pathway data: {pathway_drug_df['drug_name'].nunique()}")