import requests
import pandas as pd
import os

print("Starting drug-gene data download...")
print("="*50)

# ── 1. CIViC DATA ─────────────────────────────────
print("\n[1/2] Downloading CIViC data...")
print("      Using CIViC bulk download (most reliable method)")

# CIViC provides bulk TSV downloads - much more reliable than API
CIVIC_EVIDENCE_URL = "https://civicdb.org/downloads/nightly/nightly-ClinicalEvidenceSummaries.tsv"
CIVIC_GENES_URL = "https://civicdb.org/downloads/nightly/nightly-GeneSummaries.tsv"

# Download evidence summaries
print("   Downloading evidence summaries...")
response = requests.get(CIVIC_EVIDENCE_URL)
with open("data/raw/civic/civic_evidence_raw.tsv", "wb") as f:
    f.write(response.content)
print(f"   Saved raw evidence file ({len(response.content):,} bytes)")

# Load and filter for ovarian cancer
civic_df = pd.read_csv("data/raw/civic/civic_evidence_raw.tsv", sep="\t", low_memory=False)
print(f"   Total CIViC evidence items: {len(civic_df)}")

# Filter for ovarian cancer related entries
ovarian_mask = civic_df["disease"].str.contains(
    "ovarian|Ovarian|OVARIAN", na=False, case=False
)
civic_ovarian = civic_df[ovarian_mask].copy()
civic_ovarian.to_csv("data/raw/civic/civic_ovarian_evidence.csv", index=False)
print(f"   Ovarian cancer evidence items: {len(civic_ovarian)}")
print(f"   Saved to data/raw/civic/civic_ovarian_evidence.csv")

# Download gene summaries
print("\n   Downloading gene summaries...")
response = requests.get(CIVIC_GENES_URL)
with open("data/raw/civic/civic_genes_raw.tsv", "wb") as f:
    f.write(response.content)

genes_df = pd.read_csv("data/raw/civic/civic_genes_raw.tsv", sep="\t", low_memory=False)
genes_df.to_csv("data/raw/civic/civic_genes.csv", index=False)
print(f"   Saved {len(genes_df)} genes to data/raw/civic/civic_genes.csv")

# Show sample of ovarian cancer data
print("\n   Sample ovarian cancer entries:")
if len(civic_ovarian) > 0:
    cols = ["gene", "variant", "disease", "drugs", "evidence_type", "evidence_level"]
    available_cols = [c for c in cols if c in civic_ovarian.columns]
    print(civic_ovarian[available_cols].head(3).to_string())

# ── 2. DGIdb DATA ─────────────────────────────────
print("\n[2/2] Downloading DGIdb drug-gene interactions...")
print("      (Drug Gene Interaction Database - completely free)")

# DGIdb bulk download - most reliable
DGIDB_URL = "https://dgidb.org/data/monthly_tsvs/2024-Jan/interactions.tsv"

print("   Downloading DGIdb interactions...")
response = requests.get(DGIDB_URL)

if response.status_code == 200:
    with open("data/raw/drugbank/dgidb_raw.tsv", "wb") as f:
        f.write(response.content)
    
    dgidb_df = pd.read_csv(
        "data/raw/drugbank/dgidb_raw.tsv",
        sep="\t",
        low_memory=False
    )
    print(f"   Total DGIdb interactions: {len(dgidb_df)}")
    
    # Filter for our key ovarian cancer genes
    ovarian_genes = [
        "BRCA1", "BRCA2", "TP53", "PTEN", "PIK3CA",
        "VEGFA", "EGFR", "ERBB2", "CDK4", "CDK6",
        "ATM", "RAD51", "PALB2", "RAD51C", "BRIP1"
    ]
    
    dgidb_filtered = dgidb_df[
        dgidb_df["gene_name"].isin(ovarian_genes)
    ].copy()
    
    dgidb_filtered.to_csv("data/raw/drugbank/dgidb_interactions.csv", index=False)
    print(f"   Ovarian cancer gene interactions: {len(dgidb_filtered)}")
    print(f"   Saved to data/raw/drugbank/dgidb_interactions.csv")
    
    # Show sample
    print("\n   Sample interactions:")
    print(dgidb_filtered[["gene_name", "drug_name", "interaction_types"]].head(5).to_string())

else:
    print(f"   DGIdb bulk download failed (status {response.status_code})")
    print("   Trying alternative URL...")
    
    # Alternative - try latest
    ALT_URL = "https://dgidb.org/data/monthly_tsvs/2023-Jan/interactions.tsv"
    response = requests.get(ALT_URL)
    
    if response.status_code == 200:
        with open("data/raw/drugbank/dgidb_raw.tsv", "wb") as f:
            f.write(response.content)
        dgidb_df = pd.read_csv("data/raw/drugbank/dgidb_raw.tsv", sep="\t", low_memory=False)
        
        ovarian_genes = [
            "BRCA1", "BRCA2", "TP53", "PTEN", "PIK3CA",
            "VEGFA", "EGFR", "ERBB2", "CDK4", "CDK6",
            "ATM", "RAD51", "PALB2", "RAD51C", "BRIP1"
        ]
        dgidb_filtered = dgidb_df[dgidb_df["gene_name"].isin(ovarian_genes)].copy()
        dgidb_filtered.to_csv("data/raw/drugbank/dgidb_interactions.csv", index=False)
        print(f"   Saved {len(dgidb_filtered)} interactions")
    else:
        print("   Could not download DGIdb data. Will handle in next step.")

# ── SUMMARY ───────────────────────────────────────
print("\n" + "="*50)
print("Summary:")
print(f"   CIViC ovarian evidence:    {len(civic_ovarian)}")
print(f"   CIViC genes:               {len(genes_df)}")
print("\nFiles saved:")
for root, dirs, files in os.walk("data/raw"):
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath)
        print(f"   {filepath} ({size:,} bytes)")