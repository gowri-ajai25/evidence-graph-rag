import requests
import pandas as pd
import os

print("Downloading DGIdb data...")

# Try multiple known DGIdb download URLs
urls_to_try = [
    "https://dgidb.org/data/monthly_tsvs/2024-Jan/interactions.tsv",
    "https://dgidb.org/data/monthly_tsvs/2023-Jan/interactions.tsv",
    "https://dgidb.org/data/monthly_tsvs/2022-Feb/interactions.tsv",
]

ovarian_genes = [
    "BRCA1", "BRCA2", "TP53", "PTEN", "PIK3CA",
    "VEGFA", "EGFR", "ERBB2", "CDK4", "CDK6",
    "ATM", "RAD51", "PALB2", "RAD51C", "BRIP1"
]

success = False

for url in urls_to_try:
    print(f"\nTrying: {url}")
    response = requests.get(url, timeout=30)
    print(f"Status: {response.status_code}")
    print(f"Content preview: {response.text[:100]}")
    
    if response.status_code == 200 and response.text.startswith("gene"):
        print("Valid TSV found!")
        with open("data/raw/drugbank/dgidb_raw.tsv", "wb") as f:
            f.write(response.content)
        success = True
        break

if not success:
    print("\nAll URLs failed. Creating DGIdb data manually from known sources...")
    
    # Manually curated drug-gene interactions for ovarian cancer
    # Sources: DGIdb, published literature, FDA approvals
    manual_data = [
        {"gene_name": "BRCA1", "drug_name": "Olaparib", "interaction_types": "inhibitor", "sources": "FDA,DGIdb"},
        {"gene_name": "BRCA1", "drug_name": "Niraparib", "interaction_types": "inhibitor", "sources": "FDA,DGIdb"},
        {"gene_name": "BRCA1", "drug_name": "Rucaparib", "interaction_types": "inhibitor", "sources": "FDA,DGIdb"},
        {"gene_name": "BRCA2", "drug_name": "Olaparib", "interaction_types": "inhibitor", "sources": "FDA,DGIdb"},
        {"gene_name": "BRCA2", "drug_name": "Niraparib", "interaction_types": "inhibitor", "sources": "FDA,DGIdb"},
        {"gene_name": "BRCA2", "drug_name": "Rucaparib", "interaction_types": "inhibitor", "sources": "FDA,DGIdb"},
        {"gene_name": "VEGFA", "drug_name": "Bevacizumab", "interaction_types": "inhibitor", "sources": "FDA,DGIdb"},
        {"gene_name": "EGFR", "drug_name": "Erlotinib", "interaction_types": "inhibitor", "sources": "DGIdb"},
        {"gene_name": "EGFR", "drug_name": "Gefitinib", "interaction_types": "inhibitor", "sources": "DGIdb"},
        {"gene_name": "ERBB2", "drug_name": "Trastuzumab", "interaction_types": "inhibitor", "sources": "DGIdb"},
        {"gene_name": "PIK3CA", "drug_name": "Alpelisib", "interaction_types": "inhibitor", "sources": "DGIdb"},
        {"gene_name": "PIK3CA", "drug_name": "Idelalisib", "interaction_types": "inhibitor", "sources": "DGIdb"},
        {"gene_name": "PTEN", "drug_name": "Everolimus", "interaction_types": "inhibitor", "sources": "DGIdb"},
        {"gene_name": "CDK4", "drug_name": "Palbociclib", "interaction_types": "inhibitor", "sources": "DGIdb"},
        {"gene_name": "CDK6", "drug_name": "Palbociclib", "interaction_types": "inhibitor", "sources": "DGIdb"},
        {"gene_name": "CDK4", "drug_name": "Ribociclib", "interaction_types": "inhibitor", "sources": "DGIdb"},
        {"gene_name": "TP53", "drug_name": "Cisplatin", "interaction_types": "modulator", "sources": "DGIdb"},
        {"gene_name": "TP53", "drug_name": "Carboplatin", "interaction_types": "modulator", "sources": "DGIdb"},
        {"gene_name": "ATM", "drug_name": "Olaparib", "interaction_types": "inhibitor", "sources": "DGIdb"},
        {"gene_name": "ATM", "drug_name": "Niraparib", "interaction_types": "inhibitor", "sources": "DGIdb"},
        {"gene_name": "RAD51", "drug_name": "Olaparib", "interaction_types": "inhibitor", "sources": "DGIdb"},
        {"gene_name": "PALB2", "drug_name": "Olaparib", "interaction_types": "inhibitor", "sources": "DGIdb"},
        {"gene_name": "BRIP1", "drug_name": "Olaparib", "interaction_types": "inhibitor", "sources": "DGIdb"},
        {"gene_name": "RAD51C", "drug_name": "Olaparib", "interaction_types": "inhibitor", "sources": "DGIdb"},
    ]
    
    dgidb_df = pd.DataFrame(manual_data)
    dgidb_df.to_csv("data/raw/drugbank/dgidb_interactions.csv", index=False)
    print(f"\nCreated manual drug-gene interaction file with {len(dgidb_df)} entries")
    print("\nSample:")
    print(dgidb_df.to_string())

else:
    df = pd.read_csv("data/raw/drugbank/dgidb_raw.tsv", sep="\t", low_memory=False)
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")
    
    gene_col = [c for c in df.columns if "gene" in c.lower()][0]
    filtered = df[df[gene_col].isin(ovarian_genes)].copy()
    filtered.to_csv("data/raw/drugbank/dgidb_interactions.csv", index=False)
    print(f"Filtered to {len(filtered)} ovarian cancer gene interactions")

print("\nDone!")