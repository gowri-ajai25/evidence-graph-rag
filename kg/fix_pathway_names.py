"""
Fetches human-readable names for all KEGG pathway IDs
and updates the gene_pathways.csv file.
"""

import requests
import pandas as pd
import time

df = pd.read_csv("data/raw/pathways/gene_pathways.csv")
print(f"Total rows: {len(df)}")

unique_ids = df["pathway_id"].unique()
print(f"Unique pathway IDs to name: {len(unique_ids)}")

pathway_names = {}

for i, pid in enumerate(unique_ids, 1):
    try:
        r = requests.get(f"https://rest.kegg.jp/list/{pid}", timeout=15)
        time.sleep(0.2)
        if r.status_code == 200 and "\t" in r.text:
            name = r.text.strip().split("\t")[1]
            # Clean up the name — KEGG appends "- Homo sapiens (human)"
            name = name.replace(" - Homo sapiens (human)", "").strip()
            pathway_names[pid] = name
        else:
            pathway_names[pid] = pid
    except:
        pathway_names[pid] = pid

    if i % 20 == 0:
        print(f"   Fetched {i}/{len(unique_ids)} pathway names...")

print(f"\nFetched {len(pathway_names)} pathway names")
print("\nSample:")
for pid, name in list(pathway_names.items())[:10]:
    print(f"   {pid} -> {name}")

# Update the CSV
df["pathway_name"] = df["pathway_id"].map(pathway_names)
df.to_csv("data/raw/pathways/gene_pathways.csv", index=False)
print(f"\nUpdated gene_pathways.csv with real pathway names")