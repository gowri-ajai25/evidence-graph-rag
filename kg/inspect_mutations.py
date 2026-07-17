"""
Inspects the downloaded MAF file to see what columns are available
so we know exactly how to extract patient-gene mutation relationships.
"""

import pandas as pd
import gzip
import os

mutation_dir = "data/raw/mutations"
files = os.listdir(mutation_dir)
print("Files found in mutations folder:")
for f in files:
    print(f"   {f}")

# Find the MAF file (could be .maf or .maf.gz)
maf_file = None
for f in files:
    if f.endswith(".maf.gz") or f.endswith(".maf"):
        maf_file = os.path.join(mutation_dir, f)
        break

if not maf_file:
    print("\nNo MAF file found!")
else:
    print(f"\nInspecting: {maf_file}")

    # MAF files have comment lines starting with # before the real header
    if maf_file.endswith(".gz"):
        df = pd.read_csv(maf_file, sep="\t", comment="#", compression="gzip", low_memory=False, nrows=20)
    else:
        df = pd.read_csv(maf_file, sep="\t", comment="#", low_memory=False, nrows=20)

    print(f"\nTotal columns: {len(df.columns)}")
    print("\nKey columns we care about:")
    relevant_keywords = ["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification",
                          "Variant_Type", "Chromosome", "Start_Position", "HGVSp"]
    for col in df.columns:
        if any(keyword.lower() in col.lower() for keyword in relevant_keywords):
            print(f"   {col}")

    print("\nSample rows (first 5):")
    sample_cols = [c for c in df.columns if any(k.lower() in c.lower() for k in relevant_keywords)]
    print(df[sample_cols].head(5).to_string())

    print(f"\nUnique patient barcodes in this sample: {df['Tumor_Sample_Barcode'].nunique() if 'Tumor_Sample_Barcode' in df.columns else 'N/A'}")