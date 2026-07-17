import requests
import json
import os
import pandas as pd

# GDC API base URL - this is the public cancer genomics database
GDC_API = "https://api.gdc.cancer.gov"

print("Starting TCGA-OV data download...")
print("="*50)

# ── 1. CLINICAL DATA ──────────────────────────────
print("\n[1/4] Downloading clinical data...")

clinical_params = {
    "filters": json.dumps({
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "project.project_id", "value": "TCGA-OV"}},
            {"op": "=", "content": {"field": "data_type", "value": "Clinical Supplement"}}
        ]
    }),
    "format": "JSON",
    "fields": "submitter_id,demographic.gender,demographic.vital_status,demographic.days_to_death,diagnoses.age_at_diagnosis,diagnoses.tumor_stage",
    "size": "600"
}

response = requests.get(f"{GDC_API}/cases", params=clinical_params)
clinical_data = response.json()

cases = []
for case in clinical_data["data"]["hits"]:
    row = {"patient_id": case.get("submitter_id", "")}
    
    demo = case.get("demographic", {})
    row["gender"] = demo.get("gender", "")
    row["vital_status"] = demo.get("vital_status", "")
    row["days_to_death"] = demo.get("days_to_death", "")
    
    diagnoses = case.get("diagnoses", [{}])
    if diagnoses:
        row["age_at_diagnosis"] = diagnoses[0].get("age_at_diagnosis", "")
        row["tumor_stage"] = diagnoses[0].get("tumor_stage", "")
    
    cases.append(row)

clinical_df = pd.DataFrame(cases)
clinical_df.to_csv("data/raw/clinical/clinical_data.csv", index=False)
print(f"   Saved {len(clinical_df)} patients to data/raw/clinical/clinical_data.csv")

# ── 2. SOMATIC MUTATIONS ──────────────────────────
print("\n[2/4] Downloading somatic mutation file list...")

mutation_params = {
    "filters": json.dumps({
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-OV"}},
            {"op": "=", "content": {"field": "data_type", "value": "Masked Somatic Mutation"}},
            {"op": "=", "content": {"field": "data_format", "value": "MAF"}}
        ]
    }),
    "format": "JSON",
    "fields": "file_id,file_name,cases.submitter_id",
    "size": "10"
}

response = requests.get(f"{GDC_API}/files", params=mutation_params)
mutation_files = response.json()

hits = mutation_files["data"]["hits"]
print(f"   Found {len(hits)} mutation files")

# Download first mutation file as sample
if hits:
    file_id = hits[0]["file_id"]
    file_name = hits[0]["file_name"]
    
    print(f"   Downloading {file_name}...")
    download_url = f"{GDC_API}/data/{file_id}"
    r = requests.get(download_url, stream=True)
    
    save_path = f"data/raw/mutations/{file_name}"
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"   Saved to {save_path}")

# ── 3. GENE EXPRESSION ────────────────────────────
print("\n[3/4] Downloading gene expression file list...")

expression_params = {
    "filters": json.dumps({
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-OV"}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}}
        ]
    }),
    "format": "JSON",
    "fields": "file_id,file_name,cases.submitter_id",
    "size": "5"
}

response = requests.get(f"{GDC_API}/files", params=expression_params)
expression_files = response.json()

hits = expression_files["data"]["hits"]
print(f"   Found {len(hits)} expression files")

if hits:
    file_id = hits[0]["file_id"]
    file_name = hits[0]["file_name"]
    
    print(f"   Downloading {file_name}...")
    download_url = f"{GDC_API}/data/{file_id}"
    r = requests.get(download_url, stream=True)
    
    save_path = f"data/raw/expression/{file_name}"
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"   Saved to {save_path}")

# ── 4. COPY NUMBER VARIATION ──────────────────────
print("\n[4/4] Downloading copy number variation file list...")

cnv_params = {
    "filters": json.dumps({
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-OV"}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Level Copy Number"}},
        ]
    }),
    "format": "JSON",
    "fields": "file_id,file_name,cases.submitter_id",
    "size": "5"
}

response = requests.get(f"{GDC_API}/files", params=cnv_params)
cnv_files = response.json()

hits = cnv_files["data"]["hits"]
print(f"   Found {len(hits)} CNV files")

if hits:
    file_id = hits[0]["file_id"]
    file_name = hits[0]["file_name"]
    
    print(f"   Downloading {file_name}...")
    download_url = f"{GDC_API}/data/{file_id}"
    r = requests.get(download_url, stream=True)
    
    save_path = f"data/raw/cnv/{file_name}"
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"   Saved to {save_path}")

print("\n" + "="*50)
print("Download complete! Your data folder structure:")
for root, dirs, files in os.walk("data/raw"):
    level = root.replace("data/raw", "").count(os.sep)
    indent = "   " * level
    print(f"{indent}{os.path.basename(root)}/")
    for file in files:
        size = os.path.getsize(os.path.join(root, file))
        print(f"{indent}   {file} ({size:,} bytes)")

