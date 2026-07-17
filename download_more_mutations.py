"""
Downloads MAF mutation files for 50 TCGA-OV patients.
Each file represents one patient's somatic mutations.
"""

import requests
import json
import os

GDC_API = "https://api.gdc.cancer.gov"
SAVE_DIR = "data/raw/mutations"

print("Fetching list of TCGA-OV mutation files...")

params = {
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
    "size": "200"
}

response = requests.get(f"{GDC_API}/files", params=params)
files_data = response.json()
hits = files_data["data"]["hits"]

print(f"Found {len(hits)} mutation files. Downloading...\n")

downloaded = 0
for i, hit in enumerate(hits, 1):
    file_id = hit["file_id"]
    file_name = hit["file_name"]
    save_path = os.path.join(SAVE_DIR, file_name)

    if os.path.exists(save_path):
        print(f"[{i}/{len(hits)}] Already have {file_name}, skipping")
        downloaded += 1
        continue

    download_url = f"{GDC_API}/data/{file_id}"
    r = requests.get(download_url, stream=True)

    if r.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[{i}/{len(hits)}] Downloaded {file_name}")
        downloaded += 1
    else:
        print(f"[{i}/{len(hits)}] Failed: {file_name} (status {r.status_code})")

print(f"\nDone! {downloaded}/{len(hits)} mutation files in {SAVE_DIR}")