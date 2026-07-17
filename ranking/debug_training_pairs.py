from dotenv import load_dotenv
import pickle
import sys
import os

sys.path.append(os.path.dirname(__file__))

load_dotenv()

with open("data/processed/hetero_graph.pkl", "rb") as f:
    saved = pickle.load(f)

data = saved["data"]
gene_idx = saved["gene_idx"]
drug_idx = saved["drug_idx"]

idx_to_gene = {v: k for k, v in gene_idx.items()}
idx_to_drug = {v: k for k, v in drug_idx.items()}

tp53_idx = gene_idx["TP53"]
cisplatin_idx = drug_idx["Cisplatin"]

print("All edges from TP53 to Cisplatin in the exported graph:\n")

for edge_type in [("gene", "targeted_by", "drug"), ("gene", "has_clinical_evidence", "drug")]:
    edge_index = data[edge_type].edge_index
    edge_attr = data[edge_type].edge_attr.squeeze(-1)
    for i in range(edge_index.shape[1]):
        g = edge_index[0, i].item()
        d = edge_index[1, i].item()
        if g == tp53_idx and d == cisplatin_idx:
            print(f"   {edge_type}: score = {edge_attr[i].item()}")

# Reproduce the FIXED sum-based merge logic from train.py
gene_to_drugs = {}
for edge_type in [("gene", "targeted_by", "drug"), ("gene", "has_clinical_evidence", "drug")]:
    edge_index = data[edge_type].edge_index
    edge_attr = data[edge_type].edge_attr.squeeze(-1)
    for i in range(edge_index.shape[1]):
        g = edge_index[0, i].item()
        d = edge_index[1, i].item()
        score = edge_attr[i].item()
        gene_to_drugs.setdefault(g, {})
        gene_to_drugs[g][d] = gene_to_drugs[g].get(d, 0.0) + score

for g in gene_to_drugs:
    for d in gene_to_drugs[g]:
        gene_to_drugs[g][d] = max(-1.0, min(1.0, gene_to_drugs[g][d]))

print(f"\nAfter train.py's FIXED sum-based merge logic:")
print(f"   TP53 -> Cisplatin final training label = {gene_to_drugs[tp53_idx][cisplatin_idx]}")