"""
Checks how positive (patient, drug) labels are distributed across patients,
WITHOUT re-running GNN training. Reuses train.py's own pair-building
functions so the counts are guaranteed consistent with what training
actually used.

Place this file in the same folder as train.py (e.g. ranking/), then run:
    python ranking/check_positive_distribution.py
"""

import pickle
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(__file__))
from train import load_graph, build_gene_drug_evidence, build_patient_gene_map, build_patient_drug_pairs


def main():
    print("Loading graph and rebuilding training pairs (no training performed)...")
    print("=" * 70)

    data, patient_idx, gene_idx, drug_idx = load_graph()
    num_patients, num_genes, num_drugs = len(patient_idx), len(gene_idx), len(drug_idx)
    print(f"Total patients in graph: {num_patients}")

    gene_to_drugs = build_gene_drug_evidence(data, gene_idx, drug_idx)
    patient_to_genes = build_patient_gene_map(data)

    positives, negatives = build_patient_drug_pairs(
        patient_to_genes, gene_to_drugs, num_drugs, num_patients
    )

    print(f"\nTotal positive pairs: {len(positives)}")
    print(f"Total negative pairs: {len(negatives)}")

    # How many DISTINCT patients have at least one positive (net-positive-labeled) pair?
    # NOTE: positives here includes ALL reachable (patient, drug, score) tuples,
    # both positive-score and negative-score, per build_patient_drug_pairs()'s
    # own definition ("positives" = reachable drugs, not necessarily net-positive).
    # We specifically want patients with at least one score >= 0 pair, to match
    # evaluate.py's "candidates" (net-positive) ground truth definition.

    pos_scores_only = [(p, d, s) for (p, d, s) in positives if s >= 0]

    patients_with_any_reachable = set(p for p, d, s in positives)
    patients_with_net_positive = set(p for p, d, s in pos_scores_only)

    print(f"\nPatients with >=1 reachable drug (any score, +/-): {len(patients_with_any_reachable)} / {num_patients}")
    print(f"Patients with >=1 NET-POSITIVE drug (score >= 0):    {len(patients_with_net_positive)} / {num_patients}")

    frac_net_positive = len(patients_with_net_positive) / num_patients
    expected_in_34 = round(34 * frac_net_positive, 1)

    print(f"\nFraction of all patients with a net-positive drug: {frac_net_positive:.3f}")
    print(f"Expected number evaluable in a 34-patient validation split: {expected_in_34}")
    print("\n(Compare this expected number to the actual 7/34 evaluate.py reports.")
    print(" If they're close, the 7/34 result is genuine data skew, not a bug.")
    print(" If expected is much higher than 7, there's still a real mismatch to find.)")


if __name__ == "__main__":
    main()