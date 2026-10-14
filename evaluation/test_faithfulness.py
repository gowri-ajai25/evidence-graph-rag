# evaluation/test_faithfulness.py
"""
Evaluates the faithfulness verifier two ways:

  1. Clean rate   - average faithfulness of the LLM's real explanations
                    (data/processed/explanations_*.json written by
                    explanation/llm_explainer.py after verification was added)
  2. Detection    - takes claims the verifier accepted, injects a known error,
                    and measures how often the verifier flags the corrupted claim

Corruptions injected:
  fake_citation   cites an evidence ID that does not exist
  no_citation     removes the citation
  flipped_stance  swaps supports <-> opposes
  wrong_number    adds an evidence score that is not in the cited edge
  foreign_drug    names a different drug
  foreign_gene    names a gene that is not in the retrieved evidence
  wrong_mutation  quotes a protein change that is not on the cited edge
  wrong_source    attributes the claim to a database that is not the cited source

Run from the repo root:
  python evaluation/test_faithfulness.py
If no real explanation files exist yet, a small synthetic example is used so
the script still runs.
"""

import os
import sys
import glob
import json
import copy
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "explanation"))
from faithfulness import verify_explanation, clean_vocab  # noqa: E402

PROCESSED = os.path.join(ROOT, "data", "processed")


# ── load real explanations (or fall back to a synthetic one) ──

def load_entries():
    """Returns a list of (drug, evidence_index, claims) from saved explanations."""
    entries = []
    for path in sorted(glob.glob(os.path.join(PROCESSED, "explanations_*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"   skipped {os.path.basename(path)}: {e}")
            continue
        for key in ("candidates", "contraindications"):
            for item in payload.get(key, []):
                ver = item.get("verification") or {}
                if item.get("evidence_index") and ver.get("claims"):
                    claims = [
                        {"text": c["text"], "cites": c["cites"], "stance": c["stance"]}
                        for c in ver["claims"]
                    ]
                    entries.append((item["drug"], item["evidence_index"], claims))
    return entries


def synthetic_entries():
    index = {
        "E1": {"side": "SUPPORTING", "gene": "BRCA1", "drug": "Olaparib", "evidence_score": 0.850,
               "source": "CIViC", "protein_change": "p.C61G"},
        "E2": {"side": "OPPOSING", "gene": "TP53", "drug": "Olaparib", "evidence_score": -0.300,
               "source": "DGIdb", "protein_change": "p.R196*"},
    }
    claims = [
        {"text": "BRCA1 loss is linked to Olaparib sensitivity (evidence score 0.850).",
         "cites": ["E1"], "stance": "supports"},
        {"text": "TP53 mutation tempers confidence in this suggestion.",
         "cites": ["E2"], "stance": "opposes"},
    ]
    return [("Olaparib", index, claims)]


def load_known():
    """Gene and drug vocabularies from the exported graph, if it can be loaded."""
    try:
        with open(os.path.join(PROCESSED, "hetero_graph.pkl"), "rb") as f:
            saved = pickle.load(f)
        return clean_vocab(saved["gene_idx"]), clean_vocab(saved["drug_idx"])
    except Exception as e:  # noqa: BLE001  (torch_geometric missing, file missing, ...)
        print(f"   could not load hetero_graph.pkl ({type(e).__name__}); using a small built-in vocabulary")
        return (["BRCA1", "BRCA2", "TP53", "PIK3CA", "PTEN", "KRAS", "ATM", "ARID1A"],
                ["Olaparib", "Cisplatin", "Carboplatin", "Bevacizumab", "Niraparib"])


# ── corruptions ───────────────────────────────────────────────

def _free_number(evidence_index):
    scores = [abs(e["evidence_score"]) for e in evidence_index.values()]
    for cand in (0.123, 0.777, 0.456):
        if all(abs(cand - s) > 0.05 for s in scores):
            return cand
    return 0.314


def corruptions(claim, drug, evidence_index, known_genes, known_drugs):
    """Yields (name, corrupted_claim) for every corruption that applies."""
    c = copy.deepcopy(claim)
    c["cites"] = ["E99"]
    yield "fake_citation", c

    c = copy.deepcopy(claim)
    c["cites"] = []
    yield "no_citation", c

    if claim["stance"] in ("supports", "opposes"):
        c = copy.deepcopy(claim)
        c["stance"] = "opposes" if claim["stance"] == "supports" else "supports"
        yield "flipped_stance", c

    c = copy.deepcopy(claim)
    c["text"] = c["text"].rstrip(". ") + f" (evidence score {_free_number(evidence_index):.3f})."
    yield "wrong_number", c

    other_drug = next((d for d in known_drugs if d.lower() != drug.lower()), None)
    if other_drug:
        c = copy.deepcopy(claim)
        c["text"] = c["text"].rstrip(". ") + f", and {other_drug} shows similar benefit."
        yield "foreign_drug", c

    index_genes = {e["gene"] for e in evidence_index.values()}
    other_gene = next((g for g in known_genes if g not in index_genes), None)
    if other_gene:
        c = copy.deepcopy(claim)
        c["text"] = c["text"].rstrip(". ") + f" {other_gene} is also implicated."
        yield "foreign_gene", c

    if all("protein_change" in e for e in evidence_index.values()):
        c = copy.deepcopy(claim)
        c["text"] = c["text"].rstrip(". ") + " (p.Z999Q)."
        yield "wrong_mutation", c

    if all("source" in e for e in evidence_index.values()):
        cited_src = " ".join(evidence_index[i]["source"] for i in claim["cites"]
                             if i in evidence_index).lower()
        wrong = next((x for x in ("CIViC", "DGIdb", "OncoKB") if x.lower() not in cited_src), None)
        if wrong:
            c = copy.deepcopy(claim)
            c["text"] = c["text"].rstrip(". ") + f", as reported by {wrong}."
            yield "wrong_source", c


# ── main ──────────────────────────────────────────────────────

def main():
    print("Faithfulness verifier evaluation")
    print("=" * 60)

    entries = load_entries()
    source = "real explanations"
    if not entries:
        entries = synthetic_entries()
        source = "synthetic example (no verified explanation files found yet)"
    print(f"Source: {source}")

    known_genes, known_drugs = load_known()

    # Clean rate
    total_claims = supported = partial = unsupported = 0
    for drug, index, claims in entries:
        res = verify_explanation({"claims": claims}, index, drug, known_genes, known_drugs)
        total_claims += len(res["claims"])
        supported += res["n_supported"]
        partial += res["n_partial"]
        unsupported += res["n_unsupported"]

    print(f"\n[1] Clean explanations: {len(entries)} drug explanation(s), {total_claims} claim(s)")
    if total_claims:
        print(f"    supported   {supported:>4}  ({supported / total_claims:.1%})")
        print(f"    partial     {partial:>4}  ({partial / total_claims:.1%})")
        print(f"    unsupported {unsupported:>4}  ({unsupported / total_claims:.1%})")

    # Detection rate on injected errors
    tested, caught = {}, {}
    for drug, index, claims in entries:
        for claim in claims:
            base = verify_explanation({"claims": [claim]}, index, drug, known_genes, known_drugs)
            if base["claims"][0]["status"] != "supported":
                continue   # only corrupt claims the verifier accepted
            for name, bad in corruptions(claim, drug, index, known_genes, known_drugs):
                res = verify_explanation({"claims": [bad]}, index, drug, known_genes, known_drugs)
                tested[name] = tested.get(name, 0) + 1
                if res["claims"][0]["status"] == "unsupported":
                    caught[name] = caught.get(name, 0) + 1

    print("\n[2] Detection of injected errors (claims first accepted as 'supported')")
    results = {}
    for name in ("fake_citation", "no_citation", "flipped_stance",
                 "wrong_number", "foreign_drug", "foreign_gene",
                 "wrong_mutation", "wrong_source"):
        t, c = tested.get(name, 0), caught.get(name, 0)
        rate = (c / t) if t else None
        results[name] = {"tested": t, "caught": c, "rate": rate}
        shown = f"{rate:.1%}" if rate is not None else "n/a"
        print(f"    {name:<16} {c:>3}/{t:<3}  {shown}")

    out = {
        "source": source,
        "clean": {"claims": total_claims, "supported": supported,
                  "partial": partial, "unsupported": unsupported},
        "detection": results,
    }
    os.makedirs(PROCESSED, exist_ok=True)
    out_path = os.path.join(PROCESSED, "faithfulness_eval.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {out_path}")
    except OSError as e:
        print(f"\nCould not save results: {e}")


if __name__ == "__main__":
    main()