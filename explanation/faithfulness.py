# explanation/faithfulness.py
"""
Evidence-grounded faithfulness verification for LLM-generated explanations.

The explainer asks the LLM to return structured claims, each citing the IDs
(E1, E2, ...) of the retrieved knowledge-graph edges it relies on. This module
checks every claim against those edges and flags what cannot be traced back:

  1. Citation     - the claim cites at least one edge, and every cited ID exists
  2. Entities     - every gene / drug named in the claim is in the retrieved
                    evidence (a gene that is in the evidence but not cited by
                    this claim is a softer, 'partial' flag)
  3. Direction    - the claim's stance (supports / opposes) matches the side
                    (SUPPORTING / OPPOSING) of the cited edges
  4. Numbers      - every decimal quoted in the claim matches the evidence
                    score of a cited edge
  5. Mutation     - any protein change quoted (e.g. p.R196*) matches the
                    protein change recorded on a cited edge
  6. Source       - any database named (CIViC, DGIdb, ...) is the source of a
                    cited edge

Status per claim:
  supported   - no issues
  partial     - only soft issues (gene named but its edge not cited)
  unsupported - at least one hard issue

This is a deterministic consistency check. It verifies that claims are
attributable to retrieved edges; it does not judge whether the reasoning in a
sentence follows logically from the evidence.
"""

import re

NUM_RE = re.compile(r"[-+]?\d*\.\d+")
PROT_RE = re.compile(r"p\.([A-Za-z0-9_]+(?:\*\d*)?)")
SOURCES = ["CIViC", "DGIdb", "OncoKB", "DrugBank", "KEGG"]
SCORE_TOL = 0.006   # scores are shown to 3 dp; allow rounding to 2 dp


# ── helpers ───────────────────────────────────────────────────

def clean_vocab(names):
    """Keeps only real, non-empty string names. Graph exports can contain NaN or
    other non-string keys (e.g. a gene row with a missing symbol), which would
    break regex matching."""
    return sorted({n.strip() for n in names if isinstance(n, str) and n.strip()})


def _mentions(token, text, ignore_case=False):
    """Whole-token match, so 'MET' does not match inside 'METABOLISM'."""
    flags = re.IGNORECASE if ignore_case else 0
    pattern = rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])"
    return re.search(pattern, text, flags) is not None


def _norm_prot(value):
    """'p.R196*' -> 'R196*' so recorded and quoted protein changes compare equal."""
    v = str(value).strip()
    return v[2:] if v.lower().startswith("p.") else v


def _normalise_cites(cites):
    """Accepts ['E1'], 'E1', ['[E1]'], ['e1, E2'] and returns ['E1', 'E2']."""
    if cites is None:
        return []
    if isinstance(cites, str):
        cites = [cites]
    out = []
    for c in cites:
        for part in re.split(r"[,\s]+", str(c).strip("[] ")):
            part = part.strip("[]").upper()
            if part:
                out.append(part)
    return out


# ── per-claim verification ────────────────────────────────────

def verify_claim(claim, evidence_index, drug, known_genes, known_drugs):
    """
    claim          : {"text": str, "cites": [str], "stance": "supports|opposes|neutral"}
    evidence_index : {"E1": {"side": "SUPPORTING"|"OPPOSING", "gene": str,
                             "drug": str, "evidence_score": float}, ...}
    drug           : the drug this explanation is about
    known_genes    : iterable of every gene symbol in the knowledge graph
    known_drugs    : iterable of every drug name in the knowledge graph
    """
    text = str(claim.get("text", ""))
    cites = _normalise_cites(claim.get("cites"))
    stance = str(claim.get("stance", "neutral")).lower()
    hard, soft = [], []

    # 1. Citation: present, and every cited ID exists
    if not cites:
        hard.append("no evidence cited")
    unknown = [c for c in cites if c not in evidence_index]
    if unknown:
        hard.append(f"cites non-existent evidence {unknown}")
    cited = [evidence_index[c] for c in cites if c in evidence_index]

    # 2. Entities: named genes / drugs must be traceable to the evidence
    index_genes = {e["gene"] for e in evidence_index.values()}
    cited_genes = {e["gene"] for e in cited}
    for g in known_genes:
        if _mentions(g, text):
            if g not in index_genes:
                hard.append(f"gene {g} is not in the retrieved evidence")
            elif g not in cited_genes:
                soft.append(f"names {g} without citing its evidence")
    for d in known_drugs:
        if str(d).lower() != str(drug).lower() and _mentions(d, text, ignore_case=True):
            hard.append(f"names a drug not under discussion: {d}")

    # 3. Direction: stance must match the side of the cited edges
    if cited and stance in ("supports", "opposes"):
        wanted = "SUPPORTING" if stance == "supports" else "OPPOSING"
        sides = {e["side"] for e in cited}
        if wanted not in sides:
            hard.append(f"claim {stance} the drug but cited evidence is {'/'.join(sorted(sides))}")

    # 4. Numbers: every decimal quoted must match a cited evidence score
    cited_scores = [abs(float(e["evidence_score"])) for e in cited]
    for m in NUM_RE.findall(text):
        if not any(abs(abs(float(m)) - s) <= SCORE_TOL for s in cited_scores):
            hard.append(f"number {m} does not match any cited evidence score")

    # 5. Mutation: a quoted protein change must be recorded on a cited edge.
    #    (Skipped for evidence indexes saved before protein changes were tracked.)
    if cited and all("protein_change" in e for e in cited):
        recorded = {_norm_prot(e["protein_change"]) for e in cited if e["protein_change"]}
        for m in PROT_RE.findall(text):
            if m not in recorded:
                hard.append(f"protein change p.{m} is not recorded on the cited evidence")

    # 6. Source: a database named in the claim must be the source of a cited edge
    if cited and all("source" in e for e in cited):
        cited_sources = " ".join(str(e["source"]) for e in cited).lower()
        for src in SOURCES:
            if _mentions(src, text, ignore_case=True) and src.lower() not in cited_sources:
                hard.append(f"attributes the claim to {src}, which is not the source of the cited evidence")

    if hard:
        status = "unsupported"
    elif soft:
        status = "partial"
    else:
        status = "supported"

    return {
        "text": text,
        "cites": cites,
        "stance": stance,
        "status": status,
        "issues": hard + soft,
    }


# ── whole-explanation verification ────────────────────────────

def verify_explanation(parsed, evidence_index, drug, known_genes, known_drugs):
    """
    parsed : the LLM's JSON output, {"claims": [...], "evidence_strength": ...}
    Returns per-claim results plus an overall faithfulness score
    (supported claims / total claims).
    """
    known_genes = clean_vocab(known_genes)
    known_drugs = clean_vocab(known_drugs)

    claims = [
        verify_claim(c, evidence_index, drug, known_genes, known_drugs)
        for c in parsed.get("claims", [])
        if isinstance(c, dict)
    ]
    n = len(claims)
    n_supported = sum(c["status"] == "supported" for c in claims)
    n_partial = sum(c["status"] == "partial" for c in claims)
    n_unsupported = sum(c["status"] == "unsupported" for c in claims)

    return {
        "claims": claims,
        "n_supported": n_supported,
        "n_partial": n_partial,
        "n_unsupported": n_unsupported,
        "faithfulness": (n_supported / n) if n else None,
    }