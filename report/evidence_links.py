# report/evidence_links.py
"""
Resolves a knowledge-graph edge (gene -> drug, source CIViC or DGIdb) to the
real source records behind it, so every piece of evidence in the dashboard and
in the exported report links out to where it came from.

The graph stores only the aggregated edge (score, level, source). The
underlying records live in the source files the graph was built from, so links
are looked up from those files at report time:

  CIViC : data/raw/civic/civic_ovarian_evidence.csv
          -> evidence item (CIViC URL), PubMed article, trial (NCT) IDs
  DGIdb : data/raw/drugbank/dgidb_interactions.csv
          -> DGIdb gene page, ChEMBL / DrugBank compound page, source databases

Matching mirrors kg/build_graph.py so the links correspond to the edges that
were actually loaded: CIViC gene = first word of the molecular profile, and
therapies are split on commas / " and ".
"""

import os
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIVIC_CSV = os.path.join(ROOT, "data", "raw", "civic", "civic_ovarian_evidence.csv")
DGIDB_CSV = os.path.join(ROOT, "data", "raw", "drugbank", "dgidb_interactions.csv")

LEVEL_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def _clean(v):
    v = "" if v is None else str(v).strip()
    return "" if v.lower() in ("nan", "none") else v


def _split_therapies(raw):
    """Same splitting as kg/build_graph.py::load_civic_evidence."""
    return [d.strip() for d in _clean(raw).replace(" and ", ",").split(",") if d.strip()]


def _read_csv(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error) as e:
        print(f"[evidence_links] could not read {path}: {e}")
        return []


def _compound_url(concept_id):
    """drugbank:DB00001 / chembl:CHEMBL25 -> a compound page URL; else None."""
    cid = _clean(concept_id)
    if ":" not in cid:
        return None
    prefix, ident = cid.split(":", 1)
    prefix = prefix.lower()
    if prefix == "drugbank":
        return f"https://go.drugbank.com/drugs/{ident}"
    if prefix == "chembl":
        return f"https://www.ebi.ac.uk/chembl/compound_report_card/{ident.upper()}/"
    return None


class EvidenceLinker:
    def __init__(self, civic_csv=CIVIC_CSV, dgidb_csv=DGIDB_CSV):
        self._civic = {}    # (gene, drug_lower) -> [row, ...]
        self._dgidb = {}    # (gene, drug_lower) -> [row, ...]

        for row in _read_csv(civic_csv):
            profile = _clean(row.get("molecular_profile"))
            if not profile:
                continue
            gene = profile.split()[0]
            for drug in _split_therapies(row.get("therapies")):
                self._civic.setdefault((gene, drug.lower()), []).append(row)

        for row in _read_csv(dgidb_csv):
            gene, drug = _clean(row.get("gene_name")), _clean(row.get("drug_name"))
            if gene and drug:
                self._dgidb.setdefault((gene, drug.lower()), []).append(row)

    # ── CIViC ─────────────────────────────────────────────────

    def _civic_links(self, gene, drug, max_items):
        rows = self._civic.get((gene, drug.lower()), [])
        rows = sorted(
            rows,
            key=lambda r: (LEVEL_ORDER.get(_clean(r.get("evidence_level")).upper(), 9),
                           -(float(_clean(r.get("rating")) or 0))),
        )
        out = []
        for r in rows[:max_items]:
            eid = _clean(r.get("evidence_id"))
            url = _clean(r.get("evidence_civic_url")) or (
                f"https://civicdb.org/links/evidence_items/{eid}" if eid else "")
            detail = ", ".join(x for x in (
                f"level {_clean(r.get('evidence_level'))}" if _clean(r.get("evidence_level")) else "",
                _clean(r.get("evidence_direction")),
                _clean(r.get("significance")),
            ) if x)
            if url:
                out.append({"kind": "CIViC", "label": f"CIViC EID{eid}" if eid else "CIViC evidence",
                            "url": url, "detail": detail})

            pmid = _clean(r.get("citation_id"))
            if pmid and _clean(r.get("source_type")).lower() == "pubmed":
                cite = _clean(r.get("citation"))
                out.append({"kind": "PubMed", "label": f"PubMed {pmid}",
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            "detail": cite})

            for nct in _clean(r.get("nct_ids")).replace(";", ",").split(","):
                nct = nct.strip()
                if nct.upper().startswith("NCT"):
                    out.append({"kind": "Trial", "label": nct.upper(),
                                "url": f"https://clinicaltrials.gov/study/{nct.upper()}",
                                "detail": "clinical trial"})
        return out

    # ── DGIdb ─────────────────────────────────────────────────

    def _dgidb_links(self, gene, drug):
        rows = self._dgidb.get((gene, drug.lower()), [])
        out = [{"kind": "DGIdb", "label": f"DGIdb: {gene}",
                "url": f"https://dgidb.org/genes/{gene}",
                "detail": "gene page"}]
        if not rows:
            return out

        dbs = sorted({_clean(r.get("interaction_source_db_name")) for r in rows} - {""})
        if dbs:
            out[0]["detail"] = "reported by " + ", ".join(dbs)

        seen = set()
        for r in rows:
            url = _compound_url(r.get("drug_concept_id"))
            if url and url not in seen:
                seen.add(url)
                name = "DrugBank" if "drugbank" in url else "ChEMBL"
                out.append({"kind": name, "label": f"{name}: {drug}", "url": url,
                            "detail": "compound page"})
        return out

    # ── public API ────────────────────────────────────────────

    def links_for(self, gene, drug, source, max_civic=3):
        """
        Returns a list of {"kind", "label", "url", "detail"} for one graph edge.
        `source` is the edge's source property ('CIViC' or 'DGIdb').
        An empty list means no source record could be matched.
        """
        gene, drug, source = _clean(gene), _clean(drug), _clean(source).lower()
        if not gene or not drug:
            return []
        if "civic" in source:
            links = self._civic_links(gene, drug, max_civic)
        elif "dgidb" in source:
            links = self._dgidb_links(gene, drug)
        else:
            links = self._civic_links(gene, drug, max_civic) + self._dgidb_links(gene, drug)

        # several CIViC items often cite the same article / trial: keep each URL once
        seen, unique = set(), []
        for link in links:
            if link["url"] not in seen:
                seen.add(link["url"])
                unique.append(link)
        return unique

    def links_for_entries(self, entries, drug, max_civic=3):
        """For a list of retrieved gene entries (supporting or opposing genes of one
        drug). Returns [(entry, [links])] in the same order."""
        return [(e, self.links_for(e.get("gene"), drug, e.get("evidence_source"), max_civic))
                for e in entries]