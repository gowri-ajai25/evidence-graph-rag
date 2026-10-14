# report/build_report.py
"""
Builds the exportable clinician-facing PDF report for one patient.

The report contains, in order:
  1. patient header, research-prototype disclaimer, summary counts
  2. the patient-specific evidence subgraph figure
  3. mutated genes (with GO function context)
  4. ranked treatments table
  5. per-drug sections: supporting / opposing evidence with clickable source
     links (CIViC, PubMed, DGIdb, ChEMBL / DrugBank, trials) and the
     LLM explanation, one claim at a time, each marked supported / partial /
     unsupported by the faithfulness verifier, with the evidence IDs it cites
  6. numbered reference list of every source link used
  7. method and limitation notes

Requires:  pip install reportlab
"""

import io
import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    KeepTogether, PageBreak,
)

GOOD = colors.HexColor("#1a7f4b")
WARN = colors.HexColor("#b26a00")
BAD = colors.HexColor("#b42318")
INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#616e7c")
LINE = colors.HexColor("#cbd2d9")
HEAD_BG = colors.HexColor("#e4e7eb")
LINK = "#1a56db"

DISCLAIMER = (
    "Research and educational prototype. This is exploratory decision support for clinician "
    "review. It is not a certified clinical tool, not an autonomous prescribing system, and "
    "must not be used to inform real treatment decisions."
)

METHOD_NOTES = [
    "Evidence scores are computed when the knowledge graph is built and combine evidence level, "
    "number of independent sources and approval status, signed by the direction of the evidence "
    "(supportive or resistance).",
    "The GNN score comes from a GraphSAGE ranker that sees graph structure only; it is not an "
    "evidence-weighted score and is not shown to the language model.",
    "Each explanation claim is checked against the retrieved graph edges it cites: the cited "
    "evidence must exist, named genes, drugs, mutations and databases must appear in it, the "
    "stated direction must match, and quoted scores must match.",
    "Verification confirms that claims are traceable to retrieved evidence. It does not confirm "
    "that the evidence is clinically correct, and it cannot detect wording that overstates what "
    "the evidence shows.",
]


# ── text helpers ──────────────────────────────────────────────

def _txt(value):
    """Escapes for reportlab markup and drops characters the built-in PDF fonts cannot draw."""
    s = "" if value is None else str(value)
    for bad, good in (("‑", "-"), (" ", " "), (" ", " "), (" ", " "),
                      ("​", ""), ("−", "-")):
        s = s.replace(bad, good)
    s = s.encode("cp1252", "replace").decode("cp1252")
    return escape(s)


def _link(url, label):
    return f'<a href="{escape(url, {chr(34): "&quot;"})}" color="{LINK}">{_txt(label)}</a>'


def _styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=18, textColor=INK, alignment=0, spaceAfter=2),
        "sub": ParagraphStyle("sub", parent=base["Normal"], fontSize=9, textColor=MUTED),
        "h1": ParagraphStyle("h1", parent=base["Heading2"], fontSize=13, textColor=INK, spaceBefore=10, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=base["Heading3"], fontSize=11, textColor=INK, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9, leading=12, textColor=INK),
        "small": ParagraphStyle("small", parent=base["Normal"], fontSize=7.5, leading=10, textColor=MUTED),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8, leading=10, textColor=INK),
        "cellb": ParagraphStyle("cellb", parent=base["Normal"], fontSize=8, leading=10, textColor=INK, fontName="Helvetica-Bold"),
        "box": ParagraphStyle("box", parent=base["Normal"], fontSize=8.5, leading=11, textColor=BAD),
        "claim": ParagraphStyle("claim", parent=base["Normal"], fontSize=9, leading=12, textColor=INK, leftIndent=0),
    }
    return s


def _table(data, col_widths, header=True, zebra=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), HEAD_BG))
    if zebra:
        for r in range(1, len(data)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f5f7fa")))
    t.setStyle(TableStyle(style))
    return t


# ── reference registry (numbered, de-duplicated) ──────────────

class _Refs:
    def __init__(self):
        self.items = []     # [{"label","url","detail"}]
        self._idx = {}

    def add(self, link):
        url = link["url"]
        if url not in self._idx:
            self.items.append(link)
            self._idx[url] = len(self.items)
        return self._idx[url]


# ── sections ──────────────────────────────────────────────────

def _evidence_block(title, entries, drug, linker, refs, st):
    """Table of gene-level evidence for one drug, with numbered source links."""
    if not entries:
        return []
    rows = [[Paragraph(h, st["cellb"]) for h in ("Gene / mutation", "Score", "Source", "Source records")]]
    for entry, links in (linker.links_for_entries(entries, drug) if linker else [(e, []) for e in entries]):
        mut = _txt(entry.get("mutation_type") or "")
        prot = entry.get("protein_change")
        gene_cell = f"<b>{_txt(entry.get('gene'))}</b><br/>{mut}" + (f" ({_txt(prot)})" if prot and str(prot) not in ("nan", "None") else "")
        parts = []
        for link in links:
            n = refs.add(link)
            parts.append(f"[{n}] " + _link(link["url"], link["label"]))
        links_cell = "<br/>".join(parts) if parts else "no source record matched"
        try:
            score = f"{float(entry.get('evidence_score')):+.3f}"
        except (TypeError, ValueError):
            score = "n/a"
        rows.append([Paragraph(gene_cell, st["cell"]), Paragraph(score, st["cell"]),
                     Paragraph(_txt(entry.get("evidence_source") or ""), st["cell"]),
                     Paragraph(links_cell, st["cell"])])
    return [Paragraph(title, st["h2"]), _table(rows, [38 * mm, 16 * mm, 20 * mm, 100 * mm])]


def _explanation_block(exp, st):
    """Claim-by-claim explanation with verifier status. exp may be a dict, a str, or None."""
    out = [Paragraph("Explanation", st["h2"])]
    if exp is None:
        out.append(Paragraph("No explanation has been generated for this drug.", st["small"]))
        return out
    if isinstance(exp, str):
        out.append(Paragraph(_txt(exp), st["body"]))
        out.append(Paragraph("Not verified (generated before faithfulness verification was added).", st["small"]))
        return out

    ver = exp.get("verification")
    if not ver:
        out.append(Paragraph(_txt(exp.get("explanation", "")), st["body"]))
        out.append(Paragraph("Not verified (generated before faithfulness verification was added).", st["small"]))
        return out
    if ver.get("faithfulness") is None:
        out.append(Paragraph(_txt(exp.get("explanation", "")), st["body"]))
        out.append(Paragraph(f"Could not be verified: {_txt(ver.get('error', 'unknown error'))}", st["small"]))
        return out

    colour = {"supported": GOOD, "partial": WARN, "unsupported": BAD}
    index = exp.get("evidence_index", {})
    for c in ver["claims"]:
        hexcol = colour[c["status"]].hexval().replace("0x", "#")
        cites = []
        for eid in c["cites"]:
            e = index.get(eid)
            cites.append(f"{eid}: {_txt(e['gene'])} ({e['side'].lower()}, {e['evidence_score']:+.3f})" if e else f"{eid}: not found")
        block = [Paragraph(
            f'<font color="{hexcol}"><b>[{c["status"].upper()}]</b></font> {_txt(c["text"])}', st["claim"])]
        block.append(Paragraph("Cites: " + (" | ".join(cites) if cites else "nothing"), st["small"]))
        for issue in c["issues"]:
            block.append(Paragraph(f'<font color="{BAD.hexval().replace("0x", "#")}">Flag: {_txt(issue)}</font>', st["small"]))
        block.append(Spacer(1, 3))
        out.append(KeepTogether(block))

    summary = (f"Faithfulness {ver['faithfulness']:.0%}: {ver['n_supported']} supported, "
               f"{ver['n_partial']} partial, {ver['n_unsupported']} unsupported.")
    if exp.get("evidence_strength"):
        summary += f" Evidence strength stated: {_txt(exp['evidence_strength'])}."
    out.append(Paragraph(summary, st["small"]))
    return out


def build_pdf(patient_id, summary, ranking_rows, drug_sections, linker=None,
              subgraph_png=None, genes_rows=None, coverage_note=None):
    """
    patient_id     : str
    summary        : dict from SubgraphRetriever.get_patient_subgraph_summary
    ranking_rows   : list of dicts with keys 'GNN rank', 'Drug', 'GNN score', 'Status', 'Net evidence score'
    drug_sections  : list of dicts {"drug", "status" ('CANDIDATE'|'CONTRAINDICATION'),
                     "gnn_score" (float|None), "evidence" (candidate/contra entry), "explanation" (dict|str|None)}
    linker         : report.evidence_links.EvidenceLinker (optional; no source links without it)
    subgraph_png   : PNG bytes of the evidence subgraph figure (optional)
    genes_rows     : list of dicts with keys 'gene', 'mutation_type', 'protein_change', 'GO function'
    coverage_note  : short text on GNN coverage for this patient (optional)
    Returns the PDF as bytes.
    """
    st = _styles()
    refs = _Refs()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"Evidence-Aware Graph-RAG report - {patient_id}",
                            author="Evidence-Aware Graph-RAG")
    story = []

    # 1. header
    story.append(Paragraph("Evidence-Aware Graph-RAG: treatment evidence report", st["title"]))
    story.append(Paragraph(
        f"Patient {_txt(patient_id)} &nbsp;|&nbsp; generated {datetime.date.today().isoformat()}"
        + (f" &nbsp;|&nbsp; {_txt(coverage_note)}" if coverage_note else ""), st["sub"]))
    story.append(Spacer(1, 6))
    box = Table([[Paragraph(_txt(DISCLAIMER), st["box"])]], colWidths=[178 * mm])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, BAD), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef3f2")),
                             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.append(box)
    story.append(Spacer(1, 6))
    counts = [[Paragraph(f"<b>{summary.get('num_mutated_genes', 0)}</b><br/>mutated genes in graph", st["cell"]),
               Paragraph(f"<b>{summary.get('num_candidate_drugs', 0)}</b><br/>candidate drugs", st["cell"]),
               Paragraph(f"<b>{summary.get('num_contraindications', 0)}</b><br/>contraindication flags", st["cell"])]]
    story.append(_table(counts, [59 * mm] * 3, header=False, zebra=False))

    # 2. subgraph figure
    if subgraph_png:
        story.append(Paragraph("Patient-specific evidence subgraph", st["h1"]))
        try:
            from reportlab.lib.utils import ImageReader
            img = ImageReader(io.BytesIO(subgraph_png))
            w, h = img.getSize()
            width = 178 * mm
            story.append(Image(io.BytesIO(subgraph_png), width=width, height=width * h / w))
            story.append(Paragraph("Green edges: supporting evidence. Red edges: opposing evidence. "
                                   "Grey edges: patient mutations.", st["small"]))
        except Exception as e:  # noqa: BLE001
            story.append(Paragraph(f"Subgraph figure unavailable: {_txt(e)}", st["small"]))

    # 3. mutated genes
    if genes_rows:
        story.append(Paragraph("Mutated genes", st["h1"]))
        rows = [[Paragraph(h, st["cellb"]) for h in ("Gene", "Mutation type", "Protein change", "GO function")]]
        for g in genes_rows:
            rows.append([Paragraph(_txt(g.get("gene")), st["cellb"]), Paragraph(_txt(g.get("mutation_type")), st["cell"]),
                         Paragraph(_txt(g.get("protein_change") if str(g.get("protein_change")) not in ("nan", "None") else ""), st["cell"]),
                         Paragraph(_txt(g.get("GO function")), st["cell"])])
        story.append(_table(rows, [22 * mm, 36 * mm, 30 * mm, 90 * mm]))

    # 4. ranked treatments
    if ranking_rows:
        story.append(Paragraph("Ranked treatments", st["h1"]))
        story.append(Paragraph("GNN score is structural relevance from the trained GraphSAGE ranker. Status and net "
                               "evidence score come from the evidence in the knowledge graph.", st["small"]))
        rows = [[Paragraph(h, st["cellb"]) for h in ("GNN rank", "Drug", "GNN score", "Status", "Net evidence")]]
        for r in ranking_rows[:25]:
            def fmt(v, spec):
                try:
                    return format(float(v), spec)
                except (TypeError, ValueError):
                    return "n/a"
            try:
                rank_txt = str(int(float(r.get("GNN rank"))))
            except (TypeError, ValueError):
                rank_txt = ""   # None / NaN when the GNN has no ranking for this patient
            rows.append([Paragraph(rank_txt, st["cell"]),
                         Paragraph(_txt(r.get("Drug")), st["cellb"]),
                         Paragraph(fmt(r.get("GNN score"), "+.3f"), st["cell"]),
                         Paragraph(_txt(r.get("Status")), st["cell"]),
                         Paragraph(fmt(r.get("Net evidence score"), "+.3f"), st["cell"])])
        story.append(_table(rows, [20 * mm, 70 * mm, 28 * mm, 34 * mm, 26 * mm]))

    # 5. per-drug sections
    if drug_sections:
        story.append(PageBreak())
        story.append(Paragraph("Evidence and explanations", st["h1"]))
    for sec in drug_sections:
        drug = sec["drug"]
        status = sec.get("status", "")
        col = GOOD if status == "CANDIDATE" else BAD
        hexcol = col.hexval().replace("0x", "#")
        score = sec.get("gnn_score")
        score_txt = f" &nbsp;(GNN score {score:+.3f})" if isinstance(score, (int, float)) else ""
        head = Paragraph(f'<b>{_txt(drug)}</b> &nbsp;<font color="{hexcol}"><b>{_txt(status)}</b></font>{score_txt}', st["h1"])
        ev = sec.get("evidence") or {}
        story.append(KeepTogether([head]))
        story += _evidence_block("Supporting evidence", ev.get("supporting_genes", []), drug, linker, refs, st)
        story += _evidence_block("Opposing evidence", ev.get("opposing_genes", []), drug, linker, refs, st)
        not_applied = ev.get("non_discriminating_genes", [])
        if not_applied:
            names = ", ".join(sorted({_txt(g["gene"]) for g in not_applied}))
            story.append(Paragraph(
                f"<i>Not applied: {names} has literature evidence against this drug, but the gene is "
                "mutated in most patients, so it cannot single out this patient and is excluded from "
                "the score.</i>", st["cell"]))
        story += _explanation_block(sec.get("explanation"), st)
        story.append(Spacer(1, 8))

    # 6. references
    if refs.items:
        story.append(PageBreak())
        story.append(Paragraph("References", st["h1"]))
        rows = []
        for i, item in enumerate(refs.items, start=1):
            rows.append([Paragraph(f"[{i}]", st["cell"]),
                         Paragraph(_link(item["url"], item["label"]) + (f" - {_txt(item['detail'])}" if item.get("detail") else ""), st["cell"])])
        story.append(_table(rows, [14 * mm, 164 * mm], header=False))

    # 7. method notes
    story.append(Paragraph("Method and limitations", st["h1"]))
    for note in METHOD_NOTES:
        story.append(Paragraph("- " + _txt(note), st["small"]))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(16 * mm, 9 * mm, f"Research prototype - not for clinical use - patient {patient_id}")
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()