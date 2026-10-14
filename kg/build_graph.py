"""
Builds the Evidence-Aware Knowledge Graph in Neo4j.

Loads:
  - Genes (from CIViC + our key ovarian cancer gene list)
  - Drugs (from DGIdb, cleaned and merged with CIViC names -- see prepare_dgidb)
  - GENE -> DRUG relationships, weighted by composite evidence score
  - CIViC evidence items linking genes/variants to drugs with clinical evidence levels
"""

from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd
import os
import sys
import re

# Allow importing evidence_score.py from the same kg/ folder
sys.path.append(os.path.dirname(__file__))
from evidence_score import composite_score, dgidb_score

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# ── Data preparation (no database access; testable on its own) ─────────────

# Brand / synonym names that should resolve to one drug node
DRUG_ALIASES = {
    "taxol": "Paclitaxel",
    "doxorubicin hydrochloride": "Doxorubicin",
    "doxorubicin hydrochloride liposome": "Doxorubicin Liposomal",
    "topotecan hydrochloride": "Topotecan",
    "gemcitabine hydrochloride": "Gemcitabine",
}

# The DGIdb export maps some concept IDs to the wrong drug_name (all Trastuzumab
# rows carry drug_name RITUXIMAB, rxcui:121191). For these, trust the claim name.
CLAIM_NAME_OVERRIDES = {"trastuzumab": "Trastuzumab", "bevacizumab": "Bevacizumab"}

# FDA biosimilar suffixes: BEVACIZUMAB-AWWB -> BEVACIZUMAB
BIOSIMILAR_RE = re.compile(r"^(.*[A-Za-z])-[A-Za-z]{4}$")

# Unresolved identifiers DGIdb uses when a compound has no proper name
JUNK_DRUG_RE = re.compile(r"^(CHEMBL:|SCHEMBL|CID\s*\d)", re.IGNORECASE)

# Generic class labels (not a specific agent) cannot be recommended as a drug
CLASS_SUFFIXES = ("inhibitor", "compound", "antibody", "agent", "receptor")


def is_class_name(name):
    """'Platinum Compound', 'Taxane Compound', 'Mitogen-Activated Protein Kinase
    Kinase Inhibitor' are classes. 'CDK Inhibitor SNS-032' names a specific
    agent (it carries a code), so it is kept."""
    n = name.strip().lower()
    return n.endswith(CLASS_SUFFIXES) and not re.search(r"\d", n)


def canonical_drug(name, spelling=None):
    """One display name per drug: alias map, then the CIViC spelling if CIViC
    has the drug, else Title Case for ALL-CAPS names (codes with digits such as
    PD-98059 keep their form)."""
    n = str(name).strip()
    m = BIOSIMILAR_RE.match(n)
    if m and not re.search(r"\d", n):
        n = m.group(1)
    if n.lower() in DRUG_ALIASES:
        return DRUG_ALIASES[n.lower()]
    if spelling and n.lower() in spelling:
        return spelling[n.lower()]
    if n.isupper() and not re.search(r"\d", n):
        return n.title()
    return n


def load_fda_list(path="data/raw/fda/fda_ovarian_approved.csv"):
    """FDA-approved ovarian cancer drugs (NCI approved-drug list, derived from FDA
    approvals). Returns a DataFrame; empty if the file is missing."""
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as e:
        print(f"   Could not read FDA list ({e}); continuing without it")
        return pd.DataFrame(columns=["drug_name", "brand_names", "source", "source_url"])


# ── CIViC profile scope ─────────────────────────────────────────────────────
# A CIViC molecular profile is "GENE <what>". Only gene-level profiles can be
# matched to a patient from a mutation call alone.
#   gene        any qualifying mutation in the gene (Mutation, Loss-of-function, ...)
#   expression  needs expression / copy-number data (Overexpression, Amplification, Loss)
#   variant     one specific variant (R273L, V600E, Exon 4 Deletion, ...): must not be
#               generalised to patients who carry a different variant of the gene
GENE_LEVEL_PROFILES = {"mutation", "mut", "alteration", "loss-of-function", "loss of function",
                       "lof", "deleterious mutation", "oncogenic mutation"}
EXPRESSION_PROFILES = {"amplification", "loss", "deletion", "overexpression", "underexpression",
                       "expression", "copy number loss", "copy number gain"}


def civic_scope(molecular_profile):
    parts = str(molecular_profile).strip().split(None, 1)
    rest = parts[1].strip().lower() if len(parts) > 1 else ""
    if not rest or rest in GENE_LEVEL_PROFILES:
        return "gene"
    if rest in EXPRESSION_PROFILES:
        return "expression"
    return "variant"


def split_therapies(raw):
    return [d.strip() for d in str(raw).replace(" and ", ",").split(",") if d.strip()]


def civic_spellings(civic_df):
    """lower-case name -> CIViC spelling, for specific drugs only."""
    out = {}
    for raw in civic_df["therapies"].dropna():
        for d in split_therapies(raw):
            if not is_class_name(d):
                out[DRUG_ALIASES.get(d.lower(), d).lower()] = DRUG_ALIASES.get(d.lower(), d)
    return out


def prepare_dgidb(dgidb_df, civic_df, fda_names=frozenset()):
    """
    Cleans and aggregates the DGIdb export to one row per (gene, drug):
      - drops rows with missing names, unresolved identifiers (CHEMBL:...),
        and generic class labels
      - keeps drugs that are approved, annotated anti-neoplastic, or present in CIViC
      - merges synonyms (Taxol -> Paclitaxel) and case variants
      - counts DISTINCT source databases, takes the max interaction score,
        converts it to a percentile rank, and scores with dgidb_score()
    """
    spelling = civic_spellings(civic_df)
    df = dgidb_df.dropna(subset=["gene_name", "drug_name"]).copy()
    claim = df["drug_claim_name"].astype(str).str.strip().str.lower()
    df["drug_name"] = [CLAIM_NAME_OVERRIDES.get(c, n) for c, n in zip(claim, df["drug_name"])]
    df = df[~df["drug_name"].astype(str).str.match(JUNK_DRUG_RE)]
    df["drug"] = df["drug_name"].map(lambda n: canonical_drug(n, spelling))
    df = df[~df["drug"].map(is_class_name)]

    for col in ("approved", "anti_neoplastic"):
        df[col] = df[col].astype(str).str.lower().eq("true")
    in_civic = df["drug"].str.lower().isin(spelling.keys())
    df["approved"] = df["approved"] | df["drug"].str.lower().isin(fda_names)
    df = df[df["approved"] | df["anti_neoplastic"] | in_civic]

    agg = (
        df.groupby(["gene_name", "drug"])
        .agg(
            num_sources=("interaction_source_db_name", "nunique"),
            max_score=("interaction_score", "max"),
            approved=("approved", "any"),
            anti_neoplastic=("anti_neoplastic", "any"),
            interaction_type=("interaction_type",
                              lambda x: ",".join(sorted({str(v) for v in x.dropna()}))),
            sources=("interaction_source_db_name",
                     lambda x: ",".join(sorted({str(v) for v in x.dropna()}))),
        )
        .reset_index()
    )
    agg["interaction_pct"] = agg["max_score"].rank(pct=True)
    agg["evidence_score"] = [
        dgidb_score(r.num_sources, r.approved, r.interaction_pct, r.anti_neoplastic)
        for r in agg.itertuples()
    ]
    return agg


def prepare_civic(civic_df, spelling=None, fda_names=frozenset(), return_excluded=False):
    """
    One row per (gene, drug) from CIViC. Class labels are dropped and drug names
    canonicalised. Only GENE-scope profiles become gene->drug edges (see
    civic_scope); expression-level and single-variant evidence is not applied
    to patients it cannot be matched to, and is returned separately when
    return_excluded=True. When a pair has several evidence items, the one with
    the largest |score| is kept (so a strong resistance item is not silently
    overwritten by whichever row happened to come last).
    """
    spelling = spelling or civic_spellings(civic_df)
    rows = []
    for _, row in civic_df.iterrows():
        if pd.isna(row.get("molecular_profile")) or pd.isna(row.get("therapies")):
            continue
        gene = str(row["molecular_profile"]).split()[0].strip()
        for d in split_therapies(row["therapies"]):
            if is_class_name(d):
                continue
            drug = canonical_drug(d, spelling)
            score = composite_score(
                evidence_level=row.get("evidence_level"),
                num_sources=1,
                has_fda_approval=drug.lower() in fda_names,
                clinical_significance=row.get("significance"),
            )
            rows.append({
                "gene": gene, "drug": drug, "score": score,
                "level": str(row.get("evidence_level")), "disease": str(row.get("disease")),
                "significance": str(row.get("significance")),
                "profile": str(row["molecular_profile"]), "scope": civic_scope(row["molecular_profile"]),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return (out, out) if return_excluded else out
    excluded = out[out["scope"] != "gene"].copy()
    out = out[out["scope"] == "gene"].copy()
    out["abs"] = out["score"].abs()
    out = out.sort_values("abs", ascending=False).drop_duplicates(["gene", "drug"]).drop(columns="abs")
    out["direction"] = out["score"].map(
        lambda v: "resistant" if v < 0 else ("supportive" if v > 0 else "neutral"))
    out = out.reset_index(drop=True)
    return (out, excluded.reset_index(drop=True)) if return_excluded else out


class KnowledgeGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        """Wipes the graph clean before rebuilding. Use with caution."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("   Cleared existing graph data")

    def create_constraints(self):
        """Ensures no duplicate nodes for the same gene/drug name."""
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT gene_name IF NOT EXISTS FOR (g:Gene) REQUIRE g.name IS UNIQUE")
            session.run("CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE")
        print("   Created uniqueness constraints")

    def load_genes_from_dgidb(self, dgidb_df):
        """Creates Gene nodes from the prepared DGIdb table."""
        with self.driver.session() as session:
            genes = dgidb_df["gene_name"].unique()
            for gene in genes:
                session.run("MERGE (g:Gene {name: $name})", name=gene)
        print(f"   Created {len(genes)} Gene nodes")

    def load_drugs_from_dgidb(self, dgidb_df):
        """Creates Drug nodes from the prepared DGIdb table."""
        with self.driver.session() as session:
            drugs = dgidb_df["drug"].unique()
            for drug in drugs:
                session.run("MERGE (d:Drug {name: $name})", name=drug)
        print(f"   Created {len(drugs)} Drug nodes")

    def load_gene_drug_relationships(self, dgidb_df):
        """GENE -[:TARGETED_BY {evidence_score}]-> DRUG, one edge per gene-drug pair."""
        with self.driver.session() as session:
            count = 0
            for r in dgidb_df.itertuples():
                session.run(
                    """
                    MATCH (g:Gene {name: $gene})
                    MATCH (d:Drug {name: $drug})
                    MERGE (g)-[r:TARGETED_BY]->(d)
                    SET r.evidence_score = $score,
                        r.interaction_type = $interaction_type,
                        r.num_sources = $num_sources,
                        r.approved = $approved,
                        r.db_sources = $db_sources,
                        r.source = 'DGIdb'
                    """,
                    gene=r.gene_name, drug=r.drug, score=r.evidence_score,
                    interaction_type=r.interaction_type, num_sources=int(r.num_sources),
                    approved=bool(r.approved), db_sources=r.sources,
                )
                count += 1
        print(f"   Created {count} GENE -> DRUG relationships")

    def load_civic_evidence(self, civic_prepared):
        """GENE -[:HAS_CLINICAL_EVIDENCE]-> DRUG from the prepared CIViC table."""
        with self.driver.session() as session:
            count = 0
            for r in civic_prepared.itertuples():
                session.run("MERGE (g:Gene {name: $gene})", gene=r.gene)
                session.run("MERGE (d:Drug {name: $drug})", drug=r.drug)
                session.run(
                    """
                    MATCH (g:Gene {name: $gene})
                    MATCH (d:Drug {name: $drug})
                    MERGE (g)-[r:HAS_CLINICAL_EVIDENCE]->(d)
                    SET r.evidence_score = $score,
                        r.evidence_level = $level,
                        r.disease = $disease,
                        r.clinical_significance = $significance,
                        r.source = 'CIViC',
                        r.direction = $direction
                    """,
                    gene=r.gene, drug=r.drug, score=r.score, level=r.level,
                    disease=r.disease, significance=r.significance, direction=r.direction,
                )
                count += 1
        print(f"   Created {count} CIViC clinical evidence relationships")

    def load_fda_approved(self, fda_df):
        """Creates (or flags) a Drug node for every FDA-approved ovarian drug and
        records where the approval came from."""
        with self.driver.session() as session:
            for r in fda_df.itertuples():
                session.run(
                    """
                    MERGE (d:Drug {name: $name})
                    SET d.fda_ovarian_approved = true,
                        d.fda_source = $source,
                        d.brand_names = $brands
                    """,
                    name=r.drug_name, source=str(r.source),
                    brands="" if pd.isna(r.brand_names) else str(r.brand_names),
                )
        print(f"   Flagged {len(fda_df)} FDA-approved ovarian drugs")

    def print_summary(self):
        """Prints basic stats about what's now in the graph."""
        with self.driver.session() as session:
            gene_count = session.run("MATCH (g:Gene) RETURN count(g) AS c").single()["c"]
            drug_count = session.run("MATCH (d:Drug) RETURN count(d) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

            print("\n" + "=" * 50)
            print("KNOWLEDGE GRAPH SUMMARY")
            print("=" * 50)
            print(f"   Gene nodes:        {gene_count}")
            print(f"   Drug nodes:        {drug_count}")
            print(f"   Total relationships: {rel_count}")


def main():
    # NOTE: this wipes the whole graph. Re-run load_patients, load_pathways and
    # load_go_terms afterwards (see README), then ranking/export_graph.py.
    print("Building Evidence-Aware Knowledge Graph...")
    print("=" * 50)

    # Load our collected data
    print("\n[1/5] Loading data files...")
    try:
        dgidb_raw = pd.read_csv("data/raw/drugbank/dgidb_interactions.csv")
        civic_raw = pd.read_csv("data/raw/civic/civic_ovarian_evidence.csv")
    except (OSError, pd.errors.ParserError) as e:
        print(f"   Could not read input files: {e}")
        sys.exit(1)
    fda_df = load_fda_list()
    fda_names = {n.lower() for n in fda_df["drug_name"]}
    dgidb_df = prepare_dgidb(dgidb_raw, civic_raw, fda_names)
    civic_df, civic_excluded = prepare_civic(civic_raw, fda_names=fda_names, return_excluded=True)
    print(f"   CIViC rows not applied to mutation-based matching: "
          f"{civic_excluded['scope'].value_counts().to_dict() if len(civic_excluded) else {}}")
    try:
        os.makedirs("data/processed", exist_ok=True)
        civic_excluded.to_csv("data/processed/civic_not_applied.csv", index=False)
    except OSError as e:
        print(f"   Could not write civic_not_applied.csv: {e}")
    print(f"   DGIdb: {len(dgidb_raw)} raw rows -> {len(dgidb_df)} gene-drug pairs "
          f"({dgidb_df['drug'].nunique()} drugs)")
    print(f"   CIViC: {len(civic_raw)} raw rows -> {len(civic_df)} gene-drug pairs")

    # Connect to Neo4j
    print("\n[2/5] Connecting to Neo4j...")
    kg = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # Clear and set up
    print("\n[3/5] Preparing graph...")
    kg.clear_database()
    kg.create_constraints()

    # Load nodes and relationships
    print("\n[4/5] Loading nodes and relationships...")
    kg.load_genes_from_dgidb(dgidb_df)
    kg.load_drugs_from_dgidb(dgidb_df)
    kg.load_gene_drug_relationships(dgidb_df)
    kg.load_civic_evidence(civic_df)
    kg.load_fda_approved(fda_df)

    # Summary
    print("\n[5/5] Done!")
    kg.print_summary()

    kg.close()


if __name__ == "__main__":
    main()