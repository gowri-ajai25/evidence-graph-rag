import os, csv, time, requests
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
URI  = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD  = os.getenv("NEO4J_PASSWORD")


def get_genes():
    d = GraphDatabase.driver(URI, auth=(USER, PWD))
    with d.session() as s:
        rows = s.run("MATCH (g:Gene) RETURN g.name AS s").data()
    d.close()
    return sorted({r["s"] for r in rows if r["s"]})


def as_list(x):
    return [] if x is None else (x if isinstance(x, list) else [x])


def fetch(batch):
    r = requests.post(
        "https://mygene.info/v3/query",
        data={"q": ",".join(batch), "scopes": "symbol",
              "species": "human", "fields": "symbol,go.BP,go.MF"},
        timeout=60)
    r.raise_for_status()
    hits = r.json()
    # /query can return several hits per gene; keep the first one that has GO data
    best = {}
    for h in hits:
        if h.get("notfound") or "go" not in h:
            continue
        best.setdefault(h["query"], h)
    return list(best.values())


genes = get_genes()
print(f"{len(genes)} genes in KG")
os.makedirs("data/raw/go", exist_ok=True)

rows_written = 0
genes_with_go = set()

with open("data/raw/go/gene_go_terms.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["gene", "go_id", "term", "category", "evidence"])
    for i in range(0, len(genes), 200):
        for hit in fetch(genes[i:i + 200]):
            if hit.get("notfound") or "go" not in hit:
                continue
            for cat in ("BP", "MF"):
                for t in as_list(hit["go"].get(cat)):
                    w.writerow([hit["query"], t.get("id"), t.get("term"),
                                cat, t.get("evidence")])
                    rows_written += 1
                    genes_with_go.add(hit["query"])
        time.sleep(0.5)

print(f"done: {rows_written} annotations for {len(genes_with_go)}/{len(genes)} genes")