from dotenv import load_dotenv
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))
from subgraph_retriever import SubgraphRetriever

load_dotenv()

retriever = SubgraphRetriever(
    os.getenv("NEO4J_URI"), os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")
)

rows = retriever.get_patient_subgraph("TCGA-13-1497")

print(f"Total rows returned: {len(rows)}\n")
for row in rows:
    print(row)

retriever.close()