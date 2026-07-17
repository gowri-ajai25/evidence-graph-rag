from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

load_dotenv()
driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))

with driver.session() as session:
    result = session.run("""
        MATCH (g:Gene {name: "TP53"})-[r]->(d:Drug {name: "Cisplatin"})
        RETURN type(r) AS rel_type, r.evidence_score AS score, r.source AS source, r.direction AS direction
    """)
    print("TP53 -> Cisplatin, all relationships:")
    for record in result:
        print(f"   {record['rel_type']}: score={record['score']}, source={record['source']}, direction={record['direction']}")

driver.close()