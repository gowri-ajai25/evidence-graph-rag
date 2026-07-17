from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

load_dotenv()
driver = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD')))

with driver.session() as session:
    result = session.run('''
        MATCH (g:Gene)-[r:HAS_CLINICAL_EVIDENCE]->(d:Drug)
        WHERE r.direction = "resistant"
        RETURN g.name AS gene, d.name AS drug, r.evidence_score AS score, r.clinical_significance AS sig
        LIMIT 10
    ''')
    print('Resistance relationships (should have negative scores):')
    for record in result:
        print(f'   {record["gene"]} -> {record["drug"]}: score={record["score"]}, sig={record["sig"]}')

    result = session.run('''
        MATCH ()-[r:HAS_CLINICAL_EVIDENCE]->()
        RETURN r.direction AS direction, count(*) AS count
    ''')
    print()
    print('Direction breakdown:')
    for record in result:
        print(f'   {record["direction"]}: {record["count"]}')

driver.close()