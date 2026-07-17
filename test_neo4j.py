from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

print(f"Connecting to: {uri}")

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print("Neo4j connection successful!")
    
    # Create a simple test node
    with driver.session() as session:
        session.run("CREATE (t:Test {name: 'connection_test'}) RETURN t")
        result = session.run("MATCH (t:Test) RETURN t.name AS name")
        for record in result:
            print(f"Test node found: {record['name']}")
        
        # Clean up test node
        session.run("MATCH (t:Test) DELETE t")
        print("Test node cleaned up")
    
    driver.close()
    print("\nNeo4j is ready!")

except Exception as e:
    print(f"Connection failed: {e}")