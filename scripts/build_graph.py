import os
import sys
import json
import logging
from typing import List, Dict, Any
from neo4j import GraphDatabase, basic_auth
import dotenv
from src.graph.schema_graph_builder import GenericSchemaGraphBuilder
dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



def load_schema_from_file(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Schema file must contain a JSON array.")
    return data

def clear_neo4j_database(uri: str, user: str, password: str) -> None:
    logger.info("Clearing Neo4j database before rebuilding graph.")
    driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()
    logger.info("Neo4j database cleared successfully.")

def main():
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    db_name = "financial"
    neo4j_password = os.getenv("NEO4J_PASSWORD", "your_password")
    schema_file_path = f"bird_data/converted_schemas/{db_name}.json"
    logger.info(f"Loading schema from {schema_file_path}")
    schema_data = load_schema_from_file(schema_file_path)
    cache_dir = os.getenv("SCHEMA_KG_CACHE_DIR", "cache")
    cache_dir = os.path.join(cache_dir, db_name)
    builder = GenericSchemaGraphBuilder(neo4j_uri, neo4j_user, neo4j_password, cache_dir=cache_dir)
    try:
        clear_neo4j_database(neo4j_uri, neo4j_user, neo4j_password)
        builder.create_indexes() 
        builder.build_graph(schema_data)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
    finally:
        builder.close()

if __name__ == "__main__":
    main()