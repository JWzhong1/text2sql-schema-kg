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

def main():
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "your_password")
    schema_file_path = "bird_data/converted_schemas/california_schools.json"
    logger.info(f"Loading schema from {schema_file_path}")
    schema_data = load_schema_from_file(schema_file_path)
    builder = GenericSchemaGraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
    try:
        # 修改：先创建索引，确保即使后续数据导入失败，索引也存在
        builder.create_indexes() 
        builder.build_graph(schema_data)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
    finally:
        builder.close()

if __name__ == "__main__":
    main()