import logging
import os
import json
import time
from src.graph.schema_graph_retriever import GraphRAGRetriever
import dotenv

dotenv.load_dotenv()

# 创建日志目录
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"retrieval_{time.strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 设置 neo4j 库的日志级别为 WARNING，屏蔽 INFO 级别的 DBMS 通知
logging.getLogger("neo4j").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "your_password")
    schema_json_path="bird_data/converted_schemas/california_schools.json"
    retriever = GraphRAGRetriever(neo4j_uri, neo4j_user, neo4j_password, schema_json_path)

    golden_link_path = "bird_data/golden_link/golden_schema_link_test.json"
    with open(golden_link_path, 'r', encoding='utf-8') as f:
        golden_links = json.load(f)
    logger.info(f"Loaded {len(golden_links)} golden links from {golden_link_path}") 
    for link in golden_links[:1]:  # 仅测试第一个链接
        nl_query = link['question']
        back_knowledge = link['evidence']
        query = {
            "question": nl_query,
            "evidence": back_knowledge
        }
        logger.info(f"Processing query: {nl_query} \n\nEvidence: {back_knowledge}")
        table_column_map = retriever.retrieve_schema_subgraph(query)
        logger.info(f"Table-Column Map: {json.dumps(table_column_map, ensure_ascii=False)}")