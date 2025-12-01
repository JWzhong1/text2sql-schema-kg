import json
import os
import logging
import sys
from collections import defaultdict
import dotenv

# Add project root to sys.path to allow importing src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.schema_graph_retriever import GraphRAGRetriever

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress neo4j logs
logging.getLogger("neo4j").setLevel(logging.WARNING)

def calculate_metrics(retrieved, golden):
    """
    Calculate TP, FP, FN for tables and columns.
    """
    # Normalize golden to set of tables and set of columns (table.col)
    golden_tables = set(golden.keys())
    golden_columns = set()
    for tbl, cols in golden.items():
        for col in cols:
            golden_columns.add(f"{tbl}.{col}")

    # Normalize retrieved
    retrieved_tables = set(retrieved.keys())
    retrieved_columns = set()
    for tbl, cols in retrieved.items():
        for col in cols:
            retrieved_columns.add(f"{tbl}.{col}")

    # Tables
    tbl_tp = len(retrieved_tables & golden_tables)
    tbl_fp = len(retrieved_tables - golden_tables)
    tbl_fn = len(golden_tables - retrieved_tables)

    # Columns
    col_tp = len(retrieved_columns & golden_columns)
    col_fp = len(retrieved_columns - golden_columns)
    col_fn = len(golden_columns - retrieved_columns)

    return {
        "tbl_tp": tbl_tp, "tbl_fp": tbl_fp, "tbl_fn": tbl_fn,
        "col_tp": col_tp, "col_fp": col_fp, "col_fn": col_fn
    }

def evaluate(test_file_path):
    logger.info(f"Loading test cases from {test_file_path}")
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    # Group by db_id to minimize retriever re-initialization
    cases_by_db = defaultdict(list)
    for case in test_cases:
        cases_by_db[case['db_id']].append(case)

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "your_password")

    total_metrics = {
        "tbl_tp": 0, "tbl_fp": 0, "tbl_fn": 0,
        "col_tp": 0, "col_fp": 0, "col_fn": 0
    }
    
    count = 0
    total_cases = len(test_cases)

    for db_id, cases in cases_by_db.items():
        schema_path = f"bird_data/converted_schemas/{db_id}.json"
        if not os.path.exists(schema_path):
            logger.warning(f"Schema file not found: {schema_path}, skipping {len(cases)} cases for {db_id}.")
            continue
            
        logger.info(f"Initializing retriever for database: {db_id} ({len(cases)} cases)")
        try:
            retriever = GraphRAGRetriever(neo4j_uri, neo4j_user, neo4j_password, schema_path)
        except Exception as e:
            logger.error(f"Failed to initialize retriever for {db_id}: {e}")
            continue

        for case in cases:
            count += 1
            question = case['question']
            evidence = case.get('evidence', '')
            golden_link = case['golden_schema_link']
            
            query = {
                "question": question,
                "evidence": evidence
            }
            
            logger.info(f"Processing [{count}/{total_cases}] Q: {question[:60]}...")
            try:
                retrieved_link = retriever.retrieve_schema_subgraph(query)
                metrics = calculate_metrics(retrieved_link, golden_link)
                
                for k in total_metrics:
                    total_metrics[k] += metrics[k]
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
        
        retriever.close()

    # Calculate global metrics
    def calc_f1(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    tbl_p, tbl_r, tbl_f1 = calc_f1(total_metrics['tbl_tp'], total_metrics['tbl_fp'], total_metrics['tbl_fn'])
    col_p, col_r, col_f1 = calc_f1(total_metrics['col_tp'], total_metrics['col_fp'], total_metrics['col_fn'])

    print("\n" + "="*40)
    print("RETRIEVAL EVALUATION REPORT")
    print("="*40)
    print(f"Total Questions Evaluated: {count}")
    print("-" * 20)
    print(f"TABLE LEVEL:")
    print(f"  Precision : {tbl_p:.4f}")
    print(f"  Recall    : {tbl_r:.4f}")
    print(f"  F1 Score  : {tbl_f1:.4f}")
    print("-" * 20)
    print(f"COLUMN LEVEL:")
    print(f"  Precision : {col_p:.4f}")
    print(f"  Recall    : {col_r:.4f}")
    print(f"  F1 Score  : {col_f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    # Default path, can be overridden or passed as arg if needed
    test_file = "bird_data/golden_link/golden_schema_link_test.json"
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]

    if os.path.exists(test_file):
        evaluate(test_file)