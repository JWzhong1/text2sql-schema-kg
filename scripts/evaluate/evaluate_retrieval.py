import json
import os
import logging
import sys
from collections import defaultdict
import dotenv
import concurrent.futures
import threading
from tqdm import tqdm
import time

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

# Locks for thread safety
file_lock = threading.Lock()
print_lock = threading.Lock()
eval_lock = threading.Lock()
progress_lock = threading.Lock()

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

def process_single_db(db_id, cases, saved_results, output_file, eval_results, eval_output_file, neo4j_config, progress_bar):
    """
    Process all cases for a single database in a thread.
    """
    neo4j_uri, neo4j_user, neo4j_password = neo4j_config
    
    local_metrics = {
        "tbl_tp": 0, "tbl_fp": 0, "tbl_fn": 0,
        "col_tp": 0, "col_fp": 0, "col_fn": 0
    }
    
    # Check if we need to initialize retriever (if there are any cases not in cache)
    cases_needing_retrieval = [c for c in cases if str(c.get('question_id')) not in saved_results]
    
    retriever = None
    if cases_needing_retrieval:
        schema_path = f"bird_data/converted_schemas/{db_id}.json"
        if not os.path.exists(schema_path):
            logger.warning(f"Schema file not found: {schema_path}, skipping retrieval for {len(cases_needing_retrieval)} cases.")
            # Update progress for skipped cases
            with progress_lock:
                progress_bar.update(len(cases))
            return local_metrics
        else:
            logger.info(f"Initializing retriever for database: {db_id} ({len(cases_needing_retrieval)} new)")
            try:
                retriever = GraphRAGRetriever(neo4j_uri, neo4j_user, neo4j_password, schema_path)
            except Exception as e:
                logger.error(f"Failed to initialize retriever for {db_id}: {e}")
                retriever = None
                # Update progress for skipped cases
                with progress_lock:
                    progress_bar.update(len(cases))
                return local_metrics
    
    for case in cases:
        question = case['question']
        evidence = case.get('evidence', '')
        golden_link = case['golden_schema_link']
        question_id = str(case.get('question_id'))
        
        query = {
            "question": question,
            "evidence": evidence
        }
        
        retrieved_link = {}

        # Check cache (thread-safe read)
        if question_id in saved_results:
            retrieved_link = saved_results[question_id]
        else:
            if retriever is None:
                with progress_lock:
                    progress_bar.update(1)
                continue

            logger.info(f"Retrieving Q: {question[:60]}...")
            try:
                retrieved_link = retriever.retrieve_schema_subgraph(query)
                
                # Save result with lock to prevent race conditions on file write
                with file_lock:
                    saved_results[question_id] = retrieved_link
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(saved_results, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                with progress_lock:
                    progress_bar.update(1)
                continue

        try:
            metrics = calculate_metrics(retrieved_link, golden_link)

            g_tbls = set(golden_link.keys())
            r_tbls = set(retrieved_link.keys())
            g_cols = set(f"{t}.{c}" for t, cs in golden_link.items() for c in cs)
            r_cols = set(f"{t}.{c}" for t, cs in retrieved_link.items() for c in cs)

            differences = {
                "missing_tables": sorted(g_tbls - r_tbls),
                "extra_tables": sorted(r_tbls - g_tbls),
                "missing_columns": sorted(g_cols - r_cols),
                "extra_columns": sorted(r_cols - g_cols)
            }
            has_mismatch = any(differences.values())

            if has_mismatch:
                match_info = {
                    "question": question,
                    "db_id": db_id,
                    "golden": golden_link,
                    "retrieved": retrieved_link,
                    "metrics": metrics,
                    "differences": differences
                }

                with eval_lock:
                    eval_results[question_id] = match_info
                    with open(eval_output_file, 'w', encoding='utf-8') as f:
                        json.dump(eval_results, f, ensure_ascii=False, indent=2)

            if has_mismatch:
                with print_lock:
                    print(f"\n[MISMATCH] DB: {db_id}")
                    print(f"  Question: {question}")
                    if differences["missing_tables"]:
                        print(f"  Missing Tables: {differences['missing_tables']}")
                    if differences["extra_tables"]:
                        print(f"  Extra Tables  : {differences['extra_tables']}")
                    if differences["missing_columns"]:
                        print(f"  Missing Columns: {differences['missing_columns']}")
                    if differences["extra_columns"]:
                        print(f"  Extra Columns  : {differences['extra_columns']}")
                    print("-" * 40)

            for k in local_metrics:
                local_metrics[k] += metrics[k]
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
        finally:
            # Update progress after processing each question
            with progress_lock:
                progress_bar.update(1)
    
    if retriever:
        retriever.close()
        
    return local_metrics

def evaluate(db_name, test_file_path, retrieval_cache_dir, eval_report_path, max_workers=16):
    logger.info(f"Loading test cases from {test_file_path}")
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    # Load existing results
    if os.path.exists(retrieval_cache_dir):
        logger.info(f"Loading cached results from {retrieval_cache_dir}")
        with open(retrieval_cache_dir, 'r', encoding='utf-8') as f:
            saved_results = json.load(f)
    else:
        saved_results = {}

    # Prepare evaluation report file
    eval_output_file = eval_report_path
    os.makedirs(os.path.dirname(eval_output_file), exist_ok=True)
    eval_results = {}
    with open(eval_output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    # Group by db_id to minimize retriever re-initialization
    cases_by_db = defaultdict(list)
    for case in test_cases:
        cases_by_db[case['db_id']].append(case)

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "your_password")
    neo4j_config = (neo4j_uri, neo4j_user, neo4j_password)

    total_metrics = {
        "tbl_tp": 0, "tbl_fp": 0, "tbl_fn": 0,
        "col_tp": 0, "col_fp": 0, "col_fn": 0
    }
    
    total_cases = len(test_cases)
    logger.info(f"Starting evaluation with {max_workers} workers for {total_cases} questions across {len(cases_by_db)} databases.")

    # Create progress bar for total questions
    with tqdm(total=total_cases, desc="Processing Questions", unit="question") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for db_id, cases in cases_by_db.items():
                futures.append(
                    executor.submit(
                        process_single_db,
                        db_id,
                        cases,
                        saved_results,
                        retrieval_cache_dir,
                        eval_results,
                        eval_output_file,
                        neo4j_config,
                        progress_bar
                    )
                )

            try:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        local_metrics = future.result()
                        for k in total_metrics:
                            total_metrics[k] += local_metrics[k]
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")
            except KeyboardInterrupt:
                logger.warning("Evaluation interrupted by user")
                executor.shutdown(wait=False, cancel_futures=True)
                raise

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
    print(f"Total Questions Evaluated: {total_cases}")
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
    db_name = "financial"
    test_file = "bird_data/golden_link/golden_schema_link_financial.json"
    retrieval_cache_dir = f"scripts/evaluate/cache/retrieval_results_{db_name}.json"
    report_file = f"scripts/evaluate/result/eval_report_{db_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]

    if os.path.exists(test_file):
        evaluate(db_name, test_file, retrieval_cache_dir, report_file)