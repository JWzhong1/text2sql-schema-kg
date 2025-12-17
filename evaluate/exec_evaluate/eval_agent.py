import os
import sys
import logging
import dotenv
from pathlib import Path
import json
import datetime
import concurrent.futures

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.text2sql_agent import Text2SQLAgent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_case(idx, case, agent, db_name):
    question = case["question"]
    evidence = case.get("evidence", "")
    golden_sql = case.get("SQL", "")
    case_db_name = case.get("db_id", db_name)

    start_time = datetime.datetime.now()
    result = agent.solve(question, evidence)
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    generated_sql = result.get("sql", "")
    if generated_sql is None:
        generated_sql = ""
    generated_sql_clean = " ".join(generated_sql.split())
    golden_sql_clean = " ".join(golden_sql.split())

    predict_result = (str(idx), f"{generated_sql_clean}\t----- bird -----\t{case_db_name}")
    gold_sql = f"{golden_sql_clean}\t{case_db_name}"
    case_record = {
        "id": idx,
        "db_id": case_db_name,
        "question": question,
        "evidence": evidence,
        "golden_sql": golden_sql,
        "generated_sql": generated_sql,
        "status": result.get("status"),
        "error": result.get("error"),
        "duration_seconds": duration,
        "intermediate_steps": result.get("history", [])
    }
    return predict_result, gold_sql, case_record, result

def main():
    # 加载环境变量
    dotenv.load_dotenv()
    
    # 配置 Neo4j 连接
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "your_password")
    neo4j_config = (neo4j_uri, neo4j_user, neo4j_password)

    # 设定测试参数
    db_name = "california_schools" 
    
    # 构造 SQLite 路径
    project_root = Path(__file__).parent.parent
    db_path = f"bird_data/bird/llm/data/dev_databases/{db_name}/{db_name}.sqlite"
    
    if not Path(db_path).exists():
        logging.error(f"Database file not found at {db_path}")
        return

    # 初始化 Agent
    agent = Text2SQLAgent(
        db_name=db_name,
        db_path=db_path,
        cache_dir="src/agent/schema_retrieval_cache_dev",
        neo4j_config=(neo4j_uri, neo4j_user, neo4j_password)
    )

    # 准备输出目录
    output_dir = Path("evaluate/exec_evaluate/output_dev_o3")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 输出文件路径 - 符合 evaluation.py 的格式要求
    predict_json_file = output_dir / f"predict_dev.json"  # 预测结果 JSON
    gold_sql_file = output_dir / f"dev_gold.sql"          # Ground Truth SQL
    detail_file = output_dir / f"test_details_{timestamp}.json"  # 详细结果（可选）

    # 用于 evaluation.py 的输出格式
    predict_results = {}  # {idx: "sql\t----- bird -----\tdb_name"}
    gold_sqls = []        # ["sql\tdb_name", ...]
    results_summary = []  # 详细结果

    try:
        # 测试问题
        jsonl_path = "bird_data/golden_link/golden_schema_link_california_schools.json"
        if not os.path.exists(jsonl_path):
             logging.error(f"JSONL file not found at {jsonl_path}")
             return

        with open(jsonl_path, "r", encoding="utf-8") as f:
            dev_jsonl = json.load(f)

        test_cases = dev_jsonl  # 只测试前5个题目
        
        # 多线程并发执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_idx = {
                executor.submit(process_case, idx, case, agent, db_name): idx
                for idx, case in enumerate(test_cases)
            }
            results_buffer = [None] * len(test_cases)
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                predict_result, gold_sql, case_record, result = future.result()
                results_buffer[idx] = (predict_result, gold_sql, case_record, result)

        # 按顺序写入结果
        for idx, (predict_result, gold_sql, case_record, result) in enumerate(results_buffer):
            predict_results[predict_result[0]] = predict_result[1]
            gold_sqls.append(gold_sql)
            results_summary.append(case_record)

            print(f"\n{'='*50}")
            print(f"Processing Question {idx + 1}/{len(test_cases)}: {case_record['question']}")
            print(f"{'='*50}\n")
            print(f"Status: {result['status']}")
            if result["status"] == "success":
                print(f"Generated SQL: \n{case_record['generated_sql']}\n")
            else:
                print(f"Error: {result.get('error')}")

        # 保存预测结果 JSON (evaluation.py 格式)
        with open(predict_json_file, "w", encoding="utf-8") as f:
            json.dump(predict_results, f, indent=4, ensure_ascii=False)
        
        # 保存 Ground Truth SQL 文件 (evaluation.py 格式)
        with open(gold_sql_file, "w", encoding="utf-8") as f:
            f.write("\n".join(gold_sqls))
        
        # 保存详细结果
        with open(detail_file, "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=4, ensure_ascii=False)
        
        print(f"\n{'='*50}")
        print(f"Evaluation files saved:")
        print(f"  - Predictions: {predict_json_file}")
        print(f"  - Gold SQL: {gold_sql_file}")
        print(f"  - Details: {detail_file}")
        print(f"{'='*50}")
        print(f"\nTo run evaluation:")
        print(f"python bird_data/bird/llm/src/evaluation.py \\")
        print(f"  --predicted_sql_path src/agent/output/ \\")
        print(f"  --ground_truth_path src/agent/output/ \\")
        print(f"  --db_root_path bird_data/bird/llm/data/dev_databases/ \\")
        print(f"  --data_mode dev \\")
        print(f"  --diff_json_path <your_diff_json_path>")
        print(f"{'='*50}\n")

    finally:
        agent.close()

if __name__ == "__main__":
    main()