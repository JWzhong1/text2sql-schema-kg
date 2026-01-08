import os
import sys
import logging
import dotenv
from pathlib import Path
import json
import datetime
import sqlite3

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.text2sql_agent import Text2SQLAgent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # 加载环境变量
    dotenv.load_dotenv()
    
    # 配置 Neo4j 连接
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "your_password")
    neo4j_config = (neo4j_uri, neo4j_user, neo4j_password)

    # 设定测试参数
    # 这里以 california_schools 为例，你可以修改为你 workspace 中存在的其他 DB
    db_name = "financial" 
    
    # 构造 SQLite 路径 (根据你的 workspace 结构)
    project_root = Path(__file__).parent.parent
    db_path = f"bird_data/dev_databases/{db_name}/{db_name}.sqlite"
    
    if not Path(db_path).exists():
        logging.error(f"Database file not found at {db_path}")
        return

    # 初始化 Agent
    agent = Text2SQLAgent(
        db_name=db_name,
        db_path=db_path,
        cache_dir="src/agent/schema_retrieval_cache",
        neo4j_config=(neo4j_uri, neo4j_user, neo4j_password)
    )

    # 准备输出目录
    output_dir = Path("src/agent/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"test_results_{timestamp}.json"

    results_summary = []

    try:
        # 测试问题
        json_path = "bird_data/dev_20251106/financial.json"
        if not os.path.exists(json_path):
             logging.error(f"JSONL file not found at {json_path}")
             return

        with open(json_path, "r", encoding="utf-8") as f:
            dev_jsonl = json.load(f)

        # 只测试前5个题目
        test_cases = dev_jsonl
        
        for idx, case in enumerate(test_cases):
            question = case["question"]
            if case["question_id"] != 103:
                continue
            evidence = case.get("evidence", "")
            golden_sql = case.get("SQL", "")    
            golden_result = case.get("execution_result", [])  

            print(f"\n{'='*50}")
            print(f"Processing Question {idx + 1}/5: {question}")
            print(f"{'='*50}\n")

            # 记录开始时间
            start_time = datetime.datetime.now()
            
            result = agent.solve(question, evidence)
            
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()

            # --- 修改开始: 参照 evaluation.py 执行 Golden SQL 并使用集合比较 ---
            generated_result = result.get("result", [])

            # 1. 执行 Golden SQL 获取真实结果 (Evaluation 逻辑)
            golden_result_executed = []
            try:
                # 使用 sqlite3 连接执行
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(golden_sql)
                    golden_result_executed = cursor.fetchall()
            except Exception as e:
                logging.warning(f"Golden SQL executon failed: {e}")
                golden_result_executed = f"Error: {str(e)}"

            # 2. 比较逻辑: 转为集合比较 (忽略顺序)
            def normalize_to_set(rows):
                if not isinstance(rows, list):
                    return set()
                # 确保行转为 tuple (可哈希)，单值转为单元素 tuple
                return set(tuple(row) if isinstance(row, (list, tuple)) else (row,) for row in rows)

            # 仅当两者均为列表结果时进行比较
            if isinstance(generated_result, list) and isinstance(golden_result_executed, list):
                is_correct = normalize_to_set(generated_result) == normalize_to_set(golden_result_executed)
            else:
                is_correct = False
            # --- 修改结束 ---

            case_record = {
                "id": idx,
                "question": question,
                "evidence": evidence,
                "golden_sql": golden_sql,
                "golden_result_original": golden_result, # 保留原 JSON 中的参考值
                "golden_result_executed": golden_result_executed, # 新增: 实际 Golden SQL 执行结果
                "generated_sql": result.get("sql"),
                "generated_result": generated_result,
                "status": result.get("status"),
                "error": result.get("error"),
                "is_correct": is_correct,
                "duration_seconds": duration,
                # 如果 agent.solve 返回了中间步骤 (例如 history 或 thoughts)，可以在这里保存
                "intermediate_steps": result.get("history", []) 
            }
            
            results_summary.append(case_record)

            print(f"Status: {result['status']}")
            print(f"Correct: {is_correct}")
            if result["status"] == "success":
                print(f"Generated SQL: \n{result['sql']}\n")
            else:
                print(f"Error: {result.get('error')}")

        # 保存结果到文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=4, ensure_ascii=False)
        
        print(f"\n{'='*50}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*50}\n")

    finally:
        agent.close()

if __name__ == "__main__":
    main()