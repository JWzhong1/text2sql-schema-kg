import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 推荐添加 tqdm 以显示进度条

def extract_tables_and_fields(sql:str, schema_json:dict):
    """
    从SQL语句中提取使用到的表及字段。
    """
    sys_prompt = """
你是一个SQL解析助手，负责从sql语句中提取使用到的表及字段，严格按照JSON格式返回结果。

"""
    user_prompt = f"""
参考数据库的schema文档，从以下SQL语句中提取使用到的表及字段：
SQL: {sql} \n\n
数据库schema文档: {json.dumps(schema_json, ensure_ascii=False)} \n\n
请返回一个JSON对象，包含两个字段,示例如下
```json
{{
  "frpm": [
    "CDSCode",
    "Low Grade",
    "School Name"
  ],
  "schools": [
    "CDSCode",
    "City",
    "State",
    "Latitude"
  ]
}}
```
"""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    from src.llm.client import get_competition_json
    try:
        resp = get_competition_json(messages)
        return resp
    except Exception as e:
        print(f"Error during extraction: {e}")
        return {}

def process_single_item(item, db_schema):
    """
    处理单个条目（已限定到某一个 db）。
    """
    question_id = item.get("question_id")
    question = item.get("question")
    db_id = item.get("db_id")
    sql = item.get("SQL")
    evidence = item.get("evidence", "")
    difficulty = item.get("difficulty", "")
    
    if not sql:
        return None

    extracted = extract_tables_and_fields(sql, db_schema)
    try:
        golden_link = json.loads(extracted) if isinstance(extracted, str) else extracted
        return {
            "question_id": question_id,
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "SQL": sql,
            "difficulty": difficulty,
            "golden_schema_link": golden_link
        }
    except Exception as e:
        print(f"解析提取结果失败 question_id {question_id}: {e}")
        return None

def process_dev_file(dev_file_path, schema_file_path: Path, output_file_path):
    """
    仅处理与 schema_file_path 对应 db 的条目。
    """
    with open(dev_file_path, 'r', encoding='utf-8') as dev_file:
        dev_data = json.load(dev_file)
    
    if not schema_file_path.exists():
        print(f"schema 文件不存在: {schema_file_path}")
        return
    
    try:
        with open(schema_file_path, 'r', encoding='utf-8') as f:
            db_schema = json.load(f)
    except json.JSONDecodeError:
        print(f"schema 文件解析失败: {schema_file_path}")
        return

    # 从文件名推断 db_id（去除扩展名）
    target_db_id = schema_file_path.stem

    items_to_process = [item for item in dev_data if item.get("db_id") == target_db_id]
    if not items_to_process:
        print(f"dev.json 中没有匹配 db_id={target_db_id} 的条目。")
        return

    results = []
    max_workers = 8

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_single_item, item, db_schema): item for item in items_to_process}
        for future in tqdm(as_completed(future_to_item), total=len(items_to_process), desc=f"Processing {target_db_id}"):
            result = future.result()
            if result:
                results.append(result)

    results.sort(key=lambda x: x.get("question_id", 0))

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(results, output_file, indent=4, ensure_ascii=False)
    print(f"提取结果已保存到 {output_file_path}")

if __name__ == "__main__":
    # 仅处理单个 db
    dev_file = "bird_data/bird/llm/data/dev.json"
    # 改为具体 schema 文件
    schema_file = Path("bird_data/converted_schemas/financial.json")
    output_file = "bird_data/golden_link/golden_schema_link_financial.json"
    process_dev_file(dev_file, schema_file, output_file)