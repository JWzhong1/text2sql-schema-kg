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

def process_single_item(item, schema_dir):
    """
    处理单个条目的辅助函数，用于多线程调用。
    """
    question_id = item.get("question_id")
    question = item.get("question")
    db_id = item.get("db_id")
    sql = item.get("SQL")
    evidence = item.get("evidence", "")
    difficulty = item.get("difficulty", "")
    
    schema_file_path = schema_dir / f"{db_id}.json"

    if not schema_file_path.exists():
        print(f"警告: 数据库ID {db_id} 的schema文件未在 {schema_file_path} 找到，跳过该条目。")
        return None
    
    with open(schema_file_path, 'r', encoding='utf-8') as f:
        try:
            db_schema = json.load(f)
        except json.JSONDecodeError:
            print(f"警告: schema文件 {schema_file_path} 解析失败，跳过。")
            return None

    if sql:
        extracted = extract_tables_and_fields(sql, db_schema)
        try:
            # extract_tables_and_fields 返回的可能是字符串形式的 JSON，也可能是字典（取决于 get_competition_json 的实现）
            # 这里假设它返回的是字符串，如果已经是字典则不需要 json.loads
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
    return None

def process_dev_file(dev_file_path, schema_json_path, output_file_path):
    """
    处理dev.json文件，提取SQL中使用到的表及字段，并保存到JSON文件。
    """
    with open(dev_file_path, 'r', encoding='utf-8') as dev_file:
        dev_data = json.load(dev_file)
    
    results = []
    schema_dir = Path("data/schemas")
    
    # 设置最大线程数，例如 10
    max_workers = 10
    
    items_to_process = dev_data 

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_item = {executor.submit(process_single_item, item, schema_dir): item for item in items_to_process}
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(future_to_item), total=len(items_to_process), desc="Processing"):
            result = future.result()
            if result:
                results.append(result)
    
    # 按 question_id 排序以保持顺序（可选）
    results.sort(key=lambda x: x.get("question_id", 0))

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(results, output_file, indent=4, ensure_ascii=False)
    print(f"提取结果已保存到 {output_file_path}")

if __name__ == "__main__":
    dev_file = "data/bird/llm/data/dev_tied_append.json"
    output_file = "data/golden_schema_link.json"
    schema_json  = "data/converted_schema.json"
    process_dev_file(dev_file, schema_json, output_file)