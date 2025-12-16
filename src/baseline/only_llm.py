import json
import os
import sqlite3
from typing import Dict, Any, List, Union
from llm.client import get_competition_json

# 全局缓存，避免重复读取相同schema文件
SCHEMA_CACHE = {}

def load_schema(db_id: str, base_path: str = "bird_data/converted_schemas") -> str:
    """
    根据db_id加载对应的schema文件，并进行缓存。
    """
    if db_id in SCHEMA_CACHE:
        return SCHEMA_CACHE[db_id]
    
    schema_path = os.path.join(base_path, f"{db_id}.json")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
        # 将schema转为字符串缓存，因为后续使用也是需要字符串
        schema_str = json.dumps(schema, ensure_ascii=False)
        SCHEMA_CACHE[db_id] = schema_str
        return schema_str

def execute_sql(sql: str, db_path: str) -> Union[List[Any], str]:
    """
    执行SQL语句并返回结果。
    """
    if not os.path.exists(db_path):
        return f"Database file not found: {db_path}"
        
    try:
        # 使用只读模式连接数据库，防止意外修改
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return result
    except sqlite3.Error as e:
        return f"SQLite Error: {e}"
    except Exception as e:
        return f"Error executing SQL: {e}"

def compare_results(generated_result: Any, golden_result: Any) -> bool:
    """
    比较生成结果和标准结果。
    """
    if isinstance(generated_result, str) and generated_result.startswith("Error"):
        return False
        
    # 将sqlite返回的tuple列表转换为list列表，以便与json中的list列表比较
    if isinstance(generated_result, list):
        # 处理结果中的浮点数精度问题，这里简单处理，如果需要更精确的比较可能需要专门的逻辑
        formatted_result = []
        for row in generated_result:
            formatted_row = []
            for val in row:
                # 如果是tuple，转为list (虽然fetchall返回的是list of tuples)
                if isinstance(val, tuple):
                    formatted_row.append(list(val))
                else:
                    formatted_row.append(val)
            formatted_result.append(formatted_row)
        
        # 简单的全等比较
        # 注意：这里没有处理列的顺序或者行的顺序可能不同的情况，
        # 也没有处理浮点数微小差异的情况。
        return formatted_result == golden_result
        
    return False

def schema_link(question: Dict[str, Any], schema: str) -> Dict[str, Any]:
    """
    调用LLM生成SQL。
    """
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个数据分析专家，精通使用SQL查询关系型数据库。"
                "根据用户的问题，结合提供的数据库模式(schema)，生成准确的SQL查询语句,可用于sqlite查询。"
                "以json格式返回结果，包含字段 'sql'，其值为生成的SQL查询语句。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题: {question['question']}\n\n"
                f"背景知识: {question['evidence']}\n\n"
                f"请基于下列数据库模式和用户问题，生成相应的SQL查询语句。注意：使用schema中的original_name\n"
                f"数据库模式(schema):\n{schema}\n\n"
                "请以json格式返回结果，输出示例：\n"
                "{\"sql\": \"SELECT * FROM table_name WHERE condition;\", \"reasoning\": \"解释说明\"}"
            ),
        },
    ]
    
    try:
        response_str = get_competition_json(messages)
        # 尝试解析返回的JSON字符串
        response_json = json.loads(response_str)
        return response_json
    except json.JSONDecodeError:
        print(f"Error decoding JSON from LLM response: {response_str}")
        return {"sql": "Error generating SQL", "reasoning": "JSON Decode Error"}
    except Exception as e:
        print(f"Error in schema_link: {e}")
        return {"sql": "Error generating SQL", "reasoning": str(e)}

def main():
    test_file_path = "test.jsonl"
    if not os.path.exists(test_file_path):
        print(f"Test file not found: {test_file_path}")
        return

    # 读取整个文件内容
    with open(test_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 解析包含多个JSON对象的文件（可能是pretty-printed JSONL）
    items = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        try:
            # 跳过空白字符
            while pos < len(content) and content[pos].isspace():
                pos += 1
            if pos >= len(content):
                break
            
            obj, end = decoder.raw_decode(content, idx=pos)
            items.append(obj)
            pos = end
        except json.JSONDecodeError:
            # 尝试跳过非JSON内容或报错
            print(f"Error decoding JSON at position {pos}")
            break

    # 处理前5条测试数据
    for i, item in enumerate(items):
        if i >= 5:
            break
            
        db_id = item.get("db_id")
        if not db_id:
            print(f"Skipping item without db_id: {item}")
            continue

        print(f"--- Test Case ID: {item.get('question_id', 'Unknown')} ---")
        print("Question:", item["question"])
        print("Evidence:", item["evidence"])
        print("DB ID:", db_id)

        try:
            schema_str = load_schema(db_id)
            result = schema_link(item, schema_str)
            generated_sql = result.get("sql")
            print("Generated SQL:", generated_sql)
            print("Reasoning:", result.get("reasoning"))
            
            if generated_sql and generated_sql != "Error generating SQL":
                db_path = f"bird_data/bird/llm/data/dev_databases/{db_id}/{db_id}.sqlite"
                exec_result = execute_sql(generated_sql, db_path)
                
                # 简单的结果展示
                if isinstance(exec_result, list):
                    print(f"Execution Result (First 3 rows): {exec_result[:3]}")
                else:
                    print(f"Execution Result: {exec_result}")
                
                golden_result = item.get("execution_result")
                is_match = compare_results(exec_result, golden_result)
                print(f"Match Golden: {is_match}")
                
                if not is_match:
                    # 如果不匹配，打印Golden Result的前几行以便对比
                    if isinstance(golden_result, list):
                         print(f"Golden Result (First 3 rows): {golden_result[:3]}")
                    else:
                         print(f"Golden Result: {golden_result}")

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"An error occurred processing item: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    main()
        
