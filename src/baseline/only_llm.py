import json
import os
from typing import Dict, List
from src.llm.client import get_competition_json

def schema_linking(nl_query: str,evidence: str, schema_file_path: str) -> Dict[str, List[str]]:
    """
    调用LLM执行schema linking
    :param nl_query: 自然语言查询
    :param evidence: 查询说明
    :param schema_file_path: schema.json文件路径
    :return: JSON格式的schema link结果
    """
    # 1. 读取schema文件
    with open(schema_file_path, 'r', encoding='utf-8') as f:
        schema_data = json.load(f)
    
    # 2. 构造LLM提示词
    sys_prompt = f"""你是一个数据库专家，请根据用户查询准确识别涉及的表和字段。
要求：
1. 仅输出JSON格式结果，不要任何解释
2. 仅包含查询中明确提及或强烈暗示的表和字段
3. 字段名必须严格匹配schema中的原始名称
4. 输出格式示例：
{{
    "table1": ["column1", "column2"],
    "table2": ["column3"]
}}
"""
    user_prompt = f"""
    ## 用户查询: {nl_query}
## 查询说明：{evidence}
## 数据库Schema:

{json.dumps(schema_data, indent=2, ensure_ascii=False)}

"""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # 3. 调用LLM (此处为模拟，实际替换为真实API调用)
    llm_response = get_competition_json(messages)
    
    # 4. 解析LLM响应
    try:
        schema_link_result = json.loads(llm_response)
        return schema_link_result
    except json.JSONDecodeError:
        print("LLM返回的结果无法解析为JSON")        
    
    # 5. 失败时返回空结果
    return {}

# ===== 演示使用 =====
if __name__ == "__main__":
    db_name = "financial"
    schema_json_path = f"bird_data/converted_schemas/{db_name}.json"
    dev_path = f"bird_data/dev_20251106/{db_name}.json"
    output_path = f"bird_data/golden_link/golden_schema_link_{db_name}_dev2025.json"
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_jsonl = json.load(f)

    for item in dev_jsonl:
        nl_query = item["question"]
        evidence = item.get("evidence", "")
        schema_link_result = schema_linking(nl_query, evidence, schema_json_path)
        item["golden_schema_link"] = schema_link_result
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(dev_jsonl, out_f, ensure_ascii=False, indent=2)