from llm.client import get_competition_json

def schema_link(question: str, schema: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个数据分析专家，精通使用SQL查询关系型数据库。"
                "根据用户的问题，结合提供的数据库模式(schema)，为当前问题找出sql查询语句所需的表和列字段。"
                "以json"
            ),
        },
        {
            "role": "user",
            "content": (
                f"数据库模式(schema):\n{schema}\n\n"
                f"用户问题: {question}\n\n"
                "请生成相应的SQL查询语句。"
            ),
        },
    ]
    response = get_competition_json(messages)
    return response