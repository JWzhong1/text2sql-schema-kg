import openai
import os
from dotenv import load_dotenv
import dashscope

load_dotenv()
def get_competition(messages: list) -> str:
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_API_MODEL", "qwen-plus"),
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message.content

def get_competition_json(messages: list) -> str:
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_API_MODEL", "qwen-plus"),
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content

def get_competition_embedding(text: str) -> list:
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),  
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.embeddings.create(
        model="text-embedding-v4",
        input=text,
        dimensions=1024
    )

    return completion.data[0].embedding


def get_competition_rerank(query: str,
                           documents: list,
                           top_n: int | None = None,
                           return_documents: bool = True,
                           model: str | None = None):
    """
    调用阿里 DashScope 文本重排模型 (默认 gte-rerank-v2 / 可通过环境变量 RERANK_MODEL 指定，如 qwen3-rerank)。
    返回列表: [{"index": 原始索引, "relevance_score": 分数, "text": 文本}]
    """

    model = model or os.getenv("RERANK_MODEL", "qwen3-rerank")
    if top_n is None:
        top_n = len(documents)

    resp = dashscope.TextReRank.call(
        model=model,
        query=query,
        documents=[d if isinstance(d, str) else str(d) for d in documents],
        top_n=top_n,
        return_documents=return_documents
    )
    if getattr(resp, "status_code", 500) != 200:
        raise RuntimeError(f"Rerank 调用失败: {getattr(resp,'message',resp)}")

    results = resp.output.get("results", [])
    parsed = []
    for r in results:
        parsed.append({
            "index": r.get("index"),
            "relevance_score": r.get("relevance_score"),
            "text": (r.get("document") or {}).get("text", "")
        })
    return parsed