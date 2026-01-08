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
        api_key=os.getenv("DASHSCOPE_API_KEY"),  
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.embeddings.create(
        model="text-embedding-v4",
        input=text,
        dimensions=1024
    )

    return completion.data[0].embedding

def get_competition_from_coder(messages: list) -> str:
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    response = client.chat.completions.create(
        model=os.getenv("CODER_MODEL", "qwen3-coder-plus"),
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content