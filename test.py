from openai import OpenAI

client = OpenAI(
    base_url="http://222.20.98.63:3010/v1",
    api_key="sk-LGwJMc2YynUHJt7a5fB93b9e9b0f473891A7FcFc76FcEeB5"
)

response = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role": "user", "content": "Hello, world!"}
    ],
    max_tokens=1024,
    stream=False
)

print(response.choices[0].message.content)

if response.usage:
    print(f"Using Model:{response.model}\n"
          f"Prompt tokens: {response.usage.prompt_tokens}\n"
          f"Completion tokens: {response.usage.completion_tokens}\n"
          f"Total tokens: {response.usage.total_tokens}")
