from openai import OpenAI

client = OpenAI(
    api_key="sk-KojeCDC5OULi2MOn7bDf42243bD9402c8217F8Ea43B4A978",
    base_url="http://222.20.98.63:3010/v1"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.1,
)
print(response.choices[0].message.content)