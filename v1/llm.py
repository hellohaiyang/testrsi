from openai import OpenAI
from pathlib import Path

from settings import settings
from .tools import tools, TOOLS, execute_tool


def call_llm(messages):
    client = OpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        stream=False,
        reasoning_effort="high",
        tools=TOOLS,
        extra_body={"thinking": {"type": "enabled"}},
    )
    return response.choices[0].message, response.usage.total_tokens
