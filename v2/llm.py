from openai import OpenAI

from settings import settings
from .tools import TOOLS


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
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", 0) if usage is not None else 0
    return response.choices[0].message, total_tokens
