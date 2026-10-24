"""LLM 调用封装。

相对当前版本的变化（都很小、默认行为不变）：

* 模型名 / reasoning_effort 可通过 settings 或环境变量覆盖，便于用**不同模型**
  交叉验证 agent 的行为——执行分析里指出“内层 agent 与外层 agent 是同一个模型”
  会让“天花板效应”这类结论无法外推，可配置模型是消除该偏差的最低成本手段。
* 响应结构异常（没有 choices / choices[0].message 为空）时抛出明确异常，
  让 Agent.run() 记为 ``status="failed"`` 并留下可读的 error，
  而不是在循环深处抛 AttributeError。
"""

import os

from openai import OpenAI

from settings import settings
from .tools import TOOLS

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_REASONING_EFFORT = "high"


def _resolve(name, default):
    value = getattr(settings, name, None)
    if value is None:
        value = os.environ.get(name)
    return value or default


def call_llm(messages):
    client = OpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model=_resolve("RSI_MODEL", DEFAULT_MODEL),
        messages=messages,
        stream=False,
        reasoning_effort=_resolve("RSI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT),
        tools=TOOLS,
        extra_body={"thinking": {"type": "enabled"}},
    )
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", 0) if usage is not None else 0
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("LLM 响应中没有 choices，无法取得消息（可能是服务端异常）。")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError("LLM 响应的 choices[0].message 为空，无法取得消息。")
    return message, total_tokens
