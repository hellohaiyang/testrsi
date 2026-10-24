"""LLM 调用封装。

相对当前版本的变化：

* 模型名 / reasoning_effort 可通过 settings 或环境变量覆盖，便于用**不同模型**
  交叉验证 agent 的行为——执行分析里指出“内层 agent 与外层 agent 是同一个模型”
  会让“天花板效应”这类结论无法外推，可配置模型是消除该偏差的最低成本手段。
* 响应结构异常（没有 choices / choices[0].message 为空）时抛出明确异常，
  让 Agent.run() 记为 ``status="failed"`` 并留下可读的 error，
  而不是在循环深处抛 AttributeError。

v4 追加（对应执行分析第六节 F / V5 的“回显无上限”）：

* ``prepare_messages``：按**字符数预算**裁剪上下文（不是 token 精确计数，属启发式，
  在 trace 里如实写成 ``estimated``）。默认不改动任何消息；只有超过软上限时才把
  非保护消息（system 与最后一条除外）的长内容截断，并留下明确的截断标记。
* ``ContextOverflowError``：裁剪后仍超过硬上限时抛出，由 Agent 记为
  ``status="failed"`` + ``error="context_overflow: ..."`` 并照常写轨迹，
  而不是等 API 端报错或在循环深处炸掉（V5 期望的就是“降级为显式失败”）。
"""

import os

from openai import OpenAI

from settings import settings
from .tools import TOOLS

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_REASONING_EFFORT = "high"

# 上下文预算（字符数，启发式；1 token ≈ 2~3 字符中文 / 4 字符英文，这里取保守值）
SOFT_CHAR_LIMIT = 400_000
HARD_CHAR_LIMIT = 700_000
# 触发裁剪后，单条非保护消息允许保留的最大字符数
PER_MESSAGE_CHAR_LIMIT = 20_000


class ContextOverflowError(RuntimeError):
    """裁剪后上下文仍超出硬上限——调用方应记为 status="failed"。"""


def _resolve(name, default):
    value = getattr(settings, name, None)
    if value is None:
        value = os.environ.get(name)
    return value or default


def _resolve_int(name, default):
    value = _resolve(name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _content_text(message):
    """取出消息里参与“体积”计算的文本部分（兼容 dict / pydantic 对象）。"""
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts)
    return ""


def estimate_chars(messages):
    """粗估上下文体积（字符数）。工具调用参数也计入。"""
    total = 0
    for message in messages:
        total += len(_content_text(message))
        tool_calls = (
            message.get("tool_calls")
            if isinstance(message, dict)
            else getattr(message, "tool_calls", None)
        )
        for call in tool_calls or []:
            function = (
                call.get("function")
                if isinstance(call, dict)
                else getattr(call, "function", None)
            )
            if function is None:
                continue
            arguments = (
                function.get("arguments")
                if isinstance(function, dict)
                else getattr(function, "arguments", "")
            )
            total += len(str(arguments or ""))
    return total


def _as_plain(message):
    """把消息转成可安全改写内容、且能被 OpenAI 客户端接受的 dict。"""
    if isinstance(message, dict):
        return dict(message)
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", exclude_none=True)
        except TypeError:
            return dump(exclude_none=True)
    role = getattr(message, "role", None)
    return {"role": role or "assistant", "content": _content_text(message)}


def _truncate_text(text, limit):
    if not isinstance(text, str) or len(text) <= limit:
        return text, False
    head = text[:limit]
    return (
        head
        + f"\n...(因上下文长度预算被截断：原始 {len(text)} 字符，保留前 {limit} 字符；"
        "被截断的部分未验证，不要据其断言)",
        True,
    )


def prepare_messages(
    messages,
    soft_limit=None,
    hard_limit=None,
    per_message_limit=None,
):
    """按字符预算准备本次要送给模型的消息。

    返回 ``(messages_for_call, info)``：

    * 未超过软上限时**原样返回**（默认行为不变，也不做 dict 化，避免任何格式风险）；
    * 超过软上限时，保护首条（system）与最后一条消息，其余消息的长内容截断，
      并在 info 里如实记录体积、截断条数（这些数字都是估算，不是精确 token）；
    * 裁剪后仍超过硬上限 → 抛 ``ContextOverflowError``。
    """
    soft_limit = (
        _resolve_int("RSI_CONTEXT_SOFT_CHARS", SOFT_CHAR_LIMIT)
        if soft_limit is None
        else soft_limit
    )
    hard_limit = (
        _resolve_int("RSI_CONTEXT_HARD_CHARS", HARD_CHAR_LIMIT)
        if hard_limit is None
        else hard_limit
    )
    per_message_limit = (
        _resolve_int("RSI_CONTEXT_PER_MESSAGE_CHARS", PER_MESSAGE_CHAR_LIMIT)
        if per_message_limit is None
        else per_message_limit
    )

    before = estimate_chars(messages)
    info = {
        "estimated_chars_before": before,
        "estimated_chars_after": before,
        "soft_limit_chars": soft_limit,
        "hard_limit_chars": hard_limit,
        "truncated_messages": 0,
        "trimmed": False,
        "estimate_kind": "chars_heuristic",  # 不是精确 token 计数
    }
    if before <= soft_limit:
        return messages, info

    protected = set()
    if messages:
        protected.add(0)
        protected.add(len(messages) - 1)
    prepared = []
    truncated = 0
    for index, message in enumerate(messages):
        plain = _as_plain(message)
        if index not in protected:
            content, was_truncated = _truncate_text(
                plain.get("content"), per_message_limit
            )
            if was_truncated:
                plain["content"] = content
                truncated += 1
            elif isinstance(content, str):
                plain["content"] = content
        prepared.append(plain)

    after = estimate_chars(prepared)
    info.update(
        {
            "estimated_chars_after": after,
            "truncated_messages": truncated,
            "trimmed": True,
        }
    )
    if after > hard_limit:
        raise ContextOverflowError(
            f"上下文体积估算 {after} 字符，超过硬上限 {hard_limit}"
            f"（裁剪前 {before} 字符，已截断 {truncated} 条消息仍不够）。"
            "这不是模型能力问题：环境无法在不丢失关键回显的前提下继续，"
            "请在最终回答中说明“因上下文超限未能完成/未验证”，不要臆造结果。"
        )
    return prepared, info


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
