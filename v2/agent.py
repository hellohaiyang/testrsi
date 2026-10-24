import json

from pathlib import Path
from uuid import uuid4

from .llm import call_llm
from .tools import TOOLS, execute_tool

MAX_STEPS = 200

POLICY_REJECT_PREFIX = "[约束拒绝]"

SYSTEM_PROMPT = """
你是一个可以调用工具完成任务的助手。

必须遵守的规则：
1. 只依据【工具回显】和你实际读到的内容做断言。凡是无法通过工具验证的结论，必须显式标注为“未验证”，不得臆断或夸大。
2. 工具的确切语义以工具 description 与工具回显为准。默认情况下 edit 只替换【第一处】匹配；如果回显显示本次只替换了 1 处、或文件里还剩匹配项，则不得声称“已替换全部”。
3. 任务对工具的限制（允许使用哪些工具、每个工具最多调用几次）由执行环境强制。被拒绝的调用会以“[约束拒绝]”开头回显，且该调用【没有执行】，不得把它当作已完成。
4. 如果任务要求（例如“替换所有”）与工具能力/调用次数限制冲突，导致无法在一次调用内完成，应当直接说明“只能完成一部分”以及实际完成的数量，而不是宣称任务完成。
5. 最终回答必须与工具回显一致，并明确给出实际替换处数。
"""


class Agent:
    def __init__(
        self,
        task,
        trace_path,
        policy=None,
        eval_context=None,
        max_steps=MAX_STEPS,
    ):
        """
        task:        发给模型的任务 prompt（进入 messages）。
        trace_path:  轨迹保存路径。
        policy:      可选的执行约束，形如
                     {"allowed_tools": ["edit"], "max_calls": {"edit": 1},
                      "max_total_calls": 1}
                     由 harness 强制执行，而不是只写在 prompt 里靠模型自觉。
        eval_context: 评测侧上下文（答案钥匙、评分标准等），
                     只写入轨迹、不进入 messages，避免把答案泄露给模型。
        max_steps:   最大循环步数。
        """
        self.run_id = str(uuid4())
        self.trace_path = Path(trace_path)
        self.status = "created"
        self.error = None
        self.task = task
        self.eval_context = eval_context
        self.max_steps = max_steps
        self.policy = self._normalize_policy(policy)
        self.final_answer = None
        self.stats = {
            "steps": 0,
            "tool_calls": [],
            "per_tool_count": {},
            "rejected": [],
            "total_tokens": 0,
            "final_answer": None,
        }
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": task,
            }
        ]

    @staticmethod
    def _normalize_policy(policy):
        policy = dict(policy or {})
        allowed = policy.get("allowed_tools")
        if allowed is not None:
            allowed = list(allowed)
        max_calls = dict(policy.get("max_calls") or {})
        return {
            "allowed_tools": allowed,
            "max_calls": max_calls,
            "max_total_calls": policy.get("max_total_calls"),
        }

    def _consume_budget(self, name):
        """在校验通过时占用一次调用额度，返回 None；否则返回拒绝原因。"""
        allowed = self.policy.get("allowed_tools")
        if allowed is not None and name not in allowed:
            return (
                f"{POLICY_REJECT_PREFIX} 本任务只允许使用工具 {allowed}，"
                f"你调用了 {name}，该调用【未执行】。请勿声称该操作已完成。"
            )
        limit = self.policy.get("max_calls", {}).get(name)
        used = self.stats["per_tool_count"].get(name, 0)
        if limit is not None and used >= limit:
            return (
                f"{POLICY_REJECT_PREFIX} 工具 {name} 最多调用 {limit} 次，"
                f"额度已用完（已调用 {used} 次），该调用【未执行】。"
                "请基于已有回显如实回答，不要假设后续调用发生了。"
            )
        max_total = self.policy.get("max_total_calls")
        total_used = sum(self.stats["per_tool_count"].values())
        if max_total is not None and total_used >= max_total:
            return (
                f"{POLICY_REJECT_PREFIX} 工具调用总次数上限为 {max_total}，"
                f"额度已用完（已调用 {total_used} 次），该调用【未执行】。"
            )
        self.stats["per_tool_count"][name] = used + 1
        return None

    @staticmethod
    def _short(value, limit=4000):
        text = str(value)
        if len(text) <= limit:
            return text
        return text[:limit] + f"...(已截断，共 {len(text)} 字符)"

    def save(self):
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        messages = [
            message.model_dump(mode="json")
            if hasattr(message, "model_dump")
            else message
            for message in self.messages
        ]
        data = {
            "run_id": self.run_id,
            "status": self.status,
            "error": self.error,
            "messages": messages,
            "final_answer": self.final_answer,
            "task": self.task,
            "policy": self.policy,
            "max_steps": self.max_steps,
            "stats": self.stats,
            "eval_context": self.eval_context,
        }
        with self.trace_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _handle_tool_call(self, tool_call, step):
        name = tool_call.function.name
        tool_call_id = tool_call.id
        raw_arguments = tool_call.function.arguments
        record = {
            "step": step,
            "id": tool_call_id,
            "name": name,
            "arguments": None,
            "rejected": False,
            "ok": False,
            "result": None,
        }
        arguments = None
        try:
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON 对象")
        except (json.JSONDecodeError, TypeError) as exc:
            tool_res = f"参数解析失败：{exc}，请提交合法的 JSON 对象参数后重试。"
            record["result"] = self._short(tool_res)
            self.stats["tool_calls"].append(record)
            return tool_call_id, tool_res

        record["arguments"] = arguments
        rejection = self._consume_budget(name)
        if rejection:
            tool_res = rejection
            record["rejected"] = True
            self.stats["rejected"].append(
                {"step": step, "id": tool_call_id, "name": name,
                 "arguments": arguments, "reason": rejection}
            )
        else:
            try:
                tool_res = execute_tool(name, arguments)
                record["ok"] = True
            except Exception as exc:
                tool_res = (
                    f"工具 {name} 调用失败："
                    f"{type(exc).__name__}: {exc}。"
                    "请根据工具定义修正参数后重试。"
                )
        record["result"] = self._short(tool_res)
        self.stats["tool_calls"].append(record)
        return tool_call_id, tool_res

    def _run_loop(self):
        for step in range(1, self.max_steps + 1):
            self.stats["steps"] = step
            message, token = call_llm(self.messages)
            try:
                self.stats["total_tokens"] += int(token or 0)
            except (TypeError, ValueError):
                pass
            self.messages.append(message)
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    tool_call_id, tool_res = self._handle_tool_call(tool_call, step)
                    print(str(tool_res))
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": str(tool_res)
                    })
            else:
                final_answer = message.content
                print(final_answer)
                self.final_answer = final_answer
                self.stats["final_answer"] = final_answer
                self.status = "completed"
                return final_answer

        # 步数耗尽：显式给出“未收敛”信号，而不是隐式返回 None
        self.status = "max_steps"
        self.final_answer = None
        self.stats["final_answer"] = None
        self.error = f"达到最大步数 {self.max_steps} 仍未给出最终回答"
        return None

    def run(self):
        self.status = "running"
        self.error = None
        try:
            return self._run_loop()
        except Exception as exc:
            self.status = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.save()

# agent = Agent(r"读取这个文件的内容D:/★面试/项目/rsi/new/ceshi.txt")
# agent.run()
