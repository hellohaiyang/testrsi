import json

from openai import OpenAI
from pathlib import Path

from .llm import call_llm
from .tools import tools, TOOLS, execute_tool
from uuid import uuid4

MAX_STEPS = 200
SYSTEM_PROMPT = """
你是一个可以调用工具完成任务的助手。
"""


class Agent:
    def __init__(self, task, trace_path):
        self.run_id = str(uuid4())
        self.trace_path = Path(trace_path)
        self.status = "created"
        self.error = None
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
        }
        with self.trace_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _run_loop(self):
        for _ in range(MAX_STEPS):
            message, token = call_llm(self.messages)
            self.messages.append(message)
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    tool_call_id = tool_call.id
                    arguments = tool_call.function.arguments
                    try:
                        arguments = json.loads(arguments)
                        if not isinstance(arguments, dict):
                            raise ValueError("工具参数必须是 JSON 对象")
                        tool_res = execute_tool(name, arguments)
                    except json.JSONDecodeError as exc:
                        tool_res = f"参数 JSON 解析失败：{exc}，请修正后重试。"
                    except Exception as exc:
                        tool_res = (
                            f"工具 {name} 调用失败："
                            f"{type(exc).__name__}: {exc}。"
                            "请根据工具定义修正参数后重试。"
                        )
                    print(str(tool_res))
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": str(tool_res)
                    })
            else:
                final_answer = message.content
                print(final_answer)
                self.status = "completed"
                return final_answer
        self.status = "max_steps"

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
