import json
from pathlib import Path
from uuid import uuid4

from openai import OpenAI

from settings import settings


class MiniAgent:
    def __init__(self, trace_path, allowed_paths):
        self.tools = {
            "read": read,
            "list_files": list_files,
            "pwd": pwd,
        }
        self.run_id = str(uuid4())
        self.trace_path = Path(trace_path)
        self.allowed_paths = [
            Path(path).resolve()
            for path in allowed_paths
        ]

    def call_llm_qwen(self, messages, tools=None):
        client = OpenAI(
            api_key=settings.QWEN_API_KEY,
            base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model="qwen3.8-flash",
            messages=messages,
            tools=tools,
            extra_body={"enable_thinking": True},
        )
        return completion.choices[0].message, completion.usage.total_tokens

    def save(self, messages):
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        serialized_messages = [
            message.model_dump(mode="json")
            if hasattr(message, "model_dump")
            else message
            for message in messages
        ]
        data = {
            "run_id": self.run_id,
            "messages": serialized_messages,
        }
        with self.trace_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def execute_tools(self, name, arguments):
        if name not in self.tools:
            return "工具未注册，无法使用。"
        if name in {"read", "list_files"}:
            allowed_text = "\n".join(
                str(path) for path in self.allowed_paths
            )
            path = arguments.get("path")
            # 请求默认目录时，直接告知本轮允许访问的位置
            if path in (None, "", "."):
                return f"本轮允许读取的目录或文件：\n{allowed_text}"
            target = Path(path).resolve()
            allowed = any(
                target == root
                or (root.is_dir() and root in target.parents)
                for root in self.allowed_paths
            )
            if not allowed:
                return (
                    f"不允许访问：{target}\n"
                    f"本轮允许读取的目录或文件：\n{allowed_text}"
                )
            arguments = {
                **arguments,
                "path": str(target),
            }
        tool_res = self.tools[name](**arguments)
        if len(tool_res) > 200000:
            tool_res = (
                    tool_res[:80000]
                    + "【工具太长被截断】"
                    + tool_res[-80000:]
            )
        return tool_res

    def run(self, system_prompt, task):
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": task
            }
        ]
        for _ in range(200):
            message, token = self.call_llm_qwen(messages, TOOLS)
            messages.append(message)
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    tool_call_id = tool_call.id
                    arguments = tool_call.function.arguments
                    print(name, arguments)
                    try:
                        arguments = json.loads(arguments)
                        if not isinstance(arguments, dict):
                            raise ValueError("工具参数必须是 JSON 对象")
                        tool_res = self.execute_tools(name, arguments)
                    except json.JSONDecodeError as exc:
                        tool_res = f"参数 JSON 解析失败：{exc}，请修正后重新调用。"
                    except Exception as exc:
                        tool_res = (
                            f"工具 {name} 调用失败："
                            f"{type(exc).__name__}: {exc}。"
                            "请检查工具参数定义，补齐必填参数后重新调用。"
                        )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": str(tool_res)
                    })
                self.save(messages)
            else:
                final_answer = message.content
                print(final_answer)
                self.save(messages)
                return final_answer


def read(path, start_line=1, end_line=None):
    """读取文件。行号从1开始，包含结束行；默认读取全部。"""
    try:
        if start_line < 1:
            return "起始行号必须大于等于1"
        if end_line is not None and end_line < start_line:
            return "结束行号不能小于起始行号"
        content = Path(path).read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)
        return "".join(lines[start_line - 1:end_line])
    except Exception as e:
        return str(e)


def list_files(path=".", recursive=False):
    """列出文件和子目录，recursive=True时递归列出。"""
    try:
        directory = Path(path)
        if not directory.is_dir():
            return "目录不存在或路径不是目录"
        if recursive:
            items = directory.rglob("*")
        else:
            items = directory.iterdir()
        results = []
        for item in sorted(items):
            name = item.relative_to(directory)
            if item.is_dir():
                results.append(f"[目录] {name}")
            else:
                results.append(f"[文件] {name}")
        return "\n".join(results) or "目录为空"
    except Exception as e:
        return str(e)


def pwd():
    """返回当前工作目录。"""
    return str(Path.cwd())


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "读取文件内容。行号从1开始，包含结束行。不传行号时读取全部内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "起始行号，默认是1"
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "结束行号，不能小于起始行号；不传则读取到文件末尾"
                    }
                },
                "required": ["path"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出目录下的文件和子目录，可选择是否递归列出所有层。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "目录路径，默认是当前工作目录"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "是否递归列出，默认是false"
                    }
                },
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pwd",
            "description": "返回当前工作目录的绝对路径。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    }
]

if __name__ == "__main__":
    miniagent = MiniAgent()
    ans = miniagent.run("你是一个能用工具的助手。", "帮我看一下rsi.py的内容。")
