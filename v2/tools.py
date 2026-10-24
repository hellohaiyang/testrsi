import subprocess

from pathlib import Path


def read(path):
    try:
        path = Path(path)
        res = path.read_text(encoding="utf-8")
        return res
    except Exception as e:
        return str(e)


def write(path, content):
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return "写入成功"
    except Exception as e:
        return str(e)


def edit(path, old_text, new_text, replace_all=False):
    """按字面量替换文件文本。

    默认只替换【第一处】匹配；需要替换全部必须显式传 replace_all=True。
    返回值包含：本次实际替换数、原有匹配总数、替换后剩余匹配数。
    这三项是让调用方在无法 read 的情况下也能自我校正的关键信息。
    """
    try:
        path = Path(path)
        if not isinstance(old_text, str) or old_text == "":
            return "修改失败：old_text 不能为空（未改动文件）。"
        content = path.read_text(encoding="utf-8")
        total = content.count(old_text)
        if total == 0:
            return "未找到要修改的文本（0 处匹配，文件未改动）。"
        if replace_all:
            replaced = total
            new_content = content.replace(old_text, new_text)
        else:
            replaced = 1
            new_content = content.replace(old_text, new_text, 1)
        path.write_text(new_content, encoding="utf-8")
        remaining = total - replaced
        return (
            f"修改成功：本次替换 {replaced} 处；"
            f"替换前文件中共有 {total} 处匹配，替换后仍剩 {remaining} 处。"
        )
    except Exception as e:
        return str(e)


def bash(command):
    try:
        res = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120
        )
        return res.stdout + res.stderr
    except Exception as e:
        return str(e)


tools = {
    "bash": bash,
    "read": read,
    "write": write,
    "edit": edit,
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read",
        "description": "读取文件内容，返回文件全文（不带行号）。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"}
            },
            "required": ["path"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "write",
        "description": "把 content 完整写入 path（覆盖已有内容，父目录不存在时自动创建）。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "写入内容"}
            },
            "required": ["path", "content"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "edit",
        "description": (
            "按字面量（非正则）替换文件中的文本。"
            "默认只替换【第一处】匹配，需要替换全部匹配时必须显式传 replace_all=true。"
            "返回值会告诉本次实际替换了几处、文件中原有几处匹配、替换后还剩几处，"
            "请以返回值为准，不要假设一次 edit 已经替换了全部。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "old_text": {"type": "string", "description": "被替换的原文（字面量，不支持正则）"},
                "new_text": {"type": "string", "description": "替换后的新文本"},
                "replace_all": {
                    "type": "boolean",
                    "description": "默认 false，只替换第一处；true 表示替换全部匹配"
                }
            },
            "required": ["path", "old_text", "new_text"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "执行 Shell 命令，返回 stdout 与 stderr 的拼接结果。",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell 命令"}
            },
            "required": ["command"]
        }
    }
}]

_SCHEMAS = {
    item["function"]["name"]: item["function"].get("parameters", {})
    for item in TOOLS
}


def validate_arguments(name, arguments):
    """按 TOOLS 里的 schema 做显式参数校验，返回错误信息或 None。"""
    if not isinstance(arguments, dict):
        return "工具参数必须是 JSON 对象。"
    params = _SCHEMAS.get(name, {})
    properties = params.get("properties", {})
    required = params.get("required", [])
    missing = [key for key in required if key not in arguments]
    if missing:
        return f"工具 {name} 缺少必需参数：{missing}；需要 {required}。"
    unknown = [key for key in arguments if key not in properties]
    if unknown:
        return (
            f"工具 {name} 不支持参数：{unknown}；"
            f"支持的参数为 {sorted(properties)}。"
        )
    for key, value in arguments.items():
        expected = properties[key].get("type")
        if expected == "string" and not isinstance(value, str):
            return f"工具 {name} 的参数 {key} 应为字符串。"
        if expected == "boolean" and not isinstance(value, bool):
            return f"工具 {name} 的参数 {key} 应为布尔值。"
    return None


def execute_tool(name, arguments):
    if name not in tools:
        return (
            f"工具不存在：{name}；"
            f"可用工具为 {sorted(tools)}。"
        )
    error = validate_arguments(name, arguments)
    if error:
        return error
    res = tools[name](**arguments)
    return res
