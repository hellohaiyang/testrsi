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


def edit(path, old_text, new_text):
    try:
        path = Path(path)
        content = path.read_text(encoding="utf-8")
        if old_text not in content:
            return "未找到要修改的文本"
        content = content.replace(old_text, new_text, 1)
        path.write_text(content, encoding="utf-8")
        return "修改成功"
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
        "description": "读取文件内容",
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
        "description": "写入文件内容",
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
        "description": "替换文件中的指定文本",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "old_text": {"type": "string", "description": "原文本"},
                "new_text": {"type": "string", "description": "新文本"}
            },
            "required": ["path", "old_text", "new_text"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "执行 Shell 命令",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell 命令"}
            },
            "required": ["command"]
        }
    }
}]


def execute_tool(name, arguments):
    if name not in tools:
        return "工具不存在。"
    res = tools[name](**arguments)
    return res
