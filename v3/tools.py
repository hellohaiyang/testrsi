"""工具层：文件读写 / 字面量编辑 / shell 执行。

本版针对执行分析里已实测的缺陷做了修复，主要变化：

1. ``edit`` 的回显不再只给一个 ``remaining = total - replaced`` 的算术值，
   而是对**替换后的全文重新计数**（修复“new_text 含 old_text 时谎报仍剩 0 处”）。
2. ``edit`` 的回显给出**每一处匹配的行号 + 该行内容**，而不是只给 ``共 K 处``
   这个无从定位的数字——这是“old_text 命中意料之外子串”时唯一的求救信号。
3. ``edit`` 新增 ``whole_line`` / ``line`` / ``occurrence`` 三个可选参数，
   提供行锚定能力，不必再靠手动拼 ``"web\\nweb-auth"`` 这种脆弱技巧。
4. ``read`` / ``write`` / ``edit`` 改为 ``read_bytes`` / ``write_bytes``，
   不再让 ``Path.read_text(newline=None)`` 静默把 CRLF 归一化成 LF（数据破坏）。
5. 失败回显统一带 ``[参数错误]`` / ``[未执行]`` / ``[未找到]`` / ``[失败]`` 前缀，
   让 harness 能据前缀判定 ``ok`` 与是否计费。
6. ``read`` 回显附带不可见字符提示（CRLF / BOM / 行尾空白 / tab / 末尾无换行）。
"""

import difflib
import subprocess

from pathlib import Path

# ---------------------------------------------------------------------------
# 统一错误前缀：agent 层据前缀判定本次调用是否成功
# ---------------------------------------------------------------------------
ERR_ARG = "[参数错误]"
ERR_NOT_EXEC = "[未执行]"
ERR_NOT_FOUND = "[未找到]"
ERR_FAIL = "[失败]"

ERROR_PREFIXES = (ERR_ARG, ERR_NOT_EXEC, ERR_NOT_FOUND, ERR_FAIL)

# 回显里最多列出多少处匹配 / 多少行改动
MAX_MATCH_ROWS = 20
MAX_DIFF_ROWS = 10

# 会把某个文件路径作为参数并可能改动的工具
PATH_TOOLS = ("read", "write", "edit")


def is_error_result(text):
    """回显是否代表失败（用于 agent 层设置 record["ok"]）。"""
    if not isinstance(text, str):
        return True
    return text.startswith(ERROR_PREFIXES)


# ---------------------------------------------------------------------------
# 字节安全的读写（保留原始换行，不做 newline 归一化）
# ---------------------------------------------------------------------------
def _read_text(path):
    return Path(path).read_bytes().decode("utf-8")


def _write_text(path, text):
    Path(path).write_bytes(text.encode("utf-8"))


def _visible(text, limit=160):
    """把回车/换行/tab/控制字符转成可见转义，便于回显定位。"""
    out = []
    for ch in str(text):
        if ch == "\r":
            out.append("\\r")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 32:
            out.append("\\x%02x" % ord(ch))
        else:
            out.append(ch)
    text = "".join(out)
    if len(text) > limit:
        text = text[:limit] + f"...(该行共 {len(text)} 字符)"
    return text


def _eol_note(text):
    """行尾 / BOM 提示。"""
    notes = []
    if "\r\n" in text:
        notes.append("检测到 CRLF（\\r\\n）行尾，读写均按原字节保留，不会被改成 LF")
    elif "\r" in text:
        notes.append("检测到单独的 CR（\\r）行尾，读写均按原字节保留")
    if text.startswith("\ufeff"):
        notes.append("文件以 BOM（\\ufeff）开头，已原样保留")
    if not notes:
        return ""
    return "\n" + "；".join(notes) + "。"


def _line_spans(text):
    """返回 [(行起始偏移, 行内容结束偏移, 行含行尾符结束偏移)]，保留原始行尾符。"""
    spans = []
    i = 0
    n = len(text)
    start = 0
    while i < n:
        ch = text[i]
        if ch == "\r":
            if i + 1 < n and text[i + 1] == "\n":
                spans.append((start, i, i + 2))
                i += 2
            else:
                spans.append((start, i, i + 1))
                i += 1
            start = i
        elif ch == "\n":
            spans.append((start, i, i + 1))
            i += 1
            start = i
        else:
            i += 1
    if start < n:
        spans.append((start, n, n))
    return spans


def _line_texts(text):
    return [text[s:ce] for s, ce, _ in _line_spans(text)]


def _line_no_of(spans, pos):
    for index, (start, content_end, span_end) in enumerate(spans):
        if start <= pos < span_end:
            return index + 1
    return len(spans) if spans else 1


def _match_positions(text, old_text):
    """非重叠字面量匹配的起始偏移列表（与 str.count 口径一致）。"""
    positions = []
    start = 0
    while True:
        index = text.find(old_text, start)
        if index < 0:
            break
        positions.append(index)
        start = index + len(old_text)
    return positions


def _describe_positions(content, spans, positions, limit=MAX_MATCH_ROWS):
    """把每个匹配渲染成“第 n 行：'该行内容'”，这是模型唯一能定位的手段。"""
    if not positions:
        return "  （无）"
    rows = []
    for index, pos in enumerate(positions[:limit], start=1):
        line_no = _line_no_of(spans, pos)
        start, content_end, _ = spans[line_no - 1]
        rows.append(
            f"  {index}) 第 {line_no} 行：{_visible(content[start:content_end])!r}"
        )
    if len(positions) > limit:
        rows.append(f"  ...还有 {len(positions) - limit} 处未列出")
    return "\n".join(rows)


def _changed_rows(old_content, new_content, limit=MAX_DIFF_ROWS):
    """按行 diff 出“第 n 行：旧 → 新”，替代易被误读的纯数字回显。"""
    old_lines = _line_texts(old_content)
    new_lines = _line_texts(new_content)
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    rows = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for offset in range(i2 - i1):
                rows.append(
                    f"  第 {i1 + offset + 1} 行：{_visible(old_lines[i1 + offset])!r}"
                    f" → {_visible(new_lines[j1 + offset])!r}"
                )
        elif tag == "replace":
            rows.append(
                f"  第 {i1 + 1}-{i2} 行（{i2 - i1} 行）被替换为"
                f" {j2 - j1} 行：{_visible(chr(10).join(new_lines[j1:j2]))!r}"
            )
        elif tag == "delete":
            rows.append(
                f"  第 {i1 + 1}-{i2} 行被删除：{_visible(chr(10).join(old_lines[i1:i2]))!r}"
            )
        else:  # insert
            rows.append(
                f"  在第 {i1} 行之后插入 {j2 - j1} 行："
                f"{_visible(chr(10).join(new_lines[j1:j2]))!r}"
            )
    if len(rows) > limit:
        extra = len(rows) - limit
        rows = rows[:limit] + [f"  ...还有 {extra} 行改动未列出"]
    return rows


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------
def read(path):
    try:
        target = Path(path)
        if not target.exists():
            return f"{ERR_NOT_EXEC} 文件不存在：{target}。"
        if not target.is_file():
            return f"{ERR_NOT_EXEC} 不是普通文件：{target}。"
        try:
            content = _read_text(target)
        except UnicodeDecodeError as exc:
            return f"{ERR_FAIL} 文件不是合法 UTF-8，无法以文本方式读取：{exc}"
        lines = _line_spans(content)
        notes = []
        if content.startswith("\ufeff"):
            notes.append("文件以 BOM（\\ufeff）开头")
        if "\r\n" in content:
            notes.append("行尾为 CRLF（\\r\\n）——回显里看不见 \\r，用 \"\\n\" 拼接的跨行 old_text 会匹配失败")
        elif "\r" in content:
            notes.append("行尾含单独的 CR（\\r）")
        if content and not content.endswith(("\n", "\r")):
            notes.append("文件末尾没有换行符")
        notes.append(f"共 {len(lines)} 行、{len(content)} 字符")
        trailing = [
            str(index + 1)
            for index, (start, content_end, _) in enumerate(lines)
            if content[start:content_end] != content[start:content_end].rstrip()
        ]
        if trailing:
            notes.append("以下行含行尾空白：第 " + "、".join(trailing[:10]) + " 行")
        tabs = [
            str(index + 1)
            for index, (start, content_end, _) in enumerate(lines)
            if "\t" in content[start:content_end]
        ]
        if tabs:
            notes.append("以下行含 tab：第 " + "、".join(tabs[:10]) + " 行")
        footer = (
            "\n--- 读取提示（不属于文件内容，请勿复制进 old_text / new_text）---\n"
            + "\n".join("- " + note for note in notes)
        )
        return content + footer
    except Exception as exc:  # 保留原有“不抛异常”的风格，但带回显前缀
        return f"{ERR_FAIL} 读取失败：{type(exc).__name__}: {exc}"


def write(path, content):
    try:
        if not isinstance(content, str):
            return f"{ERR_ARG} content 必须是字符串（实际为 {type(content).__name__}，未改动文件）。"
        target = Path(path)
        existed = target.exists()
        before = None
        if existed and target.is_file():
            try:
                before = _read_text(target)
            except Exception:
                before = None
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_text(target, content)
        message = (
            f"写入成功：{'覆盖' if existed else '新建'}文件 {target}，"
            f"写入 {len(content)} 字符、{len(_line_spans(content))} 行。"
        )
        if "\r\n" in content:
            message += "（内容含 CRLF，已按原字节写入，未做换行归一化）"
        if before is not None and before != content:
            rows = _changed_rows(before, content)
            if rows:
                message += "\n改动明细（相对于覆盖前）：\n" + "\n".join(rows)
        return message
    except Exception as exc:
        return f"{ERR_FAIL} 写入失败：{type(exc).__name__}: {exc}"


def _edit_warnings(total, replaced, remaining, first_line, old_text, new_text, whole_line=False):
    warnings = []
    if whole_line and total > replaced:
        warnings.append(
            f"说明：old_text 作为字面子串在文件中出现 {total} 次，"
            f"但整行内容恰好等于 old_text 的只有一部分；本次按整行模式只替换了 {replaced} 处"
            "（子串式的误伤已被排除）。"
        )
    elif total > replaced:
        warnings.append(
            f"警告：old_text 在文件中作为字面子串共出现 {total} 次，"
            f"本次只替换了其中 {replaced} 处（第一处在第 {first_line} 行，见上方匹配列表）。"
            "如果你预期只应有 1 处匹配，说明 old_text 命中了意料之外的子串或位置"
            "（例如目标是整行 `web`，而 `web` 同时是 `webhook` / `web-cache` 的前缀），"
            "必须立即 read 复核，并改用唯一上下文 / whole_line=true / line=n 后重试；"
            "在复核之前不得断言任务完成。"
        )
    elif total > 1 and replaced == total:
        warnings.append(
            f"注意：本次按 replace_all 一次性替换了 {replaced} 处。"
            f"请核对你的预期处数是否也是 {replaced} 处；"
            "若你只需要改其中一处，说明 old_text 命中了意料之外的子串。"
        )
    if remaining > 0 and isinstance(old_text, str) and isinstance(new_text, str) and old_text in new_text:
        warnings.append(
            f"说明：对替换后的全文重新计数 old_text 仍为 {remaining} 处，"
            "是因为 new_text 本身也包含 old_text（字面量计数会把新写入的内容算进去），"
            "不代表还有未替换的原始匹配。"
        )
    elif remaining > 0:
        warnings.append(
            f"注意：对替换后的全文重新计数，old_text 仍剩 {remaining} 处未替换。"
        )
    return warnings


def edit(path, old_text, new_text, replace_all=False, occurrence=None, line=None, whole_line=False):
    """按字面量替换文件文本（子串替换，无行锚定、无词边界、无正则）。

    参数：
      old_text:     被替换的字面量。
      new_text:     替换后的文本。
      replace_all:  默认 False，只替换【第一处】匹配；True 替换全部匹配。
      whole_line:   True 表示只匹配“整行内容恰好等于 old_text”的行（消除子串误伤）。
      line:         1 起始行号，只在该行内操作。
      occurrence:   1 起始序号，只替换第 n 处字面匹配。

    回显包含：本次替换处数、替换前字面匹配总数、**替换后对全文重新计数**的剩余数、
    每一处匹配的行号与行内容、改动明细（第 n 行：旧 → 新）。
    """
    try:
        target = Path(path)
        if not isinstance(old_text, str) or old_text == "":
            return f"{ERR_ARG} old_text 不能为空（未改动文件）。"
        if not isinstance(new_text, str):
            return f"{ERR_ARG} new_text 必须是字符串（未改动文件）。"
        if not isinstance(replace_all, bool):
            return (
                f"{ERR_ARG} replace_all 必须是布尔值 true/false，"
                f"实际为 {type(replace_all).__name__}：{replace_all!r}（未改动文件）。"
            )
        if line is not None and (isinstance(line, bool) or not isinstance(line, int)):
            return f"{ERR_ARG} line 必须是整数，实际为 {line!r}（未改动文件）。"
        if occurrence is not None and (isinstance(occurrence, bool) or not isinstance(occurrence, int)):
            return f"{ERR_ARG} occurrence 必须是整数，实际为 {occurrence!r}（未改动文件）。"
        if not isinstance(whole_line, bool):
            return f"{ERR_ARG} whole_line 必须是布尔值（未改动文件）。"
        if not target.exists():
            return f"{ERR_NOT_EXEC} 文件不存在：{target}（未改动文件）。"
        if not target.is_file():
            return f"{ERR_NOT_EXEC} 不是普通文件：{target}（未改动文件）。"
        try:
            content = _read_text(target)
        except UnicodeDecodeError as exc:
            return f"{ERR_FAIL} 文件不是合法 UTF-8，无法编辑：{exc}（未改动文件）。"

        spans = _line_spans(content)
        line_texts = [content[start:content_end] for start, content_end, _ in spans]
        positions = _match_positions(content, old_text)
        total = len(positions)
        eol_note = _eol_note(content)

        if whole_line:
            # ---- 整行模式：只匹配整行内容等于 old_text 的行 ----
            if occurrence is not None:
                return (
                    f"{ERR_ARG} whole_line=true 不能与 occurrence 同时使用，"
                    "请改用 line 或 replace_all（未改动文件）。"
                )
            if replace_all and line is not None:
                return (
                    f"{ERR_ARG} whole_line=true 时 replace_all 与 line 不能同时使用"
                    "（未改动文件）。"
                )
            candidates = list(range(len(line_texts)))
            if not line_texts:
                return (
                    f"{ERR_NOT_FOUND} 文件为空（0 行），没有整行内容恰好等于 "
                    f"{old_text!r} 的行（未改动文件）。"
                )
            if line is not None:
                if line < 1 or line > max(len(line_texts), 1):
                    return (
                        f"{ERR_NOT_EXEC} 行号 {line} 超出范围"
                        f"（文件共 {len(line_texts)} 行，未改动文件）。"
                    )
                candidates = [line - 1]
            exact = [index for index in candidates if line_texts[index] == old_text]
            if not exact:
                scope = f"第 {line} 行" if line is not None else "文件"
                return (
                    f"{ERR_NOT_FOUND} {scope}没有整行内容恰好等于 {old_text!r} 的行"
                    f"（文件共 {len(line_texts)} 行）。\n"
                    f"作为字面子串，old_text 在文件中出现 {total} 处，位置如下：\n"
                    f"{_describe_positions(content, spans, positions)}\n"
                    f"（未改动文件）{eol_note}"
                )
            selected = exact if replace_all else exact[:1]
            selected_set = set(selected)
            parts = []
            for index, (start, content_end, span_end) in enumerate(spans):
                parts.append(new_text if index in selected_set else content[start:content_end])
                parts.append(content[content_end:span_end])
            new_content = "".join(parts)
            _write_text(target, new_content)
            replaced = len(selected)
            remaining = new_content.count(old_text)
            remaining_whole = sum(1 for text in _line_texts(new_content) if text == old_text)
            rows = _changed_rows(content, new_content)
            header = (
                f"修改成功（whole_line 整行模式）：本次替换 {replaced} 处整行；"
                f"替换前字面匹配 {total} 处、整行匹配 {len(exact)} 行；"
                f"替换后对全文重新计数，字面仍剩 {remaining} 处、整行仍剩 {remaining_whole} 行。"
            )
            if not replace_all and len(exact) > 1:
                header += (
                    f"\n警告：整行内容等于 old_text 的行共有 {len(exact)} 行"
                    f"（第 {'、'.join(str(i + 1) for i in exact[:10])} 行），"
                    f"但 replace_all=false，只替换了第 {selected[0] + 1} 行。"
                )
            body = (
                f"\n匹配位置（替换前，字面子串，共 {total} 处）：\n"
                f"{_describe_positions(content, spans, positions)}"
                f"\n改动明细：\n" + ("\n".join(rows) if rows else "  （无）")
            )
            warnings = _edit_warnings(
                total, replaced, remaining, selected[0] + 1, old_text, new_text,
                whole_line=True,
            )
            tail = ("\n" + "\n".join(warnings)) if warnings else ""
            return header + body + tail + eol_note

        # ---- 子串模式 ----
        if total == 0:
            extra = ""
            if "\r\n" in content:
                extra = (
                    "\n提示：该文件使用 CRLF（\\r\\n）行尾，回显里看不见 \\r；"
                    "如果你用 \"\\n\" 拼接跨行 old_text，将无法匹配。"
                )
            return (
                f"{ERR_NOT_FOUND} 未找到要修改的文本（old_text={old_text!r}，"
                f"0 处匹配，文件共 {len(line_texts)} 行，未改动文件）。{extra}"
            )
        if replace_all and (line is not None or occurrence is not None):
            return (
                f"{ERR_ARG} replace_all=true 不能与 line / occurrence 同时使用，"
                "请改用其中一种定位方式（未改动文件）。"
            )

        first_line = _line_no_of(spans, positions[0])
        if line is not None:
            if line < 1 or line > max(len(line_texts), 1):
                return (
                    f"{ERR_NOT_EXEC} 行号 {line} 超出范围"
                    f"（文件共 {len(line_texts)} 行，未改动文件）。"
                )
            selected = [
                index
                for index, pos in enumerate(positions)
                if _line_no_of(spans, pos) == line
            ]
            if not selected:
                return (
                    f"{ERR_NOT_EXEC} old_text={old_text!r} 未出现在第 {line} 行"
                    f"（该行内容：{_visible(line_texts[line - 1])!r}）；"
                    f"old_text 在文件中共 {total} 处，位置如下：\n"
                    f"{_describe_positions(content, spans, positions)}（未改动文件）"
                )
            if not replace_all:
                selected = selected[:1]
            first_line = _line_no_of(spans, positions[selected[0]])
        elif occurrence is not None:
            if occurrence < 1 or occurrence > total:
                return (
                    f"{ERR_NOT_EXEC} occurrence={occurrence} 超出范围"
                    f"（文件中共 {total} 处匹配，未改动文件）。\n"
                    f"匹配位置：\n{_describe_positions(content, spans, positions)}"
                )
            selected = [occurrence - 1]
            first_line = _line_no_of(spans, positions[selected[0]])
        elif replace_all:
            selected = list(range(total))
        else:
            selected = [0]

        replaced = len(selected)
        new_content = content
        for index in sorted(selected, reverse=True):
            pos = positions[index]
            new_content = (
                new_content[:pos] + new_text + new_content[pos + len(old_text):]
            )
        _write_text(target, new_content)
        # 关键修复：remaining 不是 total - replaced 的算术值，而是对全文重新计数
        remaining = new_content.count(old_text)
        rows = _changed_rows(content, new_content)
        header = (
            f"修改成功：本次替换 {replaced} 处；"
            f"替换前文件中共有 {total} 处匹配，"
            f"替换后对全文重新计数仍剩 {remaining} 处。"
        )
        body = (
            f"\n匹配位置（替换前，字面子串，共 {total} 处）：\n"
            f"{_describe_positions(content, spans, positions)}"
            f"\n改动明细：\n" + ("\n".join(rows) if rows else "  （无）")
        )
        warnings = _edit_warnings(total, replaced, remaining, first_line, old_text, new_text)
        tail = ("\n" + "\n".join(warnings)) if warnings else ""
        return header + body + tail + eol_note
    except Exception as exc:
        return f"{ERR_FAIL} 编辑失败：{type(exc).__name__}: {exc}（文件可能已部分改动，请 read 复核）。"


def bash(command):
    try:
        res = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120
        )
        output = (res.stdout or "") + (res.stderr or "")
        if res.returncode != 0:
            output += f"\n[退出码 {res.returncode}]"
        return output
    except Exception as exc:
        return f"{ERR_FAIL} 命令执行失败：{type(exc).__name__}: {exc}"


tools = {
    "bash": bash,
    "read": read,
    "write": write,
    "edit": edit,
}

TOOL_NAMES = set(tools)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read",
        "description": (
            "读取文件内容，返回文件全文。"
            "文件内容之后会附一段“读取提示”（不属于文件内容），"
            "其中报告行数、字符数、是否 CRLF/BOM、哪些行含行尾空白或 tab。"
            "读到的内容与文件字节一致，不会做换行归一化。"
        ),
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
        "description": "把 content 完整写入 path（覆盖已有内容，父目录不存在时自动创建；按原字节写入，不做换行归一化）。",
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
            "注意：匹配方式是【字面子串】，没有行锚定、没有词边界、没有正则——"
            "old_text 可能命中意料之外的其它文本（例如目标是整行 `web`，但 `web` 同时是 `webhook` 的前缀）。\n"
            "默认只替换【第一处】匹配；replace_all=true 才替换全部匹配。"
            "要精确改某一整行，用 whole_line=true（只匹配整行内容恰好等于 old_text 的行）；"
            "也可用 line=n（只在该行内替换）或 occurrence=n（只替换第 n 处字面匹配）。\n"
            "返回值包含：本次实际替换处数、替换前字面匹配总数、**替换后对全文重新计数**的剩余匹配数、"
            "每一处匹配的行号与行内容、以及逐行改动明细。"
            "请核对“替换前共有 K 处”是否与你预期的处数一致：不一致说明命中了意外子串，"
            "必须 read 复核后再断言，不要假设一次 edit 已经完成了任务。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "old_text": {"type": "string", "description": "被替换的原文（字面量，不支持正则）"},
                "new_text": {"type": "string", "description": "替换后的新文本"},
                "replace_all": {
                    "type": "boolean",
                    "description": "默认 false，只替换第一处；true 表示替换全部匹配（不能与 line / occurrence 同用）"
                },
                "whole_line": {
                    "type": "boolean",
                    "description": "默认 false；true 表示只匹配“整行内容恰好等于 old_text”的行，用于消除子串误伤"
                },
                "line": {
                    "type": "integer",
                    "description": "1 起始的行号，只在这一行内替换"
                },
                "occurrence": {
                    "type": "integer",
                    "description": "1 起始的序号，只替换第 n 处字面匹配"
                }
            },
            "required": ["path", "old_text", "new_text"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "执行 Shell 命令，返回 stdout 与 stderr 的拼接结果（非 0 退出码会额外标注）。"
            "命令由 /bin/sh 执行（不是 bash），不支持 bash 特有语法"
            "（如 ${var:0:1} 子串切片、[[ ]]、数组）；需要时请改用 POSIX 写法或显式调用 bash -c。"
        ),
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
        if value is None:
            continue  # 可选参数显式传 null 视同未传
        expected = properties[key].get("type")
        if expected == "string" and not isinstance(value, str):
            return (
                f"工具 {name} 的参数 {key} 应为字符串，"
                f"实际为 {type(value).__name__}：{value!r}。"
            )
        if expected == "boolean" and not isinstance(value, bool):
            return (
                f"工具 {name} 的参数 {key} 应为布尔值 true/false（JSON 布尔），"
                f"不要传字符串 \"true\"/\"false\"；实际为 {type(value).__name__}：{value!r}。"
            )
        if expected == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            return (
                f"工具 {name} 的参数 {key} 应为整数，"
                f"实际为 {type(value).__name__}：{value!r}。"
            )
        if expected == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            return (
                f"工具 {name} 的参数 {key} 应为数字，"
                f"实际为 {type(value).__name__}：{value!r}。"
            )
    return None


def execute_tool_detailed(name, arguments):
    """执行工具并返回 (ok, 回显文本)。

    ok 由回显内容决定（是否带错误前缀），而不是“有没有抛异常”——
    旧版把“参数非法 / 校验失败 / 工具不存在”都当成 ok=True，污染了轨迹统计。
    """
    if name not in tools:
        return False, f"{ERR_ARG} 工具不存在：{name}；可用工具为 {sorted(tools)}。"
    error = validate_arguments(name, arguments)
    if error:
        return False, f"{ERR_ARG} {error}"
    try:
        res = tools[name](**arguments)
    except TypeError as exc:
        return False, f"{ERR_ARG} 参数与工具签名不匹配：{type(exc).__name__}: {exc}"
    except Exception as exc:
        return False, f"{ERR_FAIL} 工具 {name} 执行异常：{type(exc).__name__}: {exc}"
    if not isinstance(res, str):
        res = str(res)
    return (not is_error_result(res)), res


def execute_tool(name, arguments):
    """兼容旧接口：只返回回显文本。"""
    return execute_tool_detailed(name, arguments)[1]
