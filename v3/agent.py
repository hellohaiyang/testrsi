"""Agent 主循环。

相对当前版本（v2）的改动（对应执行分析里的缺口）：

* 先校验、后扣额度：参数错误（含类型写错，如 "replace_all": "true"）不再静默烧掉
  唯一一次 edit 额度；回显明确写出“本次未执行、不消耗额度”。
* ``record["ok"]`` 反映真实结果（由回显前缀 + 约束是否触发决定），不再“只要没抛异常就 True”。
* 工具回显尾部附带额度状态，让模型能正确归因“为什么下一次调用被拒绝”。
* system prompt 拆成“通用规则 + edit 专项规则”，并按任务内容决定是否注入；
  新增两条关键规则：(a) 回显“共 K 处匹配”必须与预期处数一致，不一致立即 read 复核；
  (b) 任何数量型结论必须与实际验证次数一致，外推必须标注“未验证”。
  同时修正 v2 规则 2 的反噬：剩余匹配数 > 0 不等于“还没做完”。
* 轨迹里新增 ``stats["file_snapshots"]`` / 每条工具调用的 before/after sha256，
  使轨迹在事后可重判（旧版 trace 不存文件快照，无法复算终态）。
"""

import hashlib
import json

from pathlib import Path
from uuid import uuid4

from .llm import call_llm
from .tools import (
    ERR_ARG,
    PATH_TOOLS,
    TOOL_NAMES,
    TOOLS,
    execute_tool_detailed,
    validate_arguments,
)

MAX_STEPS = 200

POLICY_REJECT_PREFIX = "[约束拒绝]"

# stats 里单条回显的最大保留长度（超出会带明确标记，不用静默截断骗人）
STATS_RESULT_LIMIT = 100_000
# 文件快照里保留全文的最大长度
SNAPSHOT_TEXT_LIMIT = 20_000

GENERAL_RULES = [
    "只依据【工具回显】和你实际读到的内容做断言。凡是无法通过工具验证的结论，"
    "必须显式标注为“未验证”，不得臆断或夸大。",
    "工具的确切语义以工具 description 与工具回显为准。不要在回显之外假设文件长什么样；"
    "回显说失败就不要说成功。",
    "任务对工具的限制（允许使用哪些工具、每个工具最多调用几次）由执行环境强制。"
    "被拒绝的调用会以“[约束拒绝]”开头回显，且该调用【没有执行】，不得把它当作已完成；"
    "每次工具回显末尾都会附带当前额度状态，请以它为准。",
    "以“[参数错误]”开头的调用【没有执行】，且【不消耗】调用额度；"
    "修正参数后可以重试。以“[失败]”“[未执行]”“[未找到]”开头的调用也没有达成目标，"
    "同样不得声称成功。",
    "任何数量型结论（例如“共 N 次运行”“N/N 通过”“替换了 N 处”）"
    "必须与实际执行的验证次数 / 工具回显数字一致；没有逐个验证的必须写成“外推 / 未验证”，"
    "不得把外推说成实测。",
    "一旦你发现自己的某次测量/操作的前提已经失效（例如当前文件内容已不是那次操作的终态），"
    "必须重做测量，而不是补一句解释。",
    "若你已经造成了不可逆的改动，且额度用尽或工具被禁用导致无法回滚，"
    "必须明确写出“我破坏了哪几行、目标是否达成、当前无法回滚”，不得用“已完成”掩盖。",
]

EDIT_RULES = [
    "edit 是【字面量子串】替换：没有行锚定、没有词边界、没有正则、没有“第 n 处”之外的精确定位。"
    "默认只替换【第一处】匹配；replace_all=true 才会替换全部匹配。",
    "核对回显里的“替换前文件中共有 K 处匹配”：K 必须与你预期的处数一致。"
    "若 K 大于你预期的处数，说明 old_text 命中了意料之外的子串或位置"
    "（典型情况：目标是一整行 `web`，但 `web` 同时是 `webhook` / `web-cache` 的前缀）。"
    "此时必须立即 read 复核，并改用唯一上下文（把相邻字符/整行包含进 old_text）"
    "或 whole_line=true / line=n / occurrence=n 重试；复核之前不得断言完成。",
    "如果回显把“共 4 处匹配”和你心里预期的“1 处”摆在一起，这个差值本身就是唯一的告警信号。"
    "不要只看“本次替换了 1 处”就认为做对了；要看“命中的是哪一行”（回显会列出每处匹配的行号与行内容）。",
    "不要仅因为回显里还有剩余匹配就宣称“还没做完”。如果任务语义本来就只需要改一处"
    "（其余匹配是子串或本来就该保留），“仍剩 K-1 处”是正常结果。"
    "此时正确说法是“我只改了第 X 行，其余匹配是有意保留的”，"
    "而不是“还有 3 处没替换完”。但也不要因此声称“已替换全部”——除非回显确实如此。",
    "如果任务要求（例如“替换所有”，或“只改某一整行”）与工具能力 / 调用次数限制冲突，"
    "导致无法在一次调用内完成、或无法无副作用地完成，应当直接说明“只能完成一部分 / 无法完成”"
    "以及实际完成的数量与已造成的破坏，而不是宣称任务完成。",
    "最终回答必须与工具回显一致，并给出回显里实际替换处数（不要自己重算），"
    "同时说明改动落在哪一行、其它行是否被改动。",
]

EDIT_KEYWORDS = ("edit", "替换", "修改", "改写", "改掉", "改为", "改成")

SYSTEM_PROMPT = (
    "你是一个可以调用工具完成任务的助手。\n\n必须遵守的规则：\n"
    + "\n".join(f"{i}. {rule}" for i, rule in enumerate(GENERAL_RULES, 1))
    + "\n"
)


def _needs_edit_rules(task, policy):
    """只有任务涉及文本替换、或策略里显式放行 edit 时，才注入 edit 专项规则。

    这样元任务（例如“去评测另一个 agent”）不会被 edit 专用规则污染。
    """
    text = str(task or "").lower()
    if any(keyword in text for keyword in EDIT_KEYWORDS):
        return True
    allowed = (policy or {}).get("allowed_tools")
    return bool(allowed is not None and "edit" in allowed)


def build_system_prompt(task, policy=None):
    rules = list(GENERAL_RULES)
    if _needs_edit_rules(task, policy):
        rules.extend(EDIT_RULES)
    body = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1))
    return f"你是一个可以调用工具完成任务的助手。\n\n必须遵守的规则：\n{body}\n"


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
                      "max_total_calls": 1, "watch_paths": ["/tmp/x.txt"]}
                     由 harness 强制执行，而不是只写在 prompt 里靠模型自觉。
                     watch_paths 可选：额外指定需要做前后快照的文件
                     （即使模型从未 read/edit 它们）。
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
            "not_executed": [],
            "total_tokens": 0,
            "final_answer": None,
            "file_snapshots": {},
            "budget": {},
        }
        self.messages = [
            {
                "role": "system",
                "content": build_system_prompt(task, self.policy),
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
        watch_paths = policy.get("watch_paths") or []
        if isinstance(watch_paths, str):
            watch_paths = [watch_paths]
        return {
            "allowed_tools": allowed,
            "max_calls": max_calls,
            "max_total_calls": policy.get("max_total_calls"),
            "watch_paths": [str(item) for item in watch_paths],
        }

    # ------------------------------------------------------------------
    # 额度
    # ------------------------------------------------------------------
    def _budget_text(self):
        """当前额度状态（没有配置任何限制时返回空串）。"""
        parts = []
        for name, limit in (self.policy.get("max_calls") or {}).items():
            used = self.stats["per_tool_count"].get(name, 0)
            remaining = max(int(limit) - used, 0) if isinstance(limit, int) else "?"
            parts.append(f"{name} 已用 {used}/{limit}（剩余 {remaining}）")
        max_total = self.policy.get("max_total_calls")
        if max_total is not None:
            total_used = sum(self.stats["per_tool_count"].values())
            parts.append(f"总调用 已用 {total_used}/{max_total}")
        return "；".join(parts)

    def _append_budget(self, text):
        budget = self._budget_text()
        if not budget:
            return text
        return f"{text}\n[额度] {budget}"

    def _allowed_rejection(self, name):
        """仅检查 allowed_tools，不消耗额度。"""
        allowed = self.policy.get("allowed_tools")
        if allowed is not None and name not in allowed:
            return (
                f"{POLICY_REJECT_PREFIX} 本任务只允许使用工具 {allowed}，"
                f"你调用了 {name}，该调用【未执行】，也没有消耗任何额度。"
                "请勿声称该操作已完成。"
            )
        return None

    def _consume_budget(self, name):
        """在【参数校验通过之后】占用一次调用额度，返回 None；否则返回拒绝原因。"""
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
                "请基于已有回显如实回答，不要假设后续调用发生了；"
                "如果上一次调用已经造成了破坏，请明确说明当前无法回滚。"
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

    # ------------------------------------------------------------------
    # 参数校验（先于扣额度）
    # ------------------------------------------------------------------
    def _validate_call(self, name, arguments):
        if name not in TOOL_NAMES:
            return (
                f"{ERR_ARG} 工具不存在：{name}；可用工具为 {sorted(TOOL_NAMES)}。"
                "本次调用【未执行】，也未消耗额度，请改用可用工具。"
            )
        error = validate_arguments(name, arguments)
        if error:
            return (
                f"{ERR_ARG} {error} 本次调用【未执行】，且【不消耗】调用额度；"
                "请修正参数后重试。"
            )
        return None

    # ------------------------------------------------------------------
    # 文件快照（让轨迹事后可重判）
    # ------------------------------------------------------------------
    @staticmethod
    def _cap_text(raw, limit=SNAPSHOT_TEXT_LIMIT):
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            return None
        if len(text) > limit:
            return text[:limit] + f"...(已截断，共 {len(text)} 字符)"
        return text

    def _track_path(self, path):
        key = str(path)
        snapshots = self.stats["file_snapshots"]
        if key in snapshots:
            return snapshots[key]
        entry = {
            "path": key,
            "existed_before": None,
            "initial_sha256": None,
            "initial_content": None,
            "final_sha256": None,
            "final_content": None,
            "final_captured": False,
            "changed": None,
        }
        try:
            target = Path(path)
            entry["existed_before"] = target.exists()
            if target.is_file():
                raw = target.read_bytes()
                entry["initial_sha256"] = hashlib.sha256(raw).hexdigest()
                entry["initial_content"] = self._cap_text(raw)
        except Exception as exc:
            entry["read_error"] = f"{type(exc).__name__}: {exc}"
        snapshots[key] = entry
        return entry

    @staticmethod
    def _digest(path):
        try:
            target = Path(path)
            if not target.is_file():
                return None
            return hashlib.sha256(target.read_bytes()).hexdigest()
        except Exception:
            return None

    def _finalize_snapshots(self):
        for key, entry in self.stats.get("file_snapshots", {}).items():
            try:
                target = Path(key)
                if target.is_file():
                    raw = target.read_bytes()
                    entry["final_sha256"] = hashlib.sha256(raw).hexdigest()
                    entry["final_content"] = self._cap_text(raw)
                entry["final_captured"] = True
                entry["changed"] = entry.get("initial_sha256") != entry.get("final_sha256")
            except Exception as exc:
                entry["final_error"] = f"{type(exc).__name__}: {exc}"

    def _finalize_stats(self):
        for path in self.policy.get("watch_paths") or []:
            self._track_path(path)
        self._finalize_snapshots()
        used = dict(self.stats["per_tool_count"])
        limits = dict(self.policy.get("max_calls") or {})
        remaining = {
            name: (int(limit) - used.get(name, 0))
            for name, limit in limits.items()
            if isinstance(limit, int)
        }
        self.stats["budget"] = {
            "allowed_tools": self.policy.get("allowed_tools"),
            "max_calls": limits,
            "max_total_calls": self.policy.get("max_total_calls"),
            "used": used,
            "remaining": remaining,
            "total_used": sum(used.values()),
        }

    # ------------------------------------------------------------------
    # 轨迹
    # ------------------------------------------------------------------
    @staticmethod
    def _short(value, limit=STATS_RESULT_LIMIT):
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

    # ------------------------------------------------------------------
    # 单次工具调用
    # ------------------------------------------------------------------
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
            "not_executed": False,
            "result": None,
            "result_chars": 0,
            "file_before_sha256": None,
            "file_after_sha256": None,
        }

        def finish(text):
            text = self._append_budget(str(text))
            record["result"] = self._short(text)
            record["result_chars"] = len(text)
            self.stats["tool_calls"].append(record)
            return tool_call_id, text

        # 1) 解析 JSON 参数
        arguments = None
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON 对象")
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            record["not_executed"] = True
            return finish(
                f"{ERR_ARG} 参数解析失败：{exc}。本次调用【未执行】，也未消耗额度；"
                "请提交合法的 JSON 对象参数后重试。"
            )

        record["arguments"] = arguments

        # 2) 允许列表（不消耗额度）
        rejection = self._allowed_rejection(name)
        if rejection:
            record["rejected"] = True
            self.stats["rejected"].append(
                {"step": step, "id": tool_call_id, "name": name,
                 "arguments": arguments, "reason": rejection}
            )
            return finish(rejection)

        # 3) 工具是否存在 + 参数校验（【先校验，后扣额度】）
        problem = self._validate_call(name, arguments)
        if problem:
            record["not_executed"] = True
            self.stats["not_executed"].append(
                {"step": step, "id": tool_call_id, "name": name,
                 "arguments": arguments, "reason": problem}
            )
            return finish(problem)

        # 4) 文件快照（执行前）
        path = arguments.get("path") if name in PATH_TOOLS else None
        if isinstance(path, str) and path:
            self._track_path(path)
            record["file_before_sha256"] = self._digest(path)

        # 5) 占用额度（校验已通过）
        rejection = self._consume_budget(name)
        if rejection:
            record["rejected"] = True
            self.stats["rejected"].append(
                {"step": step, "id": tool_call_id, "name": name,
                 "arguments": arguments, "reason": rejection}
            )
            return finish(rejection)

        # 6) 执行（ok 由回显内容 / 异常共同决定）
        try:
            ok, tool_res = execute_tool_detailed(name, arguments)
        except Exception as exc:  # 防御：execute_tool_detailed 理论上不抛
            ok, tool_res = False, (
                f"[失败] 工具 {name} 调用失败：{type(exc).__name__}: {exc}。"
                "请根据工具定义修正参数后重试。"
            )
        record["ok"] = ok
        if not ok:
            record["not_executed"] = False  # 已执行但未达成目标
            self.stats["not_executed"].append(
                {"step": step, "id": tool_call_id, "name": name,
                 "arguments": arguments, "reason": self._short(str(tool_res), 500)}
            )

        # 7) 文件快照（执行后）
        if isinstance(path, str) and path:
            record["file_after_sha256"] = self._digest(path)

        return finish(tool_res)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
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
            try:
                self._finalize_stats()
            except Exception as exc:  # 统计/快照失败不得覆盖原始 status/error
                self.stats["finalize_error"] = f"{type(exc).__name__}: {exc}"
            self.save()

# agent = Agent(r"读取这个文件的内容D:/★面试/项目/rsi/new/ceshi.txt")
# agent.run()
