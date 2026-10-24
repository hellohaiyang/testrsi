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

v4 追加（对应执行分析第六节 A / B / F / G）：

* **额度改为“预扣 + 有条件退还”**：先校验参数、先记 before sha，再占用额度；
  执行后若 ``ok=False`` 且目标文件确实没变（before == after），就把额度退还，
  并在回显里写清 ``是否计费``。这样“CRLF 拼接写错一次 = 任务永久不可完成”
  这种由**计量语义**造成的死局被消除；同时仍不允许“失败后无限重试”——因为
  只要文件被改动（无法回滚）就照常计费，退还只发生在“失败且零副作用”时。
* 回显尾部新增机器可读的 ``reason_code``（tool_not_allowed / budget_exhausted /
  arg_error / not_executed / not_found / exec_failed / nonzero_exit / ok），
  ``stats["tool_calls"][*]`` 与 ``stats["attempts_per_tool"]`` 记录**含被拒在内**的尝试次数，
  让 G5 / G7 这类判据不必再去解析中文 prose。
* ``policy.watch_paths`` 改为**进入循环之前**预先快照（``initial_source="pre_run"``），
  修掉“watch 路径恒为 changed=False”的缺陷；收尾时只补齐从未被 track 过的路径，
  并把这种兜底快照标成 ``initial_source="finalize_fallback"``（这种快照不可信）。
* 文件快照不再同时重复保存 initial/final 两份全文（未改动时只留一份），
  上下文与轨迹体积按分析里的建议收敛；``llm.prepare_messages`` 负责字符级预算裁剪，
  超硬上限时以 ``status="failed"`` + ``error="context_overflow: ..."`` 显式降级。
"""

import hashlib
import json

from pathlib import Path
from uuid import uuid4

from .llm import ContextOverflowError, call_llm, prepare_messages
from .tools import (
    ERR_ARG,
    PATH_TOOLS,
    TOOL_NAMES,
    TOOLS,
    execute_tool_ex,
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
    "同样不得声称成功。注意：这类**已执行但失败**的调用默认仍然占用额度，"
    "只有回显里明确写出“已全额退还额度（是否计费=no）”时才不占额度；"
    "请以回显的额度行为准，不要在额度贴脸时靠“失败了就免费”来试错。",
    "任何数量型结论（例如“共 N 次运行”“N/N 通过”“替换了 N 处”）"
    "必须与实际执行的验证次数 / 工具回显数字一致；没有逐个验证的必须写成“外推 / 未验证”，"
    "不得把外推说成实测。",
    "一旦你发现自己的某次测量/操作的前提已经失效（例如当前文件内容已不是那次操作的终态），"
    "必须重做测量，而不是补一句解释。",
    "若你已经造成了不可逆的改动，且额度用尽或工具被禁用导致无法回滚，"
    "必须明确写出“我破坏了哪几行、目标是否达成、当前无法回滚”，不得用“已完成”掩盖。",
    "集合型断言（例如“N 个文件都读完了”“含行尾空白的行号是 …”“全文共 N 处匹配”）"
    "必须逐项有回显证据；任何一项没有回显支撑，就要写成“未验证”，"
    "不得用“应该 / 大概 / 看起来”把外推说成实测。",
    "写入 / 删除只允许发生在任务工作区内；需要临时脚本时放到工作区内的 scratch 子目录，"
    "并在最终回答里列出所有新建 / 覆盖 / 删除过的路径。不要把临时文件写进被测代码仓或共享目录。",
    "额度贴脸（例如每个工具只剩 1 次）时，不要在同一轮里并行发多个调用："
    "并行调用会各自预占一份额度，而且你在拿到回显之前无法根据结果调整下一步。"
    "先发一个、读回显、再决定下一个。",
    "工具回显里凡出现“另有 N 处 / N 行未列出”“已截断”这类字样，都表示对应部分【未被验证】，"
    "回答时必须保留这个限定。",
]

EDIT_RULES = [
    "edit 是【字面量子串】替换：没有行锚定、没有词边界、没有正则、没有“第 n 处”之外的精确定位。"
    "默认只替换【第一处】匹配；replace_all=true 才会替换全部匹配。",
    "核对回显里的“替换前文件中共有 K 处匹配”：K 必须与你预期的处数一致。"
    "若 K 大于你预期的处数，说明 old_text 命中了意料之外的子串或位置"
    "（典型情况：目标是一整行 `web`，但 `web` 同时是 `webhook` / `web-cache` 的前缀）。"
    "此时的处理分两种：【没用】定位符（line / occurrence / whole_line）就必须复核——"
    "read 一次或用唯一上下文 / whole_line / line=n 重新定位后重试；"
    "【已经用了】定位符、且回显写明“命中就在你限定的行内，其余匹配在范围之外，属预期保留”时，"
    "这就是正常结果，不必再花一次 read，也不要因为“仍剩 K 处”就说没做完。"
    "如果额度不足以复核，就在最终回答里写清“未复核（未验证）”，不要假定自己改对了。",
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

EDIT_KEYWORDS = ("edit", "替换", "修改", "改写", "改掉", "改为", "改成", "换成", "变成", "重写")

# 任务里出现这些“像是要动文件”的线索时，也注入文本替换规则。
# 说明：这仍然是**启发式**（分析里指出“规则注入靠关键词匹配”本身就是脆弱前提），
# 所以同时把决策理由写进 trace（stats["rule_injection"]），让评测能看见“这次为什么有/没有
# edit 规则”，而不是靠猜。
PATH_HINTS = ("/", "\\", ".conf", ".txt", ".csv", ".json", ".md", ".log", ".yaml", ".yml", ".ini", "文件")

SYSTEM_PROMPT = (
    "你是一个可以调用工具完成任务的助手。\n\n必须遵守的规则：\n"
    + "\n".join(f"{i}. {rule}" for i, rule in enumerate(GENERAL_RULES, 1))
    + "\n"
)


def _edit_rules_reason(task, policy):
    """返回注入 edit 专项规则的理由；None 表示不注入。"""
    text = str(task or "").lower()
    for keyword in EDIT_KEYWORDS:
        if keyword in text:
            return f"keyword:{keyword}"
    allowed = (policy or {}).get("allowed_tools")
    if allowed is not None:
        writable = [name for name in ("edit", "write") if name in allowed]
        if writable:
            return "allowed_tools:" + ",".join(writable)
    for hint in PATH_HINTS:
        if hint in text:
            return f"path_hint:{hint}"
    return None


def _needs_edit_rules(task, policy):
    """是否注入 edit 专项规则（文本替换 / 放行可写的工具 / 任务里像是有文件）。"""
    return _edit_rules_reason(task, policy) is not None


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
            # 含被拒 / 失败 / 已退还在内的“尝试次数”，与 per_tool_count（实际计费次数）区分
            "attempts_per_tool": {},
            "rejected": [],
            "not_executed": [],
            "refunds": [],
            "total_tokens": 0,
            "final_answer": None,
            "file_snapshots": {},
            "budget": {},
            "context": {},
            "rule_injection": {
                "edit_rules": False,
                "reason": None,
                "general_rule_count": len(GENERAL_RULES),
                "edit_rule_count": len(EDIT_RULES),
            },
        }
        rule_reason = _edit_rules_reason(task, self.policy)
        self.stats["rule_injection"].update(
            {
                "edit_rules": rule_reason is not None,
                "reason": rule_reason,
                "note": (
                    "规则注入是启发式的（关键词 / allowed_tools / 路径线索）；"
                    "reason=None 表示未注入 edit 专项规则"
                ),
            }
        )
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

    def _billing_enabled(self):
        """是否配置了任何额度限制——没有限制时“是否计费”这种提示纯属噪声。"""
        return bool(self.policy.get("max_calls")) or (
            self.policy.get("max_total_calls") is not None
        )

    def _refund_budget(self, name, step, tool_call_id, reason):
        """退还一次额度（只用于“失败且文件未被改动”的调用）。

        语义：额度核算的是“对环境的净消耗”。失败且无副作用 → 退还并写进 stats["refunds"]；
        失败但文件已被改动（无法回滚）→ 不退，按计费处理。
        """
        used = self.stats["per_tool_count"].get(name, 0)
        if used <= 0:
            return False
        self.stats["per_tool_count"][name] = used - 1
        self.stats["refunds"].append(
            {
                "step": step,
                "id": tool_call_id,
                "name": name,
                "reason": reason,
                "per_tool_count_after": self.stats["per_tool_count"][name],
            }
        )
        return True

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

    def _track_path(self, path, source="first_touch"):
        """登记文件快照（同一个路径只取第一次，保证 initial 是本次 run 之前的态）。

        ``source`` 记录这次 initial 是什么时候取的：
          * ``pre_run``          —— run() 进入循环前对 watch_paths 预快照（可信）；
          * ``first_touch``      —— 模型第一次调用带 path 的工具之前（可信）；
          * ``finalize_fallback``—— 收尾时才补记（此时 initial 可能已是终态，不可信）。
        """
        key = str(path)
        snapshots = self.stats["file_snapshots"]
        if key in snapshots:
            return snapshots[key]
        entry = {
            "path": key,
            "initial_source": source,
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

    def _presnapshot_watch_paths(self):
        """进入循环之前对 policy.watch_paths 预快照。

        旧版在 ``_finalize_stats`` 里才 track watch_paths（此刻 initial 就是终态），
        于是“模型没碰过 / 只被 bash 改过”的文件永远显示 ``changed=False``。
        """
        for path in self.policy.get("watch_paths") or []:
            self._track_path(path, source="pre_run")

    def _finalize_snapshots(self):
        for key, entry in self.stats.get("file_snapshots", {}).items():
            try:
                target = Path(key)
                if target.is_file():
                    raw = target.read_bytes()
                    entry["final_sha256"] = hashlib.sha256(raw).hexdigest()
                    final_text = self._cap_text(raw)
                    if final_text is not None and final_text == entry.get("initial_content"):
                        # 未改动就不再重复存一份全文，避免轨迹里同一内容出现多次
                        entry["final_content"] = None
                        entry["final_content_omitted"] = "与 initial_content 相同（文件未改动）"
                    else:
                        entry["final_content"] = final_text
                entry["final_captured"] = True
                entry["changed"] = entry.get("initial_sha256") != entry.get("final_sha256")
            except Exception as exc:
                entry["final_error"] = f"{type(exc).__name__}: {exc}"

    def _finalize_stats(self):
        # 只兜底“整轮都没被 track 过”的 watch 路径，并显式标成不可信来源
        for path in self.policy.get("watch_paths") or []:
            if str(path) not in self.stats["file_snapshots"]:
                self._track_path(path, source="finalize_fallback")
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
            # 尝试次数（含被拒 / 失败 / 已退还）与计费次数分开记，便于机判“预算预判失败”
            "attempts": dict(self.stats.get("attempts_per_tool") or {}),
            "refunded_calls": len(self.stats.get("refunds") or []),
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
        # 尝试次数（含被拒 / 失败 / 已退还）：与计费次数分开统计，供 G7 类判据机读
        self.stats["attempts_per_tool"][name] = (
            self.stats["attempts_per_tool"].get(name, 0) + 1
        )
        record = {
            "step": step,
            "id": tool_call_id,
            "name": name,
            "arguments": None,
            "rejected": False,
            "ok": False,
            "not_executed": False,
            "reason_code": None,
            "billed": False,
            "refunded": False,
            "exit_code": None,
            "result": None,
            "result_chars": 0,
            "file_before_sha256": None,
            "file_after_sha256": None,
        }
        billing_note = []

        def finish(text):
            text = str(text)
            if billing_note and self._billing_enabled():
                text = text + "\n" + "\n".join(billing_note)
            text = self._append_budget(text)
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
            record["reason_code"] = "arg_error"
            billing_note.append("是否计费=no（参数解析失败，调用未执行）")
            return finish(
                f"{ERR_ARG} 参数解析失败：{exc}。本次调用【未执行】，也未消耗额度；"
                "请提交合法的 JSON 对象参数后重试。"
            )

        record["arguments"] = arguments

        # 2) 允许列表（不消耗额度）
        rejection = self._allowed_rejection(name)
        if rejection:
            record["rejected"] = True
            record["reason_code"] = "tool_not_allowed"
            record["billed"] = False
            billing_note.append("是否计费=no（工具不在允许列表，调用未执行）")
            self.stats["rejected"].append(
                {"step": step, "id": tool_call_id, "name": name,
                 "arguments": arguments, "reason": rejection,
                 "reason_code": "tool_not_allowed"}
            )
            return finish(rejection)

        # 3) 工具是否存在 + 参数校验（【先校验，后扣额度】）
        problem = self._validate_call(name, arguments)
        if problem:
            record["not_executed"] = True
            record["reason_code"] = "arg_error"
            record["billed"] = False
            billing_note.append("是否计费=no（参数错误，调用未执行）")
            self.stats["not_executed"].append(
                {"step": step, "id": tool_call_id, "name": name,
                 "arguments": arguments, "reason": problem,
                 "reason_code": "arg_error", "executed": False}
            )
            return finish(problem)

        # 4) 文件快照（执行前；同一个路径只取第一次，保证 initial 是本次 run 之前的态）
        path = arguments.get("path") if name in PATH_TOOLS else None
        if isinstance(path, str) and path:
            self._track_path(path, source="first_touch")
            record["file_before_sha256"] = self._digest(path)

        # 5) 占用额度（校验已通过；先预扣，执行后按“是否产生副作用”决定是否退还）
        rejection = self._consume_budget(name)
        if rejection:
            record["rejected"] = True
            record["reason_code"] = "budget_exhausted"
            record["billed"] = False
            billing_note.append("是否计费=no（额度已用完，调用未执行）")
            self.stats["rejected"].append(
                {"step": step, "id": tool_call_id, "name": name,
                 "arguments": arguments, "reason": rejection,
                 "reason_code": "budget_exhausted"}
            )
            return finish(rejection)
        record["billed"] = True

        # 6) 执行（ok 由工具回显 / exit_code / 异常共同决定，见 tools.execute_tool_ex）
        try:
            ok, tool_res, meta = execute_tool_ex(name, arguments)
        except Exception as exc:  # 防御：execute_tool_ex 理论上不抛
            ok, meta = False, {"reason_code": "exec_failed"}
            tool_res = (
                f"[失败] 工具 {name} 调用失败：{type(exc).__name__}: {exc}。"
                "请根据工具定义修正参数后重试。"
            )
        record["ok"] = ok
        record["reason_code"] = (meta or {}).get("reason_code") or ("ok" if ok else "failed")
        if "exit_code" in (meta or {}):
            record["exit_code"] = meta["exit_code"]

        # 7) 文件快照（执行后）
        after_sha = None
        if isinstance(path, str) and path:
            after_sha = self._digest(path)
            record["file_after_sha256"] = after_sha

        # 8) 失败且【没有产生任何副作用】→ 退还额度（预扣 + 有条件退还）
        if not ok:
            unchanged = record["file_before_sha256"] == after_sha
            if unchanged and self._refund_budget(
                name, step, tool_call_id, f"{record['reason_code']}:{name}"
            ):
                record["refunded"] = True
                record["billed"] = False
                billing_note.append(
                    "是否计费=no：本次调用【失败】且"
                    + ("目标文件字节未发生变化" if path else "无可观测的文件改动")
                    + "，已全额退还额度（本次失败不占用额度，可在额度内修正后重试）。"
                )
                if not path:
                    billing_note.append(
                        "注意：bash 的副作用无法核验；若该命令其实已经改动了文件，"
                        "退还额度并不代表改动被回滚。"
                    )
            else:
                billing_note.append(
                    "是否计费=yes：本次调用【失败】"
                    + (
                        "但目标文件已经发生变化（before/after sha 不同，无法确认可回滚），因此照常占用额度。"
                        if path
                        else "（未能证明无副作用），因此照常占用额度。"
                    )
                )
            record["not_executed"] = False  # 已执行但未达成目标
            self.stats["not_executed"].append(
                {"step": step, "id": tool_call_id, "name": name,
                 "arguments": arguments, "reason": self._short(str(tool_res), 500),
                 "reason_code": record["reason_code"], "executed": True,
                 "billed": record["billed"]}
            )
        else:
            billing_note.append("是否计费=yes：本次调用成功执行并占用额度。")

        return finish(tool_res)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _run_loop(self):
        for step in range(1, self.max_steps + 1):
            self.stats["steps"] = step
            # 上下文预算：默认原样发送；超过软上限才裁剪，超硬上限抛 ContextOverflowError
            prepared, context_info = prepare_messages(self.messages)
            self.stats["context"] = context_info
            if context_info.get("trimmed"):
                self.stats.setdefault("context_trims", []).append(
                    {"step": step, **context_info}
                )
            message, token = call_llm(prepared)
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
            # 关键：watch_paths 必须在进入循环【之前】快照，
            # 否则收尾时才 track 会让 initial == final，changed 恒为 False。
            self._presnapshot_watch_paths()
            return self._run_loop()
        except ContextOverflowError as exc:
            # 显式降级：不是“优雅收敛”，而是把 status 记为 failed 并写清原因，
            # 让上层能区分“能力不足”和“环境给不了更多上下文”。
            self.status = "failed"
            self.error = f"context_overflow: {exc}"
            self.final_answer = None
            self.stats["final_answer"] = None
            self.stats["context_overflow"] = {"detail": str(exc)}
            print(self.error)
            return None
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
