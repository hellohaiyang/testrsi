import importlib
from pathlib import Path
from uuid import uuid4
import shutil
import json

from miniagent import MiniAgent


def load_agent(version):
    importlib.invalidate_caches()
    module = importlib.import_module(f"{version}.agent")
    return module.Agent


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class RSI:
    def __init__(self):
        self.history_tasks = []
        self.version = "v3"
        self.root_dir = Path(__file__).resolve().parent
        self.experiment_id = uuid4().hex
        self.round_index = 0

    def step1_create_task(self, trace_path):
        miniagent = MiniAgent(
            trace_path=trace_path,
            allowed_paths=[
                self.root_dir / self.version,
            ],
        )
        system_prompt = "你是agent自进化的评测者。"
        task = f"想一个具体的任务，让agent完成。最好能让agent做错，发现能力不足的任务。之前做过的任务{self.history_tasks}。你可以用工具查看agent的代码。"
        content = miniagent.run(system_prompt, task)
        self.history_tasks.append(content)
        return content

    # def step3_analyze_result(self, trace_path):
    #     miniagent = MiniAgent(trace_path=trace_path)
    #     system_prompt = "你是agent自进化的评测者。"
    #     task = f"分析agent的执行流程，给出可以改进的点。你可以用工具查看agent的代码。"
    #     content = miniagent.run(system_prompt, task)
    #     self.history_tasks.append(content)
    #     return content
    def step3_analyze_result(
            self,
            task,
            agent_version,
            agent_trace_path,
            trace_path,
    ):
        agent_trace_path = Path(agent_trace_path)
        if not agent_trace_path.is_file():
            raise FileNotFoundError(
                f"找不到待分析的执行轨迹：{agent_trace_path}"
            )
        code_dir = self.root_dir / agent_version
        miniagent = MiniAgent(
            trace_path=trace_path,
            allowed_paths=[
                code_dir,
                agent_trace_path,
            ],
        )
        system_prompt = "你是 agent 自进化的评测者，请基于代码和执行轨迹分析问题。"
        analysis_task = f"""
            请分析下面这次 agent 执行。
        
            原始任务：
            {task}
        
            执行任务的 agent 版本：
            {agent_version}
        
            该版本的代码目录：
            {code_dir}
        
            该次执行的轨迹文件：
            {agent_trace_path}
        
            请使用工具读取上述轨迹和相关代码，然后给出：
            1. 任务是否完成，依据是什么。
            2. 执行中出现了什么具体问题。
            3. 哪些问题与 agent 的代码有关。
            4. 建议修改哪些文件、哪些逻辑，以及修改原因。
        
            请区分已经观察到的问题和推测。
            如果文件无法读取，请明确说明。
            """
        return miniagent.run(system_prompt, analysis_task)

    def step4_evolve_agent(self, task, analysis, trace_path):
        source_dir = self.root_dir / self.version
        # 找到第一个尚未使用的版本目录名
        version_number = int(self.version[1:])+1
        while (self.root_dir / f"v{version_number}").exists():
            version_number += 1
        candidate_version = f"v{version_number}"
        candidate_dir = self.root_dir / candidate_version
        # 复制完整版本，跳过 Python 缓存目录
        shutil.copytree(
            source_dir,
            candidate_dir,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        evolution_task = f"""
            请根据执行分析，改进候选目录中的 agent 代码。
        
            原始任务：
            {task}
        
            分析结论：
            {analysis}
        
            当前版本的源代码目录：
            {source_dir}
        
            候选版本的代码目录：
            {candidate_dir}
        
            程序已经把当前版本完整复制到了候选目录。
        
            请先读取候选目录中的 agent.py、llm.py、tools.py，
            再根据分析结论修改候选目录中的代码。
        
            要求：
            1. 只向候选目录内写入或修改代码。
            2. 不得删除任何文件，不得修改源代码目录和共享配置。
            3. 保留 Agent(task, trace_path) 构造接口。
            4. 保留 run() 方法和向指定 trace_path 保存 JSON 轨迹的功能。
            5. 同版本模块之间使用相对导入。
            6. 完成后说明修改了什么，以及预期解决什么问题。
            7. 保留轨迹中的 status、error、messages 字段及其含义，
            正常结束为 completed，异常为 failed，耗尽步数为 max_steps。
            """
        AgentClass = load_agent(self.version)
        evolver = AgentClass(
            evolution_task,
            trace_path=trace_path,
        )
        evolver.run()
        if evolver.status != "completed":
            raise RuntimeError(
                f"生成候选代码的过程未正常完成：{evolver.status}"
            )
        return candidate_version

    # def step6_judge_evolution(self, trace_path):
    #     miniagent = MiniAgent(trace_path=trace_path)
    #     system_prompt = "你是agent自进化的评测者。"
    #     task = f"对比自进化前后的agent的代码，和执行任务的结果，你认为是否要启用进化后的版本，你可以通过工具查看执行轨迹和agent代码，直接输出yes或no。"
    #     content = miniagent.run(system_prompt, task)
    #     self.history_tasks.append(content)
    #     return content
    def step6_judge_evolution(
            self,
            task,
            baseline_version,
            candidate_version,
            baseline_trace,
            candidate_trace,
            trace_path,
    ):
        for path in (baseline_trace, candidate_trace):
            if not Path(path).is_file():
                raise FileNotFoundError(f"找不到待评判的轨迹：{path}")
        baseline_dir = self.root_dir / baseline_version
        candidate_dir = self.root_dir / candidate_version
        miniagent = MiniAgent(
            trace_path=trace_path,
            allowed_paths=[
                baseline_dir,
                candidate_dir,
                baseline_trace,
                candidate_trace,
            ],
        )
        system_prompt = (
            "你是 agent 自进化的评测者。"
            "请根据实际代码和执行轨迹判断改进是否有效。"
            "最终回答只能是 yes 或 no。"
        )
        judge_task = f"""
            原始任务：
            {task}
        
            旧版本：{baseline_version}
            旧版本代码目录：{baseline_dir}
            旧版本执行轨迹：{baseline_trace}
        
            候选版本：{candidate_version}
            候选版本代码目录：{candidate_dir}
            候选版本执行轨迹：{candidate_trace}
        
            请使用工具读取两份执行轨迹和相关代码，比较：
            1. 两个版本实际完成任务的情况。
            2. 候选版本是否解决了旧版本暴露的问题。
            3. 候选版本是否引入了新的明显问题。
        
            不要仅凭候选版本声称“已完成”就认定成功，
            需要检查轨迹中的工具调用和结果。
        
            只有证据支持候选版本在本轮任务中有所改善，
            且没有发现明显退步时，才回答 yes。
            证据不足或没有改善时回答 no。
        
            最终回答只能是 yes 或 no。
            """
        content = miniagent.run(system_prompt, judge_task)
        decision = (content or "").strip().lower()
        if decision not in {"yes", "no"}:
            raise ValueError(f"评判结果格式不正确：{content!r}")
        return decision == "yes"

    def run_one_round(self, round_index):
        round_dir = (
                self.root_dir
                / "artifacts"
                / "runs"
                / self.experiment_id
                / f"round_{round_index:04d}"
        )
        trace_dir = round_dir / "traces"
        # 1.创建任务，可以读取agent的代码。
        task = self.step1_create_task(trace_path=trace_dir / "create_task.json")
        print(task)
        # 2.agent执行任务，返回任务id，把轨迹保存。
        baseline_trace = trace_dir / "baseline.json"
        AgentClass = load_agent(self.version)
        agent1 = AgentClass(task, trace_path=baseline_trace)
        try:
            agent1.run()
        except Exception as exc:
            print(f"旧版本执行异常：{type(exc).__name__}: {exc}")
            if not baseline_trace.is_file():
                raise
            print(f"继续分析失败轨迹：{baseline_trace}")
        # 3.分析agent执行的结果，返回改进建议。可以读取agent的代码和执行文件。
        # analyze = self.step3_analyze_result(trace_path=trace_dir / "analyze.json")
        analyze = self.step3_analyze_result(
            task=task,
            agent_version=self.version,
            agent_trace_path=baseline_trace,
            trace_path=trace_dir / "analyze.json",
        )
        # 4.agent自己修改自己的代码。
        # self_evo_task = "参考分析结果，改进自己的代码。你不要改自己当前所在的目录，将所有的新的agent代码放到v2目录下。"
        # agent2 = Agent()
        candidate_version = self.step4_evolve_agent(
            task=task,
            analysis=analyze,
            trace_path=trace_dir / "evolve.json",
        )
        # 5.agent再次执行任务，查看结果。
        # agent3 = Agent(task)
        # agent3.run()
        candidate_trace = trace_dir / "candidate.json"
        try:
            CandidateAgentClass = load_agent(candidate_version)
            candidate_agent = CandidateAgentClass(
                task,
                trace_path=candidate_trace,
            )
            candidate_agent.run()
            candidate_data = json.loads(
                candidate_trace.read_text(encoding="utf-8")
            )
            status = candidate_data.get("status")
            if status != "completed":
                raise RuntimeError(f"候选版本执行状态异常：{status!r}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            # 导入或创建实例失败时，Agent 还来不及保存轨迹。
            # 由 RSI 在同一位置记录失败原因。
            if not candidate_trace.is_file():
                save_json(
                    candidate_trace,
                    {
                        "recorded_by": "RSI",
                        "version": candidate_version,
                        "status": "failed",
                        "error": error,
                        "messages": [],
                    },
                )
            print(f"拒绝候选版本 {candidate_version}：{error}")
            print(f"本轮结束，继续使用 {self.version}")
            return
        # 6.分析agent的改进是否有效，有效就换成新的版本，否则继续旧版本。
        # res = self.step6_judge_evolution(trace_path=trace_dir / "judge.json")
        baseline_version = self.version
        accepted = self.step6_judge_evolution(
            task=task,
            baseline_version=baseline_version,
            candidate_version=candidate_version,
            baseline_trace=baseline_trace,
            candidate_trace=candidate_trace,
            trace_path=trace_dir / "judge.json",
        )
        if accepted:
            self.version = candidate_version
            print(f"接受候选版本：{baseline_version} → {self.version}")
        else:
            print(f"拒绝候选版本：{candidate_version}，继续使用 {self.version}")

    def run(self, rounds=1):
        for _ in range(rounds):
            self.round_index += 1
            print(
                f"\n开始第 {self.round_index} 轮，"
                f"当前版本：{self.version}"
            )
            self.run_one_round(self.round_index)

rsi=RSI()
rsi.run(3)
