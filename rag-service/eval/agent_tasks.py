"""
Agent 任务级评估脚本 — 三层指标 + agent_eval_runs 版本化落库（module-066 / ADR-0017）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python -m eval.agent_tasks                              # agent 模式全量（默认）
    python -m eval.agent_tasks --mode chat                  # chat 模式（无工具轨迹，Trajectory 如实标注"无轨迹"）
    python -m eval.agent_tasks --sample 10 --pass_k 3       # pass^3 可靠性口径（抽样 10 条 × 3 次全成功）
    python -m eval.agent_tasks --limit 5                    # 冒烟（只跑前 5 条）
    python -m eval.agent_tasks --no-save                    # dry-run 不落库
    python -m eval.agent_tasks --fixture                    # 零 LLM/DB 启发式冒烟（不落库）

三层指标（ADR-0017 决策 1，对齐业界三层测量）:
    Outcome:    任务完成率 pass^1 / pass^k（k 次独立尝试全成功才算对，τ-bench 可靠性口径）；
                agent 多轮路径（task 为数组）单独统计
    Trajectory: 工具调用正确率（覆盖 + 无多调 + 参数类型，确定性判定，决策 4 一字不改）；
                --mode chat 无工具轨迹 → 输出"无轨迹"占位（如实标注不伪造）
    System:     平均步数（tool_count）/ 平均 token / 端到端耗时 P50/P95

判定器（确定性，不用 LLM-as-judge，ADR-0017 决策 4）:
    1. 覆盖：期望中的每个工具都被调用（顺序放宽，最后一轮前调用即算）
    2. 无多调：实际调用都在期望集合内（豁免：re_search 双组设计允许生成阶段补检）
    3. 参数类型：args 的 key 与 args_schema 必填字段一致（不判值语义）
    4. Grounding：result_ok 比例（tool_call_logs 落库读取；降级链兜底不算错）
    outcome pass = 工具覆盖（tools=[] 任务恒过）+ answer_points 关键词全部包含
    （简单子串匹配，不判语义——与 ADR-0017 诚实边界 3 一致）

预期坑（ADR-0017 决策 4 声明）: LLM 路径选择有天然方差（如 search_knowledge
直接命中不再调 search_fts）——"覆盖"只要求期望工具都出现，不要求精确顺序。

降级策略:
    - 单任务运行失败（LLM 429/超时等）→ 该任务按 fail 记录并标注 reason，其余继续
    - tool_call_logs 不可读/开关关 → Grounding 如实标 None（不伪造）
    - 数据库不可用 → --fixture 模式演示管线；真实模式如实标注"待环境"
"""
import argparse
import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path
from unittest import mock

from sqlalchemy import text

from src import observability
from src.config import settings
from src.database import async_session_factory
from agent.tool_registry import ToolRegistry, registry
from agent.react import ReactContext, _build_messages, react_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("agent_tasks")

# 本文件所在目录（eval/）
EVAL_DIR = Path(__file__).resolve().parent
TASKS_PATH = EVAL_DIR / "agent_tasks.json"

# 评测固定匿名身份（memory 路径只读不写，测后清理不污染真实记忆）
EVAL_IDENTITY = "eval-066-anon"

# 任务路径分类关键词：expected_tools=[] 时按时间/天气词区分 realtime 与 casual
_REALTIME_KEYWORDS = ("几点", "天气", "几号", "日期", "温度", "时间")

# agent_eval_runs 表 DDL（module-066，对齐 eval_runs 模式：git_commit +
# config_snapshot + scores + per_question JSONB 逐任务明细；幂等）
AGENT_EVAL_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS agent_eval_runs (
    id            BIGSERIAL    PRIMARY KEY,
    eval_type     VARCHAR(20)  NOT NULL DEFAULT 'agent_eval',
    git_commit    VARCHAR(64)  NOT NULL DEFAULT '',
    config_snapshot JSONB      NOT NULL DEFAULT '{}',
    scores        JSONB        NOT NULL DEFAULT '{}',
    per_question  JSONB        NOT NULL DEFAULT '[]',
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE agent_eval_runs IS 'Agent 任务级评估运行记录（module-066 / ADR-0017 版本化回归）';
COMMENT ON COLUMN agent_eval_runs.eval_type IS '评估类型（agent_eval）';
COMMENT ON COLUMN agent_eval_runs.git_commit IS '评估时的 git commit';
COMMENT ON COLUMN agent_eval_runs.config_snapshot IS '评估时 rag_config 快照';
COMMENT ON COLUMN agent_eval_runs.scores IS '整体指标分数（三层）';
COMMENT ON COLUMN agent_eval_runs.per_question IS '逐任务明细（判定结果/实际工具序列/步数/耗时/token）';
"""


# ==================== 数据集加载与校验 ====================


def load_agent_tasks(path: Path = TASKS_PATH) -> list[dict]:
    """加载 agent 任务集，校验结构（条数/字段/工具名合法）

    Args:
        path: agent_tasks.json 路径

    Returns:
        任务列表，每项含 id / task（str 或数组）/ expected_tools / answer_points

    Raises:
        FileNotFoundError: 任务集缺失
        ValueError: 结构非法（条数不在 30-50、id 重复、字段缺失、工具名不在
            ToolRegistry 10 工具内、answer_points 不在 1-3 个）
    """
    if not path.exists():
        raise FileNotFoundError(f"agent 任务集不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not (30 <= len(data) <= 50):
        raise ValueError(f"任务集需为 list 且 30-50 条，当前 {len(data) if isinstance(data, list) else 'N/A'}")
    valid_tools = set(registry.list_tool_names())
    seen: set[str] = set()
    for item in data:
        task = item.get("task")
        if not task or not (isinstance(task, str) or (isinstance(task, list) and task and all(isinstance(t, str) and t for t in task))):
            raise ValueError(f"任务 task 需为非空字符串或非空字符串数组: {item.get('id')}")
        tools = item.get("expected_tools")
        if not isinstance(tools, list) or not set(tools) <= valid_tools:
            raise ValueError(f"expected_tools 需为 ToolRegistry 工具名子集: {item.get('id')} -> {tools}")
        points = item.get("answer_points")
        if not isinstance(points, list) or not (1 <= len(points) <= 3) or not all(isinstance(p, str) and p for p in points):
            raise ValueError(f"answer_points 需为 1-3 个非空关键词: {item.get('id')}")
        tid = item.get("id")
        if tid in seen:
            raise ValueError(f"任务 id 重复: {tid}")
        seen.add(tid)
    return data


def classify_path(item: dict) -> str:
    """任务路径分类（数据集覆盖计数口径，六类）

    优先级：重检（expected 含 re_search）→ 记忆（含 recall_memory）→
    多轮（task 为数组）→ 空工具（realtime 时间/天气词，否则 casual）→ 单轮知识。
    """
    tools = item["expected_tools"]
    if "re_search" in tools:
        return "reselect"
    if "recall_memory" in tools:
        return "memory"
    if isinstance(item["task"], list):
        return "knowledge_multi"
    if not tools:
        q = item["task"]
        return "realtime" if any(w in q for w in _REALTIME_KEYWORDS) else "casual"
    return "knowledge_single"


def path_coverage(tasks: list[dict]) -> dict:
    """六类路径覆盖计数（≥6 类路径通过标准）"""
    counts: dict[str, int] = {}
    for t in tasks:
        p = classify_path(t)
        counts[p] = counts.get(p, 0) + 1
    return counts


# ==================== 判定器（确定性，ADR-0017 决策 4） ====================


def check_coverage(actual_names: list[str], expected_tools: list[str]) -> bool:
    """规则 1 覆盖：期望中的每个工具都被调用（顺序放宽）"""
    return set(expected_tools) <= set(actual_names)


def check_no_extra(actual_names: list[str], expected_tools: list[str]) -> bool:
    """规则 2 无多调：实际调用都在期望集合内

    豁免：re_search 双组设计允许生成阶段补检（ADR-0017 决策 4 规则 2）。
    """
    return set(actual_names) <= set(expected_tools) | {"re_search"}


def check_args_type(actual_calls: list[dict], schemas: dict) -> bool:
    """规则 3 参数类型：args 的 key 与 args_schema 必填字段一致（不判值语义）

    Args:
        actual_calls: 实际工具调用 [{"name", "args"}, ...]
        schemas: {工具名: args_schema}（来自 ToolRegistry）
    """
    for call in actual_calls:
        schema = schemas.get(call["name"])
        if not schema:
            continue  # 工具不存在由覆盖/grounding 覆盖
        required = set(schema.get("required", []))
        if required and not required <= set((call.get("args") or {}).keys()):
            return False
    return True


def outcome_pass(item: dict, answer: str, actual_names: list[str]) -> bool:
    """outcome：工具覆盖（tools=[] 任务恒过）+ answer_points 关键词全部包含"""
    if not check_coverage(actual_names, item["expected_tools"]):
        return False
    return all(p in (answer or "") for p in item["answer_points"])


def classify_failure(task: dict) -> str:
    """失败任务分类（不隐藏，如实输出）

    参数错 → 工具选错（多调）→ 工具漏调（覆盖缺）→ 路径绕（步数超出期望+1）
    → 答案缺要点（关键词未包含）。
    """
    if task.get("args_ok") is False:
        return "参数错"
    if task.get("no_extra") is False:
        return "工具选错"
    if task.get("coverage") is False:
        return "工具漏调"
    if task.get("tool_count", 0) > len(task["expected_tools"]) + 1:
        return "路径绕"
    return "答案缺要点"


# ==================== 指标聚合（纯函数） ====================


def _percentile(sorted_vals: list, p: float) -> float:
    """线性插值百分位（P50/P95 口径）"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def compute_scores(per_task: list[dict]) -> dict:
    """三层指标聚合（纯函数）

    Args:
        per_task: 逐任务明细（见 _run_task）

    Returns:
        scores dict：pass^1 / 工具正确率 / 无多调率 / 参数正确率 / grounding /
        平均步数 / 平均 token / P50-P95 耗时 / per_path 子集统计
    """
    n = len(per_task)
    if not n:
        return {"count": 0, "pass_1": 0.0, "tool_correct_rate": None,
                "no_extra_rate": None, "args_rate": None, "grounding": None,
                "avg_tool_count": 0.0, "avg_tokens": None,
                "p50_ms": 0.0, "p95_ms": 0.0, "per_path": {}}
    pass_n = sum(1 for t in per_task if t["pass"])
    # Trajectory：agent 模式有轨迹；chat 模式 tool_correct=None → 如实"无轨迹"
    traj = [t for t in per_task if t["tool_correct"] is not None]
    grounding_vals = [t["grounding"] for t in per_task if t.get("grounding") is not None]
    tokens = [t["tokens"] for t in per_task if t.get("tokens") is not None]
    durs = sorted(t["duration_ms"] for t in per_task)
    per_path: dict[str, dict] = {}
    for t in per_task:
        bucket = per_path.setdefault(t["path"], {"count": 0, "pass": 0})
        bucket["count"] += 1
        bucket["pass"] += 1 if t["pass"] else 0
    scores = {
        "count": n,
        "pass_1": round(pass_n / n, 4),
        "tool_correct_rate": round(sum(1 for t in traj if t["tool_correct"]) / len(traj), 4) if traj else None,
        "no_extra_rate": round(sum(1 for t in traj if t["no_extra"]) / len(traj), 4) if traj else None,
        "args_rate": round(sum(1 for t in traj if t["args_ok"]) / len(traj), 4) if traj else None,
        "grounding": round(sum(grounding_vals) / len(grounding_vals), 4) if grounding_vals else None,
        "avg_tool_count": round(sum(t["tool_count"] for t in per_task) / n, 2),
        "avg_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None,
        "p50_ms": round(_percentile(durs, 0.5), 1),
        "p95_ms": round(_percentile(durs, 0.95), 1),
        "per_path": {p: {"count": v["count"], "pass_rate": round(v["pass"] / v["count"], 4)}
                     for p, v in per_path.items()},
    }
    return scores


# ==================== 运行器 ====================


def _tool_schemas() -> dict:
    """{工具名: args_schema}（判定器规则 3 用）"""
    return {t.name: t.args_schema for t in registry.list_tools()}


def _sum_usage(usage: dict) -> int:
    """usage {provider: {prompt, completion}} → 总 token"""
    return sum(v.get("prompt", 0) + v.get("completion", 0) for v in usage.values())


async def _load_grounding(trace_id: str) -> float | None:
    """从 tool_call_logs 读该次运行的 result_ok 比例（ADR 决策 1 数据来源）

    不可读/无行 → None（如实标注，不伪造）。
    """
    try:
        from src.database import ensure_tool_call_logs_table
        await ensure_tool_call_logs_table()
        async with async_session_factory() as session:
            rows = (await session.execute(
                text("SELECT result_ok FROM tool_call_logs WHERE trace_id = :t"),
                {"t": trace_id})).mappings().all()
        if not rows:
            return None
        return round(sum(1 for r in rows if r["result_ok"]) / len(rows), 4)
    except Exception as e:
        logger.warning("读取 tool_call_logs 失败，grounding 标 None: %s", e)
        return None


async def _run_agent_once(item: dict, k: int, fixture: bool) -> dict:
    """单次 agent 模式运行：react_loop → 收集工具序列/答案/耗时/token/grounding

    trace_id 自设 eval-<id>-<k>（observability 上下文），tool_call_logs
    以该 trace_id 落库，grounding 事后读回——真实模式闭环。
    """
    trace_id = f"eval-{item['id']}-{k}"
    observability.init_request(trace_id)
    rounds = item["task"] if isinstance(item["task"], list) else [item["task"]]
    calls: list[dict] = []
    answer = ""
    t0 = time.perf_counter()
    history: list[dict] = []
    error: str | None = None
    try:
        for q in rounds:
            ctx = ReactContext(q, identity=EVAL_IDENTITY, history=history)
            if fixture:
                events = await _run_loop_fixture(ctx, q, item, is_last=(q == rounds[-1]))
            else:
                events = await _run_loop_real(ctx, q)
            round_answer = ""
            for evt in events:
                if evt["type"] == "tool_call":
                    calls.append({"name": evt["name"], "args": evt["args"]})
                elif evt["type"] == "done":
                    round_answer = evt.get("answer", "") or ""
            answer = round_answer
            history.append({"role": "user", "content": q})
            if round_answer:
                history.append({"role": "assistant", "content": round_answer})
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.error("[%s] 任务运行失败: %s", item["id"], error)
    duration_ms = int((time.perf_counter() - t0) * 1000)
    tokens = _sum_usage(observability.get_request_stats().get("usage", {}))
    grounding = None if fixture else await _load_grounding(trace_id)
    return _build_task_result(item, calls, answer, duration_ms, tokens,
                              grounding, error)


async def _run_loop_real(ctx: ReactContext, q: str) -> list[dict]:
    """真实模式：react_loop（真实 LLM + 真实工具 + DB 检索）"""
    return [evt async for evt in react_loop(
        ctx, _build_messages(ctx), settings.max_agent_tools)]


async def _run_loop_fixture(ctx: ReactContext, q: str, item: dict,
                            is_last: bool) -> list[dict]:
    """fixture 模式：假 LLM 按期望工具序列回放 + 假工具，零 LLM/DB

    假 LLM 逐次返回 expected_tools 中的工具调用（参数取 args_schema 必填
    字段占位），计划耗尽后直接输出答案；最后一轮答案包含全部 answer_points
    （fixture 仅演示管线，不代表真实质量，如实标注）。
    """
    plan = [(name, _args_for(name)) for name in item["expected_tools"]]
    final_answer = "、".join(item["answer_points"]) + "。"
    client = _FixtureClient(plan, final_answer if is_last else "（fixture 中间轮回答）")
    with mock.patch("agent.react.LLMFactory.get_client", return_value=client):
        return [evt async for evt in react_loop(
            ctx, _build_messages(ctx), settings.max_agent_tools,
            tools=_fixture_registry())]


async def _run_chat_once(item: dict, k: int) -> dict:
    """单次 chat 模式运行：engine.chat 生产流水线（无工具轨迹，Trajectory 无）"""
    from rag.schemas import ChatRequest
    from rag.engine import rag_engine
    trace_id = f"eval-{item['id']}-{k}"
    observability.init_request(trace_id)
    rounds = item["task"] if isinstance(item["task"], list) else [item["task"]]
    answer = ""
    t0 = time.perf_counter()
    history: list[dict] = []
    error: str | None = None
    try:
        for q in rounds:
            result = await rag_engine.chat(
                ChatRequest(query=q, history=history), identity=EVAL_IDENTITY)
            answer = result.answer or ""
            history.append({"role": "user", "content": q})
            if answer:
                history.append({"role": "assistant", "content": answer})
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.error("[%s] chat 任务运行失败: %s", item["id"], error)
    duration_ms = int((time.perf_counter() - t0) * 1000)
    tokens = _sum_usage(observability.get_request_stats().get("usage", {}))
    return _build_task_result(item, [], answer, duration_ms, tokens,
                              None, error, has_tools=False)


def _build_task_result(item: dict, calls: list[dict], answer: str,
                       duration_ms: int, tokens: int, grounding,
                       error: str | None, has_tools: bool = True) -> dict:
    """聚合单次运行结果：判定器四规则 + outcome（纯逻辑，可单测）

    has_tools=False（chat 模式）：无工具轨迹层——覆盖/无多调/参数/Grounding
    置 None（如实标注"无轨迹"），outcome 只按 answer_points 判定。
    """
    names = [c["name"] for c in calls]
    if not has_tools:
        return {
            "task_id": item["id"],
            "path": classify_path(item),
            "expected_tools": item["expected_tools"],
            "pass": (error is None) and all(
                p in (answer or "") for p in item["answer_points"]),
            "coverage": None, "no_extra": None, "args_ok": None,
            "tool_correct": None, "grounding": None,
            "actual_names": [], "tool_count": 0,
            "tokens": tokens, "duration_ms": duration_ms,
            "answer": (answer or "")[:200], "fail_reason": error,
        }
    schemas = _tool_schemas()
    coverage = check_coverage(names, item["expected_tools"])
    no_extra = check_no_extra(names, item["expected_tools"])
    args_ok = check_args_type(calls, schemas)
    tool_correct = coverage and no_extra and args_ok
    return {
        "task_id": item["id"],
        "path": classify_path(item),
        "expected_tools": item["expected_tools"],
        "pass": (error is None) and outcome_pass(item, answer, names),
        "coverage": coverage,
        "no_extra": no_extra,
        "args_ok": args_ok,
        "tool_correct": tool_correct,
        "grounding": grounding,
        "actual_names": names,
        "tool_count": len(names),
        "tokens": tokens,
        "duration_ms": duration_ms,
        "answer": (answer or "")[:200],
        "fail_reason": error,
    }


async def run_eval(tasks: list[dict], mode: str, pass_k: int,
                   fixture: bool) -> tuple[list[dict], dict]:
    """运行任务集：每任务 pass_k 次独立尝试，聚合三层指标

    pass^k 口径：k 次全成功才算该任务 pass（τ-bench 可靠性思想）；
    Trajectory/System 取首次运行结果（诚实标注口径）。

    Returns:
        (per_question, scores)
    """
    per_question: list[dict] = []
    for i, item in enumerate(tasks):
        runs: list[dict] = []
        for k in range(1, pass_k + 1):
            if mode == "chat":
                runs.append(await _run_chat_once(item, k))
            else:
                runs.append(await _run_agent_once(item, k, fixture))
        first = dict(runs[0])
        first["pass"] = all(r["pass"] for r in runs)  # k 次全成功才算过
        per_question.append(first)
        logger.info("[%d/%d] %s pass=%s tools=%s",
                    i + 1, len(tasks), item["id"], first["pass"], first["actual_names"])
    scores = compute_scores(per_question)
    scores.update({
        "mode": mode,
        "pass_k": pass_k,
        "fixture": fixture,
        "dataset_size": len(tasks),
        "trajectory": "无轨迹（--mode chat 无工具调用明细，如实标注）"
        if mode == "chat" else "有轨迹",
    })
    return per_question, scores


# ==================== fixture 假 LLM / 假工具 ====================


def _args_for(name: str) -> dict:
    """工具参数占位：args_schema 必填字段（判定器规则 3 通过所需）"""
    tool = registry.get(name)
    if not tool:
        return {}
    return {k: "fixture" for k in (tool.args_schema.get("required") or [])}


def _fixture_registry() -> ToolRegistry:
    """fixture stub 工具注册表：返回固定文本，不依赖 DB/LLM"""
    async def _stub(ctx, args):
        return "（fixture 结果：模拟工具输出）"

    reg = ToolRegistry()
    for name in ("search_knowledge", "generate_answer", "re_search", "recall_memory"):
        tool = registry.get(name)
        reg.register(name, tool.description, tool.args_schema, _stub,
                     group=list(tool.group))
    return reg


class _FixtureClient:
    """fixture 假 LLM：按计划回放 tool_calls，计划耗尽后直接输出答案"""

    def __init__(self, plan: list, answer: str):
        self._plan = list(plan)
        self._answer = answer

    async def chat_with_tools(self, messages, tools):
        if self._plan:
            name, args = self._plan.pop(0)
            return {
                "content": "",
                "tool_calls": [{"id": f"f-{name}", "name": name, "args": args}],
                "message": {"role": "assistant", "content": "",
                            "tool_calls": [{"id": f"f-{name}", "type": "function",
                                            "function": {"name": name,
                                                         "arguments": json.dumps(args, ensure_ascii=False)}}]},
            }
        return {"content": self._answer, "tool_calls": [],
                "message": {"role": "assistant", "content": self._answer}}

    async def chat(self, messages):
        return self._answer


# ==================== agent_eval_runs 落库 ====================


async def ensure_agent_eval_runs_table() -> None:
    """幂等创建 agent_eval_runs 表（与 eval_runs 同款拆分执行模式）"""
    statements = [s.strip() for s in AGENT_EVAL_RUNS_DDL.split(";") if s.strip()]
    async with async_session_factory() as session:
        for stmt in statements:
            await session.execute(text(stmt))
        await session.commit()


async def save_agent_eval_run(git_commit: str, config_snapshot: dict,
                              scores: dict, per_question: list[dict]) -> int:
    """记录一次 Agent 评估运行到 agent_eval_runs 表（失败返回 0，不中断）"""
    try:
        await ensure_agent_eval_runs_table()
        async with async_session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO agent_eval_runs
                        (eval_type, git_commit, config_snapshot, scores, per_question)
                    VALUES ('agent_eval', :git_commit,
                            CAST(:config_snapshot AS jsonb),
                            CAST(:scores AS jsonb),
                            CAST(:per_question AS jsonb))
                    RETURNING id
                """),
                {
                    "git_commit": git_commit,
                    "config_snapshot": json.dumps(config_snapshot, ensure_ascii=False),
                    "scores": json.dumps(scores, ensure_ascii=False),
                    "per_question": json.dumps(per_question, ensure_ascii=False),
                },
            )
            await session.commit()
            row = result.fetchone()
            return int(row[0]) if row else 0
    except Exception as e:
        logger.warning("记录 agent_eval_runs 失败（评估结果仍有效）: %s", e)
        return 0


async def _cleanup_eval_memory() -> None:
    """清理评测身份的记忆残留（react_loop 直连不写记忆，防御性清理）"""
    try:
        async with async_session_factory() as session:
            await session.execute(
                text("DELETE FROM documents WHERE source LIKE :p"),
                {"p": f"memory:{EVAL_IDENTITY}:%"})
            await session.commit()
    except Exception as e:
        logger.warning("评测记忆清理失败（fail-open）: %s", e)


# ==================== 报告 ====================


def print_report(scores: dict, per_question: list[dict], saved_id: int,
                 commit: str) -> None:
    """打印三层指标报告 + 失败案例分类（不隐藏）"""
    print("\n" + "=" * 64)
    title = "Agent Tasks Eval"
    if scores.get("fixture"):
        title += "  [fixture 模式：假 LLM 回放，非真实指标；仅演示管线]"
    print(title)
    print("=" * 64)
    print(f"Dataset: {scores['dataset_size']} tasks | Mode: {scores['mode']}"
          f" | pass_k: {scores['pass_k']} | Evaluated: {scores['count']}")
    print("-" * 64)
    print(f"[Outcome]    pass^{scores['pass_k']}: {scores['pass_1']:.4f}")
    for p, v in sorted(scores.get("per_path", {}).items()):
        print(f"             {p:<18} n={v['count']:<3} pass_rate={v['pass_rate']:.4f}")
    if scores.get("trajectory") == "有轨迹":
        print(f"[Trajectory] 工具正确率: {scores['tool_correct_rate']:.4f}"
              f" | 无多调率: {scores['no_extra_rate']:.4f}"
              f" | 参数正确率: {scores['args_rate']:.4f}")
        g = scores.get("grounding")
        print(f"             Grounding: {f'{g:.4f}' if g is not None else '无数据（tool_call_logs 不可读，如实标注）'}")
    else:
        print(f"[Trajectory] {scores['trajectory']}")
    print(f"[System]     平均步数: {scores['avg_tool_count']} | 平均 token: "
          f"{scores['avg_tokens'] if scores['avg_tokens'] is not None else 'N/A'}"
          f" | 耗时 P50/P95: {scores['p50_ms']}/{scores['p95_ms']} ms")
    print("-" * 64)
    failed = [t for t in per_question if not t["pass"]]
    if failed:
        print(f"失败案例分类（{len(failed)} 个，不隐藏）：")
        counts: dict[str, int] = {}
        for t in failed:
            cat = classify_failure(t)
            counts[cat] = counts.get(cat, 0) + 1
            reason = f" [{t['fail_reason']}]" if t.get("fail_reason") else ""
            print(f"  {cat:<8} {t['task_id']} tools={t['actual_names']}"
                  f" expect={t['expected_tools']}{reason}")
        print("  —— 分类计数:", counts)
    else:
        print("失败案例：无")
    print("=" * 64)
    if saved_id:
        print(f"Saved to agent_eval_runs (id={saved_id}, commit={commit[:8]})")
    else:
        print("Not saved to agent_eval_runs")
    print()


# ==================== CLI ====================


async def main() -> None:
    """评测脚本入口"""
    parser = argparse.ArgumentParser(description="Agent 任务级评估：三层指标 + 版本化落库")
    parser.add_argument("--mode", choices=["chat", "agent"], default="agent",
                        help="运行模式：agent=ReAct 循环（三层指标）；chat=engine 流水线（无轨迹，Trajectory 占位）")
    parser.add_argument("--sample", type=int, default=0,
                        help="抽样条数（0=全量；固定种子 42 可复现）")
    parser.add_argument("--pass_k", type=int, default=1,
                        help="每任务独立尝试次数，k 次全成功才算过（默认 1）")
    parser.add_argument("--limit", type=int, default=0,
                        help="冒烟：只跑前 N 条")
    parser.add_argument("--no-save", action="store_true", help="不记录 agent_eval_runs 表")
    parser.add_argument("--fixture", action="store_true",
                        help="fixture 模式：假 LLM 回放（零 LLM/DB，仅演示管线，不落库）")
    args = parser.parse_args()

    pass_k = max(1, int(args.pass_k or 1))
    tasks = load_agent_tasks()  # 先校验任务集（结构非法立即报错退出）
    if args.limit:
        tasks = tasks[:max(0, args.limit)]
    if args.sample:
        tasks = random.Random(42).sample(tasks, min(args.sample, len(tasks)))
        tasks.sort(key=lambda t: t["id"])  # 固定顺序输出

    logger.info("任务集 %d 条，路径覆盖: %s", len(tasks), path_coverage(tasks))
    if args.fixture:
        settings.tool_call_logs_enabled = False  # fixture 零 DB，不落 tool_call_logs
    elif args.mode == "agent":
        # 真实 agent 模式依赖 tool_call_logs（Grounding 数据来源）：脚本独立
        # 运行（无服务端 init_db）时先幂等建表，防首任务落库失败（fail-open）
        try:
            from src.database import ensure_tool_call_logs_table
            await ensure_tool_call_logs_table()
        except Exception as e:
            logger.warning("tool_call_logs 建表失败（grounding 将标 None）: %s", e)

    per_question, scores = await run_eval(tasks, args.mode, pass_k, args.fixture)

    saved_id = 0
    commit = ""
    if args.fixture:
        print("[fixture] 强制跳过 agent_eval_runs 落库（fixture 模式不依赖 DB）")
    elif not args.no_save:
        from eval.golden.golden_retrieval import get_git_commit, load_rag_config
        commit = get_git_commit()
        config_snapshot = await load_rag_config()
        saved_id = await save_agent_eval_run(commit, config_snapshot,
                                             scores, per_question)
    if args.mode == "agent" and not args.fixture:
        await _cleanup_eval_memory()  # 测后清理评测身份记忆残留（防御性）
    print_report(scores, per_question, saved_id, commit)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
