"""
Agent ReAct 循环 — 工具编排核心（module-028）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

把固定流水线（意图路由→检索→反思→生成）升级为 Agentic ReAct 循环：
LLM 自己决定调用什么工具、以什么顺序，直到信息足够直接回答，或达到
工具总调用次数预算。

核心流程：
  while 未回答 and 工具调用数 < budget:
    LLM 调用（含已收集工具结果作上下文）
    if tool_calls:
      逐个执行 → 结果追加到消息 → 工具数 +1
    else:
      LLM 直接输出答案 → 结束
  budget 耗尽 → 用已收集 docs 兜底生成（reflector.generate_answer）

设计要点：
  1. react_loop 是异步生成器，逐事件产出（tool_call/tool_result/token/done），
     react_agent（非流式，供测试/调用）与 main.py 的 SSE 端点共用同一核心，
     避免逻辑重复。
  2. 工具结果作为上下文追加到 messages（OpenAI dict 格式：assistant 消息带
     原样 tool_calls + reasoning_content，后接逐条 tool 结果消息），
     LLM 每一轮都能看到历史工具结果。
  3. 工具总次数预算（非单工具次数）防空转烧钱；预算=0 时 LLM 不带工具直接回答。
  4. 消息用 OpenAI dict 格式；assistant 消息保留 reasoning_content
     （deepseek thinking 模式回传要求）并只含实际执行的 tool_calls，
     避免预算截断时出现无对应 tool 结果的孤立声明。
"""
import json
import logging
import time
from typing import AsyncGenerator, Optional

from src.config import settings
from src.observability import get_trace_id
from llm.client import LLMFactory
from agent.reflector import reflector
from agent.tool_registry import ToolRegistry, registry

logger = logging.getLogger(__name__)

# ReAct 系统提示词：指导 LLM 自主决定工具调用顺序
_SYSTEM_PROMPT = """你是知识库问答系统的 Agentic RAG 问答助手。用户的问题需要检索知识库来回答，
你可以通过 function calling 调用工具，自主决定调用哪些工具、以什么顺序，直到信息足够回答问题。

可用工具：
- search_knowledge: 混合检索（关键词 + 语义向量），推荐首选
- search_fts: 精确关键词全文检索（适合专有名词、代码、精确术语）
- search_vector: 语义向量检索（适合概念性、同义表述查询）
- search_graph: 知识图谱检索（实体关系图遍历）
- extract_entities: 从查询/文本中提取技术实体
- recall_memory: 召回该用户的跨会话长期记忆
- generate_answer: 基于已检索到的全部文档生成带引用标注的最终答案
- verify_answer: 逐句验证已生成答案是否被检索文档支持，标注可信度
- re_search: 检索不足时自动改写查询重检
- note_to_self: 记录中间发现或推理结论到工作笔记，后续轮次可参考

使用规则：
1. 优先用 search_knowledge 做一次检索；结果不足时再换 search_fts / search_vector /
   search_graph，或改用更精确的查询词重试
2. 检索工具会自动累积已检索文档；信息足够后调用 generate_answer 生成带引用答案，
   或直接输出最终答案
3. 工具返回空结果不代表出错，可能是知识库无相关内容，请判断是继续检索还是如实告知用户
4. 用中文回答，严格基于检索到的文档内容，禁止编造
5. 检索结果与问题不相关时，调用 re_search 自动改写查询重检，
   无需手动换 search_fts/search_vector（与 engine 流水线的自动反思对齐）"""


class ReactContext:
    """ReAct 循环的会话上下文（每请求独立，多会话并发安全）

    Attributes:
        query: 用户当前问题
        identity: 请求身份标识（user_id 优先，否则 client_ip；记忆按身份隔离，
            module-032/036 语义，原名 client_ip 已过时）
        history: 历史对话列表 [{"role", "content"}, ...]
        docs: 检索工具累积的文档（按 doc id 去重）
        memory: recall_memory 工具召回的记忆文本（供 generate_answer 使用）
        scratchpad: note_to_self 工具记录的工作笔记列表，按写入序（module-041）
        phase: 工具执行阶段（module-058 / ADR-0012 方案 A）——初始 "retrieval"；
            本轮调用过 generate_answer/verify_answer → 下一轮切 "generation"
        executed_fingerprints: 幂等指纹集合（module-083 WP-B，每请求独立、跨请求
            不共享；同参只读检索二次调用拦截）
    """

    def __init__(self, query: str, identity: str = "unknown",
                 history: Optional[list[dict]] = None):
        self.query = query
        self.identity = identity
        self.history = history or []
        self.docs: list[dict] = []
        self._seen_ids: set = set()
        self.memory = ""
        self.scratchpad: list[str] = []  # module-041: Agent 工作笔记，按写入序
        self.phase: str = "retrieval"    # module-058: 工具执行阶段状态机
        self.retrieval_rounds: int = 0   # module-068: 检索阶段未切换轮次计数（防空转兜底）
        self.phase_count: dict[str, int] = {"retrieval": 0, "generation": 0}  # module-068: 各阶段实际执行工具数
        self.last_research_query: str = ""  # module-073: re_search 最近一次改写 query（同改写守卫，防 LLM 空转）
        self.executed_fingerprints: set[str] = set()  # module-083（WP-B）：幂等指纹集合（每请求独立，同参只读检索二次调用拦截）
    def add_note(self, note: str) -> bool:
        """记录一条工作笔记到 scratchpad（module-041/073）

        module-073：完全一致去重（strip 后逐字比较，不做近似去重——scratchpad
        重复来自 LLM 同参数机械重复调用，措辞变体是正常产出不应拦截），重复
        已存在时不再追加。

        Args:
            note: 笔记内容（调用方已负责截断，此处仅 strip）

        Returns:
            True=新增成功；False=内容与既有笔记完全一致，未重复记录
        """
        note = note.strip()
        if note in self.scratchpad:
            return False
        self.scratchpad.append(note)
        return True

    def add_docs(self, docs: list[dict]) -> None:
        """按 doc id 去重累积检索文档（供 generate_answer / 兜底生成使用）"""
        for d in docs or []:
            did = d.get("id")
            if did is not None and did not in self._seen_ids:
                self.docs.append(d)
                self._seen_ids.add(did)


def _build_messages(ctx: ReactContext) -> list:
    """构造 ReAct 会话消息（OpenAI dict 格式）：system + history + 当前问题"""
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(ctx.history or [])
    messages.append({"role": "user", "content": ctx.query})
    return messages


def _assistant_message(response: dict, executed_ids: set) -> dict:
    """构造 assistant 工具调用消息（保留 reasoning_content + 原样 tool_calls）

    DeepSeek thinking 模式要求把 reasoning_content 原样回传（否则 400），
    tool_calls 用模型原始 arguments 字符串（不重新序列化，保持格式一致）。
    只保留本轮实际执行的 tool_calls（预算截断时避免出现无对应 tool 结果）。
    """
    raw = response.get("message") or {}
    msg = {"role": "assistant", "content": raw.get("content") or ""}
    reasoning = raw.get("reasoning_content")
    if reasoning:
        msg["reasoning_content"] = reasoning
    calls = [c for c in (raw.get("tool_calls") or []) if c.get("id") in executed_ids]
    if calls:
        msg["tool_calls"] = calls
    return msg


# ─── 工具阶段切分公共辅助（module-058 / ADR-0012 方案 A，两条循环共用） ───
# 阶段判定：以"是否已调用过 generate_answer/verify_answer"为界（非 docs
# 非空——后者会切断"生成后发现不足→再补检"能力）；generation 内调 re_search
# 不回退（单向前进，防死循环）。归组见 tool_registry.register_builtin_tools。
_GENERATION_GATE_TOOLS = {"generate_answer", "verify_answer"}

# module-068：检索命中即切 generation 的命中工具清单（task-brief 6 个，
# 不含 re_search——双组补检工具，命中判定排除；零 LLM 判断，066 已证 LLM
# 行为性不可靠）。推进规则见 advance_phase docstring。
_RETRIEVAL_HIT_TOOLS = {
    "search_knowledge", "search_fts", "search_vector", "search_graph",
    "extract_entities", "recall_memory",
}
# 空结果标记（与 tool_registry.py 文案耦合——红线不碰 tool_registry，若未来
# 改文案此处判定失效，解耦方案见 changelog backlog）
_EMPTY_RESULT_MARKERS = ("（无检索结果）", "（无相关历史记忆）")


def _retrieval_hit(name: str, result: str) -> bool:
    """检索命中判定（module-068，确定性零 LLM 判断）

    规则：工具名 ∈ 检索命中集合 + 结果非空 + 非空结果标记（"（无检索结果）"/
    "（无相关历史记忆）" 均为非空字符串，bool(result) 会误判命中）+ extract_
    entities 解析 JSON 判 entities 非空（解析失败/无 entities 键按非空文本判定）。

    Args:
        name: 工具名
        result: 工具执行返回的结果文本

    Returns:
        是否命中（命中 → 下一轮切 generation）
    """
    if name not in _RETRIEVAL_HIT_TOOLS:
        return False
    if not result:
        return False
    if result in _EMPTY_RESULT_MARKERS:
        return False
    if name == "extract_entities":
        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            data = None
        if isinstance(data, dict) and "entities" in data:
            return bool(data["entities"])
    return True


def _phase_budget(phase: str) -> int:
    """阶段预算（module-068 WP-B）：检索 ≤ agent_retrieval_budget / 生成 ≤ agent_generation_budget"""
    if phase == "generation":
        return settings.agent_generation_budget
    return settings.agent_retrieval_budget


def schemas_for_phase(tools: ToolRegistry, ctx: ReactContext) -> list[dict]:
    """按当前阶段选工具 schema（开关 false → 全量 10 个，零回归逃生口）

    两条 ReAct 循环（react_loop + langgraph_react_loop）共用本函数，只改
    一处 = 回归（防两处漂移）。
    """
    if settings.tool_phase_split:
        return tools.to_llm_schemas(group=ctx.phase)
    return tools.to_llm_schemas()


def advance_phase(ctx: ReactContext, executed_names: list[str],
                  executed_results: Optional[list[str]] = None) -> None:
    """本轮触发阶段推进 → 下一轮切 generation（单向前进）

    module-068 扩展：原条件（本轮调用过生成工具）保留；新增确定性分支——
    任一检索命中工具本轮返回非空真实结果 → 切 generation（零 LLM 判断，打破
    "检索阶段 schema 无生成工具 → LLM 无法调 generate_answer → 永不切
    generation"死锁）。防空转兜底：检索阶段轮次 ≥ agent_retrieval_max_rounds
    且始终未命中 → 强制切 generation（阈值判定在本轮未因其他条件切换之后）。

    executed_results 缺省 None 时行为 = 旧逻辑（仅生成工具判定）——存量
    test_advance_phase_unit 单列表调用零改动（向后兼容红线）。

    Args:
        ctx: 会话上下文（phase 原地更新，跨轮次/跨节点可见）
        executed_names: 本轮实际执行的工具名列表（含预算截断后实际执行者）
        executed_results: 与 executed_names 同序的结果文本列表；None = 旧行为
    """
    if ctx.phase != "retrieval":
        return
    if any(n in _GENERATION_GATE_TOOLS for n in executed_names):
        ctx.phase = "generation"
        return
    if executed_results is not None and any(
            _retrieval_hit(n, r)
            for n, r in zip(executed_names, executed_results)):
        ctx.phase = "generation"
        return
    ctx.retrieval_rounds += 1
    if ctx.retrieval_rounds >= settings.agent_retrieval_max_rounds:
        ctx.phase = "generation"


# ─── 工具执行 + tool_call_logs 落库（module-066 / ADR-0017 决策 2） ───
# 两条 ReAct 循环（react_loop + langgraph_react_loop）共用本辅助，只改一处
# = 回归（防两处漂移，对齐 schemas_for_phase 模式）。落库语义：
#   - 只记录实际执行的 tool_calls（预算截断掉的 LLM 提议不记，无对应结果）
#   - 工具不存在/run 抛出异常 → result_ok=false；AgentTool.run 返回空串属
#     正常路径（run 内部捕获失败），result_ok=true
#   - 落库失败 fail-open（不阻断工具执行循环，对齐 save_request_log 哲学）
#   - 开关 tool_call_logs_enabled=false 时零开销跳过（不构造记录）


async def record_tool_call(name: str, args: dict, result_ok: bool,
                           result: str, duration_ms: int) -> None:
    """落库 tool_call_logs 一行（fail-open：失败仅日志告警，不阻断循环）

    建表走 init_db 自愈幂等 DDL（ensure_tool_call_logs_table），本函数不建表；
    trace_id 从观测上下文读取（module-058 contextvar，无请求上下文时为空串）。

    Args:
        name: 工具名
        args: 工具参数（非 JSON 序列化时兜底 {}）
        result_ok: 执行成功标记（工具不存在/异常才 false）
        result: 工具结果文本（截断 200 字符）
        duration_ms: 单次工具执行耗时（毫秒）
    """
    if not settings.tool_call_logs_enabled:
        return
    try:
        from sqlalchemy import text
        from src.database import async_session_factory

        try:
            args_json = json.dumps(args, ensure_ascii=False)
        except TypeError:  # 防御：个别供应商传入非 JSON 序列化参数 → 兜底 {}
            args_json = "{}"
        async with async_session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO tool_call_logs
                        (trace_id, tool_name, args, result_ok,
                         result_preview, duration_ms)
                    VALUES (:trace_id, :tool_name, CAST(:args AS jsonb),
                            :result_ok, :result_preview, :duration_ms)
                """),
                {
                    "trace_id": get_trace_id() or "",
                    "tool_name": name,
                    "args": args_json,
                    "result_ok": result_ok,
                    "result_preview": (result or "")[:200],
                    "duration_ms": int(duration_ms or 0),
                },
            )
            await session.commit()
    except Exception as e:
        logger.warning("tool_call_logs 落库失败（fail-open，不影响工具执行）: %s", e)


def _phase_allows(name: str, ctx: ReactContext) -> bool:
    """执行层 schema 守门：工具名是否在当前阶段允许集合内（2026-08-20）

    066 Tester 实测发现"执行层不校验 schema 暴露"：LLM 可强行调用 schema 外
    工具（at-002 强行调 generate_answer 致 15s 超时）。语义闭环——系统提示词
    列全 10 工具是"手册"、tools 参数是"门禁"、本函数是执行层"守门"：
    schema 外调用拒绝执行并返回可读提示（喂回 LLM 判断），不再真执行。

    Args:
        name: 工具名
        ctx: ReAct 会话上下文（取 phase）

    Returns:
        True 允许执行；False 当前阶段不可用（tool_phase_split=false → 全放行
        零回归）
    """
    if not settings.tool_phase_split:
        return True
    allowed = {s.get("function", {}).get("name") for s in schemas_for_phase(registry, ctx)}
    return name in allowed


async def execute_tool_with_log(name: str, args: dict, tool,
                                ctx: ReactContext,
                                allowed_tools: Optional[set[str]] = None) -> str:
    """执行单个工具并落库 tool_call_logs（module-066 / ADR-0017 决策 2）

    计时包住 tool.run，result_ok 语义：工具不存在/run 抛出异常才 false
    （AgentTool.run 内部捕获失败返回空串属正常路径，result_ok=true）。
    执行层二维守门（module-083 WP-E + 066 实测补齐）：工具存在但不在当前
    阶段允许集合（_phase_allows）或不在 Agent 权限白名单（allowed_tools）→
    拒绝执行，返回可读提示，result_ok=false（审计可见越权尝试），喂回 LLM
    判断——闭环 066 实测"执行层不校验 schema"漏洞。两维独立判因：阶段粒度
    （058/ADR-0012）与 Agent 粒度（083 WP-E，None=全量放行向后兼容）。

    Args:
        name: 工具名
        args: 工具参数
        tool: 工具实例（tools.get 未命中为 None）
        ctx: ReAct 会话上下文
        allowed_tools: Agent 权限白名单（module-083 WP-E）；None = 全量放行
    Returns:
        工具结果文本（与旧 `"" if tool is None else await tool.run(...)` 等价）
    """
    started = time.perf_counter()
    result_ok = tool is not None
    result = ""
    if tool is not None and (not _phase_allows(name, ctx)
                             or (allowed_tools is not None and name not in allowed_tools)):
        result_ok = False
        if allowed_tools is not None and name not in allowed_tools:
            result = f"（工具 {name} 不在当前 Agent 权限白名单，请按可用工具选择）"
        else:
            result = f"（工具 {name} 当前阶段不可用，请按可用工具列表选择）"
        logger.warning("工具 %s 被权限/阶段守门拒绝（allowed=%s phase=%s）",
                       name, allowed_tools, ctx.phase)
    elif tool is not None:
        try:
            result = await tool.run(args, ctx)
        except Exception as e:
            result_ok = False
            logger.warning("工具 %s 执行异常（tool_call_logs result_ok=false）: %s",
                           name, e)
    duration_ms = int((time.perf_counter() - started) * 1000)
    await record_tool_call(name, args, result_ok, result, duration_ms)
    return result


async def react_agent(
    query: str,
    history: Optional[list[dict]] = None,
    identity: str = "unknown",
    budget: Optional[int] = None,
    tools: Optional[ToolRegistry] = None,
    allowed_tools: Optional[set[str]] = None,
) -> dict:
    """ReAct 循环（非流式）：自主调用工具直到可回答或达预算上限

    Args:
        query: 用户问题
        history: 历史对话列表
        identity: 请求身份标识（user_id 优先，否则 client_ip；记忆按身份隔离）
        budget: 工具总调用次数上限，None 用 settings.max_agent_tools
        tools: 工具注册表，默认全局 registry
        allowed_tools: 允许执行的工具名集合（module-083 WP-E Agent 级最小权限）；
            None = 全量放行（向后兼容）

    Returns:
        {"answer": str, "tool_count": int,
         "tool_trace": [{"name", "args", "result"}, ...]}
    """
    ctx = ReactContext(query, identity, history)
    budget = int(budget) if budget is not None else settings.max_agent_tools
    answer = ""
    tool_count = 0
    tool_trace: list[dict] = []

    async for evt in react_loop(ctx, _build_messages(ctx), budget, tools,
                                 allowed_tools=allowed_tools):
        t = evt["type"]
        if t == "tool_call":
            tool_count = evt["tool_count"]
            tool_trace.append({"name": evt["name"], "args": evt["args"]})
        elif t == "tool_result":
            if tool_trace:
                tool_trace[-1]["result"] = evt["result"][:200]
        elif t == "done":
            answer = evt.get("answer", "")
            tool_count = evt.get("tool_count", tool_count)
            break

    return {"answer": answer, "tool_count": tool_count, "tool_trace": tool_trace}


async def react_loop(
    ctx: ReactContext,
    messages: list,
    budget: int,
    tools: Optional[ToolRegistry] = None,
    allowed_tools: Optional[set[str]] = None,
    max_answer_len: int = 0,
) -> AsyncGenerator[dict, None]:
    """ReAct 循环核心（异步生成器，逐事件产出，供 react_agent 与 SSE 端点复用）

    Args:
        ctx: 会话上下文（检索工具累积 docs 到 ctx）
        messages: 会话消息（system + history + 当前问题，会追加工具结果）
        budget: 工具总调用次数上限（≥0）
        tools: 工具注册表，默认全局 registry
        allowed_tools: 允许执行的工具名集合（module-083 WP-E，None=全量放行）；
            经 execute_tool_with_log 执行层二维守门生效
        max_answer_len: 答案最大长度（0=不限制），超出截断并附加标记

    Yields 事件:
      {"type": "tool_call",   "name": str, "args": dict, "tool_count": int}
      {"type": "tool_result", "name": str, "args": dict, "result": str,
       "tool_count": int}
      {"type": "token", "content": str}             # 推理/回答文本片段
      {"type": "done", "answer": str, "tool_count": int}

    Raises:
        LLMException: 降级链所有供应商均失败（LLM 调用层面）
    """
    tools = tools or registry
    client = LLMFactory.get_client()
    budget = int(budget or 0)
    max_answer_len = int(max_answer_len or 0)
    tool_count = 0

    # 预算=0：不调用工具，LLM 直接回答（验收 §1.2「预算=0：直接生成」）
    if budget <= 0:
        answer = await client.chat(messages)
        if max_answer_len and len(answer) > max_answer_len:
            answer = answer[:max_answer_len] + "\n\n[答案过长，已截断]"
        yield {"type": "done", "answer": answer, "tool_count": 0}
        return

    while tool_count < budget:
        # module-058（ADR-0012 方案 A）：按 ctx.phase 阶段选工具 schema
        #（检索阶段 7 个 / 生成阶段 4 个；开关 false → 全量，零回归）
        response = await client.chat_with_tools(messages, schemas_for_phase(tools, ctx))
        tool_calls = response.get("tool_calls", []) or []
        content = response.get("content", "") or ""

        # 无 tool_call：LLM 认为信息足够，直接输出答案
        if not tool_calls:
            if content:
                if max_answer_len and len(content) > max_answer_len:
                    content = content[:max_answer_len] + "\n\n[答案过长，已截断]"
                yield {"type": "token", "content": content}
            yield {"type": "done", "answer": content, "tool_count": tool_count}
            return

        # 本轮 LLM 的推理文本（非最终答案），透传给前端观察进度
        if content:
            yield {"type": "token", "content": content}

        # 预算内本轮可执行的工具数（预算截断时只执行前 N 个；module-068：
        # 总预算与阶段预算取 min——阶段预算仅 tool_phase_split=true 生效，
        # false 回退纯总预算存量行为逐字）
        total_remaining = max(0, budget - tool_count)
        if settings.tool_phase_split:
            phase_remaining = max(
                0, _phase_budget(ctx.phase) - ctx.phase_count[ctx.phase])
            allowed = tool_calls[: min(total_remaining, phase_remaining)]
        else:
            allowed = tool_calls[: total_remaining]
        if not allowed:
            break  # 预算已满，无可用额度 → 兜底生成

        # 先追加 assistant 消息（保留 reasoning_content + 仅含实际执行的 tool_calls），
        # 再逐个执行并追加 tool 结果消息（OpenAI 要求 assistant 在前、tool 结果在后）
        executed_ids = {tc.get("id", "") for tc in allowed}
        messages.append(_assistant_message(response, executed_ids))

        executed_names: list[str] = []
        executed_results: list[str] = []
        for tc in allowed:
            name = tc.get("name", "")
            executed_names.append(name)
            args = tc.get("args") or {}
            if isinstance(args, str):  # 防御：个别供应商返回未解析的 JSON 字符串
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            tool = tools.get(name)
            tool_count += 1
            ctx.phase_count[ctx.phase] += 1  # module-068: 按执行时阶段计数
            yield {"type": "tool_call", "name": name, "args": args,
                   "tool_count": tool_count}
            # module-066（ADR-0017）：执行工具并落库 tool_call_logs（计时包住
            # run；工具失败时 AgentTool.run 内部返回空结果，LLM 判断继续/放弃）
            result = await execute_tool_with_log(name, args, tool, ctx,
                                                 allowed_tools=allowed_tools)
            executed_results.append(result)
            yield {"type": "tool_result", "name": name, "args": args,
                   "result": result, "tool_count": tool_count}
            # 工具结果追加到消息历史（LLM 下一轮能看到）
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                             "content": result})
        # 本轮触发阶段推进（生成工具 / 检索命中 / 防空转兜底）→ 下一轮切
        # generation（单向前进）
        advance_phase(ctx, executed_names, executed_results)

    # 预算耗尽：用已收集 docs 兜底生成
    logger.warning("工具预算耗尽 (budget=%d)，用 %d 篇已收集文档兜底生成",
                   budget, len(ctx.docs))
    answer = await reflector.generate_answer(
        ctx.query, ctx.docs, history=ctx.history, memory=ctx.memory,
        scratchpad=ctx.scratchpad,
    )
    if max_answer_len and len(answer) > max_answer_len:
        answer = answer[:max_answer_len] + "\n\n[答案过长，已截断]"
    if answer:
        yield {"type": "token", "content": answer}
    yield {"type": "done", "answer": answer, "tool_count": tool_count}
