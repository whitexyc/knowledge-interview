"""
LangGraph 版 ReAct 循环 — 实验端点（module-030）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

现有 ReAct 是手写 while 循环（agent/react.py，工作正常，生产主路径）。
本文件新增 LangGraph StateGraph 版并存（实验端点 /ai/rag/chat/agent-lg），
行为与手写版对齐（预算/工具/上下文），不动手写循环（零回归）。

图结构：
  START → [llm_call]
              ├─ 有 tool_calls → [execute_tools]
              │                    ├─ 工具数 < budget → 回到 [llm_call]
              │                    └─ 预算耗尽 → [fallback] → END
              └─ 无 tool_calls → [finalize] → END

设计要点：
  1. 复用 agent.react 的 ReactContext / _build_messages / _assistant_message
     与 agent.tool_registry 的 ToolRegistry（不重复实现工具逻辑）。
  2. 节点通过 ReActGraphState 传递状态；节点把 SSE 事件追加到 state["events"]
     （事件顺序 token → tool_call/tool_result → done，与手写 react_loop 对齐），
     ainvoke 结束后由 langgraph_react_loop 一次性产出。
  3. 条件路由：llm_call → 有 tool_call → execute_tools，无 → finalize；
     execute_tools → 工具数 < budget 继续 llm_call，否则 fallback 兜底。
  4. assistant 工具调用消息只含实际执行的 tool_calls（预算截断时避免无对应
     tool 结果的孤立声明），并保留 reasoning_content（deepseek thinking 回传）。
"""
import json
import logging
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.config import settings
from llm.client import LLMFactory
from agent.react import (
    ReactContext, _assistant_message, _build_messages,
    advance_phase, execute_tool_with_log, schemas_for_phase, _phase_budget,
)
from agent.reflector import reflector
from agent.tool_registry import ToolRegistry, registry

logger = logging.getLogger(__name__)


class ReActGraphState(TypedDict):
    """LangGraph 版 ReAct 图状态（节点间传递）

    Fields:
        ctx: ReAct 会话上下文（检索工具累积 docs / 记忆，跨节点复用）
        messages: OpenAI dict 格式会话消息（含工具结果，逐轮追加）
        budget: 工具总调用次数上限
        tool_count: 已执行的工具调用次数
        events: SSE 事件收集器（token/tool_call/tool_result/done）
        tools: 工具注册表（默认全局 registry）
        response: llm_call 节点的 LLM 工具调用响应
        answer: 最终答案（finalize/fallback 产出）
        max_answer_len: 答案最大长度（0=不限制），超出截断并附加标记
        phase_exhausted: 阶段额度耗尽标记（module-068：工具数仍 < 总预算但
            当前阶段预算已满 → 路由走 fallback，防回 llm_call 死循环）
    """
    ctx: ReactContext
    messages: list
    budget: int
    tool_count: int
    events: list
    tools: ToolRegistry
    response: dict
    answer: str
    max_answer_len: int
    phase_exhausted: bool


# ==================== Node 函数 ====================


async def llm_call(state: ReActGraphState) -> dict:
    """Node: 调用 LLM（chat_with_tools），决定本轮是否调用工具

    复用 LLMFactory 的 chat_with_tools（保留 reasoning_content 回传，
    deepseek thinking 模式要求）。有推理/回答文本时先产出 token 事件
    （与手写 react_loop 顺序一致：token 在 tool_call 之前）。

    Args:
        state: 当前图状态

    Returns:
        更新的图状态（response + events）
    """
    ctx = state["ctx"]
    messages = state["messages"]
    tools = state["tools"]
    events = state["events"]

    client = LLMFactory.get_client()
    # module-058（ADR-0012 方案 A）：按 ctx.phase 阶段选工具 schema
    #（与手写 react_loop 共用 schemas_for_phase，防两处漂移）
    response = await client.chat_with_tools(messages, schemas_for_phase(tools, ctx))

    content = response.get("content", "") or ""
    if content:
        events.append({"type": "token", "content": content})

    return {"response": response, "events": events}


async def execute_tools(state: ReActGraphState) -> dict:
    """Node: 执行本轮预算内允许的工具调用，结果追加到消息历史

    与手写 react_loop 对齐：
      - 预算内可执行的工具数 = tool_calls[:budget - tool_count]
      - 先追加 assistant 消息（保留 reasoning_content + 仅含实际执行的
        tool_calls），再逐个执行并追加 tool 结果消息
      - 工具失败由 ToolRegistry.run 统一捕获返回空串，LLM 判断继续/放弃

    Args:
        state: 当前图状态

    Returns:
        更新的图状态（messages + tool_count + events）
    """
    ctx = state["ctx"]
    messages = state["messages"]
    response = state["response"]
    budget = state["budget"]
    tool_count = state["tool_count"]
    events = state["events"]
    tools = state["tools"]

    tool_calls = response.get("tool_calls", []) or []
    # 预算内本轮可执行的工具数（module-068：总预算与阶段预算取 min——阶段预算
    # 仅 tool_phase_split=true 生效，false 回退纯总预算存量行为逐字）
    total_remaining = max(0, budget - tool_count)
    if settings.tool_phase_split:
        phase_remaining = max(
            0, _phase_budget(ctx.phase) - ctx.phase_count[ctx.phase])
        allowed = tool_calls[: min(total_remaining, phase_remaining)]
    else:
        allowed = tool_calls[: total_remaining]
    if not allowed:
        # 预算/阶段额度已满：总预算耗尽由路由判断走 fallback；阶段额度耗尽
        #（工具数仍 < 总预算）需标记 phase_exhausted 防回 llm_call 死循环
        return {"tool_count": tool_count,
                "phase_exhausted": total_remaining > 0}

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
        events.append({"type": "tool_call", "name": name, "args": args,
                       "tool_count": tool_count})
        # module-066（ADR-0017）：执行工具并落库 tool_call_logs（与手写
        # react_loop 共用 execute_tool_with_log；工具失败时 run 内部返回
        # 空结果，LLM 判断继续/放弃）
        result = await execute_tool_with_log(name, args, tool, ctx)
        executed_results.append(result)
        events.append({"type": "tool_result", "name": name, "args": args,
                       "result": result, "tool_count": tool_count})
        # 工具结果追加到消息历史（LLM 下一轮能看到）
        messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                         "content": result})
    # 本轮触发阶段推进（生成工具 / 检索命中 / 防空转兜底）→ 下一轮切
    # generation（单向前进，与手写 react_loop 共用）
    advance_phase(ctx, executed_names, executed_results)

    return {"messages": messages, "tool_count": tool_count, "events": events}


async def finalize(state: ReActGraphState) -> dict:
    """Node: LLM 直接回答（无 tool_call），产出 done 事件

    Args:
        state: 当前图状态

    Returns:
        更新的图状态（answer + events）
    """
    events = state["events"]
    answer = (state.get("response") or {}).get("content", "") or ""
    max_len = state.get("max_answer_len", 0) or 0
    if max_len and len(answer) > max_len:
        answer = answer[:max_len] + "\n\n[答案过长，已截断]"
        # 同步更新 llm_call 节点追加的 token 事件，保证 token/done 内容一致
        for evt in reversed(events):
            if evt.get("type") == "token":
                evt["content"] = answer
                break
    events.append({"type": "done", "answer": answer, "tool_count": state["tool_count"]})
    return {"answer": answer, "events": events}


async def fallback(state: ReActGraphState) -> dict:
    """Node: 预算耗尽，用已收集 docs 兜底生成（与手写 react_loop 对齐）

    Args:
        state: 当前图状态

    Returns:
        更新的图状态（answer + events）
    """
    ctx = state["ctx"]
    events = state["events"]
    logger.warning("工具预算耗尽 (budget=%d)，用 %d 篇已收集文档兜底生成",
                   state["budget"], len(ctx.docs))
    answer = await reflector.generate_answer(
        ctx.query, ctx.docs, history=ctx.history, memory=ctx.memory,
        scratchpad=ctx.scratchpad,
    )
    max_len = state.get("max_answer_len", 0) or 0
    if max_len and len(answer) > max_len:
        answer = answer[:max_len] + "\n\n[答案过长，已截断]"
    if answer:
        events.append({"type": "token", "content": answer})
    events.append({"type": "done", "answer": answer, "tool_count": state["tool_count"]})
    return {"answer": answer, "events": events}


# ==================== 路由函数 ====================


def route_after_llm(state: ReActGraphState) -> str:
    """llm_call 后路由：有 tool_call → execute_tools；无 → finalize

    Args:
        state: 当前图状态

    Returns:
        目标节点名
    """
    response = state.get("response") or {}
    tool_calls = response.get("tool_calls", []) or []
    return "execute_tools" if tool_calls else "finalize"


def route_after_tools(state: ReActGraphState) -> str:
    """execute_tools 后路由：预算/阶段额度耗尽走 fallback；否则继续 llm_call

    module-068：阶段额度耗尽（phase_exhausted=true，工具数仍 < 总预算）也走
    fallback——与手写 react_loop 的 `if not allowed: break` 语义对齐（防
    回 llm_call 后 allowed 恒空死循环）。

    Args:
        state: 当前图状态

    Returns:
        目标节点名
    """
    if state.get("phase_exhausted") or state["tool_count"] >= state["budget"]:
        return "fallback"
    return "llm_call"


# ==================== 图构建 ====================


def build_react_graph():
    """构建 LangGraph 版 ReAct StateGraph

    Returns:
        编译后的图（可 ainvoke）
    """
    graph = StateGraph(ReActGraphState)

    graph.add_node("llm_call", llm_call)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("finalize", finalize)
    graph.add_node("fallback", fallback)

    graph.set_entry_point("llm_call")

    graph.add_conditional_edges(
        "llm_call",
        route_after_llm,
        {"execute_tools": "execute_tools", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {"llm_call": "llm_call", "fallback": "fallback"},
    )
    graph.add_edge("finalize", END)
    graph.add_edge("fallback", END)

    return graph.compile()


# 全局单例（实验端点共用；节点在调用时解析模块级全局，mock 可生效）
react_graph = build_react_graph()


async def langgraph_react_loop(
    ctx: ReactContext,
    messages: list,
    budget: int,
    tools: Optional[ToolRegistry] = None,
    max_answer_len: int = 0,
):
    """LangGraph 版 ReAct 循环（异步生成器，事件与 react_loop 对齐）

    通过 StateGraph 编排：LLM 自主决定工具调用顺序，直到直接回答或达预算。
    复用现有 ReactContext + ToolRegistry（不重复实现工具逻辑）。

    Args:
        ctx: 会话上下文（检索工具累积 docs 到 ctx）
        messages: 会话消息（system + history + 当前问题，会追加工具结果）
        budget: 工具总调用次数上限（≥0）
        tools: 工具注册表，默认全局 registry
        max_answer_len: 答案最大长度（0=不限制），超出截断并附加标记

    Yields 事件（与 react_loop 一致）:
      {"type": "tool_call",   "name": str, "args": dict, "tool_count": int}
      {"type": "tool_result", "name": str, "args": dict, "result": str,
       "tool_count": int}
      {"type": "token", "content": str}
      {"type": "done", "answer": str, "tool_count": int}

    Raises:
        LLMException: 降级链所有供应商均失败（LLM 调用层面）
    """
    tools = tools or registry
    budget = int(budget or 0)
    max_answer_len = int(max_answer_len or 0)

    # 预算=0：不调用工具，LLM 直接回答（验收 §1.3「LangGraph 预算=0：直接回答」）
    if budget <= 0:
        client = LLMFactory.get_client()
        answer = await client.chat(messages)
        if max_answer_len and len(answer) > max_answer_len:
            answer = answer[:max_answer_len] + "\n\n[答案过长，已截断]"
        yield {"type": "done", "answer": answer, "tool_count": 0}
        return

    initial_state: ReActGraphState = {
        "ctx": ctx,
        "messages": messages,
        "budget": budget,
        "tool_count": 0,
        "events": [],
        "tools": tools,
        "response": {},
        "answer": "",
        "max_answer_len": max_answer_len,
        "phase_exhausted": False,
    }
    # recursion_limit 覆盖默认 25：预算大时循环步数 = 2*budget + 兜底/收尾
    final_state = await react_graph.ainvoke(
        initial_state,
        config={"recursion_limit": max(50, budget * 2 + 10)},
    )
    for evt in final_state["events"]:
        yield evt


async def langgraph_react_agent(
    query: str,
    history: Optional[list[dict]] = None,
    identity: str = "unknown",
    budget: Optional[int] = None,
    tools: Optional[ToolRegistry] = None,
) -> dict:
    """LangGraph 版 ReAct（非流式）：供测试与调用方复用

    Args:
        query: 用户问题
        history: 历史对话列表
        identity: 请求身份标识（user_id 优先，否则 client_ip；记忆按身份隔离）
        budget: 工具总调用次数上限，None 用 settings.max_agent_tools
        tools: 工具注册表，默认全局 registry

    Returns:
        {"answer": str, "tool_count": int,
         "tool_trace": [{"name", "args", "result"}, ...]}
    """
    ctx = ReactContext(query, identity, history)
    budget = int(budget) if budget is not None else settings.max_agent_tools
    answer = ""
    tool_count = 0
    tool_trace: list[dict] = []

    async for evt in langgraph_react_loop(ctx, _build_messages(ctx), budget, tools):
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
