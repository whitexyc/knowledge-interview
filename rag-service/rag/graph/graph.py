"""
RAG LangGraph — Agentic 编排层
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

将手写串行 RAG 链路重构为 LangGraph StateGraph：

  输入 → [classify_intent]
          ├─ casual_chat → [直接 LLM] → END
          ├─ realtime     → END（占位）
          └─ knowledge    → [hybrid_retrieve] → [rerank_docs]
                            → [check_sufficiency]
                              ├─ 充分 → [generate_answer] → END
                              └─ 不充分 → 改写 query → 二次检索 → [generate_answer] → END

好处：
1. 步骤天然隔离，每步是独立 Node 函数
2. 状态变更显式通过 RAGState 字段传递
3. 可加条件分支（如二次检索）无需复杂 if/else
4. 中间状态可 checkpoint、可调试
"""
import logging
import time
from typing import Literal

from langgraph.graph import StateGraph, END

from rag.state import RAGState
from rag.retrieval.retriever import hybrid_retriever
from rag.retrieval.reranker import reranker
from agent.router import router_agent
from agent.reflector import reflector
from llm.client import LLMFactory

logger = logging.getLogger(__name__)

# ─── 相关度阈值（低于此值的标记为低相关度） ───
RELEVANCE_THRESHOLD = 0.5
# ─── 低分过滤阈值（低于此值的直接丢弃） ───
FILTER_THRESHOLD = 0.6
# ─── 初始检索量（取较大值，后续过滤） ───
RETRIEVE_TOP_K = 30
# ─── Rerank 后的保留量 ───
RERANK_TOP_K = 5


def _t() -> float:
    return time.monotonic()


# ==================== Node 函数 ====================


def _make_doc_preview(docs: list[dict]) -> list[dict]:
    """生成文档预览（标题 + 前 100 字摘要 + 分数）"""
    return [
        {
            "title": d.get("title", ""),
            "snippet": d.get("content", "")[:100],
            "score": round(
                d.get("hybrid_score", d.get("rerank_score", d.get("score", 0))), 4
            ),
        }
        for d in docs[:5]
    ]


def _compute_relevance(docs: list[dict]) -> dict:
    """计算相关度统计"""
    total = len(docs)
    qualified = sum(
        1 for d in docs
        if d.get("hybrid_score", d.get("rerank_score", d.get("score", 0))) >= RELEVANCE_THRESHOLD
    )
    scores = [d.get("hybrid_score", d.get("score", 0)) for d in docs]
    return {
        "qualified": qualified,
        "total": total,
        "min_score": round(min(scores), 4) if scores else 0,
        "threshold": RELEVANCE_THRESHOLD,
    }


async def classify_intent(state: RAGState) -> dict:
    """Step 1: 意图识别"""
    logger.info("Graph: classify_intent, query=%s", state["query"][:50])
    t0 = _t()
    # module-063（WP-A）：LangGraph 编排路径同步接 history（state 含 history，
    # 空 history 零回归）
    # module-072（WP-B）：classify_intent 补传 tool_history（RAGState 可选字段，
    # 未设置 → None；LangGraph 休眠管线无生产端点调用，接线为一致性 + 单测对齐）
    result = await router_agent.classify(
        state["query"], history=state.get("history") or [],
        tool_history=state.get("tool_history"))
    intent = result.get("intent", "knowledge")
    labels = {"knowledge": "知识库", "casual_chat": "闲聊", "realtime": "实时数据"}

    steps = {**state["steps_data"]}
    steps["intent"] = {
        "label": labels.get(intent, intent),
        "confidence": round(result.get("confidence", 0), 2),
        "timing_ms": int((_t() - t0) * 1000),
    }

    return {
        "intent": intent,
        "intent_result": result,
        "steps_data": steps,
    }


async def direct_llm(state: RAGState) -> dict:
    """闲聊路径：直接 LLM 回答，不走检索"""
    logger.info("Graph: direct_llm (casual_chat)")
    client = LLMFactory.get_client()
    answer = await client.chat([
        {"role": "system", "content": "你是知识库问答系统的 AI 助手，友好地回答用户的问题。"},
        *state["history"],
        {"role": "user", "content": state["query"]},
    ])
    return {"answer": answer, "sources": []}


async def hybrid_retrieve(state: RAGState) -> dict:
    """Step 2: 混合检索

    流程：
    1. 先取较大候选池（RETRIEVE_TOP_K=30）
    2. 过滤 hybrid_score < FILTER_THRESHOLD（0.3）的低分结果
    3. 即使过滤后不足 5 条，也按实际数量返回（不强制补足）
    """
    logger.info("Graph: hybrid_retrieve")
    t0 = _t()
    try:
        all_docs = await hybrid_retriever.retrieve(state["query"], top_k=RETRIEVE_TOP_K)
    except Exception as e:
        logger.warning("检索失败: %s", e)
        all_docs = []

    initial_count = len(all_docs)

    # 低分过滤：丢弃 hybrid_score < 阈值的结果
    docs = [
        d for d in all_docs
        if d.get("hybrid_score", d.get("score", 0)) >= FILTER_THRESHOLD
    ] if all_docs else []

    steps = {**state["steps_data"]}
    top_score = round(max(
        d.get("hybrid_score", d.get("score", 0)) for d in docs
    ), 4) if docs else 0.0

    steps["retrieval"] = {
        "count": len(docs),
        "initial_count": initial_count,
        "filtered_count": initial_count - len(docs),
        "top_score": top_score,
        "timing_ms": int((_t() - t0) * 1000),
        "documents_preview": _make_doc_preview(docs),
        "relevance": _compute_relevance(docs),
    }

    return {"docs": docs, "steps_data": steps}


async def rerank_docs(state: RAGState) -> dict:
    """Step 3: Rerank 精排"""
    docs = state["docs"]
    if not docs:
        return {"docs": docs}

    logger.info("Graph: rerank_docs, before=%d", len(docs))
    t0 = _t()
    rerank_before = len(docs)
    docs = await reranker.rerank(state["query"], docs, top_k=RERANK_TOP_K)

    steps = {**state["steps_data"]}
    steps["rerank"] = {
        "before": rerank_before,
        "after": len(docs),
        "timing_ms": int((_t() - t0) * 1000),
    }

    return {"docs": docs, "steps_data": steps}


async def check_sufficiency(state: RAGState) -> dict:
    """Step 4: 反思检查 + 多次检索（最多 3 次）

    流程：
    1. 初始检索（已在 Step 2 完成）
    2. 反思：检查文档是否足够
    3. 如果不充分，改写 query 再次检索（最多额外 2 次）
    4. 每次改写后合并新文档，去重后进入生成

    为什么最多 3 次？
    多次检索可以提高召回率，但每次改写都调用 LLM（耗时 ~3-8s）。
    3 次是在"召回质量"和"响应延迟"之间的平衡点。
    """
    docs = state["docs"]
    query = state["query"]
    if not docs:
        steps = {**state["steps_data"]}
        steps["reflection"] = {"sufficient": False, "query_rewritten": False, "timing_ms": 0}
        return {"check": {"sufficient": False}, "steps_data": steps}

    logger.info("Graph: check_sufficiency")
    t0 = _t()
    max_retries = 2  # 额外检索次数（初始 1 次 + 最多 2 次改写 = 最多 3 次）
    query_rewritten = False
    current_query = query

    for attempt in range(max_retries + 1):
        check = await reflector.check_sufficiency(current_query, docs)
        if check.get("sufficient", True):
            break

        rewritten = check.get("rewritten_query", current_query)
        if not rewritten or rewritten == current_query:
            break

        query_rewritten = True
        current_query = rewritten
        logger.info("Graph: 第 %d 次改写检索, rewritten=%s", attempt + 1, rewritten)
        more_docs = await hybrid_retriever.retrieve(rewritten, top_k=10)
        if more_docs:
            more_docs = await reranker.rerank(rewritten, more_docs, top_k=3)
            existing_ids = {d.get("id") for d in docs}
            for d in more_docs:
                if d.get("id") not in existing_ids:
                    docs.append(d)
                    existing_ids.add(d.get("id"))

    steps = {**state["steps_data"]}
    steps["reflection"] = {
        "sufficient": check.get("sufficient", True),
        "query_rewritten": query_rewritten,
        "rewritten_query": check.get("rewritten_query", "") if query_rewritten else "",
        "timing_ms": int((_t() - t0) * 1000),
    }

    return {"docs": docs, "check": check, "steps_data": steps}


async def generate_answer(state: RAGState) -> dict:
    """Step 5: 生成答案 + 引用溯源"""
    docs = state["docs"]
    query = state["query"]
    history = state["history"]
    logger.info("Graph: generate_answer, docs=%d", len(docs))

    if not docs:
        client = LLMFactory.get_client()
        answer = await client.generate(
            f"用户问：{query}\n\n知识库暂无相关信息，请如实告知用户。"
        )
        return {"answer": answer, "sources": []}

    answer = await reflector.generate_answer(query, docs, history=history)

    sources = []
    for i, doc in enumerate(docs[:5]):
        sources.append({
            "id": doc.get("id"),
            "title": doc.get("title", ""),
            "content": doc.get("content", "")[:300],
            "source": doc.get("source", ""),
            "ref_index": i + 1,
        })

    return {"answer": answer, "sources": sources}


# ==================== 路由函数 ====================


def intent_router(state: RAGState) -> Literal["direct_llm", "realtime", "hybrid_retrieve"]:
    """根据意图路由到不同分支"""
    intent = state.get("intent", "knowledge")
    if intent == "casual_chat":
        return "direct_llm"
    elif intent == "realtime":
        return "realtime"
    return "hybrid_retrieve"


# ==================== 图构建 ====================


def build_rag_graph() -> StateGraph:
    """构建 RAG StateGraph"""

    graph = StateGraph(RAGState)

    # 注册节点
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("direct_llm", direct_llm)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("rerank_docs", rerank_docs)
    graph.add_node("check_sufficiency", check_sufficiency)
    graph.add_node("generate_answer", generate_answer)

    # 入口
    graph.set_entry_point("classify_intent")

    # 条件路由：根据意图走不同分支
    graph.add_conditional_edges(
        "classify_intent",
        intent_router,
        {
            "direct_llm": "direct_llm",
            "realtime": END,
            "hybrid_retrieve": "hybrid_retrieve",
        },
    )

    # 知识库路径
    graph.add_edge("hybrid_retrieve", "rerank_docs")
    graph.add_edge("rerank_docs", "check_sufficiency")
    graph.add_edge("check_sufficiency", "generate_answer")
    graph.add_edge("generate_answer", END)

    # 闲聊路径
    graph.add_edge("direct_llm", END)

    return graph.compile()


# 全局单例
rag_graph = build_rag_graph()
