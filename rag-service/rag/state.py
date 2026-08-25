"""
RAG 状态定义 — LangGraph State
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

定义在整个 RAG 图（graph）中流转的状态数据结构。
每个节点读取需要的字段，写入产出字段。

使用 TypedDict 而非 dataclass，因为 LangGraph 对 TypedDict
的序列化/反序列化支持更好（便于 checkpoint 和调试）。
"""
from typing import TypedDict, Optional


class RAGState(TypedDict):
    """RAG 图状态

    字段按职责分组：
    - 输入: query, history
    - 中间: intent, intent_result, docs, check
    - 步骤数据: steps_data（收集各步骤的统计信息）
    - 输出: answer, sources
    """
    # === 输入 ===
    query: str
    history: list[dict]
    tool_history: Optional[list]      # 工具轨迹信号（module-072 WP-B：agent 端点
                                      # 持久化轨迹，可选；未设置时 classify 传 None）

    # === 意图 ===
    intent: str                    # knowledge | casual_chat | realtime
    intent_result: dict            # 意图识别的完整结果

    # === 检索 ===
    docs: list[dict]              # 检索 + rerank 后的文档列表

    # === 反思 ===
    check: dict                   # check_sufficiency 的结果

    # === 步骤统计（供前端 Steps 面板展示） ===
    steps_data: dict

    # === 输出 ===
    answer: str
    sources: list[dict]


def make_initial_state(query: str, history: list[dict]) -> RAGState:
    """创建初始状态"""
    return RAGState(
        query=query,
        history=history or [],
        intent="knowledge",
        intent_result={},
        docs=[],
        check={},
        steps_data={
            "intent": {},
            "retrieval": {},
            "rerank": {},
            "reflection": {},
        },
        answer="",
        sources=[],
    )
