"""module-072 WP-B：WP-D 工具历史信号接线单元测试（resolve_tool_history + 三处调用点传参）

覆盖（验收 §1.2）：
- resolve_tool_history：SQL 形态（request_logs JOIN tool_call_logs，全参数化）/
  无记录 None / 无工具调用 None / 成功取工具名列表 / 异常 None / 超时 None /
  空 identity 直接 None（零 DB 访问）
- engine.chat：两 classify 分支收到 tool_history（mock resolve 返回值透传断言）；
  precise 短路分支不调 classify
- main.chat_stream：classify 收到 tool_history
- graph.classify_intent：state.get("tool_history") 透传；未设置 → None
- RAGState 新增可选字段（make_initial_state 不设默认零回归）

实现说明：全部 mock（零真实 DB/LLM）；同步用例内 asyncio.run（套件同款）。
"""
import asyncio
from unittest import mock

import rag.graph as graph_module
import rag.engine as engine_module
from rag import state as rag_state
from rag.engine import rag_engine
from rag.state import make_initial_state
from rag.schemas import ChatRequest

KB_HISTORY = [{"role": "user", "content": "什么是Java线程池？核心参数有哪些？"}]


class _FakeResult:
    """模拟 SQLAlchemy 结果：first() 取首行，迭代逐行（行支持下标）"""

    def __init__(self, rows):
        self._rows = rows

    def first(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


def _patch_session(results):
    """打桩 async_session_factory：每次 execute 依次返回 results 中对应结果

    Returns:
        (patcher, session_mock)：session_mock.execute 供断言（SQL/参数）
    """
    session = mock.MagicMock()
    session.execute = mock.AsyncMock(side_effect=results)
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=session)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    patcher = mock.patch("rag.engine.async_session_factory", return_value=cm)
    return patcher, session


# ─── resolve_tool_history：SQL 形态 / 无记录 / 异常 / 超时 / 空 identity ───

class TestResolveToolHistory:
    """request_logs JOIN tool_call_logs 按 identity 取最近一次 agent 轨迹"""

    def test_sql_shape_and_parameterized(self):
        """SQL 全参数化（无拼接无注入面）：两个查询的形态与绑定参数"""
        patcher, session = _patch_session([
            _FakeResult([("trace-1",)]),
            _FakeResult([("search_knowledge",), ("generate_answer",)]),
        ])
        with patcher:
            result = asyncio.run(engine_module.resolve_tool_history("u-1"))

        assert result == ["search_knowledge", "generate_answer"]
        sql1, params1 = session.execute.await_args_list[0].args
        sql2, params2 = session.execute.await_args_list[1].args
        # execute 收到 TextClause（SQLAlchemy），.text 取 SQL 字符串断言形态
        assert "request_logs" in sql1.text
        assert "identity = :identity" in sql1.text
        assert "IN ('agent', 'agent-lg')" in sql1.text
        assert "ORDER BY created_at DESC" in sql1.text
        assert "LIMIT 1" in sql1.text
        assert params1 == {"identity": "u-1"}
        assert "tool_call_logs" in sql2.text
        assert "trace_id = :trace_id" in sql2.text
        assert params2 == {"trace_id": "trace-1"}

    def test_no_agent_request_returns_none(self):
        """该 identity 无 agent 端点请求记录 → None（第二查询不执行）"""
        patcher, session = _patch_session([_FakeResult([])])
        with patcher:
            result = asyncio.run(engine_module.resolve_tool_history("u-1"))
        assert result is None
        assert session.execute.await_count == 1

    def test_request_without_tool_calls_returns_none(self):
        patcher, _session = _patch_session([
            _FakeResult([("trace-1",)]),
            _FakeResult([]),
        ])
        with patcher:
            result = asyncio.run(engine_module.resolve_tool_history("u-1"))
        assert result is None

    def test_db_error_returns_none(self):
        """DB 查询异常 → None（fail-open，不抛异常不阻塞路由）"""
        patcher, _session = _patch_session([RuntimeError("db down")])
        with patcher:
            result = asyncio.run(engine_module.resolve_tool_history("u-1"))
        assert result is None

    def test_timeout_returns_none(self):
        """查询超时（asyncio.TimeoutError，wait_for 2s 同源异常）→ None fail-open"""
        patcher, _session = _patch_session([asyncio.TimeoutError()])
        with patcher:
            result = asyncio.run(engine_module.resolve_tool_history("u-1"))
        assert result is None

    def test_empty_identity_returns_none_without_db(self):
        """空 identity → 直接 None，零 DB 访问"""
        patcher = mock.patch("rag.engine.async_session_factory")
        with patcher as factory:
            result = asyncio.run(engine_module.resolve_tool_history(""))
        assert result is None
        factory.assert_not_called()


# ─── 三处 classify 调用点传参断言 ───

class TestClassifyWiring:
    """engine.chat 两分支 / chat_stream / graph：classify 收到同一 tool_history"""

    FAKE_DOC = {"id": 1, "title": "t", "content": "c", "source": "s",
                "parent_id": None}

    @staticmethod
    def _chat_patches():
        return [
            mock.patch("rag.engine.resolve_tool_history",
                       new=mock.AsyncMock(return_value=["search_knowledge"])),
            mock.patch("rag.engine.router_agent.classify",
                       new=mock.AsyncMock(
                           return_value={"intent": "knowledge",
                                         "confidence": 0.9})),
            mock.patch("rag.engine.memory_service.recall",
                       new=mock.AsyncMock(return_value=[])),
            mock.patch("rag.engine.memory_service.recall_short",
                       new=mock.AsyncMock(return_value=[])),
            mock.patch("rag.engine.hybrid_retriever.retrieve",
                       new=mock.AsyncMock(return_value=[TestClassifyWiring.FAKE_DOC])),
            mock.patch("rag.engine.reranker.rerank",
                       new=mock.AsyncMock(side_effect=lambda q, d, top_k=5: d)),
            mock.patch("agent.reflector.reflector.check_sufficiency",
                       new=mock.AsyncMock(return_value={"sufficient": True})),
            mock.patch("agent.reflector.reflector.generate_answer",
                       new=mock.AsyncMock(return_value="答案")),
            mock.patch("agent.reflector.reflector.verify_answer",
                       new=mock.AsyncMock(return_value=None)),
            mock.patch.object(rag_engine, "_persist_memory", new=mock.AsyncMock()),
            mock.patch.object(rag_engine, "_persist_session", new=mock.AsyncMock()),
        ]

    def test_engine_chat_passes_tool_history(self):
        """engine.chat 默认分支：classify 收到 resolve 返回的工具轨迹（非 None）"""
        patches = self._chat_patches()
        classify_mock = patches[1].new

        async def run():
            for p in patches:
                p.start()
            try:
                await rag_engine.chat(ChatRequest(query="为什么", history=KB_HISTORY),
                                      identity="x")
            finally:
                for p in reversed(patches):
                    p.stop()

        asyncio.run(run())
        assert classify_mock.call_count == 1
        assert classify_mock.call_args.kwargs["tool_history"] == ["search_knowledge"]
        assert classify_mock.call_args.kwargs["history"] == KB_HISTORY

    def test_engine_chat_rewrite_branch_passes_tool_history(self, monkeypatch):
        """query_rewrite 开启（非 precise）分支：classify 同样收到工具轨迹"""
        monkeypatch.setattr(engine_module.settings, "query_rewrite_enabled", True)
        patches = self._chat_patches() + [
            mock.patch("rag.engine.query_rewrite.prepare",
                       new=mock.AsyncMock(return_value=(
                           "为什么", None, {"mode": "rewrite_fallback"}))),
        ]
        classify_mock = patches[1].new

        async def run():
            for p in patches:
                p.start()
            try:
                await rag_engine.chat(ChatRequest(query="为什么", history=KB_HISTORY),
                                      identity="x")
            finally:
                for p in reversed(patches):
                    p.stop()

        asyncio.run(run())
        assert classify_mock.call_count == 1
        assert classify_mock.call_args.kwargs["tool_history"] == ["search_knowledge"]

    def test_engine_chat_precise_shortcut_skips_classify(self, monkeypatch):
        """precise 短路分支不调 classify（省一次路由，module-063 语义不变）"""
        monkeypatch.setattr(engine_module.settings, "query_rewrite_enabled", True)
        patches = self._chat_patches() + [
            mock.patch("rag.engine.query_rewrite.prepare",
                       new=mock.AsyncMock(return_value=(
                           "G1垃圾收集器", None, {"mode": "precise"}))),
        ]
        classify_mock = patches[1].new

        async def run():
            for p in patches:
                p.start()
            try:
                await rag_engine.chat(ChatRequest(query="G1垃圾收集器"),
                                      identity="x")
            finally:
                for p in reversed(patches):
                    p.stop()

        asyncio.run(run())
        classify_mock.assert_not_called()

    def test_chat_stream_passes_tool_history(self):
        """chat_stream Step 1：classify 收到 resolve 返回的工具轨迹"""
        import main as main_module

        called = {}

        async def fake_classify(query, history=None, tool_history=None):
            called["query"] = query
            called["history"] = history
            called["tool_history"] = tool_history
            return {"intent": "casual_chat", "confidence": 0.9, "reason": "x"}

        async def run():
            with mock.patch("main.resolve_identity", return_value="u-1"):
                with mock.patch("main.persist_request_log", new=mock.AsyncMock()):
                    with mock.patch("main.resolve_tool_history",
                                    new=mock.AsyncMock(
                                        return_value=["generate_answer"])):
                        with mock.patch("agent.router.router_agent.classify",
                                        new=mock.AsyncMock(side_effect=fake_classify)):
                            resp = await main_module.chat_stream(
                                ChatRequest(query="为什么", history=KB_HISTORY), None)
                            await anext(resp.body_iterator)

        asyncio.run(run())
        assert called["tool_history"] == ["generate_answer"]
        assert called["history"] == KB_HISTORY

    def test_graph_classify_intent_passes_state_tool_history(self):
        """graph.classify_intent：state 含 tool_history → 透传；未设置 → None"""
        captured = {}

        async def fake_classify(query, history=None, tool_history=None):
            captured["tool_history"] = tool_history
            return {"intent": "knowledge", "confidence": 0.9, "reason": "x"}

        with mock.patch("agent.router.router_agent.classify",
                        new=mock.AsyncMock(side_effect=fake_classify)):
            state = make_initial_state("为什么", KB_HISTORY)
            state["tool_history"] = ["search_knowledge"]
            asyncio.run(graph_module.classify_intent(state))
        assert captured["tool_history"] == ["search_knowledge"]

        # 未设置 → None（make_initial_state 不设默认，零回归）
        captured.clear()
        with mock.patch("agent.router.router_agent.classify",
                        new=mock.AsyncMock(side_effect=fake_classify)):
            asyncio.run(graph_module.classify_intent(make_initial_state("为什么", [])))
        assert captured["tool_history"] is None

    def test_rag_state_has_optional_field(self):
        """RAGState 新增可选 tool_history 字段；make_initial_state 不含该键（.get 零回归）"""
        assert "tool_history" in rag_state.RAGState.__annotations__
        state = make_initial_state("为什么", [])
        assert "tool_history" not in state  # 不设默认 → classify 侧 .get() 取 None
