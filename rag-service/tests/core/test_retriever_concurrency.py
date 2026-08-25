"""Module-026 检索并发修复单元测试

覆盖（验收 §1.1 并发修复 + §4.1「并发独立 session 单测」）：
- 未传外部 session：FTS / 向量各开独立 session，gather 并行（两 session 不同）
- 传入外部 session：共享会话串行执行（不创建新 session，不并发）
- 一路失败不影响另一路（单路降级）
- 两路都失败返回空
- 独立 session 创建失败降级为共享 session 串行

实现说明：
- 用 mock.AsyncMock 打桩 session / async_session_factory / _fts_search / _vector_search，
  不依赖真实数据库
- 同步用例内 asyncio.run 执行，不依赖 pytest-asyncio（与 test_fts_search.py 同款模式）
"""
import asyncio
from unittest import mock

from rag.retriever import HybridRetriever, RetrievalException


class _FakeSessionFactory:
    """async_session_factory 打桩：按序返回 session 或抛异常

    async_session_factory() 每次调用返回一个 async 上下文管理器，
    __aenter__ 时交出 session。items 中放 session（mock.AsyncMock）或
    Exception（该次调用抛异常，模拟独立 session 创建失败）。
    """

    def __init__(self, items):
        self._items = list(items)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        item = self._items.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeCM(item)


class _FakeCM:
    """异步上下文管理器，包一层 session"""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return False


def _make_retriever():
    return HybridRetriever(embedding_service=mock.MagicMock(), alpha=0.3)


def _session():
    """返回一个可作 DB 会话的 mock session"""
    return mock.AsyncMock()


class TestExecuteIndependentSessions:
    """未传外部 session：两路独立 session 并行"""

    def test_parallel_uses_two_independent_sessions(self):
        async def run():
            retriever = _make_retriever()
            fts_sess, vec_sess = _session(), _session()
            retriever._fts_search = mock.AsyncMock(
                return_value=[{"id": 1, "score": 0.8}],
            )
            retriever._vector_search = mock.AsyncMock(
                return_value=[{"id": 2, "score": 0.6}],
            )
            factory = _FakeSessionFactory([fts_sess, vec_sess])
            with mock.patch("rag.retriever.async_session_factory", factory):
                result = await retriever._execute("问题", [0.1], 6, 3)
            return result, retriever, factory, fts_sess, vec_sess

        result, retriever, factory, fts_sess, vec_sess = asyncio.run(run())
        # 两个独立 session，互不相同（消除单连接并发限制的关键）
        assert retriever._fts_search.await_args.args[2] is fts_sess
        assert retriever._vector_search.await_args.args[2] is vec_sess
        assert fts_sess is not vec_sess
        assert factory.calls == 2
        # 两路结果都生效（并行性能保留）
        assert {d["id"] for d in result} == {1, 2}

    def test_parallel_single_channel_failure_degrades(self):
        # 向量路失败 → 仅 FTS 结果，不报错（graceful degradation）
        async def run():
            retriever = _make_retriever()
            retriever._fts_search = mock.AsyncMock(
                return_value=[{"id": 1, "score": 0.8}],
            )
            retriever._vector_search = mock.AsyncMock(
                side_effect=RetrievalException("向量通道不可用"),
            )
            factory = _FakeSessionFactory([_session(), _session()])
            with mock.patch("rag.retriever.async_session_factory", factory):
                result = await retriever._execute("问题", [0.1], 6, 3)
            return result

        result = asyncio.run(run())
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["vector_score"] == 0.0

    def test_parallel_both_fail_returns_empty(self):
        async def run():
            retriever = _make_retriever()
            retriever._fts_search = mock.AsyncMock(
                side_effect=RetrievalException("FTS 通道不可用"),
            )
            retriever._vector_search = mock.AsyncMock(
                side_effect=RetrievalException("向量通道不可用"),
            )
            factory = _FakeSessionFactory([_session(), _session()])
            with mock.patch("rag.retriever.async_session_factory", factory):
                result = await retriever._execute("问题", [0.1], 6, 3)
            return result

        result = asyncio.run(run())
        assert result == []

    def test_parallel_session_creation_failure_degrades_serial(self):
        # 独立 session 创建失败 → 降级：单共享 session 串行执行
        async def run():
            retriever = _make_retriever()
            shared_sess = _session()
            retriever._fts_search = mock.AsyncMock(
                return_value=[{"id": 1, "score": 0.8}],
            )
            retriever._vector_search = mock.AsyncMock(
                return_value=[{"id": 2, "score": 0.6}],
            )
            # 第一次创建抛异常，第二次（降级路径）返回共享 session
            factory = _FakeSessionFactory([RuntimeError("连接失败"), shared_sess])
            with mock.patch("rag.retriever.async_session_factory", factory):
                result = await retriever._execute("问题", [0.1], 6, 3)
            return result, retriever, shared_sess, factory

        result, retriever, shared_sess, factory = asyncio.run(run())
        assert factory.calls == 2
        # 降级后两路共用同一个共享 session，串行执行
        assert retriever._fts_search.await_args.args[2] is shared_sess
        assert retriever._vector_search.await_args.args[2] is shared_sess
        assert {d["id"] for d in result} == {1, 2}


class TestExecuteExternalSession:
    """传入外部 session：共享会话串行，不创建新 session"""

    def test_external_session_shared_and_no_new_session(self):
        async def run():
            retriever = _make_retriever()
            external = _session()
            retriever._fts_search = mock.AsyncMock(
                return_value=[{"id": 1, "score": 0.8}],
            )
            retriever._vector_search = mock.AsyncMock(
                return_value=[{"id": 2, "score": 0.6}],
            )
            # 若创建新 session 则断言失败（外部 session 必须被复用）
            with mock.patch(
                "rag.retriever.async_session_factory",
                mock.Mock(side_effect=AssertionError("外部 session 时不应创建新 session")),
            ):
                result = await retriever._execute("问题", [0.1], 6, 3, external)
            return result, retriever, external

        result, retriever, external = asyncio.run(run())
        # 两路共用同一个外部 session（兼容外部传入 session）
        assert retriever._fts_search.await_args.args[2] is external
        assert retriever._vector_search.await_args.args[2] is external
        assert {d["id"] for d in result} == {1, 2}

    def test_external_session_single_channel_failure_degrades(self):
        async def run():
            retriever = _make_retriever()
            external = _session()
            retriever._fts_search = mock.AsyncMock(
                return_value=[{"id": 1, "score": 0.8}],
            )
            retriever._vector_search = mock.AsyncMock(
                side_effect=RetrievalException("向量通道不可用"),
            )
            with mock.patch(
                "rag.retriever.async_session_factory",
                mock.Mock(side_effect=AssertionError("不应创建新 session")),
            ):
                result = await retriever._execute("问题", [0.1], 6, 3, external)
            return result

        result = asyncio.run(run())
        assert len(result) == 1
        assert result[0]["id"] == 1


class TestExecuteAbsCosinePassThrough:
    """module-045 WP1: 合并环 abs_cosine 透传（双命中不丢字段，fts-only 保持无字段）"""

    def test_double_hit_doc_preserves_abs_cosine(self):
        """FTS+向量双命中：合并结果含 abs_cosine（原始绝对余弦，非 min-max 相对分）"""
        async def run():
            retriever = _make_retriever()
            fts_sess, vec_sess = _session(), _session()
            retriever._fts_search = mock.AsyncMock(
                return_value=[{"id": 1, "score": 0.8}],
            )
            retriever._vector_search = mock.AsyncMock(
                return_value=[{"id": 1, "score": 0.65}],
            )
            factory = _FakeSessionFactory([fts_sess, vec_sess])
            with mock.patch("rag.retriever.async_session_factory", factory):
                result = await retriever._execute("问题", [0.1], 6, 3)
            return result

        result = asyncio.run(run())
        assert len(result) == 1
        assert result[0]["id"] == 1
        # abs_cosine 是归一化前存档的原始绝对余弦；vector_score 是 min-max 相对分
        assert result[0]["abs_cosine"] == 0.65
        assert result[0]["vector_score"] == 1.0

    def test_fts_only_doc_has_no_abs_cosine(self):
        """fts-only 文档保持无该字段（下游按 0.0 保守处理，语义不变）"""
        async def run():
            retriever = _make_retriever()
            fts_sess, vec_sess = _session(), _session()
            retriever._fts_search = mock.AsyncMock(
                return_value=[{"id": 1, "score": 0.8}],
            )
            retriever._vector_search = mock.AsyncMock(
                return_value=[{"id": 2, "score": 0.6}],
            )
            factory = _FakeSessionFactory([fts_sess, vec_sess])
            with mock.patch("rag.retriever.async_session_factory", factory):
                result = await retriever._execute("问题", [0.1], 6, 3)
            return result

        result = asyncio.run(run())
        by_id = {d["id"]: d for d in result}
        assert "abs_cosine" not in by_id[1]  # fts-only：无字段
        assert by_id[2]["abs_cosine"] == 0.6  # vector-only：带字段
