"""Module-022 检索缓存修复单元/集成测试

覆盖（验收 §1.1 + §4.1/§4.2）：
- _retrieve_cache_key：不同 top_k/min_score 生成不同 key、同参稳定、前缀不变
- delete_by_prefix：真实 Redis 前缀失效（不误删其他前缀、无匹配返回 True）
- add_document / delete_document 成功后触发全量失效（打桩 DB 层）

实现说明：
- 同步用例内 asyncio.run 执行，不依赖 pytest-asyncio（与 test_graph_store.py
  同款模式，规避既有 pytest-asyncio 缺失问题）
- delete_by_prefix 用真实 Redis（localhost:6379），其余用例 mock 掉 DB/LLM
"""
import asyncio
import hashlib
from unittest import mock

from rag.engine import _retrieve_cache_key
from src.cache import cache


# ─── cache_key 参数化（纯函数，无外部依赖） ───

class TestRetrieveCacheKey:
    """_retrieve_cache_key 参数化正确性"""

    def test_different_top_k_different_key(self):
        q = "Java线程池"
        assert _retrieve_cache_key(q, 5, 0.6) != _retrieve_cache_key(q, 10, 0.6)

    def test_different_min_score_different_key(self):
        q = "Java线程池"
        assert _retrieve_cache_key(q, 5, 0.6) != _retrieve_cache_key(q, 5, 0.3)

    def test_same_params_same_key(self):
        q = "Java线程池"
        assert _retrieve_cache_key(q, 5, 0.6) == _retrieve_cache_key(q, 5, 0.6)

    def test_different_query_different_key(self):
        assert _retrieve_cache_key("A", 5, 0.6) != _retrieve_cache_key("B", 5, 0.6)

    def test_prefix_and_acceptance_formula(self):
        q = "Java线程池"
        expected = hashlib.sha256((q + "5" + "0.6").encode()).hexdigest()[:16]
        assert _retrieve_cache_key(q, 5, 0.6) == f"rag:retrieve:{expected}"


# ─── delete_by_prefix（真实 Redis 集成） ───

class TestDeleteByPrefix:
    """delete_by_prefix 前缀失效（需 Redis 可达 localhost:6379）"""

    def test_prefix_invalidation_real_redis(self):
        from src.cache import RedisCache

        async def run():
            c = RedisCache()
            key1 = "rag:retrieve:ut-a"
            key2 = "rag:retrieve:ut-b"
            other = "rag:chat:ut-keep"  # 非检索前缀，不应被清
            await c.set(key1, [{"id": 1}])
            await c.set(key2, [{"id": 2}])
            await c.set(other, {"x": 1})

            ok = await c.delete_by_prefix("rag:retrieve:")
            assert ok is True, "delete_by_prefix 应返回 True"
            assert await c.get(key1) is None, "前缀命中 key 应被清除"
            assert await c.get(key2) is None, "前缀命中 key 应被清除"
            assert await c.get(other) is not None, "其他前缀不应被误删"

            # 无匹配前缀：返回 True 不报错
            assert await c.delete_by_prefix("rag:nonexistent:") is True
            return True

        assert asyncio.run(run()) is True


# ─── add_document / delete_document 失效接线（打桩 DB） ───

class _FakeAddSession:
    """add_document 用的假会话：execute 返回无重复，flush 赋 id，commit 成功"""

    def __init__(self):
        self._added = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, stmt):
        result = mock.MagicMock()
        result.scalar_one_or_none.return_value = None  # 无重复文档
        return result

    def add(self, obj):
        self._added.append(obj)

    async def flush(self):
        n = 0
        for o in self._added:
            if o.id is None:
                o.id = n + 1
                n += 1

    async def commit(self):
        pass

    async def rollback(self):
        pass


class TestInvalidationWiring:
    """add_document / delete_document 成功后应调用 cache.delete_by_prefix"""

    def test_add_document_invalidates_cache(self):
        from rag.engine import rag_engine

        fake_session = _FakeAddSession()
        factory = mock.MagicMock(return_value=fake_session)
        invalidation = mock.AsyncMock(return_value=True)

        async def run():
            with mock.patch("rag.engine.async_session_factory", factory), \
                 mock.patch.object(cache, "delete_by_prefix", invalidation), \
                 mock.patch("rag.engine.chunker.chunk", return_value={
                     "parents": [{"title": "section", "content": "内容"}],
                     "children": [{"title": "section", "content": "内容", "parent_index": 0}],
                 }), \
                 mock.patch("rag.engine.embedding_service.embed_documents",
                            mock.AsyncMock(return_value=[[0.1, 0.2, 0.3]])), \
                 mock.patch("rag.engine.tokenize", return_value=""), \
                 mock.patch("rag.engine.graph_store.ensure_graph",
                            mock.AsyncMock(return_value=True)), \
                 mock.patch("rag.engine.graph_extractor.extract_from_document",
                            mock.AsyncMock(return_value={"entities": [], "relations": []})):
                return await rag_engine.add_document("测试文档", "这是一段文档内容", "test")

        result = asyncio.run(run())
        assert result["duplicate"] is False
        invalidation.assert_awaited_once_with("rag:retrieve:")

    def test_delete_document_invalidates_cache(self):
        import main

        fake_session = mock.AsyncMock()
        fake_session.get = mock.AsyncMock(return_value=mock.MagicMock(title="测试文档"))
        execute_result = mock.MagicMock()
        execute_result.scalars.return_value.all.return_value = [mock.MagicMock()]
        fake_session.execute = mock.AsyncMock(return_value=execute_result)
        fake_session.delete = mock.AsyncMock()
        fake_session.commit = mock.AsyncMock()
        fake_session.__aenter__ = mock.AsyncMock(return_value=fake_session)
        fake_session.__aexit__ = mock.AsyncMock(return_value=False)

        factory = mock.MagicMock(return_value=fake_session)
        invalidation = mock.AsyncMock(return_value=True)

        async def run():
            with mock.patch("main.async_session_factory", factory), \
                 mock.patch.object(cache, "delete_by_prefix", invalidation):
                return await main.delete_document(1)

        result = asyncio.run(run())
        assert result["code"] == 0
        invalidation.assert_awaited_once_with("rag:retrieve:")
