"""Module-027 嵌入并发修复单元测试

覆盖（验收 §1.1 核心功能 + §1.2 边界 + §2.2 并发 + §4.1 单元测试）：
- 16 路并发 embed_text：不崩、全 1024 维、模型调用串行（max_active == 1）
- 8 路并发 embed_documents：不崩、各批结果正确、批量内部循环整批串行
- 空文本 embed_text 抛 EmbeddingException（既有行为回归）
- 空列表 embed_documents 返回空列表（既有行为回归）
- _retrieve 空 query 防护：返回空、不生成缓存 key（module-022 遗留）

实现说明：
- 假模型注入 EmbeddingService._model（避免加载真实 GGUF），其
  create_embedding 记录"同时活跃调用数"，断言恒为 1 即锁生效——
  若锁失效，并发会在 sleep 窗口内交错进入，max_active > 1
- 并发通过 embed_text/embed_documents 真实 async 接口 + asyncio.gather
  触发（内部 to_thread 真线程，与生产路径一致）
- 同步用例内 asyncio.run 执行，不依赖 pytest-asyncio（既有环境惯例）
"""
import asyncio
import threading
import time
from unittest import mock

from rag.embeddings import EmbeddingService, EmbeddingException
from rag.engine import rag_engine

_DIM = 1024


class _FakeModel:
    """假 Llama 模型：create_embedding 返回 1024 维向量，记录并发深度

    仅测试桩。guard 锁只用于统计活跃调用数，与 EmbeddingService 的
    串行锁相互独立，不影响被测试逻辑。
    """

    def __init__(self):
        self._guard = threading.Lock()
        self._active = 0
        self.max_active = 0

    def create_embedding(self, text):
        with self._guard:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(0.01)  # 扩大竞态窗口：无锁实现更容易暴露并发
            vec = [0.1] * _DIM  # 归一化后仍 1024 维
            return {"data": [{"embedding": vec}]}
        finally:
            with self._guard:
                self._active -= 1


def _make_service() -> EmbeddingService:
    svc = EmbeddingService()
    svc._model = _FakeModel()  # 注入假模型，_lazy_load 命中已加载直接跳过
    return svc


class TestConcurrentEmbedText:
    """16 路并发 embed_text：不崩、结果正确、模型调用串行"""

    def test_concurrent_embed_text(self):
        async def run():
            svc = _make_service()
            tasks = [svc.embed_text(f"测试文本{i}") for i in range(16)]
            results = await asyncio.gather(*tasks)
            return results, svc._model.max_active

        results, max_active = asyncio.run(run())
        assert len(results) == 16
        assert all(len(r) == _DIM for r in results)
        assert max_active == 1  # 锁保证同一时刻仅一个线程访问模型


class TestConcurrentEmbedDocuments:
    """8 路并发 embed_documents：不崩、各批结果正确、内部循环串行"""

    def test_concurrent_embed_documents(self):
        async def run():
            svc = _make_service()
            tasks = [svc.embed_documents([f"doc{i}-a", f"doc{i}-b"]) for i in range(8)]
            results = await asyncio.gather(*tasks)
            return results, svc._model.max_active

        results, max_active = asyncio.run(run())
        assert len(results) == 8
        assert all(len(batch) == 2 for batch in results)
        assert all(len(v) == _DIM for batch in results for v in batch)
        assert max_active == 1  # 批量内部循环整批持锁，同样串行


class TestEmptyInputBoundary:
    """空输入边界（既有行为回归，锁改造后仍成立）"""

    def test_embed_text_empty_raises(self):
        async def run():
            svc = _make_service()
            for bad in ("", "   "):
                try:
                    await svc.embed_text(bad)
                    return False
                except EmbeddingException:
                    continue
            return True

        assert asyncio.run(run())

    def test_embed_documents_empty_returns_empty(self):
        async def run():
            svc = _make_service()
            assert await svc.embed_documents([]) == []
            assert await svc.embed_documents([" ", ""]) == []

        asyncio.run(run())


class TestRetrieveEmptyQueryGuard:
    """_retrieve 空 query 防护（module-022 遗留）：不生成缓存 key"""

    def _assert_guard(self, query):
        async def run():
            # 空 query 若走到缓存检查 → 防护失效 → 触发断言
            cache_get = mock.AsyncMock(
                side_effect=AssertionError("空 query 不应访问缓存 / 生成缓存 key"),
            )
            with mock.patch("rag.engine.cache.get", cache_get):
                docs = await rag_engine._retrieve(query)
            return docs

        docs = asyncio.run(run())
        assert docs == []

    def test_empty_query_returns_empty(self):
        self._assert_guard("")

    def test_whitespace_query_returns_empty(self):
        self._assert_guard("   ")
