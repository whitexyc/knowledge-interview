"""Module-024 检索延迟优化单元测试

覆盖（验收 §1.1/§1.2/§1.3 + §4.1）：
- round 0 降级：向量失败 → 仅图结果；图失败 → 仅向量结果；两路都失败 → 空（不崩）
- HyDE 缓存：同一 query 第二次命中缓存（LLM 只调一次）；生成失败降级原始 query
- _hyde_cache_key：同 query 稳定、前缀 rag:hyde: 与检索缓存 rag:retrieve: 独立
- 整链路预算：超预算用已收集 docs 提前结束（不再发起新一轮检索）
- 提前终止：round 0 ≥3 篇跳过反思；<3 篇仍反思（阈值保守）

实现说明：
- 用 mock.AsyncMock 打桩 cache / hybrid_retriever / graph_store / reflector，
  不依赖真实 Redis / DB / LLM（与 test_memory.py / test_cache.py 同款模式）
- 同步用例内 asyncio.run 执行，不依赖 pytest-asyncio（规避既有环境问题）
- 假文档 parent_id=None（旧格式），_expand_to_parents 走直通分支，不触 DB
"""
import asyncio
from unittest import mock

from rag.retriever import RetrievalException
from rag.engine import rag_engine, _hyde_cache_key


# ─── 测试用假文档（parent_id=None 走 _expand_to_parents 直通分支） ───

def _doc(did, title, score):
    return {
        "id": did, "title": title, "content": f"内容{did}", "source": "test",
        "hybrid_score": score, "parent_id": None,
    }


DOC_A = _doc(1, "文档A", 0.9)
DOC_B = _doc(2, "文档B", 0.8)
DOC_C = _doc(3, "文档C", 0.7)


def _base_patches(extra=None):
    """构造 _retrieve 的公共打桩上下文

    覆盖 cache miss、HyDE 桩、round 0 三路（向量/图/实体）、反思桩。
    extra: 额外 patch 列表 [(target, new)]
    """
    patches = [
        mock.patch("rag.engine.cache.get", mock.AsyncMock(return_value=None)),
        mock.patch("rag.engine.cache.set", mock.AsyncMock(return_value=True)),
        mock.patch.object(rag_engine, "_hyde_expand", mock.AsyncMock(return_value="假HyDE")),
        mock.patch("rag.engine.hybrid_retriever.retrieve", mock.AsyncMock(return_value=[DOC_A])),
        mock.patch("rag.engine.graph_extractor.extract_from_query", mock.AsyncMock(return_value=["实体1"])),
        mock.patch("rag.engine.graph_store.search_related", mock.AsyncMock(return_value=[])),
        mock.patch("agent.reflector.reflector.check_sufficiency",
                   mock.AsyncMock(return_value={"sufficient": True})),
    ]
    for target, new in (extra or []):
        patches.append(mock.patch(target, new))
    return patches


# ─── round 0 降级 ───

class TestRound0Degradation:
    """round 0 向量/图单路失败降级，两路都失败返回空（不整链路崩溃）"""

    def test_vector_failure_degrades_to_graph(self):
        async def run():
            patches = _base_patches([
                ("rag.engine.hybrid_retriever.retrieve",
                 mock.AsyncMock(side_effect=RetrievalException("向量通道不可用"))),
                ("rag.engine.graph_store.search_related", mock.AsyncMock(return_value=[DOC_A])),
            ])
            with _patches(patches):
                docs = await rag_engine._retrieve("测试查询")
            return docs

        docs = asyncio.run(run())
        assert len(docs) == 1
        # 返回格式不变：含 id/title/content/hybrid_score
        assert {"id", "title", "content", "hybrid_score"} <= set(docs[0].keys())
        assert docs[0]["id"] == 1

    def test_vector_timeout_degrades_to_graph(self):
        # 验收 §1.1 场景 1：向量检索超时（>15s）→ 不整链路失败，降级为仅图结果
        # wait_for 超时产生 asyncio.TimeoutError，直接让 mock 抛出以模拟
        async def run():
            patches = _base_patches([
                ("rag.engine.hybrid_retriever.retrieve",
                 mock.AsyncMock(side_effect=asyncio.TimeoutError("模拟向量检索超时"))),
                ("rag.engine.graph_store.search_related", mock.AsyncMock(return_value=[DOC_A])),
            ])
            with _patches(patches):
                docs = await rag_engine._retrieve("测试查询")
            return docs

        docs = asyncio.run(run())
        assert len(docs) == 1
        assert docs[0]["id"] == 1

    def test_graph_failure_degrades_to_vector(self):
        async def run():
            patches = _base_patches([
                ("rag.engine.graph_store.search_related",
                 mock.AsyncMock(side_effect=RetrievalException("图通道不可用"))),
            ])
            with _patches(patches):
                docs = await rag_engine._retrieve("测试查询")
            return docs

        docs = asyncio.run(run())
        assert len(docs) == 1
        assert docs[0]["id"] == 1

    def test_both_fail_returns_empty(self):
        async def run():
            patches = _base_patches([
                ("rag.engine.hybrid_retriever.retrieve",
                 mock.AsyncMock(side_effect=RetrievalException("向量通道不可用"))),
                ("rag.engine.graph_store.search_related",
                 mock.AsyncMock(side_effect=RetrievalException("图通道不可用"))),
            ])
            with _patches(patches):
                docs = await rag_engine._retrieve("测试查询")
            return docs

        docs = asyncio.run(run())
        assert docs == []


# ─── HyDE 缓存 ───

class TestHydeCache:
    """HyDE 缓存：第二次命中、失败降级、key 独立"""

    def test_second_call_hits_cache(self):
        cache_store = {}

        async def fake_get(key):
            return cache_store.get(key)

        async def fake_set(key, value, ttl=300):
            cache_store[key] = value
            return True

        generated = []

        async def fake_generate(prompt):
            generated.append(prompt)
            return "假设性回答，用于检索知识库文档。"

        async def run():
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(side_effect=fake_generate)
            with mock.patch("rag.engine.cache.get", mock.AsyncMock(side_effect=fake_get)), \
                 mock.patch("rag.engine.cache.set", mock.AsyncMock(side_effect=fake_set)), \
                 mock.patch("rag.engine.LLMFactory.get_client", return_value=client):
                first = await rag_engine._hyde_expand("什么是线程池")
                second = await rag_engine._hyde_expand("什么是线程池")
            return first, second

        first, second = asyncio.run(run())
        assert first == "假设性回答，用于检索知识库文档。"
        assert second == first  # 第二次命中缓存，值一致
        assert len(generated) == 1  # LLM 只被调用一次

    def test_generation_failure_falls_back_to_query(self):
        async def run():
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(side_effect=RuntimeError("LLM 挂了"))
            with mock.patch("rag.engine.cache.get", mock.AsyncMock(return_value=None)), \
                 mock.patch("rag.engine.LLMFactory.get_client", return_value=client):
                result = await rag_engine._hyde_expand("原始问题")
            return result

        result = asyncio.run(run())
        assert result == "原始问题"


class TestHydeCacheKey:
    """_hyde_cache_key：同 query 稳定、不同 query 不同、前缀独立"""

    def test_same_query_same_key(self):
        assert _hyde_cache_key("线程池") == _hyde_cache_key("线程池")

    def test_different_query_different_key(self):
        assert _hyde_cache_key("A") != _hyde_cache_key("B")

    def test_prefix_independent_from_retrieve(self):
        key = _hyde_cache_key("X")
        assert key.startswith("rag:hyde:")
        assert not key.startswith("rag:retrieve:")


# ─── 整链路预算 ───

class TestRetrieveBudget:
    """整链路预算：超预算用已收集 docs 提前结束"""

    def test_budget_exceeded_uses_collected_docs(self):
        async def slow_check(q, d):
            await asyncio.sleep(0.1)  # 拖过 0.05s 预算
            return {"sufficient": False, "rewritten_query": "改写后的查询"}

        async def run():
            patches = _base_patches([
                ("rag.engine._RETRIEVE_BUDGET_SECONDS", 0.05),
                ("rag.engine.hybrid_retriever.retrieve",
                 mock.AsyncMock(return_value=[DOC_A, DOC_B])),
                ("agent.reflector.reflector.check_sufficiency",
                 mock.AsyncMock(side_effect=slow_check)),
            ])
            with _patches(patches):
                docs = await rag_engine._retrieve("测试")
            return docs

        docs = asyncio.run(run())
        # round 0 收集 2 篇（<3 不提前终止）→ 反思拖过预算 → 第二轮循环到点用已收集 docs 结束
        assert len(docs) == 2
        assert {d["id"] for d in docs} == {1, 2}

    def test_budget_already_expired_returns_empty(self):
        async def run():
            patches = _base_patches([("rag.engine._RETRIEVE_BUDGET_SECONDS", -1)])
            with _patches(patches):
                docs = await rag_engine._retrieve("测试")
            return docs

        docs = asyncio.run(run())
        assert docs == []  # 预算已到且无 docs → 返回空


# ─── 提前终止 ───

class TestEarlyTermination:
    """round 0 ≥3 篇跳过反思；<3 篇仍反思（阈值保守）"""

    def test_round0_sufficient_docs_skip_reflection(self):
        reflect_calls = []

        async def fake_check(q, d):
            reflect_calls.append(1)
            return {"sufficient": False, "rewritten_query": "改写后的查询"}

        async def run():
            patches = _base_patches([
                ("rag.engine.hybrid_retriever.retrieve",
                 mock.AsyncMock(return_value=[DOC_A, DOC_B, DOC_C])),
                ("agent.reflector.reflector.check_sufficiency",
                 mock.AsyncMock(side_effect=fake_check)),
            ])
            with _patches(patches):
                docs = await rag_engine._retrieve("测试")
            return docs

        docs = asyncio.run(run())
        assert reflect_calls == []  # 反思未被调用
        assert len(docs) == 3

    def test_round0_below_threshold_still_reflects(self):
        reflect_calls = []

        async def fake_check(q, d):
            reflect_calls.append(1)
            return {"sufficient": True}

        async def run():
            patches = _base_patches([
                ("rag.engine.hybrid_retriever.retrieve",
                 mock.AsyncMock(return_value=[DOC_A, DOC_B])),
                ("agent.reflector.reflector.check_sufficiency",
                 mock.AsyncMock(side_effect=fake_check)),
            ])
            with _patches(patches):
                docs = await rag_engine._retrieve("测试")
            return docs

        docs = asyncio.run(run())
        assert len(reflect_calls) == 1  # 2 篇 < 3 阈值 → 反思仍被调用
        assert len(docs) == 2


def _patches(patches):
    """把多个 mock.patch 组合成一个上下文管理器"""
    return _StackCtx(patches)


class _StackCtx:
    """批量启动 mock.patch 的组合上下文管理器"""

    def __init__(self, patches):
        self._patches = patches

    def __enter__(self):
        self._stacks = [p.__enter__() for p in self._patches]
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.__exit__(*exc)
        return False
