"""Module-054 检索降级修复单元测试

覆盖（验收 §1/§2/§7）：
- WP-1 reranker 路径：三级 dirname 解析到 ai_service/models/bge-reranker-v2-m3
  （对齐 embeddings.py 修法；真实加载冒烟在 module 冒烟脚本，本套件验证路径逻辑）
- WP-2 方案 A：hybrid/rrf/weighted 向量化失败 → warning + 向量路空（不抛整体异常），
  FTS+图谱照常融合；vector_only 保持抛错（消融语义）；正常路径零开销（无向量化
  时不额外调用向量检索）
- WP-2 方案 B：引擎 rrf 分支 retrieve() 抛 RetrievalException（方案 A 未覆盖）→
  补 _retrieve_graph_only 兜底返回图结果（对齐 hybrid 图回退）；正常路径零开销

实现说明：
- settings.retrieval_fusion_mode 用 monkeypatch 改（测试后自动还原）
- mock.AsyncMock 打桩 session / async_session_factory / 通道检索 / 引擎依赖，
  不依赖真实数据库/模型（与 test_rrf_fusion.py / test_engine_latency.py 同款模式）
- 同步用例内 asyncio.run 执行
"""
import asyncio
import os
from unittest import mock

import pytest

from src.config import settings
from rag.retrieval.reranker import CrossEncoderReranker, _LOCAL_MODEL_DIR
from rag.retrieval.retriever import HybridRetriever, RetrievalException
from rag.engine import rag_engine

# ─── 公共桩 ───


class _FakeSessionFactory:
    """async_session_factory 打桩：按序返回 session"""

    def __init__(self, items):
        self._items = list(items)

    def __call__(self):
        item = self._items.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeCM(item)


class _FakeCM:
    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return False

    def __init__(self, session):
        self._session = session


def _make_retriever(embed_ok=True):
    emb = mock.AsyncMock()
    if embed_ok:
        emb.embed_text.return_value = [0.1] * 4
    else:
        emb.embed_text.side_effect = RuntimeError("embedding 502 (mock)")
    return HybridRetriever(embedding_service=emb, alpha=0.3)


def _session():
    return mock.AsyncMock()


def _doc(doc_id, score, title=None):
    return {"id": doc_id, "title": title or f"doc-{doc_id}", "content": "x",
            "score": score}


# ─── WP-1 reranker 路径 ───

class TestRerankerPath:
    """三级 dirname 修复：解析到 ai_service/models/bge-reranker-v2-m3"""

    def test_resolves_to_ai_service_models(self):
        # 对齐 embeddings.py 修法：rag/retrieval/ 下的文件需三级 dirname
        # 才回到 ai_service/ 根（二级会落在 rag/models/...）
        assert _LOCAL_MODEL_DIR.endswith(os.path.join("models", "bge-reranker-v2-m3"))
        assert os.path.isdir(_LOCAL_MODEL_DIR), f"模型目录应存在: {_LOCAL_MODEL_DIR}"
        assert not _LOCAL_MODEL_DIR.replace(os.sep, "/").endswith(
            "rag/models/bge-reranker-v2-m3"), "二级 dirname 会错误解析到 rag/models/"

    def test_validate_model_dir_passes_on_real_dir(self):
        # 真实目录 + 权重校验通过（模型本地必备，缺则加载路径回归）
        rr = CrossEncoderReranker()
        rr._validate_model_dir()  # 不抛异常即通过


# ─── WP-2 方案 A：向量化失败降级 ───

class TestPlanA:
    """hybrid/rrf/weighted 向量化失败 → 向量路空；vector_only 保持抛错"""

    def test_hybrid_vectorization_failure_degrades_not_raise(self, caplog):
        # 方案 A：hybrid 模式向量化失败 → warning + 向量路空，不抛整体异常
        retriever = _make_retriever(embed_ok=False)
        retriever._fts_search = mock.AsyncMock(
            return_value=[_doc(1, 0.9), _doc(2, 0.7)])
        retriever._vector_search = mock.AsyncMock()
        factory = _FakeSessionFactory([_session(), _session()])
        with mock.patch("rag.retrieval.retriever.async_session_factory", factory):
            out = asyncio.run(retriever.retrieve("问题", top_k=3))
        assert len(out) == 2
        assert all(d.get("vector_score", 0.0) == 0.0 for d in out)
        # 向量检索不再被调用（None 短路，零开销）
        assert retriever._vector_search.await_count == 0
        # warning 日志
        assert any("查询向量化失败" in r.message for r in caplog.records)

    def test_rrf_vectorization_failure_still_fuses(self, monkeypatch, caplog):
        # rrf 模式：向量化失败 → 向量路空，FTS+图谱照常融合
        monkeypatch.setattr(settings, "retrieval_fusion_mode", "rrf")
        retriever = _make_retriever(embed_ok=False)
        retriever._fts_search = mock.AsyncMock(return_value=[_doc(1, 0.9)])
        retriever._vector_search = mock.AsyncMock()
        retriever._retrieve_graph_only = mock.AsyncMock(
            return_value=[{"id": 3, "title": "g3", "content": "x",
                           "hybrid_score": 0.8}])
        factory = _FakeSessionFactory([_session(), _session()])
        with mock.patch("rag.retrieval.retriever.async_session_factory", factory):
            out = asyncio.run(retriever.retrieve("问题", top_k=3, round_num=0))
        assert len(out) == 2
        assert {d["id"] for d in out} == {1, 3}
        assert all(d.get("vector_score", 0.0) == 0.0 for d in out)
        assert retriever._vector_search.await_count == 0
        assert any("查询向量化失败" in r.message for r in caplog.records)

    def test_vector_only_keeps_raising(self, caplog):
        # 消融语义：vector_only 向量化失败 → 仍抛 RetrievalException（评估区分
        # "向量通道真不可用"），changelog 声明与 hybrid 的差异
        retriever = _make_retriever(embed_ok=False)
        with pytest.raises(RetrievalException, match="查询向量化失败"):
            asyncio.run(retriever.retrieve("问题", top_k=3, mode="vector_only"))

    def test_normal_path_zero_overhead(self):
        # 正常路径（embedding 可用）：embed_text 只调一次，向量检索正常参与
        retriever = _make_retriever(embed_ok=True)
        retriever._fts_search = mock.AsyncMock(return_value=[_doc(1, 0.9)])
        retriever._vector_search = mock.AsyncMock(return_value=[_doc(2, 0.8)])
        factory = _FakeSessionFactory([_session(), _session()])
        with mock.patch("rag.retrieval.retriever.async_session_factory", factory):
            out = asyncio.run(retriever.retrieve("问题", top_k=3))
        assert retriever._embedding_service.embed_text.await_count == 1
        assert retriever._vector_search.await_count == 1
        assert {d["id"] for d in out} == {1, 2}


# ─── WP-2 方案 B：引擎 rrf 分支补图兜底 ───

def _engine_patches(extra=None):
    """engine._retrieve 公共打桩（cache miss / HyDE 桩 / 反思桩）"""
    patches = [
        mock.patch("rag.engine.cache.get", mock.AsyncMock(return_value=None)),
        mock.patch("rag.engine.cache.set", mock.AsyncMock(return_value=True)),
        mock.patch.object(rag_engine, "_hyde_expand", mock.AsyncMock(return_value="假HyDE")),
        mock.patch("agent.reflector.reflector.check_sufficiency",
                   mock.AsyncMock(return_value={"sufficient": True})),
    ]
    for target, new in (extra or []):
        patches.append(mock.patch(target, new))
    return patches


def _patches(patches):
    class _Stack:
        def __enter__(self):
            self._stacks = [p.__enter__() for p in patches]
            return self

        def __exit__(self, *exc):
            for p in reversed(patches):
                p.__exit__(*exc)
            return False
    return _Stack()


def _gdoc(did):
    return {"id": did, "title": f"g{did}", "content": f"图文档{did}",
            "hybrid_score": 0.9, "parent_id": None}


class TestPlanB:
    """rrf 分支 retrieve() 抛异常 → 引擎补图兜底（防御层，正常不触发）"""

    def test_rrf_retrieve_exception_falls_back_to_graph(self, monkeypatch):
        monkeypatch.setattr(settings, "retrieval_fusion_mode", "rrf")
        retrieve_mock = mock.AsyncMock(
            side_effect=RetrievalException("DB 不可用 (mock)"))
        graph_mock = mock.AsyncMock(return_value=[_gdoc(1), _gdoc(2), _gdoc(3)])

        async def run():
            patches = _engine_patches([
                ("rag.engine.hybrid_retriever.retrieve", retrieve_mock),
                ("rag.engine.hybrid_retriever._retrieve_graph_only", graph_mock),
            ])
            with _patches(patches):
                docs = await rag_engine._retrieve("Redis持久化")
            return docs

        docs = asyncio.run(run())
        # 兜底返回图结果（3 篇 ≥ _MIN_DOCS_SKIP_REFLECT → 提前终止，不触发反思）
        assert len(docs) == 3
        assert {d["id"] for d in docs} == {1, 2, 3}
        assert retrieve_mock.await_count == 1
        assert graph_mock.await_count == 1

    def test_graph_fallback_failure_degrades_empty(self, monkeypatch):
        monkeypatch.setattr(settings, "retrieval_fusion_mode", "rrf")
        graph_mock = mock.AsyncMock(side_effect=RuntimeError("图也挂了"))

        async def run():
            patches = _engine_patches([
                ("rag.engine.hybrid_retriever.retrieve",
                 mock.AsyncMock(side_effect=RetrievalException("DB 不可用 (mock)"))),
                ("rag.engine.hybrid_retriever._retrieve_graph_only", graph_mock),
            ])
            with _patches(patches):
                docs = await rag_engine._retrieve("Redis持久化")
            return docs

        docs = asyncio.run(run())
        assert docs == []  # 图兜底失败 → 空结果，不崩
        assert graph_mock.await_count == 1

    def test_normal_path_no_graph_fallback(self, monkeypatch):
        # 正常路径零开销：retrieve 成功时不触发 _retrieve_graph_only 兜底
        monkeypatch.setattr(settings, "retrieval_fusion_mode", "rrf")
        graph_mock = mock.AsyncMock(return_value=[])

        async def run():
            patches = _engine_patches([
                ("rag.engine.hybrid_retriever.retrieve",
                 mock.AsyncMock(return_value=[_gdoc(1), _gdoc(2), _gdoc(3)])),
                ("rag.engine.hybrid_retriever._retrieve_graph_only", graph_mock),
            ])
            with _patches(patches):
                docs = await rag_engine._retrieve("Redis持久化")
            return docs

        docs = asyncio.run(run())
        assert len(docs) == 3
        assert graph_mock.await_count == 0  # 兜底零调用
