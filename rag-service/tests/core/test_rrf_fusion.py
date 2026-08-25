"""Module-053 三通道融合（RRF / 加权）单元测试

覆盖（验收 §8 tests/test_rrf_fusion.py）：
- RRF 公式正确性：已知 rank → 已知 score（score(d) = Σ 1/(60 + rank_i(d))，k=60）
- 三通道融合排序（FTS/向量/图谱，1-based 排名）
- 开关 hybrid 零回归（默认行为不变 + round_num>0 单路混合）
- 单路失败降级（该路不参与融合不崩）
- abs_cosine 保留断言（L3 反证依赖红线）
- 加权融合（权重消融参数解析 + 权重和 + 解析失败回退默认）

实现说明：
- settings.retrieval_fusion_mode 用 monkeypatch 改（测试后自动还原）
- 用 mock.AsyncMock 打桩 session / async_session_factory / 通道检索，
  不依赖真实数据库；同步用例内 asyncio.run 执行（与 test_retriever_concurrency.py 同款模式）
- 新代码一律新路径导入 rag.retrieval.retriever
"""
import asyncio
from unittest import mock

import pytest

from src.config import Settings, settings
from rag.retrieval.retriever import HybridRetriever, RetrievalException

RRF_K = 60


class _FakeSessionFactory:
    """async_session_factory 打桩：按序返回 session 或抛异常"""

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
    emb = mock.AsyncMock()
    emb.embed_text.return_value = [0.1] * 4
    return HybridRetriever(embedding_service=emb, alpha=0.3)


def _session():
    return mock.AsyncMock()


def _doc(doc_id, score, title=None):
    return {"id": doc_id, "title": title or f"doc-{doc_id}", "content": "x",
            "score": score}


class TestRrfFormula:
    """RRF 公式正确性：已知 rank → 已知 score"""

    def test_single_channel_rank1(self):
        # 单通道：rank=1 → 1/(60+1) = 1/61
        retriever = _make_retriever()
        out = retriever._fuse_rrf(
            fts_results=[_doc(1, 0.9)],
            vector_results=[],
            graph_results=[],
        )
        assert len(out) == 1
        assert out[0]["rrf_score"] == pytest.approx(1.0 / (RRF_K + 1), rel=1e-9)
        assert out[0]["fts_score"] == pytest.approx(0.9)
        assert out[0]["vector_score"] == 0.0
        assert out[0]["graph_score"] == 0.0

    def test_known_ranks_three_channels(self):
        # 三通道已知排名：fts=[A,B,C] vec=[B,C,A] graph=[C,A]
        #   A = 1/61 + 1/63 + 1/62
        #   B = 1/62 + 1/61 + 0
        #   C = 1/63 + 1/62 + 1/61
        retriever = _make_retriever()
        out = retriever._fuse_rrf(
            fts_results=[_doc(1, 1.0), _doc(2, 0.8), _doc(3, 0.6)],
            vector_results=[_doc(2, 1.0), _doc(3, 0.7), _doc(1, 0.5)],
            graph_results=[
                {"id": 3, "title": "g3", "content": "x", "hybrid_score": 1.0},
                {"id": 1, "title": "g1", "content": "x", "hybrid_score": 0.5},
            ],
        )
        scores = {d["id"]: d["rrf_score"] for d in out}
        exp_a = 1 / 61 + 1 / 63 + 1 / 62
        exp_b = 1 / 62 + 1 / 61
        exp_c = 1 / 63 + 1 / 62 + 1 / 61
        assert scores[1] == pytest.approx(exp_a, rel=1e-9)
        assert scores[2] == pytest.approx(exp_b, rel=1e-9)
        assert scores[3] == pytest.approx(exp_c, rel=1e-9)
        # 排序：A == C > B（A、C RRF 分完全相等，稳定排序按插入序 FTS 优先）
        assert [d["id"] for d in out] == [1, 3, 2]

    def test_missing_channel_does_not_contribute(self):
        # 图谱通道缺失（空）→ 仅两通道贡献，不崩
        retriever = _make_retriever()
        out = retriever._fuse_rrf(
            fts_results=[_doc(1, 1.0)],
            vector_results=[_doc(1, 0.9)],
            graph_results=[],
        )
        assert len(out) == 1
        assert out[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 61, rel=1e-9)

    def test_rrf_score_normalized_to_hybrid_score(self):
        # 引擎 min_score 过滤兼容：hybrid_score 为 min-max 归一化（top=1.0）
        retriever = _make_retriever()
        out = retriever._fuse_rrf(
            fts_results=[_doc(1, 1.0), _doc(2, 0.5)],
            vector_results=[],
            graph_results=[],
        )
        assert out[0]["hybrid_score"] == pytest.approx(1.0)
        assert out[1]["hybrid_score"] < 1.0
        # 原始 RRF 分保留（观察用）
        assert out[0]["rrf_score"] > out[1]["rrf_score"]


class TestWeightedFusion:
    """三通道 min-max 归一化 + 权重加权"""

    def test_weighted_sum(self):
        retriever = _make_retriever()
        # fts/vec/graph 各自单文档 → 归一化全 1.0 → 加权和 = 权重和 = 1.0
        out = retriever._fuse_weighted(
            fts_results=[_doc(1, 0.9)],
            vector_results=[_doc(2, 0.8)],
            graph_results=[{"id": 3, "title": "g", "content": "x", "hybrid_score": 0.7}],
        )
        assert {d["id"]: d["hybrid_score"] for d in out} == {1: 0.3, 2: 0.6, 3: 0.1}

    def test_weight_override(self, monkeypatch):
        monkeypatch.setattr(settings, "retrieval_fusion_weights", "0.25,0.5,0.25")
        retriever = _make_retriever()
        out = retriever._fuse_weighted(
            fts_results=[_doc(1, 0.9)],
            vector_results=[_doc(2, 0.8)],
            graph_results=[{"id": 3, "title": "g", "content": "x", "hybrid_score": 0.7}],
        )
        assert {d["id"]: d["hybrid_score"] for d in out} == {1: 0.25, 2: 0.5, 3: 0.25}

    def test_invalid_weights_falls_back_to_default(self):
        retriever = _make_retriever()
        with mock.patch.object(settings, "retrieval_fusion_weights", "0.1,0.9"):
            out = retriever._fuse_weighted(
                fts_results=[_doc(1, 0.9)],
                vector_results=[_doc(2, 0.8)],
                graph_results=[],
            )
        # 回退默认 0.3,0.6,0.1
        assert {d["id"]: d["hybrid_score"] for d in out} == {1: 0.3, 2: 0.6}


class TestFusionModeValidation:
    """retrieval_fusion_mode 枚举校验（minor 修复：非法值启动即拒，防静默落入 rrf 分支）"""

    def test_invalid_mode_rejected_at_settings(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(retrieval_fusion_mode="bogus")

    def test_valid_modes_accepted(self):
        for mode in ("hybrid", "rrf", "weighted"):
            assert Settings(retrieval_fusion_mode=mode).retrieval_fusion_mode == mode


class TestFusionSwitch:
    """开关零回归：默认 hybrid 行为不变；rrf 仅 round 0 启用图谱"""

    def test_default_hybrid_uses_execute_not_graph(self, monkeypatch):
        # 默认 fusion_mode=hybrid：retrieve 走原 _execute，不触发图谱通道
        monkeypatch.setattr(settings, "retrieval_fusion_mode", "hybrid")
        retriever = _make_retriever()
        retriever._execute = mock.AsyncMock(return_value=[_doc(1, 0.7)])
        retriever._execute_fusion = mock.AsyncMock(return_value=[_doc(2, 0.9)])
        out = asyncio.run(retriever.retrieve("问题", top_k=3))
        assert retriever._execute.await_count == 1
        assert retriever._execute_fusion.await_count == 0
        assert out[0]["id"] == 1

    def test_rrf_round0_uses_fusion(self, monkeypatch):
        monkeypatch.setattr(settings, "retrieval_fusion_mode", "rrf")
        retriever = _make_retriever()
        retriever._execute = mock.AsyncMock(return_value=[_doc(1, 0.7)])
        retriever._execute_fusion = mock.AsyncMock(return_value=[_doc(2, 0.9)])
        out = asyncio.run(retriever.retrieve("问题", top_k=3, round_num=0))
        assert retriever._execute_fusion.await_count == 1
        assert retriever._execute.await_count == 0
        assert out[0]["id"] == 2

    def test_rrf_round1_2_single_channel_mixed(self, monkeypatch):
        # round 1/2 语义：fusion 模式下降级单路混合（无图谱），与引擎
        # "图谱仅 round 0 查一次"一致
        monkeypatch.setattr(settings, "retrieval_fusion_mode", "rrf")
        retriever = _make_retriever()
        retriever._execute = mock.AsyncMock(return_value=[_doc(1, 0.7)])
        retriever._execute_fusion = mock.AsyncMock(return_value=[_doc(2, 0.9)])
        out = asyncio.run(retriever.retrieve("问题", top_k=3, round_num=2))
        assert retriever._execute.await_count == 1
        assert retriever._execute_fusion.await_count == 0
        assert out[0]["id"] == 1


class TestExecuteFusion:
    """三通道并行 + 融合（_execute_fusion 全流程）"""

    def _run_fusion(self, retriever, factory_items, fts=None, vec=None, graph=None,
                    fts_err=None, vec_err=None, graph_err=None):
        if fts_err:
            retriever._fts_search = mock.AsyncMock(side_effect=fts_err)
        else:
            retriever._fts_search = mock.AsyncMock(return_value=fts or [])
        if vec_err:
            retriever._vector_search = mock.AsyncMock(side_effect=vec_err)
        else:
            retriever._vector_search = mock.AsyncMock(return_value=vec or [])
        if graph_err:
            retriever._retrieve_graph_only = mock.AsyncMock(side_effect=graph_err)
        else:
            retriever._retrieve_graph_only = mock.AsyncMock(return_value=graph or [])
        factory = _FakeSessionFactory(factory_items)
        with mock.patch("rag.retrieval.retriever.async_session_factory", factory):
            return asyncio.run(retriever._execute_fusion("问题", [0.1], 6, 3))

    def test_three_channels_merged_and_ranked(self):
        retriever = _make_retriever()
        out = self._run_fusion(
            retriever,
            [_session(), _session()],
            fts=[_doc(1, 0.9), _doc(2, 0.7)],
            vec=[_doc(3, 0.8)],
            graph=[{"id": 2, "title": "g2", "content": "x", "hybrid_score": 1.0}],
        )
        ids = [d["id"] for d in out]
        assert set(ids) == {1, 2, 3}
        # 图谱双命中 doc2：graph_score 透传
        doc2 = next(d for d in out if d["id"] == 2)
        assert doc2["graph_score"] == 1.0
        assert doc2["fts_score"] == pytest.approx(0.7)
        # rrf_score 由三路排名贡献
        assert out[0]["rrf_score"] > 0
        # 图谱单命中 doc（仅图有）也保留在结果
        assert any(d["id"] == 2 for d in out)

    def test_vector_failure_degrades_without_crash(self):
        retriever = _make_retriever()
        out = self._run_fusion(
            retriever,
            [_session(), _session()],
            fts=[_doc(1, 0.9)],
            vec_err=RetrievalException("向量通道不可用"),
            graph=[{"id": 1, "title": "g1", "content": "x", "hybrid_score": 1.0}],
        )
        assert len(out) == 1
        assert out[0]["id"] == 1
        assert out[0]["vector_score"] == 0.0
        assert out[0]["graph_score"] == 1.0

    def test_all_channels_fail_returns_empty(self):
        retriever = _make_retriever()
        out = self._run_fusion(
            retriever,
            [_session(), _session()],
            fts_err=RetrievalException("FTS 不可用"),
            vec_err=RetrievalException("向量不可用"),
            graph_err=RetrievalException("图不可用"),
        )
        assert out == []

    def test_abs_cosine_preserved(self):
        # 红线：归一化/融合前存档 abs_cosine（L3 反证依赖）；双命中透传
        retriever = _make_retriever()
        vec_docs = [{"id": 5, "title": "v", "content": "x", "score": 0.66}]
        out = self._run_fusion(
            retriever,
            [_session(), _session()],
            fts=[_doc(5, 0.9), _doc(6, 0.7)],
            vec=vec_docs,
        )
        doc5 = next(d for d in out if d["id"] == 5)
        assert doc5["abs_cosine"] == pytest.approx(0.66)
        # 仅 FTS 命中文档无 abs_cosine（下游按 0.0 保守处理）
        doc6 = next(d for d in out if d["id"] == 6)
        assert "abs_cosine" not in doc6

    def test_fusion_exception_falls_back_to_hybrid(self, monkeypatch):
        # 融合计算异常 → 回退 _execute（hybrid），不崩
        monkeypatch.setattr(settings, "retrieval_fusion_mode", "rrf")
        retriever = _make_retriever()
        retriever._fts_search = mock.AsyncMock(return_value=[_doc(1, 0.9)])
        retriever._vector_search = mock.AsyncMock(return_value=[])
        retriever._retrieve_graph_only = mock.AsyncMock(return_value=[])
        retriever._execute = mock.AsyncMock(return_value=[_doc(9, 0.5)])
        retriever._fuse_rrf = mock.Mock(side_effect=RuntimeError("融合失败"))
        factory = _FakeSessionFactory([_session(), _session()])
        with mock.patch("rag.retrieval.retriever.async_session_factory", factory):
            out = asyncio.run(retriever._execute_fusion("问题", [0.1], 6, 3))
        assert out[0]["id"] == 9
        assert retriever._execute.await_count == 1

    def test_graph_only_result_kept_in_top_k(self):
        # 图谱命中父块文档（无 embedding）也能进入融合结果
        retriever = _make_retriever()
        out = self._run_fusion(
            retriever,
            [_session(), _session()],
            graph=[{"id": 100, "title": "parent", "content": "x",
                    "hybrid_score": 0.6, "parent_id": None}],
        )
        assert len(out) == 1
        assert out[0]["id"] == 100
        assert out[0]["hybrid_score"] == pytest.approx(1.0)

    def test_weighted_full_flow(self, monkeypatch):
        # minor 修复：weighted 经 _execute_fusion 全流程（此前仅 _fuse_weighted 纯函数覆盖）
        monkeypatch.setattr(settings, "retrieval_fusion_mode", "weighted")
        retriever = _make_retriever()
        out = self._run_fusion(
            retriever,
            [_session(), _session()],
            fts=[_doc(1, 0.9)],
            vec=[_doc(2, 0.8)],
            graph=[{"id": 3, "title": "g3", "content": "x", "hybrid_score": 0.7}],
        )
        # 三路各单文档 → 各自 min-max 归一化后 1.0 → 加权分 = 权重（0.3/0.6/0.1），
        # 权重生效排序：向量通道 0.6 > FTS 0.3 > 图谱 0.1
        assert [d["id"] for d in out] == [2, 1, 3]
        assert {d["id"]: d["hybrid_score"] for d in out} == {1: 0.3, 2: 0.6, 3: 0.1}
        # 通道分透传（加权模式下 graph_score 为归一化后 1.0）
        doc3 = next(d for d in out if d["id"] == 3)
        assert doc3["graph_score"] == pytest.approx(1.0)
        assert doc3["fts_score"] == 0.0
        assert doc3["vector_score"] == 0.0
