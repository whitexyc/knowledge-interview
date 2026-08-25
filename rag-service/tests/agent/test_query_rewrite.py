"""module-049 分诊式 Query 改写单元测试（rag/query_rewrite.py + engine 接入）

覆盖（验收 §1/§2/§4/§6）：
- 分诊：FTS 命中 → precise；不命中 → vague；异常 → 保守 vague
- LLM 改写：成功 / 空 / 异常 / 超时 / 无变化 → 回退（None）
- 保真预检：余弦计算正确；嵌入失败/数量异常 → None（跳过预检）
- 择优：改写优 / 原优 / 相等回退原 / abs_cosine 缺失按 0 / 空改写回退原
- prepare 全管线：precise 零调用 / 改写失败回退 / 保真未过跳过并行 /
  并行择优（改写优/原优/单路失败/双路失败）/ 预检失败仍并行
- prepare_query：查询级管线（无并行，保真门控）
- engine 接入：chat 用并行择优文档（不再重复检索）/ 开关关闭不调用 /
  _retrieve 用改写后 query 作为 HyDE 基础

实现说明：
- 同步用例内 asyncio.run 执行，不依赖 pytest-asyncio（与套件其余用例同款）
- 打桩注入 retrieve_fn / 子函数，不依赖真实 DB / LLM / 嵌入模型
"""
import asyncio
from unittest import mock

import rag.engine as engine_module
from rag import query_rewrite as qr
from rag.engine import rag_engine

ORIG = "内存调优有没有什么好办法"
REWRITTEN = "JVM 内存调优 GC 参数"


# ─── 分诊（WP1 静态分诊） ───

class TestTriage:
    """分诊：FTS 术语命中 → precise 直接检索；不命中/失败 → vague 走改写"""

    def test_fts_hit_returns_precise(self):
        async def run():
            with mock.patch("rag.query_rewrite.fts_term_hit",
                            new=mock.AsyncMock(return_value=True)):
                return await qr.triage("G1垃圾收集器MixedGC流程")

        assert asyncio.run(run()) == "precise"

    def test_fts_miss_returns_vague(self):
        async def run():
            with mock.patch("rag.query_rewrite.fts_term_hit",
                            new=mock.AsyncMock(return_value=False)):
                return await qr.triage(ORIG)

        assert asyncio.run(run()) == "vague"

    def test_triage_error_conservative_vague(self):
        # 分诊 DB 异常 → 保守默认"模糊"走改写路径（宁多检不漏检，不中断链路）
        async def run():
            with mock.patch("rag.query_rewrite.fts_term_hit",
                            new=mock.AsyncMock(side_effect=RuntimeError("db down"))):
                return await qr.triage(ORIG)

        assert asyncio.run(run()) == "vague"


# ─── LLM 改写（WP2①，独立封装） ───

class TestLlmRewrite:
    """LLM 改写：失败/超时/空/无变化 → None（调用方回退原 query，零回归）"""

    @staticmethod
    def _patch_client(generate):
        fake = mock.MagicMock()
        fake.generate = generate
        return mock.patch.object(qr.LLMFactory, "get_client", return_value=fake)

    def test_rewrite_success(self):
        async def run():
            with self._patch_client(mock.AsyncMock(return_value=REWRITTEN)):
                return await qr.llm_rewrite(ORIG)

        assert asyncio.run(run()) == REWRITTEN

    def test_rewrite_empty_returns_none(self):
        async def run():
            with self._patch_client(mock.AsyncMock(return_value="  ")):
                return await qr.llm_rewrite(ORIG)

        assert asyncio.run(run()) is None

    def test_rewrite_error_returns_none(self):
        async def run():
            with self._patch_client(mock.AsyncMock(side_effect=RuntimeError("llm down"))):
                return await qr.llm_rewrite(ORIG)

        assert asyncio.run(run()) is None

    def test_rewrite_timeout_returns_none(self, monkeypatch):
        # 改写超时（wait_for 超时）→ 回退原 query，链路不中断（与 HyDE 降级同哲学）
        async def slow_generate(prompt):
            await asyncio.sleep(0.1)
            return REWRITTEN

        monkeypatch.setattr(qr, "_REWRITE_TIMEOUT", 0.01)

        async def run():
            with self._patch_client(mock.AsyncMock(side_effect=slow_generate)):
                return await qr.llm_rewrite(ORIG)

        assert asyncio.run(run()) is None

    def test_rewrite_unchanged_returns_none(self):
        # 改写结果与原 query 相同 → 视为无效（避免无意义并行）
        async def run():
            with self._patch_client(mock.AsyncMock(return_value=ORIG)):
                return await qr.llm_rewrite(ORIG)

        assert asyncio.run(run()) is None


# ─── 保真预检（WP2②） ───

class TestFidelityCheck:
    """保真预检：改写 vs 原 query 余弦；嵌入失败 → None（跳过预检直接并行）"""

    def test_cosine_similar(self):
        async def run():
            with mock.patch("rag.query_rewrite.embedding_service.embed_documents",
                            new=mock.AsyncMock(return_value=[[1, 0, 0], [0.6, 0.8, 0]])):
                return await qr.fidelity_check(ORIG, REWRITTEN)

        assert asyncio.run(run()) == 0.6

    def test_orthogonal_zero(self):
        async def run():
            with mock.patch("rag.query_rewrite.embedding_service.embed_documents",
                            new=mock.AsyncMock(return_value=[[1, 0, 0], [0, 1, 0]])):
                return await qr.fidelity_check(ORIG, REWRITTEN)

        assert asyncio.run(run()) == 0.0

    def test_embed_failure_returns_none(self):
        async def run():
            with mock.patch("rag.query_rewrite.embedding_service.embed_documents",
                            new=mock.AsyncMock(side_effect=RuntimeError("embed down"))):
                return await qr.fidelity_check(ORIG, REWRITTEN)

        assert asyncio.run(run()) is None

    def test_wrong_vector_count_returns_none(self):
        async def run():
            with mock.patch("rag.query_rewrite.embedding_service.embed_documents",
                            new=mock.AsyncMock(return_value=[[1, 0, 0]])):
                return await qr.fidelity_check(ORIG, REWRITTEN)

        assert asyncio.run(run()) is None


# ─── 择优（WP2④，纯函数） ───

class TestSelectBetter:
    """择优：改写 top-1 abs_cosine > 原 → 用改写；相等/缺失/空 → 回退原（保守）"""

    @staticmethod
    def _docs(abs_cosine):
        return [{"id": 1, "title": "t", "abs_cosine": abs_cosine}]

    def test_rewrite_wins(self):
        docs, used = qr.select_better(self._docs(0.55), self._docs(0.62))
        assert used is True
        assert docs[0]["abs_cosine"] == 0.62

    def test_original_wins(self):
        docs, used = qr.select_better(self._docs(0.70), self._docs(0.55))
        assert used is False
        assert docs[0]["abs_cosine"] == 0.70

    def test_equal_falls_back_to_original(self):
        # 相等 → 回退原（保守，防合并噪声）
        docs, used = qr.select_better(self._docs(0.55), self._docs(0.55))
        assert used is False
        assert docs[0]["abs_cosine"] == 0.55

    def test_missing_abs_cosine_treated_as_zero(self):
        # abs_cosine 缺失按 0 处理（module-045 口径）
        orig = [{"id": 1, "title": "t"}]
        rewritten = self._docs(0.3)
        docs, used = qr.select_better(orig, rewritten)
        assert used is True
        assert docs[0]["abs_cosine"] == 0.3

    def test_empty_rewritten_falls_back_to_original(self):
        docs, used = qr.select_better(self._docs(0.55), [])
        assert used is False
        assert docs == self._docs(0.55)


# ─── prepare 全管线（WP2③④，chat 主路径） ───

class TestPrepare:
    """prepare：分诊 → 改写 → 保真 → 并行检索择优，任一环节失败回退原 query"""

    @staticmethod
    def _stub_retrieve(orig_docs, rw_docs):
        async def _f(query):
            return rw_docs if query == REWRITTEN else orig_docs
        return _f

    def test_precise_skips_rewrite_and_retrieval(self):
        # 分诊命中 → 零 LLM、零并行、零检索（链路延迟不增加）
        calls = []

        async def _retrieve_fn(q):
            calls.append(q)
            return []

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="precise")):
                return await qr.prepare(ORIG, _retrieve_fn)

        search_query, round0, info = asyncio.run(run())
        assert search_query == ORIG
        assert round0 is None
        assert info["mode"] == "precise"
        assert calls == []

    def test_rewrite_failed_fallback(self):
        # LLM 改写失败 → 回退原 query，行为与现状完全一致
        async def _retrieve_fn(q):
            raise AssertionError("不应触发检索")

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(return_value=None)):
                    return await qr.prepare(ORIG, _retrieve_fn)

        search_query, round0, info = asyncio.run(run())
        assert search_query == ORIG
        assert round0 is None
        assert info["mode"] == "rewrite_fallback"

    def test_fidelity_reject_skips_parallel(self):
        # 保真未过（余弦 < 阈值）→ 直接用原 query 检索（省一次并行检索）
        async def _retrieve_fn(q):
            raise AssertionError("不应触发并行检索")

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(return_value=REWRITTEN)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(return_value=0.4)):
                        return await qr.prepare(ORIG, _retrieve_fn)

        search_query, round0, info = asyncio.run(run())
        assert search_query == ORIG
        assert round0 is None
        assert info["mode"] == "fidelity_reject"
        assert info["fidelity"] == 0.4

    def test_parallel_rewrite_wins(self):
        orig_docs = [{"id": 1, "abs_cosine": 0.55}]
        rw_docs = [{"id": 2, "abs_cosine": 0.62}]

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(return_value=REWRITTEN)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(return_value=0.9)):
                        return await qr.prepare(ORIG, self._stub_retrieve(orig_docs, rw_docs))

        search_query, round0, info = asyncio.run(run())
        assert search_query == REWRITTEN
        assert round0 == rw_docs
        assert info["mode"] == "parallel"
        assert info["used_rewrite"] is True
        assert info["orig_top1_abs"] == 0.55
        assert info["rewrite_top1_abs"] == 0.62

    def test_parallel_original_wins(self):
        orig_docs = [{"id": 1, "abs_cosine": 0.70}]
        rw_docs = [{"id": 2, "abs_cosine": 0.55}]

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(return_value=REWRITTEN)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(return_value=0.9)):
                        return await qr.prepare(ORIG, self._stub_retrieve(orig_docs, rw_docs))

        search_query, round0, info = asyncio.run(run())
        assert search_query == ORIG
        assert round0 == orig_docs
        assert info["used_rewrite"] is False

    def test_parallel_rewrite_side_fails_uses_original(self):
        # 并行单路失败 → 用成功路结果（对齐 round 0 降级）
        orig_docs = [{"id": 1, "abs_cosine": 0.60}]

        async def _retrieve_fn(q):
            if q == REWRITTEN:
                raise RuntimeError("rewrite 检索失败")
            return orig_docs

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(return_value=REWRITTEN)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(return_value=0.9)):
                        return await qr.prepare(ORIG, _retrieve_fn)

        search_query, round0, info = asyncio.run(run())
        assert search_query == ORIG
        assert round0 == orig_docs
        assert info["used_rewrite"] is False

    def test_parallel_original_side_fails_uses_rewritten(self):
        rw_docs = [{"id": 2, "abs_cosine": 0.62}]

        async def _retrieve_fn(q):
            if q == ORIG:
                raise RuntimeError("原检索失败")
            return rw_docs

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(return_value=REWRITTEN)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(return_value=0.9)):
                        return await qr.prepare(ORIG, _retrieve_fn)

        search_query, round0, info = asyncio.run(run())
        assert search_query == REWRITTEN
        assert round0 == rw_docs
        assert info["used_rewrite"] is True

    def test_parallel_both_fail_returns_empty(self):
        # 双路失败 → 空结果走现有无结果降级（不整链路崩溃）
        async def _retrieve_fn(q):
            raise RuntimeError("both down")

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(return_value=REWRITTEN)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(return_value=0.9)):
                        return await qr.prepare(ORIG, _retrieve_fn)

        search_query, round0, info = asyncio.run(run())
        assert search_query == ORIG
        assert round0 == []
        assert info["used_rewrite"] is False

    def test_fidelity_unavailable_still_parallel(self):
        # 保真预检失败（嵌入不可用）→ 跳过预检直接并行，让择优兜底
        orig_docs = [{"id": 1, "abs_cosine": 0.55}]
        rw_docs = [{"id": 2, "abs_cosine": 0.62}]

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(return_value=REWRITTEN)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(return_value=None)):
                        return await qr.prepare(ORIG, self._stub_retrieve(orig_docs, rw_docs))

        search_query, round0, info = asyncio.run(run())
        assert search_query == REWRITTEN
        assert round0 == rw_docs
        assert info["fidelity"] is None


# ─── prepare_query 查询级管线（流式/_retrieve 路径） ───

class TestPrepareQuery:
    """prepare_query：分诊 + 改写 + 保真门控，不做并行（无择优兜底 → 保真不可得时保守回退）"""

    def test_precise_returns_original(self):
        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="precise")):
                return await qr.prepare_query(ORIG)

        base, info = asyncio.run(run())
        assert base == ORIG
        assert info["mode"] == "precise"

    def test_rewrite_accepted(self):
        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(return_value=REWRITTEN)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(return_value=0.88)):
                        return await qr.prepare_query(ORIG)

        base, info = asyncio.run(run())
        assert base == REWRITTEN
        assert info["mode"] == "rewrite_accepted"
        assert info["fidelity"] == 0.88

    def test_fidelity_reject_falls_back(self):
        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(return_value=REWRITTEN)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(return_value=0.4)):
                        return await qr.prepare_query(ORIG)

        base, info = asyncio.run(run())
        assert base == ORIG
        assert info["mode"] == "fidelity_reject"

    def test_fidelity_unavailable_falls_back(self):
        # 本路径无并行择优兜底 → 保真不可得时保守回退原 query（改写链路
        # 任何一环失败 = 回退原 query）
        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(return_value=REWRITTEN)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(return_value=None)):
                        return await qr.prepare_query(ORIG)

        base, info = asyncio.run(run())
        assert base == ORIG
        assert info["mode"] == "fidelity_reject"

    def test_rewrite_failed_falls_back(self):
        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(return_value=None)):
                    return await qr.prepare_query(ORIG)

        base, info = asyncio.run(run())
        assert base == ORIG
        assert info["mode"] == "rewrite_fallback"


# ─── engine 接入 ───

class TestEngineChatIntegration:
    """chat 接入：round 0 用并行择优文档（不重复检索）；开关关闭不调用"""

    @staticmethod
    def _base_chat_patches():
        fake_doc = {"id": 1, "title": "t", "content": "c", "source": "s",
                    "abs_cosine": 0.7, "parent_id": None}
        return [
            mock.patch("rag.engine.router_agent.classify",
                       new=mock.AsyncMock(return_value={"intent": "knowledge", "confidence": 0.9})),
            mock.patch("rag.engine.memory_service.recall", new=mock.AsyncMock(return_value=[])),
            mock.patch("rag.engine.memory_service.recall_short", new=mock.AsyncMock(return_value=[])),
            mock.patch("rag.engine.hybrid_retriever.retrieve",
                       new=mock.AsyncMock(return_value=[fake_doc])),
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

    def test_chat_uses_rewrite_round0_docs_without_reread(self, monkeypatch):
        # 并行择优后 round 0 直接用择优文档：不再调用 hybrid_retriever.retrieve
        from rag.schemas import ChatRequest

        fake_doc = {"id": 9, "title": "改写命中", "content": "c", "source": "s",
                    "abs_cosine": 0.66, "parent_id": None}
        monkeypatch.setattr(engine_module.settings, "query_rewrite_enabled", True)

        patches = self._base_chat_patches() + [
            mock.patch("rag.engine.query_rewrite.prepare",
                       new=mock.AsyncMock(return_value=(
                           REWRITTEN, [fake_doc],
                           {"mode": "parallel", "used_rewrite": True}))),
        ]
        retrieve_mock = patches[3].new

        async def run():
            with patches[0]:
                with _enter_all(patches[1:]):
                    return await rag_engine.chat(
                        ChatRequest(query=ORIG), identity="x")

        result = asyncio.run(run())
        assert result.answer == "答案"
        assert result.message == "ok"
        # round 0 用择优文档 → 检索未被调用（零重复检索）
        retrieve_mock.assert_not_called()

    def test_chat_disabled_prepare_not_called(self, monkeypatch):
        # 开关关闭 → 不分诊不改写，行为与现状完全一致
        from rag.schemas import ChatRequest

        monkeypatch.setattr(engine_module.settings, "query_rewrite_enabled", False)

        patches = self._base_chat_patches() + [
            mock.patch("rag.engine.query_rewrite.prepare",
                       new=mock.AsyncMock(return_value=(ORIG, None, {"mode": "precise"}))),
        ]
        prepare_mock = patches[-1].new
        retrieve_mock = patches[3].new

        async def run():
            with patches[0]:
                with _enter_all(patches[1:]):
                    return await rag_engine.chat(
                        ChatRequest(query=ORIG), identity="x")

        result = asyncio.run(run())
        assert result.answer == "答案"
        prepare_mock.assert_not_called()
        # 关闭时走原检索流程（retrieve 被调用）
        assert retrieve_mock.await_count >= 1


class TestEngineRetrieveIntegration:
    """_retrieve 接入：改写后 query 作为 HyDE 扩展基础（改写与 HyDE 正交）"""

    def test_retrieve_uses_rewritten_base_for_hyde(self, monkeypatch):
        monkeypatch.setattr(engine_module.settings, "query_rewrite_enabled", True)
        hyde_args = []

        async def _fake_hyde(q):
            hyde_args.append(q)
            return f"HyDE({q})"

        patches = [
            mock.patch("rag.engine.cache.get", mock.AsyncMock(return_value=None)),
            mock.patch("rag.engine.cache.set", mock.AsyncMock(return_value=True)),
            mock.patch.object(rag_engine, "_hyde_expand",
                              mock.AsyncMock(side_effect=_fake_hyde)),
            mock.patch("rag.engine.hybrid_retriever.retrieve",
                       mock.AsyncMock(return_value=[{
                           "id": 1, "title": "t", "content": "c",
                           "hybrid_score": 0.9, "parent_id": None}])),
            mock.patch("rag.engine.graph_extractor.extract_from_query",
                       mock.AsyncMock(return_value=["实体1"])),
            mock.patch("rag.engine.graph_store.search_related",
                       mock.AsyncMock(return_value=[])),
            mock.patch("agent.reflector.reflector.check_sufficiency",
                       mock.AsyncMock(return_value={"sufficient": True})),
            mock.patch("rag.engine.query_rewrite.prepare_query",
                       mock.AsyncMock(return_value=(
                           REWRITTEN, {"mode": "rewrite_accepted", "fidelity": 0.88}))),
        ]
        retrieve_mock = patches[3].new

        async def run():
            with _enter_all(patches):
                return await rag_engine._retrieve(ORIG)

        docs = asyncio.run(run())
        assert len(docs) == 1
        # 改写 query 作为 HyDE 基础（HyDE 在改写之上继续扩展，正交）
        assert hyde_args == [REWRITTEN]
        assert retrieve_mock.await_count >= 1
        assert retrieve_mock.await_args.args[0] == f"HyDE({REWRITTEN})"


def _enter_all(patches):
    """嵌套进入多个 mock.patch（按顺序退出）"""
    return _StackCtx(patches)


class _StackCtx:
    """顺序进入 patch 的上下文管理器（替代嵌套 with，保持可读性）"""

    def __init__(self, patches):
        self._patches = patches

    def __enter__(self):
        self._entered = [p.__enter__() for p in self._patches]
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.__exit__(*exc)
        return False
