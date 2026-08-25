"""审查节点增强单元测试（module-078）

覆盖验收矩阵：阈值配置化生效（含动态调整）、审查策略三档（fail-open /
lenient / strict）、矛盾检测（dual 双确认 / 单判 / 对称回退 / fail-open）、
review_score 四层透传、结构化日志、conflict_count 汇总。

全 mock：conftest autouse 钉住 crawl_enabled=false / memory_conflict_enabled=false，
单测内显式覆盖。不加载真实模型、不依赖真实 DB、不触发真实网络请求。
"""
import asyncio
import logging

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag.crawl.crawler import (
    _check_conflict,
    _conflict_candidates,
    _judge_conflict,
    _review_content,
    _crawl_page_and_store,
    run_crawl,
    CrawlResult,
    CrawlSummary,
    ReviewResult,
)


def _mock_judges(sufficient: bool = True, score: float = 0.8) -> dict:
    """构造 reflector / hhem mock 模块（对齐 module-075 测试模式）"""
    mock_reflector = MagicMock()
    mock_reflector.check_sufficiency = AsyncMock(return_value={"sufficient": sufficient})
    mock_hhem = MagicMock()
    mock_hhem.predict = AsyncMock(return_value=[score])
    mock_ref_mod = MagicMock()
    mock_ref_mod.reflector = mock_reflector
    mock_hhem_mod = MagicMock()
    mock_hhem_mod.hhem_judge = mock_hhem
    return {"agent.reflector": mock_ref_mod, "rag.retrieval.factcheck_judge": mock_hhem_mod}


async def _review(mock_modules: dict, conflict_info: dict, **settings_kwargs):
    """运行 _review_content：mock 判定模块 + _check_conflict，返回 ReviewResult
    （settings 修改用 finally 还原，防策略/阈值泄漏到后续用例）"""
    from src.config import settings
    saved = {k: getattr(settings, k) for k in settings_kwargs}
    for k, v in settings_kwargs.items():
        setattr(settings, k, v)
    try:
        with patch.dict("sys.modules", mock_modules), \
             patch("rag.crawl.crawler._check_conflict", new_callable=AsyncMock,
                   return_value=conflict_info):
            return await _review_content("https://a.com", "content", "Title")
    finally:
        for k, v in saved.items():
            setattr(settings, k, v)


# ─── 阈值配置化（验收 1.1） ───


class TestThresholdConfig:
    @pytest.mark.asyncio
    async def test_default_threshold_zero_regression(self):
        """默认 0.3：score 0.3 通过、0.29 拒绝（module-075 硬编码值零回归）"""
        r1 = await _review(_mock_judges(score=0.3), {"conflict": False, "detail": ""})
        assert r1 == "approved"
        r2 = await _review(_mock_judges(score=0.29), {"conflict": False, "detail": ""})
        assert r2 == "rejected"
        assert r2.score == 0.29

    @pytest.mark.asyncio
    async def test_threshold_raised_changes_verdict(self):
        """阈值动态调高 0.5：score 0.4 由 approved 变 rejected（进程内修改即时生效）"""
        r1 = await _review(_mock_judges(score=0.4), {"conflict": False, "detail": ""})
        assert r1 == "approved"
        r2 = await _review(_mock_judges(score=0.4), {"conflict": False, "detail": ""},
                           crawl_hhem_threshold=0.5)
        assert r2 == "rejected"

    @pytest.mark.asyncio
    async def test_strict_uses_strict_threshold(self):
        """strict 档使用 crawl_hhem_threshold_strict（0.45）：score 0.4 拒绝、0.5 通过"""
        r1 = await _review(_mock_judges(score=0.4), {"conflict": False, "detail": ""},
                           crawl_review_policy="strict")
        assert r1 == "rejected"
        r2 = await _review(_mock_judges(score=0.5), {"conflict": False, "detail": ""},
                           crawl_review_policy="strict")
        assert r2 == "approved"

    @pytest.mark.asyncio
    async def test_hhem_unavailable_score_none(self):
        """HHEM 不可用（predict 返回 None）→ score=None 且不因分数拒绝（review_score 落 NULL）"""
        mock_hhem = MagicMock()
        mock_hhem.predict = AsyncMock(return_value=None)
        mock_hhem_mod = MagicMock()
        mock_hhem_mod.hhem_judge = mock_hhem
        modules = {"agent.reflector": MagicMock(
            reflector=MagicMock(check_sufficiency=AsyncMock(return_value={"sufficient": True})))}
        r = await _review({**modules, "rag.retrieval.factcheck_judge": mock_hhem_mod},
                          {"conflict": False, "detail": ""})
        assert r == "approved"
        assert r.score is None


# ─── 审查策略三档（验收 1.2） ───


class TestReviewPolicies:
    @pytest.mark.asyncio
    async def test_fail_open_conflict_only_recorded(self):
        """fail-open：矛盾命中仅记录（conflict=True），status 仍 approved"""
        r = await _review(_mock_judges(), {"conflict": True, "detail": "与库中文档 id=5 矛盾（判定器=dual）"})
        assert r == "approved"
        assert r.conflict is True
        assert "id=5" in r.conflict_detail

    @pytest.mark.asyncio
    async def test_lenient_conflict_rejected(self):
        r = await _review(_mock_judges(), {"conflict": True, "detail": "矛盾"},
                          crawl_review_policy="lenient")
        assert r == "rejected"

    @pytest.mark.asyncio
    async def test_strict_conflict_rejected(self):
        r = await _review(_mock_judges(score=0.9), {"conflict": True, "detail": "矛盾"},
                          crawl_review_policy="strict")
        assert r == "rejected"

    @pytest.mark.asyncio
    async def test_fail_open_exception_approved(self):
        """审查环节异常 → fail-open 默认 approved（module-075 行为零回归）"""
        r = await _review({"agent": None, "agent.reflector": None},
                          {"conflict": False, "detail": ""})
        assert r == "approved"

    @pytest.mark.asyncio
    async def test_lenient_exception_approved(self):
        """lenient：审查异常仍放行 approved"""
        r = await _review({"agent": None, "agent.reflector": None},
                          {"conflict": False, "detail": ""},
                          crawl_review_policy="lenient")
        assert r == "approved"

    @pytest.mark.asyncio
    async def test_strict_exception_rejected(self):
        """strict：审查异常 fail-closed → rejected（宁缺毋滥）"""
        r = await _review({"agent": None, "agent.reflector": None},
                          {"conflict": False, "detail": ""},
                          crawl_review_policy="strict")
        assert r == "rejected"


# ─── 矛盾检测（验收 1.3） ───


class TestConflictJudging:
    @pytest.mark.asyncio
    async def test_dual_double_confirm(self):
        """dual 双确认：nli + clf 同时 contradiction 才判矛盾"""
        with patch("rag.crawl.crawler._nli_contradicts", new_callable=AsyncMock, return_value=True), \
             patch("rag.crawl.crawler._clf_contradicts", new_callable=AsyncMock, return_value=True):
            hit, used = await _judge_conflict("p", "h", "dual")
        assert hit is True and used == "dual"

    @pytest.mark.asyncio
    async def test_dual_single_disagree_not_conflict(self):
        """dual 单判 contradiction → 不判矛盾（宁漏检也不错标）"""
        with patch("rag.crawl.crawler._nli_contradicts", new_callable=AsyncMock, return_value=True), \
             patch("rag.crawl.crawler._clf_contradicts", new_callable=AsyncMock, return_value=False):
            hit, used = await _judge_conflict("p", "h", "dual")
        assert hit is False and used == "dual"

    @pytest.mark.asyncio
    async def test_dual_nli_unavailable_falls_back_clf(self):
        """对称回退：nli 不可用 → clf 单判"""
        with patch("rag.crawl.crawler._nli_contradicts", new_callable=AsyncMock, return_value=None), \
             patch("rag.crawl.crawler._clf_contradicts", new_callable=AsyncMock, return_value=True):
            hit, used = await _judge_conflict("p", "h", "dual")
        assert hit is True and used == "clf"

    @pytest.mark.asyncio
    async def test_dual_clf_unavailable_falls_back_nli(self):
        with patch("rag.crawl.crawler._nli_contradicts", new_callable=AsyncMock, return_value=True), \
             patch("rag.crawl.crawler._clf_contradicts", new_callable=AsyncMock, return_value=None):
            hit, used = await _judge_conflict("p", "h", "dual")
        assert hit is True and used == "nli"

    @pytest.mark.asyncio
    async def test_dual_both_unavailable_skip(self):
        """双不可用 → 跳过矛盾检测（fail-open 回退基础审查）"""
        with patch("rag.crawl.crawler._nli_contradicts", new_callable=AsyncMock, return_value=None), \
             patch("rag.crawl.crawler._clf_contradicts", new_callable=AsyncMock, return_value=None):
            hit, used = await _judge_conflict("p", "h", "dual")
        assert hit is False and used == "dual"

    @pytest.mark.asyncio
    async def test_nli_mode_single(self):
        with patch("rag.crawl.crawler._nli_contradicts", new_callable=AsyncMock, return_value=True):
            hit, used = await _judge_conflict("p", "h", "nli")
        assert hit is True and used == "nli"

    @pytest.mark.asyncio
    async def test_clf_mode_single(self):
        with patch("rag.crawl.crawler._clf_contradicts", new_callable=AsyncMock, return_value=True):
            hit, used = await _judge_conflict("p", "h", "clf")
        assert hit is True and used == "clf"


class TestCheckConflict:
    @pytest.mark.asyncio
    async def test_disabled_gate_short_circuits(self, monkeypatch):
        """memory_conflict_enabled=false（conftest 钉住）→ 直接跳过（hermetic）"""
        from src.config import settings
        monkeypatch.setattr(settings, "memory_conflict_enabled", False)
        info = await _check_conflict("内容")
        assert info == {"conflict": False, "detail": ""}

    @pytest.mark.asyncio
    async def test_hit_returns_detail(self, monkeypatch):
        """候选判定矛盾 → conflict=True + detail 含候选 id/标题/判定器"""
        from src.config import settings
        monkeypatch.setattr(settings, "memory_conflict_enabled", True)
        with patch("rag.retrieval.embeddings.embedding_service.embed_text",
                   new_callable=AsyncMock, return_value=[0.1, 0.2]) as mock_emb, \
             patch("rag.crawl.crawler._conflict_candidates", new_callable=AsyncMock,
                   return_value=[{"id": 5, "title": "旧文档", "content": "premise"}]) as mock_cand, \
             patch("rag.crawl.crawler._judge_conflict", new_callable=AsyncMock,
                   return_value=(True, "dual")) as mock_judge:
            info = await _check_conflict("新内容")
        assert info["conflict"] is True
        assert "id=5" in info["detail"] and "dual" in info["detail"]
        mock_emb.assert_awaited_once()
        mock_cand.assert_awaited_once()
        mock_judge.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_candidate_no_conflict(self, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "memory_conflict_enabled", True)
        with patch("rag.retrieval.embeddings.embedding_service.embed_text",
                   new_callable=AsyncMock, return_value=[0.1]), \
             patch("rag.crawl.crawler._conflict_candidates", new_callable=AsyncMock,
                   return_value=[]):
            info = await _check_conflict("新内容")
        assert info == {"conflict": False, "detail": ""}

    @pytest.mark.asyncio
    async def test_embed_failure_fail_open(self, monkeypatch):
        """嵌入失败 → fail-open 跳过（不阻断入库主链路）"""
        from src.config import settings
        monkeypatch.setattr(settings, "memory_conflict_enabled", True)
        with patch("rag.retrieval.embeddings.embedding_service.embed_text",
                   new_callable=AsyncMock, side_effect=RuntimeError("模型挂了")):
            info = await _check_conflict("内容")
        assert info == {"conflict": False, "detail": ""}

    @pytest.mark.asyncio
    async def test_candidates_filtered_by_cosine(self, monkeypatch):
        """候选仅保留 cosine ≥ 下限；SQL 限定根父块 + embedding 非空"""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (1, "A", "内容A", 0.85),
            (2, "B", "内容B", 0.61),
            (3, "C", "内容C", 0.59),  # 低于 0.6 下限 → 丢弃
        ]
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("src.database.async_session_factory", mock_factory):
            candidates = await _conflict_candidates([0.1, 0.2])
        assert [c["id"] for c in candidates] == [1, 2]
        sql_str = str(mock_session.execute.call_args[0][0])
        assert "parent_id IS NULL" in sql_str
        assert "embedding IS NOT NULL" in sql_str


# ─── review_score 四层透传（验收 1.4） ───


class TestReviewScorePassthrough:
    @pytest.mark.asyncio
    async def test_crawl_page_to_ingest(self, monkeypatch):
        """层 1→2：_crawl_page_and_store 把 review_score/conflict 传给 ingest_document"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_enabled", True)
        with patch("rag.crawl.crawler.fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = CrawlResult(url="https://a.com", success=True, content="c", title="A")
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = ReviewResult("approved", score=0.88, conflict=True,
                                                        conflict_detail="与库中文档 id=5 矛盾（判定器=dual）")
                with patch("rag.retrieval.document_ingest.ingest_document",
                           new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 7, "chunks": 1}
                    summary = CrawlSummary()
                    await _crawl_page_and_store("https://a.com", summary)
        assert summary.crawled == 1
        assert summary.approved == 1
        assert summary.conflict_count == 1
        assert mock_ingest.call_args[1]["review_status"] == "approved"
        assert mock_ingest.call_args[1]["review_score"] == 0.88
        detail = summary.details[0]
        assert detail["review_score"] == 0.88
        assert detail["conflict"] is True

    @pytest.mark.asyncio
    async def test_ingest_to_add_document(self, monkeypatch, tmp_path):
        """层 2→3：ingest_document 把 review_score 传给 add_document"""
        from rag.retrieval import document_ingest
        from rag.retrieval.document_parser import ParsedDocument
        from rag.retrieval.document_ingest import ingest_document
        from rag.engine import rag_engine

        monkeypatch.setattr(document_ingest.settings, "upload_dir", str(tmp_path))
        monkeypatch.setattr(document_ingest.settings, "doc_dedup_semantic_enabled", False)
        monkeypatch.setattr(document_ingest.document_parser, "parse_document",
                            lambda data, filename="": ParsedDocument(text="正文", format="md",
                                                                     engine="text", page_count=None))
        monkeypatch.setattr(document_ingest.image_pipeline, "process_pdf_images",
                            lambda md, page_count=None: md)
        monkeypatch.setattr(document_ingest.document_cleaner, "clean",
                            lambda text, source_format="": text)
        monkeypatch.setattr(document_ingest.document_cleaner, "normalize",
                            lambda text, max_chars=None: text)
        monkeypatch.setattr(document_ingest, "_find_exact_duplicate",
                            AsyncMock(return_value=None))
        monkeypatch.setattr(document_ingest.document_dedup, "find_semantic_duplicate",
                            AsyncMock(return_value=None))
        monkeypatch.setattr(document_ingest.document_dedup, "compute_doc_embedding",
                            AsyncMock(return_value=None))
        add_mock = AsyncMock(return_value={"id": 1, "title": "t", "chunks": 1, "duplicate": False})
        monkeypatch.setattr(rag_engine, "add_document", add_mock)

        await ingest_document(b"bytes", "a.md", "文档A",
                              review_status="rejected", review_score=0.72)
        assert add_mock.await_args.kwargs["review_status"] == "rejected"
        assert add_mock.await_args.kwargs["review_score"] == 0.72

    @pytest.mark.asyncio
    async def test_add_document_persists_review_score(self, monkeypatch):
        """层 3→4：add_document 把 review_score 写入 Document ORM（父块+子块）"""
        from rag.engine import rag_engine

        class _FakeAddSession:
            """add_document 用假会话：execute 返回无重复，flush 赋 id，commit 成功"""
            def __init__(self):
                self._added = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def execute(self, stmt):
                result = MagicMock()
                result.scalar_one_or_none.return_value = None
                return result

            def add(self, obj):
                self._added.append(obj)

            async def flush(self):
                for i, o in enumerate(self._added):
                    if o.id is None:
                        o.id = i + 1

            async def commit(self):
                pass

            async def rollback(self):
                pass

        session = _FakeAddSession()
        factory = MagicMock(return_value=session)
        with patch("rag.engine.async_session_factory", factory), \
             patch("rag.engine.chunker.chunk", return_value={
                 "parents": [{"title": "s", "content": "内容"}],
                 "children": [{"title": "s", "content": "内容", "parent_index": 0}],
             }), \
             patch("rag.engine.embedding_service.embed_documents",
                   AsyncMock(return_value=[[0.1, 0.2, 0.3]])), \
             patch("rag.engine.tokenize", return_value=""), \
             patch("rag.engine.cache.delete_by_prefix", AsyncMock(return_value=True)), \
             patch("rag.engine.graph_store.ensure_graph", AsyncMock(return_value=True)), \
             patch("rag.engine.graph_extractor.extract_from_document",
                   AsyncMock(return_value={"entities": [], "relations": []})):
            result = await rag_engine.add_document(
                "测试文档", "这是一段文档内容", "crawl:https://x.com",
                review_status="rejected", review_score=0.72)
        assert result["duplicate"] is False
        assert session._added, "add_document 应至少 add 一个 Document"
        assert all(d.review_score == 0.72 for d in session._added)
        assert all(d.review_status == "rejected" for d in session._added)


# ─── 汇总与日志（验收 1.4/1.5） ───


class TestSummaryAndLogging:
    @pytest.mark.asyncio
    async def test_run_crawl_conflict_count(self, monkeypatch):
        """fail-open 档矛盾命中：status 仍 approved，但 conflict_count 独立计数"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_enabled", True)
        with patch("rag.crawl.crawler.fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = CrawlResult(url="https://a.com", success=True, content="c", title="A")
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = ReviewResult("approved", score=0.7, conflict=True)
                with patch("rag.retrieval.document_ingest.ingest_document",
                           new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 1, "chunks": 1}
                    result = await run_crawl([{"url_pattern": "https://a.com"}])
        assert result.approved == 1
        assert result.conflict_count == 1

    def test_review_log_line_structured(self, caplog):
        """每次审查一行结构化日志：url/status/score/sufficient/conflict/policy/elapsed_ms"""
        with patch.dict("sys.modules", _mock_judges(score=0.8)), \
             patch("rag.crawl.crawler._check_conflict", new_callable=AsyncMock,
                   return_value={"conflict": True,
                                 "detail": "与库中文档 id=5 标题='旧文档' 矛盾（判定器=dual）"}):
            with caplog.at_level(logging.INFO, logger="rag.crawl.crawler"):
                result = asyncio.run(_review_content("https://a.com", "content", "Title"))
        assert result == "approved"  # fail-open 默认
        assert "审查完成" in caplog.text
        assert "score=0.8" in caplog.text
        assert "sufficient=True" in caplog.text
        assert "conflict=True" in caplog.text
        assert "policy=fail-open" in caplog.text
        assert "elapsed_ms=" in caplog.text
        # 矛盾命中日志含候选 id / 标题 / 判定器
        assert "矛盾命中" in caplog.text
        assert "id=5" in caplog.text
        assert "dual" in caplog.text

    @pytest.mark.asyncio
    async def test_run_endpoint_includes_conflict(self, monkeypatch):
        """POST /ai/crawl/run 响应 data 含 conflict 计数"""
        import main
        with patch("rag.crawl.crawler._load_sources_from_db", new_callable=AsyncMock,
                   return_value=[{"url_pattern": "https://a.com", "enabled": True}]), \
             patch("rag.crawl.crawler.run_crawl", new_callable=AsyncMock) as mock_run:
            summary = CrawlSummary()
            summary.crawled = 1
            summary.conflict_count = 2
            mock_run.return_value = summary
            resp = await main.trigger_crawl()
        assert resp["data"]["conflict"] == 2
