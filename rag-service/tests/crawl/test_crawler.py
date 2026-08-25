"""
知识抓取流水线单元测试（module-075）

全 mock：conftest autouse 钉住 crawl_enabled=false，单测内显式覆盖。
不加载真实模型、不依赖真实 DB、不触发真实网络请求。
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from rag.crawl.crawler import (
    _is_safe_url,
    _matches_any,
    _review_content,
    fetch_page,
    run_crawl,
    CrawlResult,
    CrawlSummary,
    _load_sources_from_db,
    _scheduled_crawl_job,
    start_scheduler,
    shutdown_scheduler,
)


# ─── URL 安全校验 ───


class TestIsSafeUrl:
    def test_http_allowed(self):
        assert _is_safe_url("http://example.com") is True

    def test_https_allowed(self):
        assert _is_safe_url("https://example.com/path") is True

    def test_file_blocked(self):
        assert _is_safe_url("file:///etc/passwd") is False

    def test_ftp_blocked(self):
        assert _is_safe_url("ftp://files.example.com") is False

    def test_empty_blocked(self):
        assert _is_safe_url("") is False

    def test_case_insensitive(self):
        assert _is_safe_url("HTTPS://Example.COM") is True


# ─── 白名单/黑名单匹配 ───


class TestMatchesAny:
    def test_prefix_match(self):
        assert _matches_any("https://spring.io/docs/guide", ["https://spring.io/docs"]) is True

    def test_no_match(self):
        assert _matches_any("https://evil.com", ["https://spring.io"]) is False

    def test_empty_patterns(self):
        assert _matches_any("https://spring.io", []) is False

    def test_case_insensitive(self):
        assert _matches_any("https://SPRING.IO/docs", ["https://spring.io"]) is True


# ─── 单页抓取 ───


class TestFetchPage:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_resp = MagicMock()
        mock_resp.text = "<html><title>Test Page</title><body>Hello</body></html>"
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.raise_for_status = MagicMock()

        with patch("rag.crawl.crawler.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await fetch_page("https://example.com")
            assert result.success is True
            assert "Test Page" in result.title
            assert "Hello" in result.content

    @pytest.mark.asyncio
    async def test_unsafe_url(self):
        result = await fetch_page("file:///etc/passwd")
        assert result.success is False
        assert "不安全" in result.error

    @pytest.mark.asyncio
    async def test_timeout(self):
        import httpx
        with patch("rag.crawl.crawler.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await fetch_page("https://example.com")
            assert result.success is False
            assert "超时" in result.error

    @pytest.mark.asyncio
    async def test_http_error(self):
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("rag.crawl.crawler.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(
                side_effect=httpx.HTTPStatusError("404", request=MagicMock(), response=mock_resp)
            )
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await fetch_page("https://example.com/missing")
            assert result.success is False
            assert "404" in result.error


# ─── 审查节点 ───


class TestReviewContent:
    @pytest.mark.asyncio
    async def test_approved_when_sufficient(self):
        mock_reflector = MagicMock()
        mock_reflector.check_sufficiency = AsyncMock(return_value={"sufficient": True})
        mock_hhem = MagicMock()
        mock_hhem.predict = AsyncMock(return_value=[0.8])

        mock_ref_mod = MagicMock()
        mock_ref_mod.reflector = mock_reflector
        mock_hhem_mod = MagicMock()
        mock_hhem_mod.hhem_judge = mock_hhem

        with patch.dict("sys.modules", {
            "agent.reflector": mock_ref_mod,
            "rag.retrieval.factcheck_judge": mock_hhem_mod,
        }):
            result = await _review_content("https://example.com", "good content", "Title")
            assert result == "approved"

    @pytest.mark.asyncio
    async def test_rejected_when_reflector_insufficient(self):
        mock_reflector = MagicMock()
        mock_reflector.check_sufficiency = AsyncMock(return_value={"sufficient": False, "reason": "empty"})

        mock_ref_mod = MagicMock()
        mock_ref_mod.reflector = mock_reflector

        with patch.dict("sys.modules", {"agent.reflector": mock_ref_mod}):
            result = await _review_content("https://example.com", "empty page", "Title")
            assert result == "rejected"

    @pytest.mark.asyncio
    async def test_rejected_when_factcheck_low(self):
        mock_reflector = MagicMock()
        mock_reflector.check_sufficiency = AsyncMock(return_value={"sufficient": True})
        mock_hhem = MagicMock()
        mock_hhem.predict = AsyncMock(return_value=[0.1])

        mock_ref_mod = MagicMock()
        mock_ref_mod.reflector = mock_reflector
        mock_hhem_mod = MagicMock()
        mock_hhem_mod.hhem_judge = mock_hhem

        with patch.dict("sys.modules", {
            "agent.reflector": mock_ref_mod,
            "rag.retrieval.factcheck_judge": mock_hhem_mod,
        }):
            result = await _review_content("https://example.com", "spam", "Spam")
            assert result == "rejected"

    @pytest.mark.asyncio
    async def test_fail_open_on_import_error(self):
        """import 失败时 fail-open → approved"""
        with patch.dict("sys.modules", {"agent": None, "agent.reflector": None}):
            result = await _review_content("https://example.com", "content", "title")
            assert result == "approved"


# ─── 批量抓取 ───


class TestRunCrawl:
    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self, monkeypatch):
        """crawl_enabled=false 时直接返回空"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_enabled", False)
        result = await run_crawl([{"url_pattern": "https://example.com"}])
        assert result.crawled == 0

    @pytest.mark.asyncio
    async def test_unsafe_url_skipped(self, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_enabled", True)
        result = await run_crawl([{"url_pattern": "ftp://evil.com"}])
        assert result.skipped == 1
        assert result.crawled == 0

    @pytest.mark.asyncio
    async def test_fetch_failure_counted(self, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_enabled", True)

        with patch("rag.crawl.crawler.fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = CrawlResult(url="https://example.com", success=False, error="timeout")
            result = await run_crawl([{"url_pattern": "https://example.com"}])
            assert result.errors == 1
            assert result.crawled == 0

    @pytest.mark.asyncio
    async def test_success_with_ingest(self, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_enabled", True)

        with patch("rag.crawl.crawler.fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = CrawlResult(
                url="https://spring.io/docs", success=True,
                content="<html><title>Spring</title>body</html>", title="Spring",
            )
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = "approved"
                with patch("rag.retrieval.document_ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 42, "chunks": 3}
                    result = await run_crawl([{"url_pattern": "https://spring.io/docs", "name": "Spring Docs"}])
                    assert result.crawled == 1
                    assert result.approved == 1
                    assert result.rejected == 0
                    assert result.errors == 0

    @pytest.mark.asyncio
    async def test_rejected_still_ingested(self, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_enabled", True)

        with patch("rag.crawl.crawler.fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = CrawlResult(
                url="https://spam.com", success=True,
                content="spam content", title="Spam",
            )
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = "rejected"
                with patch("rag.retrieval.document_ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 99, "chunks": 1}
                    result = await run_crawl([{"url_pattern": "https://spam.com"}])
                    assert result.crawled == 1
                    assert result.rejected == 1
                    # rejected 但仍入库（不丢数据）
                    mock_ingest.assert_called_once()

    @pytest.mark.asyncio
    async def test_max_pages_limit(self, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_enabled", True)
        monkeypatch.setattr(settings, "crawl_max_pages_per_run", 1)

        sources = [
            {"url_pattern": "https://a.com"},
            {"url_pattern": "https://b.com"},
        ]
        with patch("rag.crawl.crawler.fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = CrawlResult(url="https://a.com", success=False, error="fail")
            result = await run_crawl(sources)
            # 只会尝试第一个（max_pages=1），然后停止
            assert mock_fetch.call_count == 1


# ─── DB 源配置加载 ───


    @pytest.mark.asyncio
    async def test_returns_rows(self):
        mock_row = (1, "https://spring.io", "Spring", True, None)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("src.database.async_session_factory", mock_factory):
            sources = await _load_sources_from_db()
            assert len(sources) == 1
            assert sources[0]["url_pattern"] == "https://spring.io"

    @pytest.mark.asyncio
    async def test_db_error_returns_empty(self):
        mock_factory = MagicMock(side_effect=Exception("db down"))
        with patch("src.database.async_session_factory", mock_factory):
            sources = await _load_sources_from_db()
            assert sources == []


# ─── 定时任务 ───


class TestScheduledJob:
    @pytest.mark.asyncio
    async def test_no_sources_skips(self):
        with patch("rag.crawl.crawler._load_sources_from_db", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = []
            # 不应抛异常
            await _scheduled_crawl_job()

    @pytest.mark.asyncio
    async def test_disabled_sources_skips(self):
        with patch("rag.crawl.crawler._load_sources_from_db", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = [{"url_pattern": "https://a.com", "enabled": False}]
            await _scheduled_crawl_job()


# ─── 调度器生命周期 ───


class TestSchedulerLifecycle:
    def test_start_when_disabled(self, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_enabled", False)
        # 不应抛异常
        start_scheduler()

    def test_shutdown_noop_when_none(self):
        # 不应抛异常
        shutdown_scheduler()
