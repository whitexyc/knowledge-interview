"""
递归抓取单元测试（module-076）

覆盖验收矩阵：深度 0/1/2、全局深度上限、A→B→A 循环、跨源去重、
黑名单（种子 + 递归）、外域丢弃、URL 规范化、链接截断、单页失败
不阻断、审查/入库 fail-open、总页数上限全树共享、空/非 HTML 终止、
入库文件名防 crawl_.txt、DB max_depth 字段加载。

全 mock：conftest autouse 钉住 crawl_enabled=false，单测内显式覆盖。
不加载真实模型、不依赖真实 DB、不触发真实网络请求。
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag.crawl.crawler import (
    _normalize_url,
    _extract_links,
    _crawl_filename,
    _blacklist_patterns,
    _load_sources_from_db,
    run_crawl,
    CrawlResult,
)


def _html(title: str, *links: str) -> str:
    """构造含指定 href 链接的简单 HTML"""
    hrefs = "".join(f'<a href="{link}">x</a>' for link in links)
    return f"<html><title>{title}</title><body>{hrefs}</body></html>"


def _enable_crawl(monkeypatch) -> None:
    """测试体内显式打开抓取开关（conftest autouse 已钉住 false）"""
    from src.config import settings
    monkeypatch.setattr(settings, "crawl_enabled", True)




# ─── URL 规范化（验收 1.2.1-1.2.3） ───


class TestNormalizeUrl:
    def test_drop_fragment(self):
        assert _normalize_url("https://example.com/path#frag") == "https://example.com/path"

    def test_drop_trailing_slash(self):
        assert _normalize_url("https://example.com/path/") == "https://example.com/path"

    def test_lowercase_scheme_host(self):
        assert _normalize_url("HTTPS://Example.COM") == "https://example.com"

    def test_keeps_port(self):
        assert _normalize_url("https://example.com:8080/x#f") == "https://example.com:8080/x"

    def test_keeps_query(self):
        assert _normalize_url("https://example.com/p?q=1#f") == "https://example.com/p?q=1"

    def test_invalid_port_fail_open(self):
        assert _normalize_url("https://example.com:bad/") == "https://example.com:bad/"


# ─── 链接提取（验收 1.2.8/1.2.9/1.2.10、1.3.5、2.2.1） ───


class TestExtractLinks:
    def test_relative_url_resolved(self):
        links = _extract_links(_html("P", "/docs/guide"), "https://example.com/a/b", 20)
        assert links == ["https://example.com/docs/guide"]

    def test_unsafe_schemes_filtered(self):
        html = (
            '<a href="mailto:a@b.com">m</a><a href="javascript:void(0)">j</a>'
            '<a href="ftp://files.example.com/x">f</a><a href="data:text/plain,hi">d</a>'
            '<a href="file:///etc/passwd">p</a><a href="https://ok.com/">o</a>'
        )
        links = _extract_links(html, "https://example.com/", 20)
        assert links == ["https://ok.com/"]

    def test_dedupe_after_normalize(self):
        html = '<a href="https://example.com/x#a">1</a><a href="https://example.com/x/">2</a>'
        links = _extract_links(html, "https://example.com/", 20)
        assert links == ["https://example.com/x"]

    def test_truncate_over_max(self):
        html = _html("P", *[f"https://example.com/p{i}" for i in range(25)])
        links = _extract_links(html, "https://example.com/", 20)
        assert len(links) == 20

    def test_empty_or_non_html(self):
        assert _extract_links("", "https://example.com/", 20) == []
        assert _extract_links("PDF binary content", "https://example.com/a.pdf", 20) == []


# ─── 入库文件名（验收 1.2.11） ───


class TestCrawlFilename:
    def test_trailing_slash_not_empty_segment(self):
        name = _crawl_filename("https://example.com/docs/")
        assert "crawl_.txt" not in name
        assert name == "crawl_docs.txt"

    def test_host_fallback(self):
        assert _crawl_filename("https://example.com/") == "crawl_example.com.txt"

    def test_query_stripped(self):
        assert _crawl_filename("https://example.com/page?id=1") == "crawl_page.txt"


# ─── 黑名单配置（验收 2.2.4） ───


class TestBlacklistPatterns:
    def test_parses_comma_separated(self, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_blacklist_patterns", "https://csdn.net, https://blog.csdn.net")
        assert _blacklist_patterns() == ["https://csdn.net", "https://blog.csdn.net"]

    def test_empty_default(self):
        assert _blacklist_patterns() == []


# ─── 递归抓取场景矩阵 ───


class TestRecursiveCrawl:
    @pytest.mark.asyncio
    async def test_depth_zero_seed_only(self, monkeypatch):
        """max_depth=0 → 仅种子页（验收 1.1.1），不跟踪链接"""
        _enable_crawl(monkeypatch)
        with patch("rag.crawl.crawler.fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = CrawlResult(
                url="https://a.com", success=True,
                content=_html("A", "https://a.com/b"), title="A",
            )
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = "approved"
                with patch("rag.retrieval.document_ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 1, "chunks": 1}
                    result = await run_crawl([{"url_pattern": "https://a.com", "max_depth": 0}])
                    assert result.crawled == 1
                    assert mock_fetch.call_count == 1
                    assert mock_ingest.call_count == 1

    @pytest.mark.asyncio
    async def test_depth_one_seed_plus_links(self, monkeypatch):
        """max_depth=1 → 种子 + 一层白名单链接（验收 1.1.2）"""
        _enable_crawl(monkeypatch)

        async def _fetch(url):
            pages = {
                "https://a.com": CrawlResult(url=url, success=True, content=_html("A", "https://a.com/b", "https://a.com/c"), title="A"),
                "https://a.com/b": CrawlResult(url=url, success=True, content=_html("B"), title="B"),
                "https://a.com/c": CrawlResult(url=url, success=True, content=_html("C"), title="C"),
            }
            return pages[url]

        with patch("rag.crawl.crawler.fetch_page", side_effect=_fetch) as mock_fetch:
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = "approved"
                with patch("rag.retrieval.document_ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 1, "chunks": 1}
                    result = await run_crawl([{"url_pattern": "https://a.com", "max_depth": 1}])
                    assert result.crawled == 3
                    assert mock_fetch.call_count == 3

    @pytest.mark.asyncio
    async def test_depth_two_two_levels(self, monkeypatch):
        """max_depth=2 → 两层递归，第三层不展开（验收 1.1.3）"""
        _enable_crawl(monkeypatch)

        async def _fetch(url):
            pages = {
                "https://a.com": CrawlResult(url=url, success=True, content=_html("A", "https://a.com/b"), title="A"),
                "https://a.com/b": CrawlResult(url=url, success=True, content=_html("B", "https://a.com/c"), title="B"),
                "https://a.com/c": CrawlResult(url=url, success=True, content=_html("C", "https://a.com/d"), title="C"),
                "https://a.com/d": CrawlResult(url=url, success=True, content=_html("D"), title="D"),
            }
            return pages[url]

        with patch("rag.crawl.crawler.fetch_page", side_effect=_fetch) as mock_fetch:
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = "approved"
                with patch("rag.retrieval.document_ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 1, "chunks": 1}
                    await run_crawl([{"url_pattern": "https://a.com", "max_depth": 2}])
                    # a(0) b(1) c(2) 被抓；d 在 depth 3 > 2 不抓
                    assert mock_fetch.call_count == 3

    @pytest.mark.asyncio
    async def test_global_depth_cap(self, monkeypatch):
        """source.max_depth=5 但全局 crawl_max_depth=2 → 实际按 2 递归（验收 1.1.8）"""
        _enable_crawl(monkeypatch)
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_max_depth", 2)

        async def _fetch(url):
            pages = {
                "https://a.com": CrawlResult(url=url, success=True, content=_html("A", "https://a.com/b"), title="A"),
                "https://a.com/b": CrawlResult(url=url, success=True, content=_html("B", "https://a.com/c"), title="B"),
                "https://a.com/c": CrawlResult(url=url, success=True, content=_html("C", "https://a.com/d"), title="C"),
                "https://a.com/d": CrawlResult(url=url, success=True, content=_html("D"), title="D"),
            }
            return pages[url]

        with patch("rag.crawl.crawler.fetch_page", side_effect=_fetch) as mock_fetch:
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = "approved"
                with patch("rag.retrieval.document_ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 1, "chunks": 1}
                    await run_crawl([{"url_pattern": "https://a.com", "max_depth": 5}])
                    assert mock_fetch.call_count == 3

    @pytest.mark.asyncio
    async def test_cycle_a_b_a(self, monkeypatch):
        """A→B→A 循环：visited 阻止 A 二次抓取（验收 1.2.4）"""
        _enable_crawl(monkeypatch)

        async def _fetch(url):
            pages = {
                "https://a.com": CrawlResult(url=url, success=True, content=_html("A", "https://a.com/b"), title="A"),
                "https://a.com/b": CrawlResult(url=url, success=True, content=_html("B", "https://a.com"), title="B"),
            }
            return pages[url]

        with patch("rag.crawl.crawler.fetch_page", side_effect=_fetch) as mock_fetch:
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = "approved"
                with patch("rag.retrieval.document_ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 1, "chunks": 1}
                    await run_crawl([{"url_pattern": "https://a.com", "max_depth": 3}])
                    assert mock_fetch.call_count == 2

    @pytest.mark.asyncio
    async def test_cross_source_shared_visited(self, monkeypatch):
        """跨源共享 visited：第二源种子页已被第一源递归抓到 → 不重复抓（验收 1.2.5）"""
        _enable_crawl(monkeypatch)

        async def _fetch(url):
            pages = {
                "https://a.com": CrawlResult(url=url, success=True, content=_html("A", "https://a.com/docs"), title="A"),
                "https://a.com/docs": CrawlResult(url=url, success=True, content=_html("Docs"), title="Docs"),
            }
            return pages[url]

        sources = [
            {"url_pattern": "https://a.com", "max_depth": 1},
            {"url_pattern": "https://a.com/docs", "max_depth": 1},
        ]
        with patch("rag.crawl.crawler.fetch_page", side_effect=_fetch) as mock_fetch:
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = "approved"
                with patch("rag.retrieval.document_ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 1, "chunks": 1}
                    await run_crawl(sources)
                    # a + docs 各抓一次（docs 被源 1 递归抓到，源 2 种子命中 visited）
                    assert mock_fetch.call_count == 2

    @pytest.mark.asyncio
    async def test_blacklist_recursive_link(self, monkeypatch):
        """递归链接命中黑名单 → 跳过不抓（验收 1.2.6）"""
        _enable_crawl(monkeypatch)
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_blacklist_patterns", "https://a.com/bad")

        async def _fetch(url):
            pages = {
                "https://a.com": CrawlResult(url=url, success=True, content=_html("A", "https://a.com/bad", "https://a.com/good"), title="A"),
                "https://a.com/good": CrawlResult(url=url, success=True, content=_html("Good"), title="Good"),
            }
            return pages[url]

        with patch("rag.crawl.crawler.fetch_page", side_effect=_fetch) as mock_fetch:
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = "approved"
                with patch("rag.retrieval.document_ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 1, "chunks": 1}
                    result = await run_crawl([{"url_pattern": "https://a.com", "max_depth": 1}])
                    # bad 被黑名单跳过不抓；good 正常抓
                    assert mock_fetch.call_count == 2
                    assert result.crawled == 2

    @pytest.mark.asyncio
    async def test_blacklisted_seed_skipped(self, monkeypatch):
        """种子命中黑名单 → 不 fetch 不计入 crawled（验收 1.2.6）"""
        _enable_crawl(monkeypatch)
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_blacklist_patterns", "https://evil.com")
        with patch("rag.crawl.crawler.fetch_page", new_callable=AsyncMock) as mock_fetch:
            result = await run_crawl([{"url_pattern": "https://evil.com/page"}])
            assert result.skipped == 1
            assert mock_fetch.call_count == 0

    @pytest.mark.asyncio
    async def test_external_link_dropped(self, monkeypatch):
        """外域链接不命中本源 url_pattern 前缀 → 丢弃不递归（验收 1.2.7）"""
        _enable_crawl(monkeypatch)

        async def _fetch(url):
            pages = {
                "https://a.com": CrawlResult(url=url, success=True, content=_html("A", "https://evil.com/x", "https://other.org/y"), title="A"),
            }
            return pages[url]

        with patch("rag.crawl.crawler.fetch_page", side_effect=_fetch) as mock_fetch:
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = "approved"
                with patch("rag.retrieval.document_ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 1, "chunks": 1}
                    await run_crawl([{"url_pattern": "https://a.com", "max_depth": 1}])
                    assert mock_fetch.call_count == 1

    @pytest.mark.asyncio
    async def test_single_page_failure_does_not_block(self, monkeypatch):
        """单页 404 → 兄弟链接继续抓（验收 1.3.1）"""
        _enable_crawl(monkeypatch)

        async def _fetch(url):
            if url == "https://a.com/b":
                return CrawlResult(url=url, success=False, error="HTTP 404")
            pages = {
                "https://a.com": CrawlResult(url=url, success=True, content=_html("A", "https://a.com/b", "https://a.com/c"), title="A"),
                "https://a.com/c": CrawlResult(url=url, success=True, content=_html("C"), title="C"),
            }
            return pages[url]

        with patch("rag.crawl.crawler.fetch_page", side_effect=_fetch) as mock_fetch:
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = "approved"
                with patch("rag.retrieval.document_ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 1, "chunks": 1}
                    result = await run_crawl([{"url_pattern": "https://a.com", "max_depth": 1}])
                    assert result.errors == 1
                    assert result.crawled == 2
                    assert mock_fetch.call_count == 3

    @pytest.mark.asyncio
    async def test_review_failure_fail_open_approved(self, monkeypatch):
        """审查节点抛异常 → fail-open 默认 approved 入库（验收 1.3.2）"""
        _enable_crawl(monkeypatch)
        with patch("rag.crawl.crawler.fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = CrawlResult(url="https://a.com", success=True, content=_html("A"), title="A")
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.side_effect = Exception("review down")
                with patch("rag.retrieval.document_ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 1, "chunks": 1}
                    result = await run_crawl([{"url_pattern": "https://a.com"}])
                    assert result.crawled == 1
                    assert result.approved == 1
                    assert mock_ingest.call_args[1].get("review_status") == "approved"

    @pytest.mark.asyncio
    async def test_ingest_failure_does_not_block(self, monkeypatch):
        """单页入库失败 → 其他链接继续抓（验收 1.3.3）"""
        _enable_crawl(monkeypatch)

        async def _fetch(url):
            pages = {
                "https://a.com": CrawlResult(url=url, success=True, content=_html("A", "https://a.com/b", "https://a.com/c"), title="A"),
                "https://a.com/b": CrawlResult(url=url, success=True, content=_html("B"), title="B"),
                "https://a.com/c": CrawlResult(url=url, success=True, content=_html("C"), title="C"),
            }
            return pages[url]

        async def _ingest(**kwargs):
            if "b" in kwargs["source"]:
                raise Exception("db down")
            return {"id": 1, "chunks": 1}

        with patch("rag.crawl.crawler.fetch_page", side_effect=_fetch):
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = "approved"
                with patch("rag.retrieval.document_ingest.ingest_document", side_effect=_ingest) as mock_ingest:
                    result = await run_crawl([{"url_pattern": "https://a.com", "max_depth": 1}])
                    assert result.errors == 1
                    assert result.crawled == 2
                    assert mock_ingest.call_count == 3

    @pytest.mark.asyncio
    async def test_total_page_limit_shared_across_tree(self, monkeypatch):
        """总页数上限全树共享：limit=3 → 全树合计 ≤ 3 页（验收 1.3.4）"""
        _enable_crawl(monkeypatch)
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_max_pages_per_run", 3)

        async def _fetch(url):
            pages = {
                "https://a.com": CrawlResult(url=url, success=True, content=_html("A", "https://a.com/b", "https://a.com/c", "https://a.com/d"), title="A"),
                "https://a.com/b": CrawlResult(url=url, success=True, content=_html("B"), title="B"),
                "https://a.com/c": CrawlResult(url=url, success=True, content=_html("C"), title="C"),
                "https://a.com/d": CrawlResult(url=url, success=True, content=_html("D"), title="D"),
            }
            return pages[url]

        with patch("rag.crawl.crawler.fetch_page", side_effect=_fetch) as mock_fetch:
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = "approved"
                with patch("rag.retrieval.document_ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 1, "chunks": 1}
                    result = await run_crawl([{"url_pattern": "https://a.com", "max_depth": 1}])
                    # 种子 a + b + c = 3 页；d 达到上限不抓
                    assert mock_fetch.call_count == 3
                    assert result.crawled == 3

    @pytest.mark.asyncio
    async def test_empty_page_terminates(self, monkeypatch):
        """空/无链接页面 → 递归自然终止（验收 1.3.5）"""
        _enable_crawl(monkeypatch)
        with patch("rag.crawl.crawler.fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = CrawlResult(url="https://a.com", success=True, content="no anchors here", title="A")
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = "approved"
                with patch("rag.retrieval.document_ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 1, "chunks": 1}
                    result = await run_crawl([{"url_pattern": "https://a.com", "max_depth": 3}])
                    assert result.crawled == 1
                    assert mock_fetch.call_count == 1

    @pytest.mark.asyncio
    async def test_rejected_recursive_still_ingested(self, monkeypatch):
        """递归页审查 rejected 仍入库（验收 1.1.9）"""
        _enable_crawl(monkeypatch)
        with patch("rag.crawl.crawler.fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = CrawlResult(url="https://a.com", success=True, content=_html("A"), title="A")
            with patch("rag.crawl.crawler._review_content", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = "rejected"
                with patch("rag.retrieval.document_ingest.ingest_document", new_callable=AsyncMock) as mock_ingest:
                    mock_ingest.return_value = {"id": 9, "chunks": 1}
                    result = await run_crawl([{"url_pattern": "https://a.com"}])
                    assert result.rejected == 1
                    assert mock_ingest.call_args[1].get("review_status") == "rejected"


# ─── DB 源配置加载（module-076 max_depth 字段） ───


class TestLoadSourcesMaxDepth:
    @pytest.mark.asyncio
    async def test_max_depth_column_loaded(self):
        mock_row = (1, "https://spring.io", "Spring", True, 2, None)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("src.database.async_session_factory", mock_factory):
            sources = await _load_sources_from_db()
            assert sources[0]["max_depth"] == 2

    @pytest.mark.asyncio
    async def test_legacy_row_without_max_depth(self):
        """存量 5 列 mock 行（module-075 兼容）：max_depth 兜底 1"""
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
            assert sources[0]["max_depth"] == 1
