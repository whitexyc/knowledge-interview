"""
优先级抓取单元测试（module-080）：crawl_priority pending 主题 → 搜索种子 → 递归抓取

覆盖验收矩阵：种子 URL 编码（特殊字符）、crawl_enabled 总闸、空队列空跑、
单轮消费上限、whitelist=None 放行、失败仍标记 processed、DB 异常 fail-open、
_scheduled_crawl_job 前置 drain（优先级先于常规源）。
全 mock：async_session_factory / _recursive_crawl，不触网不触库。
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag.crawl.priority_crawl import (
    _load_pending_topics,
    _mark_priority,
    build_seed_url,
    drain_priority_seeds,
)


def _rows_factory(rows):
    result = MagicMock()
    result.fetchall.return_value = rows
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return factory


def _session_factory(execute_result=None):
    session = AsyncMock()
    session.execute = AsyncMock(return_value=execute_result or MagicMock())
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return factory


# ─── 种子 URL 生成（验收 1.2：特殊字符 URL 编码） ───


class TestBuildSeedUrl:
    def test_quote_special_chars(self, monkeypatch):
        """中文/引号/百分号 → quote 编码，模板 {query} 占位替换"""
        from src.config import settings
        monkeypatch.setattr(
            settings, "feedback_search_url_template", "https://www.bing.com/search?q={query}")
        url = build_seed_url('Redis"持久化%原理 中文')
        assert url.startswith("https://www.bing.com/search?q=")
        assert "%E4%B8%AD%E6%96%87" in url  # 中文被编码
        assert '"' not in url.split("q=")[1]  # 引号被编码
        assert " " not in url.split("q=")[1]  # 空格被编码为 %20


# ─── 队列读取 / 状态更新（fail-open） ───


class TestQueueIO:
    @pytest.mark.asyncio
    async def test_load_pending_topics(self):
        rows = [(1, "Redis 持久化", "题目1"), (2, "MySQL 索引", "题目2")]
        with patch("rag.crawl.priority_crawl.async_session_factory", _rows_factory(rows)) as factory:
            topics = await _load_pending_topics(2)
        assert topics == [{"id": 1, "topic": "Redis 持久化", "question": "题目1"},
                          {"id": 2, "topic": "MySQL 索引", "question": "题目2"}]
        sql = str(factory.return_value.__aenter__.return_value.execute.call_args[0][0])
        assert "status='pending'" in sql
        assert "LIMIT :k" in sql

    @pytest.mark.asyncio
    async def test_load_db_error_fail_open(self):
        with patch("rag.crawl.priority_crawl.async_session_factory",
                   MagicMock(side_effect=Exception("db down"))):
            assert await _load_pending_topics(5) == []

    @pytest.mark.asyncio
    async def test_mark_priority(self):
        with patch("rag.crawl.priority_crawl.async_session_factory", _session_factory()) as factory:
            await _mark_priority(7, "processed")
        session = factory.return_value.__aenter__.return_value
        sql = str(session.execute.call_args[0][0])
        assert "UPDATE crawl_priority" in sql
        assert "processed_at=CURRENT_TIMESTAMP" in sql
        assert session.execute.call_args[0][1] == {"s": "processed", "id": 7}
    @pytest.mark.asyncio
    async def test_mark_db_error_fail_open(self):
        with patch("rag.crawl.priority_crawl.async_session_factory",
                   MagicMock(side_effect=Exception("db down"))):
            await _mark_priority(7, "processed")  # 不抛异常


# ─── 优先级抓取编排（验收 1.1 队列消费闭环） ───


class TestDrainPrioritySeeds:
    @pytest.mark.asyncio
    async def test_disabled_crawl_skips(self, monkeypatch):
        """crawl_enabled=false → 不读取队列不抓取（验收 1.3）"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_enabled", False)
        with patch("rag.crawl.priority_crawl._load_pending_topics",
                   new_callable=AsyncMock) as mock_load:
            summary = await drain_priority_seeds()
        assert summary == {"drained": 0, "errors": 0}
        mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_queue(self, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_enabled", True)
        with patch("rag.crawl.priority_crawl._load_pending_topics",
                   new_callable=AsyncMock, return_value=[]):
            summary = await drain_priority_seeds()
        assert summary == {"drained": 0, "errors": 0}

    @pytest.mark.asyncio
    async def test_processes_topics_with_whitelist_none(self, monkeypatch):
        """pending 主题 → 种子 URL → _recursive_crawl(whitelist=None) → 标记 processed"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_enabled", True)
        monkeypatch.setattr(settings, "feedback_priority_crawl_depth", 1)
        monkeypatch.setattr(settings, "crawl_max_pages_per_run", 10)
        topics = [{"id": 1, "topic": "Redis 持久化", "question": "q1"},
                  {"id": 2, "topic": "MySQL 索引", "question": "q2"}]
        with patch("rag.crawl.priority_crawl._load_pending_topics",
                   new_callable=AsyncMock, return_value=topics), \
             patch("rag.crawl.crawler._recursive_crawl",
                   new_callable=AsyncMock) as mock_crawl, \
             patch("rag.crawl.priority_crawl._mark_priority",
                   new_callable=AsyncMock) as mock_mark:
            summary = await drain_priority_seeds()
        assert summary == {"drained": 2, "errors": 0}
        assert mock_crawl.call_count == 2
        call_kwargs = mock_crawl.call_args_list[0].kwargs
        assert call_kwargs["whitelist"] is None
        assert mock_crawl.call_args_list[0].args[2] == 1  # max_depth 位置参数
        assert call_kwargs["limit"] == 10
        assert "bing.com/search?q=" in mock_crawl.call_args_list[0].args[0]
        marked = [c.args for c in mock_mark.call_args_list]
        assert (1, "processed") in marked and (2, "processed") in marked

    @pytest.mark.asyncio
    async def test_max_per_run_limits_topics(self, monkeypatch):
        """超过 feedback_priority_max_per_run → 只消费前 K 条（验收 1.2）"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_enabled", True)
        monkeypatch.setattr(settings, "feedback_priority_max_per_run", 2)
        with patch("rag.crawl.priority_crawl._load_pending_topics",
                   new_callable=AsyncMock, return_value=[]) as mock_load:
            await drain_priority_seeds()
        assert mock_load.await_args.args[0] == 2  # limit=2 传给读取

    @pytest.mark.asyncio
    async def test_crawl_failure_still_processed(self, monkeypatch):
        """单主题抓取异常 → errors+1，队列仍标记 processed（验收 1.3）"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_enabled", True)
        topics = [{"id": 9, "topic": "Redis 持久化", "question": "q1"}]
        with patch("rag.crawl.priority_crawl._load_pending_topics",
                   new_callable=AsyncMock, return_value=topics), \
             patch("rag.crawl.crawler._recursive_crawl",
                   new_callable=AsyncMock, side_effect=Exception("net down")), \
             patch("rag.crawl.priority_crawl._mark_priority",
                   new_callable=AsyncMock) as mock_mark:
            summary = await drain_priority_seeds()
        assert summary == {"drained": 0, "errors": 1}
        assert mock_mark.call_args.args == (9, "processed")


# ─── 与定时任务的集成（验收 1.1 优先级先于常规源） ───


class TestScheduledJobDrainFirst:
    @pytest.mark.asyncio
    async def test_scheduled_job_drains_before_sources(self, monkeypatch):
        """_scheduled_crawl_job：drain_priority_seeds 先于常规源执行"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_enabled", True)
        order = []

        async def fake_drain():
            order.append("drain")
            return {"drained": 0, "errors": 0}

        async def fake_load():
            order.append("load")
            return []

        with patch("rag.crawl.priority_crawl.drain_priority_seeds",
                   side_effect=fake_drain), \
             patch("rag.crawl.crawler._load_sources_from_db",
                   side_effect=fake_load):
            from rag.crawl.crawler import _scheduled_crawl_job
            await _scheduled_crawl_job()
        assert order == ["drain", "load"]
