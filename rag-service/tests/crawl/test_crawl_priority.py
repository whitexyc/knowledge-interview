"""抓取优先级测试（module-080 反向闭环）

测试待学笔记关键词匹配源时动态提升 priority、排序正确。
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from rag.crawl.crawler import _prioritize_sources, run_crawl


@pytest.fixture
def mock_sources():
    """Mock 源配置列表"""
    return [
        {"id": 1, "url_pattern": "https://redis.io/docs", "name": "Redis官方文档", "enabled": True, "max_depth": 1, "priority": 0},
        {"id": 2, "url_pattern": "https://spring.io/docs", "name": "Spring官方文档", "enabled": True, "max_depth": 1, "priority": 0},
        {"id": 3, "url_pattern": "https://kafka.apache.org", "name": "Kafka文档", "enabled": True, "max_depth": 1, "priority": 5},
    ]


@pytest.fixture
def mock_weak_topics():
    """Mock 待学笔记（关键词会匹配源 name）"""
    return [
        {"id": 1, "title": "redis", "content": "Redis持久化薄弱", "source": "weak_topic:test:"},
    ]


class TestPrioritizeSources:
    """测试优先级计算"""

    @pytest.mark.asyncio
    async def test_priority_boost_on_match(self, mock_sources, mock_weak_topics):
        """测试待学笔记关键词匹配源时提升 priority"""
        with patch('rag.memory.weak_topics.recall_weak_topics', return_value=mock_weak_topics):
            with patch('rag.crawl.crawler.settings') as mock_settings:
                mock_settings.weak_topic_priority_boost = 10
                result = await _prioritize_sources(mock_sources)

        # Redis 源 name="Redis官方文档" 匹配关键词 "redis"，priority 提升
        redis_src = next(s for s in result if s["id"] == 1)
        assert redis_src["_priority"] == 10  # 0 + 10

        # Spring 源不匹配，priority 保持
        spring_src = next(s for s in result if s["id"] == 2)
        assert spring_src["_priority"] == 0

        # Kafka 源有 DB priority=5，不匹配待学笔记，保持 5
        kafka_src = next(s for s in result if s["id"] == 3)
        assert kafka_src["_priority"] == 5

    @pytest.mark.asyncio
    async def test_sort_by_priority_desc(self, mock_sources, mock_weak_topics):
        """测试按 _priority 降序排列"""
        with patch('rag.memory.weak_topics.recall_weak_topics', return_value=mock_weak_topics):
            with patch('rag.crawl.crawler.settings') as mock_settings:
                mock_settings.weak_topic_priority_boost = 10
                result = await _prioritize_sources(mock_sources)

        # 排序后：Redis(10) > Kafka(5) > Spring(0)
        assert result[0]["id"] == 1  # Redis
        assert result[1]["id"] == 3  # Kafka
        assert result[2]["id"] == 2  # Spring

    @pytest.mark.asyncio
    async def test_no_weak_topics(self, mock_sources):
        """测试无待学笔记时按 DB priority 排序"""
        with patch('rag.memory.weak_topics.recall_weak_topics', return_value=[]):
            with patch('rag.crawl.crawler.settings') as mock_settings:
                mock_settings.weak_topic_priority_boost = 10
                result = await _prioritize_sources(mock_sources)

        # 无待学笔记，按 DB priority 排序：Kafka(5) > Redis(0) = Spring(0)
        assert result[0]["id"] == 3  # Kafka
        # Redis 和 Spring 都是 0，按原顺序
        assert result[1]["id"] in [1, 2]
        assert result[2]["id"] in [1, 2]

    @pytest.mark.asyncio
    async def test_multiple_keywords_match(self, mock_sources):
        """测试多个关键词匹配同一源"""
        weak_topics = [
            {"id": 1, "title": "redis", "content": "...", "source": "weak_topic:test:"},
            {"id": 2, "title": "docs", "content": "...", "source": "weak_topic:test:"},
        ]
        with patch('rag.memory.weak_topics.recall_weak_topics', return_value=weak_topics):
            with patch('rag.crawl.crawler.settings') as mock_settings:
                mock_settings.weak_topic_priority_boost = 10
                result = await _prioritize_sources(mock_sources)

        # Redis 源 url_pattern="https://redis.io/docs" 匹配 "redis" 和 "docs"，priority = 0 + 10*2 = 20
        redis_src = next(s for s in result if s["id"] == 1)
        assert redis_src["_priority"] == 20

    @pytest.mark.asyncio
    async def test_priority_calculation_error_fallback(self, mock_sources):
        """测试优先级计算异常时降级为默认排序"""
        with patch('rag.memory.weak_topics.recall_weak_topics', side_effect=Exception("DB error")):
            with patch('rag.crawl.crawler.settings') as mock_settings:
                mock_settings.weak_topic_priority_boost = 10
                result = await _prioritize_sources(mock_sources)

        # 降级：按 DB priority 排序
        assert result[0]["id"] == 3  # Kafka (priority=5)
        for src in result:
            assert "_priority" in src


class TestRunCrawlWithPriority:
    """测试 run_crawl 集成优先级"""

    @pytest.mark.asyncio
    async def test_run_crawl_calls_prioritize(self):
        """测试 run_crawl 调用 _prioritize_sources"""
        sources = [
            {"id": 1, "url_pattern": "https://example.com", "name": "Test", "enabled": True, "max_depth": 1, "priority": 0},
        ]

        with patch('rag.crawl.crawler._prioritize_sources', return_value=sources) as mock_prioritize:
            with patch('rag.crawl.crawler.settings') as mock_settings:
                mock_settings.crawl_enabled = False
                await run_crawl(sources)

        # crawl_enabled=False 时直接返回，不调用 _prioritize_sources
        mock_prioritize.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_crawl_enabled_calls_prioritize(self):
        """测试 crawl_enabled=True 时调用 _prioritize_sources"""
        sources = [
            {"id": 1, "url_pattern": "https://example.com", "name": "Test", "enabled": True, "max_depth": 1, "priority": 0},
        ]

        with patch('rag.crawl.crawler._prioritize_sources', return_value=sources) as mock_prioritize:
            with patch('rag.crawl.crawler.settings') as mock_settings:
                mock_settings.crawl_enabled = True
                mock_settings.crawl_max_pages_per_run = 10
                mock_settings.crawl_max_depth = 2
                mock_settings.crawl_blacklist_patterns = ""
                with patch('rag.crawl.crawler._recursive_crawl', new_callable=AsyncMock):
                    await run_crawl(sources)

        mock_prioritize.assert_called_once_with(sources)
