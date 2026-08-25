"""待学笔记测试（module-080 反向闭环）

测试待学笔记落库、去重、读取、关键词提取。
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from rag.memory.weak_topics import (
    save_weak_topic,
    recall_weak_topics,
    extract_keywords,
    _weak_topic_source,
    WEAK_TOPIC_SOURCE_PREFIX,
)


@pytest.fixture
def mock_session():
    """Mock 数据库会话"""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def mock_doc():
    """Mock Document 对象"""
    doc = MagicMock()
    doc.id = 1
    doc.title = "Redis持久化"
    doc.content = "Redis持久化\nRDB快照原理不清楚"
    doc.source = "weak_topic:test-user:"
    doc.created_at = datetime.now()
    return doc


class TestWeakTopicSource:
    """测试 source 构造"""

    def test_source_with_identity(self):
        assert _weak_topic_source("test-user") == "weak_topic:test-user:"

    def test_source_empty_identity(self):
        assert _weak_topic_source("") == "weak_topic:unknown:"

    def test_source_whitespace_identity(self):
        assert _weak_topic_source("  ") == "weak_topic:unknown:"

    def test_source_prefix(self):
        assert WEAK_TOPIC_SOURCE_PREFIX == "weak_topic:"


class TestSaveWeakTopic:
    """测试待学笔记保存"""

    @pytest.mark.asyncio
    async def test_save_new_topic(self, mock_session, mock_doc):
        """测试新增待学笔记"""
        # Mock 去重查询返回 None（无重复）
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_session.execute.return_value = mock_result

        # Mock Document 创建
        mock_doc_instance = MagicMock()
        mock_doc_instance.id = 1
        mock_doc_instance.title = "Redis持久化"
        mock_doc_instance.content = "Redis持久化\nRDB快照原理不清楚"

        with patch('rag.memory.weak_topics.async_session_factory') as mock_factory:
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
            with patch('rag.memory.weak_topics.Document', return_value=mock_doc_instance):
                result = await save_weak_topic("Redis持久化", "RDB快照原理不清楚", "test-user")

        assert result["status"] == "saved"
        assert result["title"] == "Redis持久化"
        assert result["id"] == 1

    @pytest.mark.asyncio
    async def test_save_duplicate_topic(self, mock_session, mock_doc):
        """测试重复 topic 去重（更新 context）"""
        # Mock 去重查询返回已有文档
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = mock_doc
        mock_session.execute.return_value = mock_result

        with patch('rag.memory.weak_topics.async_session_factory') as mock_factory:
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await save_weak_topic("Redis持久化", "新薄弱点", "test-user")

        assert result["status"] == "updated"
        assert result["id"] == 1

    @pytest.mark.asyncio
    async def test_save_empty_topic_raises(self):
        """测试空 topic 抛 ValueError"""
        with pytest.raises(ValueError, match="topic 不能为空"):
            await save_weak_topic("", "context", "test-user")

    @pytest.mark.asyncio
    async def test_save_whitespace_topic_raises(self):
        """测试空白 topic 抛 ValueError"""
        with pytest.raises(ValueError, match="topic 不能为空"):
            await save_weak_topic("   ", "context", "test-user")


class TestRecallWeakTopics:
    """测试待学笔记读取"""

    @pytest.mark.asyncio
    async def test_recall_all(self, mock_session, mock_doc):
        """测试读取所有待学笔记"""
        # Mock 查询结果
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_doc]
        mock_session.execute.return_value = mock_result

        with patch('rag.memory.weak_topics.async_session_factory') as mock_factory:
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
            topics = await recall_weak_topics()

        assert len(topics) == 1
        assert topics[0]["title"] == "Redis持久化"

    @pytest.mark.asyncio
    async def test_recall_by_identity(self, mock_session, mock_doc):
        """测试按身份读取待学笔记"""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_doc]
        mock_session.execute.return_value = mock_result

        with patch('rag.memory.weak_topics.async_session_factory') as mock_factory:
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
            topics = await recall_weak_topics(identity="test-user")

        assert len(topics) == 1

    @pytest.mark.asyncio
    async def test_recall_empty(self, mock_session):
        """测试无待学笔记时返回空列表"""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        with patch('rag.memory.weak_topics.async_session_factory') as mock_factory:
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
            topics = await recall_weak_topics()

        assert topics == []


class TestExtractKeywords:
    """测试关键词提取"""

    def test_extract_normal(self):
        """测试正常提取"""
        topics = [
            {"title": "Redis持久化", "content": "..."},
            {"title": "Kafka分区", "content": "..."},
        ]
        keywords = extract_keywords(topics)
        assert "redis持久化" in keywords
        assert "kafka分区" in keywords

    def test_extract_empty_title(self):
        """测试空 title 跳过"""
        topics = [{"title": "", "content": "..."}]
        keywords = extract_keywords(topics)
        assert keywords == []

    def test_extract_dedup(self):
        """测试去重"""
        topics = [
            {"title": "Redis", "content": "..."},
            {"title": "Redis", "content": "..."},
        ]
        keywords = extract_keywords(topics)
        assert len(keywords) == 1

    def test_extract_empty_list(self):
        """测试空列表"""
        keywords = extract_keywords([])
        assert keywords == []
