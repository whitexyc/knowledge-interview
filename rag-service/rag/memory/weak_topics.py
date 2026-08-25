"""
待学笔记管理（module-080 反向闭环）

低分题（feedback rating=-1）→ 待学笔记（documents 表 weak_topic:<identity>: 前缀）→
抓取优先级（source_configs.priority + 动态加权）

复用 documents 表 + 分块/嵌入/去重链路，与 memory:<identity>: 三层分层同架构。
"""
import logging
from typing import Optional

from sqlalchemy import select, text

from src.config import settings
from src.database import async_session_factory
from rag.models import Document

logger = logging.getLogger(__name__)

# 待学笔记 source 前缀（对齐 memory:<identity>: 三层分层模式）
WEAK_TOPIC_SOURCE_PREFIX = "weak_topic:"


def _weak_topic_source(identity: str) -> str:
    """构造待学笔记 source（带尾冒号分隔符，防前缀重叠身份交叉泄漏）"""
    identity = identity.strip() or "unknown"
    return f"{WEAK_TOPIC_SOURCE_PREFIX}{identity}:"


async def save_weak_topic(topic: str, context: str, identity: str) -> dict:
    """保存一条待学笔记到 documents 表

    Args:
        topic: 弱题主题关键词（如 "Redis持久化"）
        context: 薄弱点描述（如 "RDB快照原理不清楚"）
        identity: 身份标识（user_id 优先，否则 client_ip）

    Returns:
        {"id": int, "title": str, "status": "saved"} 或 {"status": "updated"}

    Raises:
        ValueError: topic 为空
        RuntimeError: 入库失败
    """
    if not topic or not topic.strip():
        raise ValueError("topic 不能为空")

    topic = topic.strip()
    context = context.strip() if context else ""
    identity = identity.strip() or "unknown"
    source = _weak_topic_source(identity)

    # 去重：同 identity + 同 topic 不重复新增（更新 context）
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(Document).where(
                    Document.source == source,
                    Document.title == topic,
                    Document.parent_id.is_(None),
                )
            )
            existing = result.scalars().first()
            if existing:
                # 追加 context 到已有记录
                new_content = existing.content
                if context and context not in existing.content:
                    new_content = f"{existing.content}\n{context}"
                await session.execute(
                    text("UPDATE documents SET content = :content WHERE id = :id"),
                    {"content": new_content, "id": existing.id},
                )
                await session.commit()
                logger.info("待学笔记已更新: topic=%s, identity=%s", topic, identity)
                return {"id": existing.id, "title": topic, "status": "updated"}
    except Exception as e:
        logger.warning("待学笔记去重查询失败，按新增处理: %s", e)

    # 新增：直接写入父块（不走 chunker/embedding，简化实现）
    content = f"{topic}\n{context}" if context else topic
    try:
        async with async_session_factory() as session:
            doc = Document(
                title=topic,
                content=content,
                source=source,
            )

            session.add(doc)
            await session.commit()
            await session.refresh(doc)
            logger.info("待学笔记已保存: id=%d, topic=%s, identity=%s", doc.id, topic, identity)
            return {"id": doc.id, "title": topic, "status": "saved"}
    except Exception as e:
        logger.error("待学笔记保存失败: %s", e, exc_info=True)
        raise RuntimeError(f"待学笔记保存失败: {e}")


async def recall_weak_topics(identity: Optional[str] = None) -> list[dict]:
    """读取待学笔记列表（供抓取优先级计算）

    Args:
        identity: 身份标识（None=读取所有身份的待学笔记）

    Returns:
        [{"id": int, "title": str, "content": str, "source": str}]
    """
    try:
        async with async_session_factory() as session:
            query = select(Document).where(
                Document.source.like(f"{WEAK_TOPIC_SOURCE_PREFIX}%"),
                Document.parent_id.is_(None),
            )
            if identity:
                source_pattern = _weak_topic_source(identity)
                query = query.where(Document.source == source_pattern)
            query = query.order_by(Document.created_at.desc()).limit(100)
            result = await session.execute(query)
            docs = result.scalars().all()
            return [
                {"id": d.id, "title": d.title, "content": d.content, "source": d.source}
                for d in docs
            ]
    except Exception as e:
        logger.warning("读取待学笔记失败: %s", e)
        return []


def extract_keywords(topics: list[dict]) -> list[str]:
    """从待学笔记提取关键词（简单：取 title 字段，去重）

    Args:
        topics: recall_weak_topics 返回的列表

    Returns:
        关键词列表（去重后）
    """
    keywords = set()
    for t in topics:
        title = t.get("title", "").strip()
        if title:
            keywords.add(title.lower())
    return list(keywords)
