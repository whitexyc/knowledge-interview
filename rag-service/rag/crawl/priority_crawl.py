"""优先级抓取（module-080）：消费 crawl_priority pending 主题。
_scheduled_crawl_job 前置调用 drain_priority_seeds()：主题 → 搜索种子 URL →
_recursive_crawl(whitelist=None，黑名单/robots/审查照常) → 无论成败标记 processed。
fail-open：DB/抓取/更新失败不抛异常。
"""
import logging
from urllib.parse import quote

from sqlalchemy import text

from src.config import settings
from src.database import async_session_factory

logger = logging.getLogger(__name__)


async def _load_pending_topics(limit: int) -> list:
    """读取 pending 主题（id 升序，前 limit 条）；失败返回 []"""
    try:
        async with async_session_factory() as session:
            rows = (await session.execute(text(
                "SELECT id, topic, question FROM crawl_priority "
                "WHERE status='pending' ORDER BY id LIMIT :k"), {"k": limit})).fetchall()
        return [{"id": r[0], "topic": r[1], "question": r[2]} for r in rows]
    except Exception as e:
        logger.warning("读取优先级队列失败（fail-open）: %s", e)
        return []


async def _mark_priority(priority_id: int, status: str) -> None:
    """更新队列状态并写 processed_at（失败仅日志）"""
    try:
        async with async_session_factory() as session:
            await session.execute(text(
                "UPDATE crawl_priority SET status=:s, processed_at=CURRENT_TIMESTAMP "
                "WHERE id=:id"), {"s": status, "id": priority_id})
            await session.commit()
    except Exception as e:
        logger.warning("优先级队列状态更新失败（fail-open）: %s", e)


def build_seed_url(topic: str) -> str:
    """种子 URL = 搜索模板.format(query=quote(topic))（特殊字符 URL 编码）"""
    return settings.feedback_search_url_template.format(query=quote(topic))


async def drain_priority_seeds() -> dict:
    """消费 pending 主题并执行优先级抓取（crawl_enabled=false 时不抓取）"""
    if not settings.crawl_enabled:
        return {"drained": 0, "errors": 0}
    topics = await _load_pending_topics(max(int(settings.feedback_priority_max_per_run), 1))
    if not topics:
        return {"drained": 0, "errors": 0}
    from rag.crawl.crawler import CrawlSummary, _recursive_crawl
    summary = {"drained": 0, "errors": 0}
    for t in topics:
        try:
            await _recursive_crawl(
                build_seed_url(t["topic"]), 0, settings.feedback_priority_crawl_depth,
                whitelist=None, visited=set(),
                limit=settings.crawl_max_pages_per_run, summary=CrawlSummary())
            summary["drained"] += 1
        except Exception as e:
            summary["errors"] += 1
            logger.warning("优先级主题抓取异常（仍标记 processed）: %s — %s", t["topic"][:40], e)
        await _mark_priority(t["id"], "processed")
    logger.info("优先级抓取完成: %s", summary)
    return summary
