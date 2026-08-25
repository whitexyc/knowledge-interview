"""反向闭环扫描器（module-080）：低分题→待学笔记→抓取优先级队列。
定时（或 POST /ai/feedback/scan）拉取 Java 低分题 → 结构化笔记 → memory_service.save
写记忆层 → 主题入 crawl_priority。fail-open：拉取/JSON/单条失败不抛异常。
"""
import logging

import httpx

from src.config import settings
from rag.memory.memory import memory_service

logger = logging.getLogger(__name__)

_WEAK_POINTS_PATH = "/api/xunzhi/v1/interview/weak-points"
_TOPIC_MAX = 30  # 主题取题目文本前 N 字符（确定性，不调 LLM）


async def fetch_low_score_questions() -> list:
    """拉取 Java 低分题列表；失败/结构异常返回 []（fail-open）"""
    url = settings.feedback_java_base_url.rstrip("/") + _WEAK_POINTS_PATH
    params = {"threshold": settings.feedback_low_score_threshold, "days": 7, "limit": 50}
    headers = ({"X-Internal-Token": settings.feedback_internal_token}
               if settings.feedback_internal_token else {})
    try:
        async with httpx.AsyncClient(timeout=settings.feedback_http_timeout_s) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as e:
        logger.warning("拉取 Java 低分题失败，fail-open 空跑: %s", e)
        return []
    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        logger.warning("Java 低分题响应结构异常，fail-open 空跑")
        return []
    return [it for it in items if isinstance(it, dict)]


def extract_topic(item: dict) -> str:
    """确定性提取主题：题目文本前 _TOPIC_MAX 字符（空白折叠），缺省回退"""
    question = (item.get("questionContent") or "").strip()
    return (" ".join(question.split())[:_TOPIC_MAX]) or "未知主题"


def build_learning_note(item: dict, topic: str) -> str:
    """结构化待学笔记（题目/得分/反馈/来源会话）"""
    return (f"【待学笔记】{topic}\n面试问题: {item.get('questionContent') or ''}\n"
            f"本题得分: {item.get('score')}/{item.get('totalScore') or '?'}\n"
            f"面试反馈: {item.get('feedback') or ''}\n"
            f"来源会话: {item.get('sessionId') or ''}")


async def enqueue_priority(item: dict, topic: str, note: str) -> bool:
    """主题入 crawl_priority（pending；同 topic pending 不重复入队；失败 False）"""
    try:
        from sqlalchemy import text
        from src.database import async_session_factory
        async with async_session_factory() as session:
            dup = await session.execute(
                text("SELECT 1 FROM crawl_priority WHERE status='pending' AND topic=:t LIMIT 1"),
                {"t": topic})
            if dup.scalar():
                return False
            await session.execute(
                text("INSERT INTO crawl_priority (topic, note, session_id, question, score) "
                     "VALUES (:t, :n, :s, :q, :sc)"),
                {"t": topic, "n": note, "s": str(item.get("sessionId") or "")[:64],
                 "q": str(item.get("questionContent") or "")[:500],
                 "sc": int(item.get("score") or 0)})
            await session.commit()
        return True
    except Exception as e:
        logger.warning("优先级入队失败（fail-open）: %s", e)
        return False


async def scan_and_generate() -> dict:
    """执行一轮反向闭环扫描：拉取→过滤高分→写笔记→入队（fail-open）"""
    items = await fetch_low_score_questions()
    summary = {"scanned": len(items), "noted": 0, "enqueued": 0, "errors": 0}
    for item in items:
        try:
            if item.get("score") is not None and int(item["score"]) >= settings.feedback_low_score_threshold:
                continue
            topic = extract_topic(item)
            note = build_learning_note(item, topic)
            await memory_service.save(note, identity=settings.feedback_learning_identity,
                                      memory_type="fact")
            summary["noted"] += 1
            if await enqueue_priority(item, topic, note):
                summary["enqueued"] += 1
        except Exception as e:
            summary["errors"] += 1
            logger.warning("低分题处理失败（fail-open，跳过该条）: %s", e)
    logger.info("反向闭环扫描完成: %s", summary)
    return summary


_feedback_scheduler = None


def setup_feedback_scheduler(enable: bool) -> None:
    """启动（enable=True）或关闭（enable=False）反向闭环定时扫描"""
    global _feedback_scheduler
    if not enable:
        if _feedback_scheduler and _feedback_scheduler.running:
            _feedback_scheduler.shutdown(wait=False)
            _feedback_scheduler = None
        return
    if not settings.feedback_reverse_enabled:
        logger.info("反向闭环已禁用（feedback_reverse_enabled=false），不启动调度器")
        return
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        _feedback_scheduler = AsyncIOScheduler()
        _feedback_scheduler.add_job(
            scan_and_generate,
            trigger=IntervalTrigger(minutes=settings.feedback_scan_interval_minutes),
            id="feedback_reverse_loop", name="反向闭环扫描", replace_existing=True)
        _feedback_scheduler.start()
        logger.info("反向闭环调度器已启动（间隔 %d 分钟）", settings.feedback_scan_interval_minutes)
    except ImportError:
        logger.warning("apscheduler 未安装，反向闭环定时扫描不可用（手动触发仍可用）")
    except Exception as e:
        logger.error("反向闭环调度器启动失败: %s", e)
