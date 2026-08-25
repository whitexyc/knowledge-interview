"""
知识抓取流水线（module-075 / module-076）

源配置 CRUD + APScheduler 定时调度 + 白/黑名单 URL 过滤 +
递归抓取（深度控制 + URL 去重 + 链接跟踪）+ 审查节点接入
reflector + factcheck_judge。

编排者决策：
- 抓取深度：module-076 起受控递归（默认 depth=1，config 可调，0=仅种子页）
- 抓取频率：默认 crawl_interval_minutes=1440（24h），config 可调
- 白名单：source_configs 表驱动（用户可配），POST /ai/crawl/sources 添加
- 审查节点：允许抓取场景适配 prompt，但不修改 reflector.py / factcheck_judge.py 共享源文件
"""
import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

# --- 安全常量 ---
_ALLOWED_SCHEMES = {"http", "https"}
_FETCH_TIMEOUT_S = 30
_USER_AGENT = "PersonalKB-Crawler/1.0"

# --- 递归抓取常量（module-076） ---
_HREF_RE = re.compile(r'href=["\']([^"\']+)')
_FILENAME_SEGMENT_MAX = 50  # 入库 filename 末段截断长度



@dataclass
class CrawlResult:
    """单页抓取结果"""
    url: str
    success: bool
    content: str = ""
    title: str = ""
    error: str = ""
    review_status: str = "approved"  # approved / rejected


@dataclass
class CrawlSummary:
    """一次抓取批次汇总"""
    crawled: int = 0
    approved: int = 0
    rejected: int = 0
    skipped: int = 0
    errors: int = 0
    details: list = field(default_factory=list)


# ─── URL 安全校验 ───


def _is_safe_url(url: str) -> bool:
    """校验 URL 协议安全性（仅允许 http/https，防 file:///etc/passwd）"""
    return url.lower().startswith(("http://", "https://"))


# ─── 白名单/黑名单过滤 ───


def _matches_any(url: str, patterns: list[str]) -> bool:
    """URL 前缀匹配任一 pattern"""
    url_lower = url.lower()
    return any(url_lower.startswith(p.lower()) for p in patterns)
# ─── 递归抓取（module-076）：URL 规范化 / 链接提取 / 文件名 ───


def _normalize_url(url: str) -> str:
    """规范化 URL：去 fragment、scheme/host 小写、去尾部斜杠（路径非空时）

    Args:
        url: 原始 URL

    Returns:
        规范化后的 URL；解析失败时原样返回（fail-open）
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname.lower() if parts.hostname else ""
        netloc = f"{host}:{parts.port}" if parts.port else host
        path = parts.path
        if len(path) > 1 and path.endswith("/"):
            path = path.rstrip("/")
        return urlunsplit((parts.scheme.lower(), netloc, path, parts.query, ""))
    except ValueError:
        return url


def _extract_links(html: str, base_url: str, max_links: int) -> list[str]:
    """从 HTML 提取 http/https 链接（标准库正则 + urljoin，纯函数）

    Args:
        html: 页面 HTML 文本
        base_url: 基准 URL（相对链接绝对化）
        max_links: 单页提取链接上限（超出截断）

    Returns:
        规范化去重后的链接列表；非 HTML（无 <a）返回空列表
    """
    if not html or "<a" not in html.lower():
        return []
    links: list[str] = []
    seen: set[str] = set()
    for href in _HREF_RE.findall(html):
        abs_url = urljoin(base_url, href.strip())
        if not _is_safe_url(abs_url):
            continue
        norm = _normalize_url(abs_url)
        if norm in seen:
            continue
        seen.add(norm)
        links.append(norm)
        if len(links) >= max_links:
            break
    return links


def _crawl_filename(url: str) -> str:
    """生成入库文件名：URL 末段为空时回退前一段（防 crawl_.txt）"""
    base = url.split("?", 1)[0].rstrip("/")
    segment = base.split("/")[-1] or "page"
    return f"crawl_{segment[:_FILENAME_SEGMENT_MAX]}.txt"


def _blacklist_patterns() -> list[str]:
    """读取黑名单 URL 前缀列表（config 逗号分隔，种子与递归链接统一过滤）"""
    return [p.strip() for p in settings.crawl_blacklist_patterns.split(",") if p.strip()]



def _is_blacklisted_url(url: str) -> bool:
    """检查 URL 是否命中黑名单（config crawl_blacklist_patterns 逗号分隔前缀）

    种子 URL 与递归链接统一过滤。
    """
    return _matches_any(url, _blacklist_patterns())



# ─── 审查节点（包装调用，不修改共享源文件） ───


async def _review_content(
    url: str,
    content: str,
    title: str,
) -> str:
    """抓取内容审查：复用 reflector.check_sufficiency + factcheck_judge

    审查不通过标记 review_status="rejected" 但仍入库（fail-open，不丢数据）。
    审查节点调用失败时默认 approved（fail-open，不误杀）。

    Args:
        url: 抓取的 URL
        content: 抓取到的文本内容
        title: 页面标题

    Returns:
        "approved" 或 "rejected"
    """
    # Step 1: 充分性检查 —— 用抓取内容模拟"文档"与自身标题比对
    try:
        from agent.reflector import reflector
        doc_for_review = {"title": title or url, "content": content[:3000]}
        result = await reflector.check_sufficiency(
            query=title or url,
            documents=[doc_for_review],
        )
        if not result.get("sufficient", True):
            logger.info("审查不充分（reflector rejected）: %s", url[:80])
            return "rejected"
    except Exception as e:
        logger.warning("reflector 审查调用失败，fail-open: %s", e)

    # Step 2: 质量打分 —— factcheck_judge 对内容做质量评分
    try:
        from rag.retrieval.factcheck_judge import hhem_judge
        scores = await hhem_judge.predict(
            docs=[content[:2000]],
            claims=[title or url],
        )
        if scores and scores[0] < 0.3:
            logger.info("审查质量低（factcheck score=%.3f）: %s", scores[0], url[:80])
            return "rejected"
    except Exception as e:
        logger.warning("factcheck_judge 调用失败，fail-open: %s", e)

    return "approved"


# ─── 单页抓取 ───


async def fetch_page(url: str) -> CrawlResult:
    """抓取单个 URL 内容

    Args:
        url: 目标 URL（必须为 http/https）

    Returns:
        CrawlResult 含 content/title 或 error
    """
    if not _is_safe_url(url):
        return CrawlResult(url=url, success=False, error="不安全的 URL 协议")

    try:
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT_S,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            text = resp.text
            # 简单标题提取（<title> 标签）
            title = ""
            lower = text.lower()
            start = lower.find("<title>")
            if start != -1:
                end = lower.find("</title>", start)
                if end != -1:
                    title = text[start + 7 : end].strip()[:200]
            return CrawlResult(url=url, success=True, content=text, title=title)
    except httpx.TimeoutException:
        return CrawlResult(url=url, success=False, error="抓取超时（>30s）")
    except httpx.HTTPStatusError as e:
        return CrawlResult(url=url, success=False, error=f"HTTP {e.response.status_code}")
    except Exception as e:
        return CrawlResult(url=url, success=False, error=str(e)[:200])


# ─── 递归抓取引擎（module-076） ───


async def _crawl_page_and_store(url: str, summary: CrawlSummary) -> list[str]:
    """抓取单页 → 审查 → 入库 → 提取子链接（单页 fail-open）

    Args:
        url: 规范化后的 URL
        summary: 批次汇总（累计计数与详情）

    Returns:
        子链接列表（供上层递归展开；单页失败不影响其他页面）
    """
    result = await fetch_page(url)
    if not result.success:
        logger.warning("递归抓取失败: %s — %s", url[:80], result.error)
        summary.errors += 1
        summary.details.append({"url": url, "status": "error", "error": result.error})
        return []

    try:
        review = await _review_content(url, result.content, result.title)
    except Exception as e:
        logger.warning("审查调用异常，fail-open approved: %s — %s", url[:80], e)
        review = "approved"

    try:
        from rag.retrieval.document_ingest import ingest_document
        ingest_result = await ingest_document(
            data=result.content.encode("utf-8"),
            filename=_crawl_filename(url),
            title=result.title or url,
            source=f"crawl:{url}",
            review_status=review,
        )
        summary.crawled += 1
        if review == "approved":
            summary.approved += 1
        else:
            summary.rejected += 1
        summary.details.append({"url": url, "status": "ok", "review": review, "doc_id": ingest_result.get("id")})
        logger.info("递归入库成功: %s (doc_id=%s, review=%s)", url[:80], ingest_result.get("id"), review)
    except Exception as e:
        logger.warning("递归入库失败: %s — %s", url[:80], e)
        summary.errors += 1
        summary.details.append({"url": url, "status": "ingest_error", "error": str(e)[:200]})

    return _extract_links(result.content, url, settings.crawl_max_links_per_page)


async def _recursive_crawl(
    url: str,
    depth: int,
    max_depth: int,
    whitelist: list[str],
    visited: set[str],
    limit: int,
    summary: CrawlSummary,
) -> None:
    """递归抓取：深度控制 + 白/黑名单 + visited 去重 + 总页数上限

    Args:
        url: 待抓取 URL（种子页或递归链接）
        depth: 当前深度（种子页=0）
        max_depth: 本源最大深度（min(source.max_depth, config 全局上限)）
        whitelist: 本源白名单前缀（不命中不递归）
        visited: 去重池（单次 run_crawl 全树共享，循环自断）
        limit: 总页数上限（全树共享计数）
        summary: 批次汇总（累计计数与详情）
    """
    if len(visited) >= limit or depth > max_depth:
        return
    url = _normalize_url(url)
    if not _matches_any(url, whitelist) or url in visited:
        return
    if _is_blacklisted_url(url):
        logger.info("链接命中黑名单，跳过: %s", url[:80])
        return
    visited.add(url)
    try:
        child_links = await _crawl_page_and_store(url, summary)
    except Exception as e:
        logger.warning("递归页处理异常: %s — %s", url[:80], e)
        summary.errors += 1
        return
    for link in child_links:
        await _recursive_crawl(link, depth + 1, max_depth, whitelist, visited, limit, summary)


async def run_crawl(
    sources: list[dict],
    *,
    max_pages: int = 0,
) -> CrawlSummary:
    """执行一次抓取批次（受控递归：深度 + 去重 + 过滤 + 总页数上限）

    Args:
        sources: source_configs 行列表 [{"url_pattern", "name", "enabled", "max_depth"}]
        max_pages: 单次最大抓取页数（0 = config 默认；递归全树共享计数）

    Returns:
        CrawlSummary 汇总
    """
    if not settings.crawl_enabled:
        logger.info("抓取功能已禁用（crawl_enabled=false）")
        return CrawlSummary()

    limit = max_pages or settings.crawl_max_pages_per_run
    summary = CrawlSummary()
    visited: set[str] = set()


    for src in sources:
        url_pattern = src.get("url_pattern", "")
        name = src.get("name", url_pattern)
        if not url_pattern or not _is_safe_url(url_pattern):
            summary.skipped += 1
            continue
        if _is_blacklisted_url(url_pattern):
            logger.info("种子命中黑名单，跳过: %s", url_pattern[:80])
            summary.skipped += 1
            continue
        raw_depth = src.get("max_depth")
        source_depth = max(int(raw_depth) if raw_depth is not None else 1, 0)
        max_depth = min(source_depth, settings.crawl_max_depth)
        whitelist = [_normalize_url(url_pattern)]
        logger.info("开始递归抓取: %s (%s, max_depth=%d)", name, url_pattern[:80], max_depth)
        await _recursive_crawl(
            url_pattern, 0, max_depth, whitelist, visited, limit, summary,
        )

    logger.info(
        "抓取批次完成: crawled=%d, approved=%d, rejected=%d, errors=%d, skipped=%d",
        summary.crawled, summary.approved, summary.rejected,
        summary.errors, summary.skipped,
    )
    return summary


# ─── APScheduler 调度器 ───


_scheduler = None


async def _scheduled_crawl_job() -> None:
    """APScheduler 定时任务回调：读取 DB 源配置并执行抓取"""
    try:
        sources = await _load_sources_from_db()
        if not sources:
            logger.info("定时抓取：无启用的源配置，跳过")
            return
        enabled = [s for s in sources if s.get("enabled", True)]
        if not enabled:
            logger.info("定时抓取：无启用的源配置，跳过")
            return
        await run_crawl(enabled)
    except Exception as e:
        logger.error("定时抓取异常: %s", e, exc_info=True)


async def _load_sources_from_db() -> list[dict]:
    """从 DB 加载所有源配置"""
    try:
        from src.database import async_session_factory
        from sqlalchemy import text
        async with async_session_factory() as session:
            result = await session.execute(
                text("SELECT id, url_pattern, name, enabled, max_depth, last_crawled_at FROM source_configs ORDER BY id")
            )
            rows = result.fetchall()
            return [
                {
                    "id": r[0], "url_pattern": r[1], "name": r[2], "enabled": r[3],
                    "max_depth": r[4] if len(r) > 4 and r[4] is not None else 1,
                    "last_crawled_at": r[5] if len(r) > 5 else None,
                }
                for r in rows
            ]
    except Exception as e:
        logger.warning("加载源配置失败: %s", e)
        return []


def start_scheduler() -> None:
    """启动 APScheduler（在 FastAPI lifespan 中调用）"""
    global _scheduler
    if not settings.crawl_enabled:
        logger.info("抓取功能已禁用，不启动调度器")
        return
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        _scheduler = AsyncIOScheduler()
        _scheduler.add_job(
            _scheduled_crawl_job,
            trigger=IntervalTrigger(minutes=settings.crawl_interval_minutes),
            id="crawl_pipeline",
            name="知识抓取定时任务",
            replace_existing=True,
        )
        _scheduler.start()
        logger.info(
            "抓取调度器已启动（间隔 %d 分钟）",
            settings.crawl_interval_minutes,
        )
    except ImportError:
        logger.warning("apscheduler 未安装，定时抓取不可用（手动触发仍可用）")
    except Exception as e:
        logger.error("抓取调度器启动失败: %s", e)


def shutdown_scheduler() -> None:
    """关闭 APScheduler（在 FastAPI lifespan 中调用）"""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("抓取调度器已关闭")
        _scheduler = None
