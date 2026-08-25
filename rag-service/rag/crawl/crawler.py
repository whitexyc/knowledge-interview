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
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlsplit, urlunsplit, urlparse

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

# --- 安全常量 ---
_ALLOWED_SCHEMES = {"http", "https"}
_FETCH_TIMEOUT_S = 30
_ROBOTS_UA = "PersonalKB-Crawler"
_ROBOTS_TIMEOUT_S = 5  # robots.txt 拉取超时（秒），铁律4：魔法数字提取为常量

# --- UA 轮换池（module-077）：~10 个主流桌面+移动浏览器 UA ---
_BUILTIN_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]

# --- 反爬模块级状态（module-077） ---
_robots_cache: dict[str, tuple[float, object]] = {}  # host → (expire_ts, RobotFileParser)
_proxy_pool: list[str] = []
_proxy_index: int = 0
_last_fetch_time: dict[int, float] = {}  # source_id → monotonic timestamp


def _ua_pool() -> list[str]:
    """返回 UA 列表：配置非空用配置，否则用内置池"""
    custom = settings.crawl_user_agents
    if custom:
        pool = [u.strip() for u in custom.split(",") if u.strip()]
        if pool:
            return pool
    return _BUILTIN_UA_POOL
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
    conflict_count: int = 0  # 检测到矛盾的文档数（module-078，与是否 rejected 独立计数）
    skipped: int = 0
    errors: int = 0
    details: list = field(default_factory=list)


class ReviewResult(str):
    """审查结果（str 子类）：值 == "approved"/"rejected"，另携带结构化审查字段

    module-078 扩展审查节点后，入库/汇总需要 score/conflict 等结构化信息，
    而 module-075 契约（_review_content 返回 str、与 "approved" 直接比较）
    必须保持零回归——故用 str 子类桥接：既可直接比较，又可取字段。
    """

    def __new__(cls, status: str, *, score=None, sufficient: bool = True, conflict: bool = False, conflict_detail: str = "", policy: str = "fail-open", elapsed_ms: int = 0) -> "ReviewResult":
        """构造 ReviewResult 实例（str 子类桥接，携带结构化审查字段）"""
        obj = str.__new__(cls, status)
        obj.status = status
        obj.score = score
        obj.sufficient = sufficient
        obj.conflict = conflict
        obj.conflict_detail = conflict_detail
        obj.policy = policy
        obj.elapsed_ms = elapsed_ms
        return obj


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



# ─── 反爬辅助函数（module-077）：robots / UA / 代理 / 限速 ───


async def _check_robots_allowed(url: str) -> bool:
    """检查 robots.txt 是否允许抓取该 URL（fail-open：失败/无 robots.txt = 允许）

    缓存策略：按源域名缓存解析结果（dict + TTL），同一域名只拉取一次 robots.txt。
    robots.txt 本身用 httpx 抓取（带超时），标准库 RobotFileParser 解析。

    Args:
        url: 目标 URL

    Returns:
        True = 允许抓取，False = robots.txt 禁止
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if not host:
            return True
        now = time.monotonic()
        ttl = settings.crawl_robots_cache_ttl
        # P3-1: ttl <= 0 = 不缓存，每次重新拉取（对齐 config 声明"0=不缓存"）
        if ttl > 0 and host in _robots_cache:
            expire_ts, rp = _robots_cache[host]
            if now < expire_ts:
                return rp.can_fetch(_ROBOTS_UA, url)
        # 拉取 robots.txt
        scheme = parsed.scheme or "https"
        robots_url = f"{scheme}://{host}/robots.txt"
        async with httpx.AsyncClient(
            timeout=_ROBOTS_TIMEOUT_S,
            follow_redirects=True,
            headers={"User-Agent": _ROBOTS_UA},  # P3-5: 带 UA 头
        ) as client:
            resp = await client.get(robots_url)
            resp.raise_for_status()
            rp = _make_robot_parser()  # P3-6: 简化构造
            rp.parse(resp.text.splitlines())
        expire_ts = now + ttl if ttl > 0 else now
        if ttl > 0:  # ttl>0 时写缓存，ttl<=0 每次拉取不缓存
            _robots_cache[host] = (expire_ts, rp)
        return rp.can_fetch(_ROBOTS_UA, url)
    except Exception as e:
        # fail-open：robots.txt 不存在/超时/网络错误 → 允许抓取
        logger.debug("robots.txt 拉取失败，fail-open: %s", e)
        return True


def _make_robot_parser():
    """创建 RobotFileParser 实例（延迟导入标准库）"""
    from urllib.robotparser import RobotFileParser
    return RobotFileParser()


def _pick_ua() -> str:
    """从 UA 池随机选取一个浏览器 User-Agent"""
    pool = _ua_pool()
    return random.choice(pool)


def _random_headers() -> dict:
    """生成带随机 UA 的浏览器风格请求头"""
    return {
        "User-Agent": _pick_ua(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
    }


def _load_proxies() -> list[str]:
    """从配置加载代理列表"""
    raw = settings.crawl_proxies
    return [p.strip() for p in raw.split(",") if p.strip()]


def _next_proxy() -> str | None:
    """Round-robin 返回下一个代理地址，空列表返回 None（直连）"""
    global _proxy_pool, _proxy_index
    if not _proxy_pool:
        _proxy_pool = _load_proxies()
    if not _proxy_pool:
        return None
    proxy = _proxy_pool[_proxy_index % len(_proxy_pool)]
    _proxy_index += 1
    return proxy


async def _rate_limit_delay(source_id: int) -> None:
    """per-source 请求间隔限速（_recursive_crawl 每个子链接 fetch 前调用）

    使用 time.monotonic() 单调时钟，不同源独立节奏互不干扰。
    """
    delay = settings.crawl_request_delay_seconds
    if delay <= 0:
        return
    last = _last_fetch_time.get(source_id)
    if last is not None:
        elapsed = time.monotonic() - last
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
    _last_fetch_time[source_id] = time.monotonic()


# ─── 审查节点（包装调用，不修改共享源文件） ───


async def _review_content(url: str, content: str, title: str) -> ReviewResult:
    """抓取内容审查：充分性 + HHEM 质量分 + 矛盾检测，按审查策略判定。
    rejected 仍入库（module-075 契约不变）；返回 ReviewResult（str 子类，
    == "approved"/"rejected"），携带 score/sufficient/conflict/elapsed_ms。"""
    t0 = time.perf_counter()
    policy = settings.crawl_review_policy
    threshold = settings.crawl_hhem_threshold_strict if policy == "strict" else settings.crawl_hhem_threshold
    status = "approved"
    score: Optional[float] = None
    sufficient = True
    conflict = False
    conflict_detail = ""
    try:  # Step 1: 充分性（复用 reflector，不修改共享源文件）
        from agent.reflector import reflector
        doc_for_review = {"title": title or url, "content": content[:3000]}
        result = await reflector.check_sufficiency(query=title or url, documents=[doc_for_review])
        sufficient = bool(result.get("sufficient", True))
        if not sufficient:
            status = "rejected"
    except Exception as e:
        logger.warning("reflector 审查调用失败: %s", e)
        if policy == "strict":
            status = "rejected"
    if status == "approved":  # Step 2: HHEM 质量分（阈值读 config）
        try:
            from rag.retrieval.factcheck_judge import hhem_judge
            scores = await hhem_judge.predict(docs=[content[:2000]], claims=[title or url])
            if scores and scores[0] is not None:
                score = float(scores[0])
                if score < threshold:
                    status = "rejected"
        except Exception as e:
            logger.warning("factcheck_judge 调用失败: %s", e)
            if policy == "strict":
                status = "rejected"
    conflict_info = await _check_conflict(content)  # Step 3: 矛盾检测（fail-open 仅记录）
    conflict = bool(conflict_info.get("conflict"))
    conflict_detail = str(conflict_info.get("detail", ""))
    if conflict and policy in ("lenient", "strict"):
        status = "rejected"
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("审查完成: url=%s status=%s score=%s sufficient=%s conflict=%s policy=%s elapsed_ms=%d",
                url[:80], status, score, sufficient, conflict, policy, elapsed_ms)
    if conflict_detail:
        logger.info("矛盾命中: %s", conflict_detail)
    return ReviewResult(status, score=score, sufficient=sufficient, conflict=conflict,
                        conflict_detail=conflict_detail, policy=policy, elapsed_ms=elapsed_ms)


async def _check_conflict(content: str) -> dict:
    """矛盾检测：embed 新内容 → 根父块向量候选 → memory_conflict_judge 判定

    任一环节失败 / 未启用（memory_conflict_enabled=false）→
    {"conflict": False, "detail": ""}（fail-open，不阻断入库主链路）。

    Args:
        content: 抓取内容（embed 取前 500 字符，判定 hypothesis 取前 2000）

    Returns:
        {"conflict": bool, "detail": str}；detail 含候选文档 id/标题/判定器
    """
    if not settings.memory_conflict_enabled:
        return {"conflict": False, "detail": ""}
    try:
        from rag.retrieval.embeddings import embedding_service
        vec = await embedding_service.embed_text(content[:500])
        if not vec:
            return {"conflict": False, "detail": "嵌入失败"}
        candidates = await _conflict_candidates(vec)
        mode = settings.memory_conflict_judge
        for cand in candidates:
            hit, judge_used = await _judge_conflict(
                cand["content"], content[:2000], mode)
            if hit:
                detail = (f"与库中文档 id={cand['id']} 标题={cand['title']!r} "
                          f"矛盾（判定器={judge_used}）")
                return {"conflict": True, "detail": detail}
        return {"conflict": False, "detail": ""}
    except Exception as e:
        logger.warning("矛盾检测失败，fail-open 跳过: %s", e)
        return {"conflict": False, "detail": ""}


async def _conflict_candidates(vec: list[float]) -> list[dict]:
    """向量查询根父块候选：cosine ≥ crawl_conflict_min_cosine，top-K

    pgvector 余弦距离 <=>（距离 = 1 - cosine），embedding 字符串绑定对齐
    retriever._vector_search 先例（规避 asyncpg 类型编解码）。

    Args:
        vec: 新内容向量（L2 归一化）

    Returns:
        [{"id", "title", "content"}]，按 cosine 降序
    """
    from sqlalchemy import text
    from src.database import async_session_factory
    vec_str = f"[{','.join(str(v) for v in vec)}]"
    sql = text("""
        SELECT id, title, content, 1 - (embedding <=> :vec) AS cosine
        FROM documents
        WHERE parent_id IS NULL AND embedding IS NOT NULL
        ORDER BY embedding <=> :vec ASC
        LIMIT :k
    """)
    async with async_session_factory() as session:
        rows = (await session.execute(
            sql, {"vec": vec_str, "k": settings.crawl_conflict_top_k})).fetchall()
    min_cosine = settings.crawl_conflict_min_cosine
    return [
        {"id": r[0], "title": r[1], "content": r[2]}
        for r in rows if float(r[3]) >= min_cosine
    ]


async def _judge_conflict(premise: str, hypothesis: str, mode: str) -> tuple[bool, str]:
    """按 memory_conflict_judge 判矛盾：nli/clf 单判，dual 双确认 + 对称回退

    Args:
        premise: 库中候选文档内容
        hypothesis: 新抓取内容
        mode: settings.memory_conflict_judge 值（nli / clf / dual）

    Returns:
        (是否矛盾, 实际使用的判定器)；判定器不可用 → (False, 判定器名)（fail-open）
    """
    if mode == "nli":
        return bool(await _nli_contradicts(premise, hypothesis)), "nli"
    if mode == "clf":
        return bool(await _clf_contradicts(premise, hypothesis)), "clf"
    # dual：双确认 contradiction 才判矛盾；任一不可用 → 另一单判（对称回退）
    nli_hit = await _nli_contradicts(premise, hypothesis)
    clf_hit = await _clf_contradicts(premise, hypothesis)
    if nli_hit is not None and clf_hit is not None:
        return nli_hit and clf_hit, "dual"
    if nli_hit is not None:
        return nli_hit, "nli"
    if clf_hit is not None:
        return clf_hit, "clf"
    return False, "dual"


async def _nli_contradicts(premise: str, hypothesis: str) -> Optional[bool]:
    """nli_judge 判定是否矛盾（三分类，contradiction → True；不可用 → None）"""
    try:
        from rag.memory.nli_judge import nli_judge
        result = await nli_judge.predict(premise=premise, hypothesis=hypothesis)
        return result == "contradiction" if result else None
    except Exception as e:
        logger.warning("nli 判定失败（fail-open None）: %s", e)
        return None


async def _clf_contradicts(premise: str, hypothesis: str) -> Optional[bool]:
    """memory_conflict_clf 判定是否矛盾（二分类，contradiction → True；不可用 → None）"""
    try:
        from rag.memory.memory_conflict_clf import memory_conflict_clf
        await memory_conflict_clf.load()
        result = await memory_conflict_clf.predict(premise=premise, hypothesis=hypothesis)
        return result == "contradiction" if result else None
    except Exception as e:
        logger.warning("clf 判定失败（fail-open None）: %s", e)
        return None


# ─── 单页抓取 ───

def _extract_title_from_html(text: str) -> str:
    """从 HTML 提取 <title> 标签内容（截断 200 字符）"""
    lower = text.lower()
    start = lower.find("<title>")
    if start == -1:
        return ""
    end = lower.find("</title>", start)
    if end == -1:
        return ""
    return text[start + 7 : end].strip()[:200]

async def fetch_page(url: str) -> CrawlResult:
    """抓取单个 URL（UA 轮换 + 代理轮换 + 429/5xx 退避重试）"""
    if not _is_safe_url(url):
        return CrawlResult(url=url, success=False, error="不安全的 URL 协议")
    max_retries = settings.crawl_retry_max
    base_delay = settings.crawl_retry_base_seconds
    last_error = ""
    for attempt in range(max_retries + 1):
        kw: dict = {"timeout": _FETCH_TIMEOUT_S, "follow_redirects": True,
                    "headers": _random_headers()}
        proxy = _next_proxy()
        if proxy:
            kw["proxy"] = proxy
        try:
            async with httpx.AsyncClient(**kw) as client:
                resp = await client.get(url)
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}"
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        await asyncio.sleep(delay + random.uniform(0, delay * 0.5))
                        continue
                    return CrawlResult(url=url, success=False, error=last_error)
                resp.raise_for_status()
                return CrawlResult(url=url, success=True, content=resp.text,
                                   title=_extract_title_from_html(resp.text))
        except httpx.TimeoutException:
            if attempt < max_retries:
                continue
            return CrawlResult(url=url, success=False,
                               error=f"抓取超时（>{_FETCH_TIMEOUT_S}s）")
        except httpx.HTTPStatusError as e:
            return CrawlResult(url=url, success=False,
                               error=f"HTTP {e.response.status_code}")
        except Exception as e:
            last_error = str(e)[:200]
            if attempt < max_retries:
                continue
            return CrawlResult(url=url, success=False, error=last_error)
    # P3-7: 防御性兜底（理论上不可达——for 循环内所有路径均已 return）
    return CrawlResult(url=url, success=False, error=last_error or "重试用尽")
# ─── 递归抓取引擎（module-076） ───


async def _crawl_page_and_store(url: str, summary: CrawlSummary) -> list[str]:
    """抓取单页→审查→入库→提取子链接（fail-open，robots/限速前置）"""
    # module-077: robots.txt 检查（fail-open：检查失败允许继续）
    if not await _check_robots_allowed(url):
        logger.info("robots.txt 禁止抓取，跳过: %s", url[:80])
        summary.skipped += 1
        summary.details.append({"url": url, "status": "robots_blocked"})
        return []
    # module-077: per-source 限速（按 URL 域名 hash 隔离限速状态，P3-4 修复）
    source_key = hash(urlparse(url).hostname or "") & 0x7FFFFFFF  # 正整数 key
    await _rate_limit_delay(source_key)
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
    review_status = str(review)
    review_score = getattr(review, "score", None)
    conflict = bool(getattr(review, "conflict", False))
    try:
        from rag.retrieval.document_ingest import ingest_document
        ingest_result = await ingest_document(
            data=result.content.encode("utf-8"), filename=_crawl_filename(url),
            title=result.title or url, source=f"crawl:{url}",
            review_status=review_status, review_score=review_score)
        summary.crawled += 1
        if review_status == "approved":
            summary.approved += 1
        else:
            summary.rejected += 1
        if conflict:
            summary.conflict_count += 1
        summary.details.append({"url": url, "status": "ok", "review": review_status,
                                "review_score": review_score, "conflict": conflict,
                                "doc_id": ingest_result.get("id")})
        logger.info("递归入库成功: %s (doc_id=%s, review=%s, score=%s, conflict=%s)",
                    url[:80], ingest_result.get("id"), review_status, review_score, conflict)
    except Exception as e:
        logger.warning("递归入库失败: %s — %s", url[:80], e)
        summary.errors += 1
        summary.details.append({"url": url, "status": "ingest_error", "error": str(e)[:200]})
    return _extract_links(result.content, url, settings.crawl_max_links_per_page)

async def _recursive_crawl(url: str, depth: int, max_depth: int, whitelist: Optional[list], visited: set[str], limit: int, summary: CrawlSummary) -> None:
    """递归抓取：深度控制 + 白/黑名单 + visited 去重 + 总页数上限

    Args:
        url: 待抓取 URL（种子页或递归链接）
        depth: 当前深度（种子页=0）
        max_depth: 本源最大深度（min(source.max_depth, config 全局上限)）
        whitelist: 本源白名单前缀（不命中不递归）；None=不限制（module-080
            优先级主题为系统显式请求，放行；黑名单/robots/审查照常生效）
        visited: 去重池（单次 run_crawl 全树共享，循环自断）
        limit: 总页数上限（全树共享计数）
        summary: 批次汇总（累计计数与详情）
    """
    if len(visited) >= limit or depth > max_depth:
        return
    url = _normalize_url(url)
    if (whitelist is not None and not _matches_any(url, whitelist)) or url in visited:
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

async def _crawl_single_source(src: dict, visited: set[str], limit: int, summary: CrawlSummary) -> None:
    """处理单个源的抓取（module-080 从 run_crawl 提取，保持行为不变）"""
    url_pattern = src.get("url_pattern", "")
    name = src.get("name", url_pattern)
    if not url_pattern or not _is_safe_url(url_pattern):
        summary.skipped += 1
        return
    if _is_blacklisted_url(url_pattern):
        logger.info("种子命中黑名单，跳过: %s", url_pattern[:80])
        summary.skipped += 1
        return
    raw_depth = src.get("max_depth")
    source_depth = max(int(raw_depth) if raw_depth is not None else 1, 0)
    max_depth = min(source_depth, settings.crawl_max_depth)
    whitelist = [_normalize_url(url_pattern)]
    logger.info("开始递归抓取: %s (%s, max_depth=%d)", name, url_pattern[:80], max_depth)
    await _recursive_crawl(url_pattern, 0, max_depth, whitelist, visited, limit, summary)


async def run_crawl(sources: list[dict], *, max_pages: int = 0) -> CrawlSummary:
    """执行一次抓取批次（受控递归：深度 + 去重 + 过滤 + 总页数上限）

    module-080：抓取前动态计算优先级（待学笔记关键词匹配源 url_pattern/name），
    高优先源先抓。

    Args:
        sources: source_configs 行列表 [{"url_pattern", "name", "enabled", "max_depth", "priority"}]
        max_pages: 单次最大抓取页数（0 = config 默认；递归全树共享计数）

    Returns:
        CrawlSummary 汇总
    """
    if not settings.crawl_enabled:
        logger.info("抓取功能已禁用（crawl_enabled=false）")
        return CrawlSummary()

    # module-080：动态优先级计算（待学笔记关键词匹配源）
    sources = await _prioritize_sources(sources)

    limit = max_pages or settings.crawl_max_pages_per_run
    summary = CrawlSummary()
    visited: set[str] = set()

    for src in sources:
        await _crawl_single_source(src, visited, limit, summary)

    logger.info(
        "抓取批次完成: crawled=%d, approved=%d, rejected=%d, errors=%d, skipped=%d",
        summary.crawled, summary.approved, summary.rejected,
        summary.errors, summary.skipped,
    )
    return summary

# ─── 待学笔记优先级加权（module-080） ───


async def _prioritize_sources(sources: list[dict]) -> list[dict]:
    """动态计算抓取优先级：待学笔记关键词匹配源 url_pattern/name 时提升 priority

    Args:
        sources: 源配置列表（含 priority 字段）

    Returns:
        按 _priority 降序排列的源列表（不修改原列表，新增 _priority 字段）
    """
    try:
        from rag.memory.weak_topics import recall_weak_topics, extract_keywords
        topics = await recall_weak_topics()
        keywords = extract_keywords(topics)
        if not keywords:
            # 无待学笔记，按 DB 静态 priority 排序
            for src in sources:
                src["_priority"] = src.get("priority", 0)
            sources.sort(key=lambda s: s["_priority"], reverse=True)
            return sources

        boost = settings.weak_topic_priority_boost
        for src in sources:
            base_priority = src.get("priority", 0)
            url_pattern = src.get("url_pattern", "").lower()
            name = src.get("name", "").lower()
            # 子串匹配：任一关键词命中 url_pattern 或 name 则提升
            matched = sum(1 for kw in keywords if kw in url_pattern or kw in name)
            src["_priority"] = base_priority + matched * boost
            if matched > 0:
                logger.info("待学笔记匹配源: %s (keywords=%d, boost=%d)", 
                           src.get("name", ""), matched, matched * boost)

        sources.sort(key=lambda s: s["_priority"], reverse=True)
        return sources
    except Exception as e:
        logger.warning("优先级计算失败，降级为默认排序: %s", e)
        for src in sources:
            src["_priority"] = src.get("priority", 0)
        sources.sort(key=lambda s: s["_priority"], reverse=True)
        return sources





_scheduler = None


async def _scheduled_crawl_job() -> None:
    """APScheduler 定时任务回调：优先级主题先抓，随后读取 DB 源配置执行常规抓取"""
    try:
        # module-080：低分题优先级主题先于常规源抓取（延迟导入防循环依赖）
        from rag.crawl.priority_crawl import drain_priority_seeds
        await drain_priority_seeds()
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
    """从 DB 加载所有源配置（含 priority，module-080）"""
    try:
        from src.database import async_session_factory
        from sqlalchemy import text
        async with async_session_factory() as session:
            result = await session.execute(
                text("SELECT id, url_pattern, name, enabled, max_depth, last_crawled_at, priority FROM source_configs ORDER BY priority DESC, id")
            )
            rows = result.fetchall()
            return [
                {
                    "id": r[0], "url_pattern": r[1], "name": r[2], "enabled": r[3],
                    "max_depth": r[4] if len(r) > 4 and r[4] is not None else 1,
                    "last_crawled_at": r[5] if len(r) > 5 else None,
                    "priority": r[6] if len(r) > 6 and r[6] is not None else 0,
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
