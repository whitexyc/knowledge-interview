"""
会话记忆服务 — 会话历史持久化（module-034 / module-046 会话摘要）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

复用 documents 表（source='memory:<identity>:session:'，无新表），
把对话历史按轮次持久化，供刷新/换设备恢复；不参与向量检索（无 embedding，
仅按 source 等值查询 + id 排序恢复）。与内存态 IP_SESSION_MESSAGES 的关系：
持久化为主（生成时优先恢复持久化会话），内存态降级为兜底缓存（/ai/chat/sessions
端点等即时读取）。

- save_session_messages(identity, messages): 写入会话消息（每消息一条，按 identity
  隔离；content_hash 去重幂等；超上限滚动删除最旧）
- get_session_messages(identity, limit): 恢复最近会话（id 升序，最近 limit 条）
- get_session_summary(identity): 读取最近一条会话摘要（module-046 WP2）

module-046 WP2 会话摘要（MemGPT 递归摘要公式）：
  超限滚动删除前，把最旧消息段 LLM 压缩成摘要（增量：新摘要 = 摘要(旧摘要 +
  新对话段)），摘要存 documents 表（source='memory:<identity>:session_summary:'，
  title='session_summary'，无 embedding，仅顺序读最新一条）。摘要 LLM 失败 →
  跳过摘要（fail-open，不阻塞滚动删除/对话）。摘要层与 session 层 source 前缀
  不同（尾冒号隔离），memory 各层精确匹配不互扰。

身份（module-032）：identity = user_id 优先，否则 client_ip（匿名降级，零回归）。
source 尾冒号分隔身份与内容（'memory:<identity>:session:'），配合身份规范化
（_normalize_identity）杜绝通配符注入绕过按身份隔离。
"""
import asyncio
import hashlib
import logging

from sqlalchemy import delete, func, select

from src.config import settings
from src.database import async_session_factory
from llm.client import LLMFactory
from rag.models import Document
from rag.memory.memory import MEMORY_SOURCE_PREFIX, _normalize_identity

logger = logging.getLogger(__name__)

# 会话层标识：source = 'memory:<identity>:session:'
SESSION_LAYER = "session"
# 会话摘要层标识：source = 'memory:<identity>:session_summary:'（module-046 WP2）
SESSION_SUMMARY_LAYER = "session_summary"

# 摘要 LLM 提示词（会议纪要式压缩 + 增量递归公式）
_SUMMARY_PROMPT = """你是会话摘要助手。请把旧对话段压缩成简洁摘要（会议纪要式），
保留关键事实、用户偏好、任务状态和重要结论，丢弃寒暄与无关细节。

已有摘要（早期对话，可能为空）:
{old_summary}

新对话段（即将被滚动删除，须并入摘要）:
{segment}

请输出合并后的新摘要（把新对话段的重要信息并入已有摘要，不要遗漏已有摘要的要点）："""

# 摘要 LLM 超时（秒）：超时降级跳过摘要（fail-open，不阻塞滚动删除）
_SUMMARY_TIMEOUT_SECONDS = 10


def _session_source(identity: str) -> str:
    """构造会话记忆 source：'memory:<identity>:session:'

    Args:
        identity: 已规范化的身份标识（user_id 优先，否则 client_ip）

    Returns:
        会话记忆 source 字符串
    """
    return f"{MEMORY_SOURCE_PREFIX}{identity}:{SESSION_LAYER}:"


def _session_summary_source(identity: str) -> str:
    """构造会话摘要 source：'memory:<identity>:session_summary:'（module-046 WP2）

    与会话层 source（'memory:<identity>:session:'）前缀不同（尾冒号隔离），
    memory 各层精确匹配（_layer_pattern）不互扰——摘要行不会被 session/短期/
    长期检索误命中。

    Args:
        identity: 已规范化的身份标识（user_id 优先，否则 client_ip）

    Returns:
        会话摘要 source 字符串
    """
    return f"{MEMORY_SOURCE_PREFIX}{identity}:{SESSION_SUMMARY_LAYER}:"


class SessionMemoryService:
    """会话记忆服务（module-034，会话历史持久化）

    职责：
    - save_session_messages: 写入会话消息（按身份隔离 + content_hash 幂等 + 超限滚动）
    - get_session_messages: 恢复最近会话（供生成 history，刷新/换设备不丢）
    """

    async def save_session_messages(
        self, identity: str, messages: list[dict],
    ) -> int:
        """保存会话消息到持久化（source='memory:<identity>:session:'）

        每消息写一条 Document（无 embedding，仅有序恢复）；按身份隔离。
        content_hash 去重：完全重复内容幂等跳过（重复保存不堆积）。写入后按
        settings.memory_session_max_messages（默认 50）控制上限，超限滚动删除
        最旧消息。任何单步失败降级日志，不影响对话响应。

        Args:
            identity: 身份标识（user_id 优先，否则 client_ip）
            messages: 会话消息列表 [{"role": "user"|"assistant", "content": str}, ...]

        Returns:
            新写入的消息条数
        """
        if not messages:
            return 0
        identity = _normalize_identity(identity)
        source = _session_source(identity)
        async with async_session_factory() as session:
            new_count = await self._ingest_messages(session, source, messages)
            if new_count:
                try:
                    await session.commit()
                except Exception as e:
                    logger.error("会话持久化提交失败: %s", e)
                    raise
            # 上限控制（超限滚动删除最旧）；失败降级不影响已保存
            try:
                await self._trim(session, identity, source)
            except Exception as e:
                logger.warning("会话上限清理失败（降级）: %s", e)
        logger.info("会话持久化: identity=%s, new=%d", identity, new_count)
        return new_count

    async def _ingest_messages(
        self, session, source: str, messages: list[dict],
    ) -> int:
        """查重后写入会话消息到 session，返回新写入条数

        content_hash 去重：先查当前 source 下已有 hash，完全重复跳过（幂等）。
        每消息构造 Document（无 embedding，id 升序恢复）。

        Args:
            session: 数据库会话
            source: 会话记忆 source 字符串
            messages: 会话消息列表 [{"role": "user"|"assistant", "content": str}, ...]

        Returns:
            新写入的消息条数
        """
        existing_hashes = set()
        try:
            rows = await session.execute(
                select(Document.content_hash).where(Document.source == source)
            )
            existing_hashes = {r[0] for r in rows.all() if r[0]}
        except Exception as e:
            logger.warning("会话去重检索失败，忽略幂等: %s", e)
        new_count = 0
        for msg in messages:
            role = str(msg.get("role") or "").strip()
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if digest in existing_hashes:
                continue
            session.add(Document(
                title=f"session:{role}" if role else "session",
                content=content,
                source=source,
                embedding=None,
                parent_id=None,
                content_hash=digest,
            ))
            existing_hashes.add(digest)
            new_count += 1
        return new_count

    async def _trim(self, session, identity: str, source: str) -> None:
        """会话上限控制：超出 settings.memory_session_max_messages 删除最旧消息

        module-046 WP2：删除前先把最旧 excess 条消息段 LLM 压缩成会话摘要
        （增量：新摘要 = 摘要(旧摘要 + 新对话段)，MemGPT 递归公式），摘要写
        documents 表（source='memory:<identity>:session_summary:'，无向量，
        仅顺序读最新一条）。摘要 LLM 失败 → 跳过摘要（fail-open，仍滚动删除，
        不阻塞）。按 id 升序取超限条数删除（id 单调递增即时间序），保持每
        identity 会话条数有界，防止 documents 表无限增长。

        Args:
            session: 数据库会话（事务由调用方 save_session_messages 持有）
            identity: 已规范化的身份标识（user_id 优先，否则 client_ip）
            source: 会话记忆 source（'memory:<identity>:session:'）
        """
        max_msgs = max(settings.memory_session_max_messages, 1)
        count = (
            await session.execute(
                select(func.count()).select_from(Document)
                .where(Document.source == source)
            )
        ).scalar() or 0
        excess = count - max_msgs
        if excess <= 0:
            return
        # module-046 WP2：滚动删除前先把最旧段压缩成摘要（内部 fail-open，不阻塞删除）
        await self._summarize_oldest_segment(session, identity, source, excess)
        rows = await session.execute(
            select(Document.id)
            .where(Document.source == source)
            .order_by(Document.id.asc())
            .limit(excess)
        )
        ids = [r[0] for r in rows.all()]
        if ids:
            await session.execute(delete(Document).where(Document.id.in_(ids)))
            await session.commit()
            logger.info("会话上限清理: source=%s, deleted=%d", source, len(ids))

    async def _summarize_oldest_segment(
        self, session, identity: str, source: str, excess: int,
    ) -> None:
        """滚动删除前：把最旧 excess 条消息段 LLM 压缩成会话摘要（module-046 WP2）

        增量更新（MemGPT 递归公式）：新摘要 = 摘要(旧摘要 + 新对话段)。
        摘要行写入 documents（source='memory:<identity>:session_summary:'，
        title='session_summary'，content=摘要文本，无 embedding、无向量，
        仅顺序读最新一条——写入时删旧摘要行，保持每 identity 至多一条）。
        任何异常（LLM 失败/超时/DB 错误）→ 日志降级，不抛出（fail-open，
        调用方 _trim 继续滚动删除，不阻塞对话）。

        Args:
            session: 数据库会话（事务由调用方 _trim 持有）
            identity: 已规范化的身份标识（user_id 优先，否则 client_ip）
            source: 会话记忆 source（'memory:<identity>:session:'）
            excess: 待删除（待摘要）的最旧消息条数
        """
        summary_source = _session_summary_source(identity)
        try:
            # 1. 取最旧 excess 条消息（即将被滚动删除的段）
            rows = await session.execute(
                select(Document)
                .where(Document.source == source)
                .order_by(Document.id.asc())
                .limit(excess)
            )
            segment = rows.scalars().all()
            if not segment:
                return
            segment_text = "\n".join(
                f"{'用户' if d.title == 'session:user' else '助手'}: {d.content}"
                for d in segment if (d.content or "").strip()
            )
            if not segment_text.strip():
                return
            # 2. 读旧摘要（仅最新一条；无摘要/失败 → 空串，增量公式照常）
            old_summary = ""
            try:
                rows = await session.execute(
                    select(Document.content)
                    .where(Document.source == summary_source)
                    .order_by(Document.id.desc())
                    .limit(1)
                )
                first = rows.first()
                old_summary = (first[0] or "").strip() if first else ""
            except Exception as e:
                logger.warning("旧摘要读取失败，按无摘要增量: %s", e)
            # 3. LLM 增量压缩：新摘要 = 摘要(旧摘要 + 新对话段)
            prompt = _SUMMARY_PROMPT.format(
                old_summary=old_summary or "（无）",
                segment=segment_text,
            )
            client = LLMFactory.get_client()
            new_summary = await asyncio.wait_for(
                client.generate(prompt), timeout=_SUMMARY_TIMEOUT_SECONDS,
            )
            new_summary = (new_summary or "").strip()
            if not new_summary:
                logger.warning("会话摘要生成返回空，跳过摘要写入")
                return
            # 4. 写摘要行：删旧摘要行 + 写新摘要行（每 identity 至多一条，仅顺序读最新）
            await session.execute(
                delete(Document).where(Document.source == summary_source)
            )
            session.add(Document(
                title="session_summary",
                content=new_summary,
                source=summary_source,
                embedding=None,
                parent_id=None,
                content_hash=hashlib.sha256(new_summary.encode("utf-8")).hexdigest(),
            ))
            await session.commit()
            logger.info("会话摘要写入: identity=%s, segment=%d, summary_len=%d",
                        identity, len(segment), len(new_summary))
        except Exception as e:
            logger.warning("会话摘要生成失败（降级跳过，不影响滚动删除）: %s", e)

    async def get_session_summary(self, identity: str) -> str:
        """读取最近一条会话摘要（module-046 WP2，仅顺序读最新一条）

        查询 source='memory:<identity>:session_summary:'，按 id 降序取最新
        一条（增量摘要只保留最新，早期摘要已并入）。无摘要/读取失败 → 返回
        空串（调用方跳过摘要段注入，零回归）。

        Args:
            identity: 身份标识（user_id 优先，否则 client_ip）

        Returns:
            摘要文本；无摘要/失败返回 ""
        """
        identity = _normalize_identity(identity)
        summary_source = _session_summary_source(identity)
        try:
            async with async_session_factory() as session:
                rows = await session.execute(
                    select(Document.content)
                    .where(Document.source == summary_source)
                    .order_by(Document.id.desc())
                    .limit(1)
                )
                first = rows.first()
                return (first[0] or "").strip() if first else ""
        except Exception as e:
            logger.warning("会话摘要读取失败，返回空: %s", e)
            return ""

    async def get_session_messages(
        self, identity: str, limit: int | None = None,
    ) -> list[dict]:
        """恢复最近会话消息（module-034，按身份隔离）

        查询 source='memory:<identity>:session:' 的全部消息，按 id 升序
        （时间序）取最近 limit 条。无记录返回空列表（调用方降级用当前请求
        history，零回归）。

        Args:
            identity: 身份标识（user_id 优先，否则 client_ip）
            limit: 返回条数（默认 settings.memory_session_history_limit）

        Returns:
            [{"role": "user"|"assistant", "content": str}, ...]（时间升序，最近 limit 条）
        """
        if limit is None:
            limit = settings.memory_session_history_limit
        identity = _normalize_identity(identity)
        source = _session_source(identity)
        try:
            async with async_session_factory() as session:
                rows = await session.execute(
                    select(Document)
                    .where(Document.source == source)
                    .order_by(Document.id.asc())
                )
                docs = rows.scalars().all()
        except Exception as e:
            logger.warning("会话恢复失败，返回空: %s", e)
            return []
        if not docs:
            return []
        recent = docs[-max(limit, 1):]
        out: list[dict] = []
        for doc in recent:
            content = (doc.content or "").strip()
            if not content:
                continue
            role = "user"
            if doc.title and doc.title.startswith("session:"):
                role = doc.title[len("session:"):].strip()
            if role not in ("user", "assistant"):
                role = "user"
            out.append({"role": role, "content": content})
        return out


# 全局单例 — 整个应用共享一个 SessionMemoryService 实例（无状态）
session_memory_service = SessionMemoryService()
