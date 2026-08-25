"""文档去重三级 — module-064 / ADR-0014 WP6

在整个 ingestion 链路中的位置：
  清洗/归一化文本 → [DocumentDedup 三级去重] → 入库（content_hash 完全重复丢弃 /
  语义重复标簇不删 / canonical 检索抑制）

三级去重：
  L1 内容哈希 —— 文档级全文本 SHA256（doc_content_hash 列）：完全相同 → 直接丢弃
  L2 文档级 embedding 余弦 —— bge-m3 + 绝对余弦口径（复用 module-035/记忆去重
     同款：embed_text 已 L2 归一化，点积=余弦）：≥ 阈值（默认 0.95）语义重复 →
     不删，标 duplicate_cluster_id + canonical 选择（留最新/结构完整），检索抑制
     （engine._expand_to_parents 只出 canonical）
  L3 SimHash-LSH —— 文档量几千+ 才上（当前 O(N²) 够用），接口 simhash_lsh 预留，
     标注"待规模"（honest：不实现不假装）

三个坑（ADR-0014 决策 6）：
  - Boilerplate 先剥离：共同页脚/免责声明主导相似度前先扒（strip_boilerplate）
  - 同源内语义去重：不跨 source 折叠（identity 与 content 分离，历史版本不折叠）
  - 文档级 embedding 存于文档根父块（parent_id IS NULL 首父块）的 embedding 列：
    复用既有列零新增 Vector 列；父块不参与向量检索（retriever 只查 parent_id
    IS NOT NULL 子块），故不污染检索（module-064 诚实声明此语义复用）
"""
import hashlib
import logging
import re
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from rag.models import Document
from rag.retrieval.embeddings import EmbeddingService, embedding_service as default_embedding_service

logger = logging.getLogger(__name__)

# 默认语义去重阈值（绝对余弦口径，对齐 module-035 记忆去重同款 0.95）
DEFAULT_DEDUP_THRESHOLD = 0.95


def exact_hash(text: str) -> str:
    """文档级全文本 SHA256（L1 内容哈希）"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Boilerplate 先剥离（相似度前） ───────────────────────────────────────
# 共同页脚/免责声明/重复导航等套话：出现在每篇文档且与主题无关，会主导相似度。
# 规则级（正则，中文+英文常见套话）；语料扩充留待后续（诚实：有限清单）。
_BOILERPLATE_LINE_RE = re.compile(
    r"^(?:"
    r"第\s*\d+\s*页"
    r"|page\s*\d+(?:\s*(?:of|/)\s*\d+)?"
    r"|(?:版权所有|版权归[^。；]*所有|Copyright|©|All rights reserved)[^。；]*"
    r"|免责声明[^。；]*"
    r"|本文档[^。；]*(?:内部|机密|仅供)[^。；]*"
    r"|(?:公司|组织|部门)[^。；]*(?:名称|logo)[^。；]*"
    r")\s*[。；]?\s*$",
    re.IGNORECASE,
)


def strip_boilerplate(text: str, enabled: Optional[bool] = None) -> str:
    """剥离 Boilerplate（页脚/免责声明等套话行），相似度计算前调用

    Args:
        text: 待比对文本
        enabled: 开关（None → 读 settings.doc_dedup_boilerplate_enabled）

    Returns:
        剥离后的文本（Boilerplate 行被移除，空行压缩）
    """
    if enabled is False or (enabled is None and not settings.doc_dedup_boilerplate_enabled):
        return text
    lines = []
    for ln in text.split("\n"):
        stripped = ln.strip()
        if stripped and _BOILERPLATE_LINE_RE.match(stripped):
            continue
        lines.append(ln)
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


# ── L2 文档级 embedding 余弦 ────────────────────────────────────────────
async def compute_doc_embedding(text: str, embedding_service: Optional[EmbeddingService] = None):
    """文档级 embedding（bge-m3 绝对余弦口径，复用 embed_text 已 L2 归一化）

    embed_text 是 async（返回协程）——本函数必须 await，否则拿到协程对象
    （真实 DB 冒烟暴露：doc_embedding=coroutine 落库报 "expected list or ndarray"）。
    失败返回 None（fail-open：语义去重跳过不阻断入库）。
    """
    svc = embedding_service or default_embedding_service
    try:
        return await svc.embed_text(text)
    except Exception as e:
        logger.warning("文档级 embedding 计算失败（语义去重跳过）: %s", e)
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    """绝对余弦（双方已 L2 归一化，点积=余弦）"""
    return sum(x * y for x, y in zip(a, b))


async def find_semantic_duplicate(
    doc_text: str,
    embedding_service: Optional[EmbeddingService] = None,
    session: Optional[AsyncSession] = None,
    threshold: Optional[float] = None,
) -> Optional[dict]:
    """在现有文档根父块中找语义重复（L2）

    文档级 embedding 存于文档根父块（parent_id IS NULL 且 embedding 非空）——
    即 module-064 之后入库的文档（首父块持有文档级向量）。存量旧文档（embedding
    NULL）不参与比较（诚实边界：语义去重对新格式文档生效，存量靠 L1/title）。

    Args:
        doc_text: 清洗/归一化后的文档全文（先剥离 Boilerplate）
        embedding_service: 嵌入服务（默认全局单例）
        session: 数据库会话（None → 自建）
        threshold: 余弦阈值（None → settings.doc_dedup_threshold）

    Returns:
        {"id": int, "title": str, "cluster_id": str, "cosine": float} 或 None
    """
    threshold = settings.doc_dedup_threshold if threshold is None else threshold
    vec = await compute_doc_embedding(strip_boilerplate(doc_text), embedding_service)
    if vec is None:
        return None

    def _query(conn):
        return conn.execute(
            select(Document).where(
                Document.parent_id.is_(None),
                Document.embedding.is_not(None),
                # 候选只出 canonical（module-064 minor-1）：非 canonical 重复副本
                # 虽存文档级 embedding，但检索侧已抑制（_expand_to_parents 只出
                # canonical），语义对齐——副本不参与比对防低概率误判
                Document.is_canonical.is_(True),
                # 同源内语义去重：不跨 source 折叠（记忆文档 source=memory:% 排除，
                # 对齐 retriever._source_condition 口径——旧格式单文档记忆
                # parent_id=None 且带向量，须排除防把知识库文档折叠进记忆簇）
                (Document.source.is_(None) | Document.source.not_like("memory:%")),
            )
        )

    if session is not None:
        result = await _query(session)
    else:
        from src.database import async_session_factory
        async with async_session_factory() as sess:
            result = await _query(sess)

    best: Optional[dict] = None
    for doc in result.scalars().all():
        emb = doc.embedding
        if not emb or len(emb) != len(vec):
            continue
        c = _cosine(vec, emb)
        if c >= threshold and (best is None or c > best["cosine"]):
            best = {
                "id": doc.id,
                "title": doc.title,
                "cluster_id": doc.duplicate_cluster_id or str(doc.id),
                "cosine": c,
            }
    return best


# ── L3 SimHash-LSH 接口预留（文档量几千+ 才启用） ────────────────────────
def simhash_lsh(embedding: list[float], num_bits: int = 64) -> Optional[int]:
    """SimHash-LSH 接口预留：大规模（文档量几千+）才启用

    当前知识库 124 篇量级 O(N²) 线性比对足够，不上 LSH（honest：接口预留，
    标注"待规模"）。embedding 维度任意，num_bits 为哈希位数。
    """
    logger.info("simhash_lsh: 接口预留，文档量几千+ 才启用（当前 O(N²) 够用）")
    return None
