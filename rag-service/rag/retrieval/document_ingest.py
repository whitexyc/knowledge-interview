"""ingestion 管线编排 — 解析 → 图片 → 清洗 → 归一化 → 原件留存 → 去重 → 入库
（module-064 / ADR-0014）

在整个 ingestion 链路中的位置（四层：解析→清洗→分块→嵌入 的 1-2 层）：
  上传原始文件字节 → [document_parser 解析层] → Markdown
  → [image_pipeline 图片三层，默认关] → [document_cleaner 五步清洗]
  → [document_cleaner.normalize 无损归一化] → [document_dedup 三级去重]
  → rag_engine.add_document（分块→嵌入→落库）

职责：
  - 单点编排以上各层，任何一层失败按对应纪律降级（fail-open，不阻断入库）
  - WP5 原件落盘（original_path）+ WP6 去重标簇（doc_content_hash / cluster）
  - 统一返回入库结果（含 duplicate / dup_kind / page_count / original_path）

诚实边界：
  - 清洗/归一化/原件落盘/去重任一失败 → 降级（跳过该步/原样入库），不因清洗
    或去重失败拒收文档（AC §8 降级验收）
  - 扫描版 PDF 无 OCR → 图片未解析提示由 image_pipeline 附加，文本照常入库
"""
import hashlib
import logging
import os
import re
from typing import Optional

from sqlalchemy import select

from src.config import settings
from rag.models import Document
from rag.retrieval import document_parser, document_cleaner, image_pipeline, document_dedup

logger = logging.getLogger(__name__)


class IngestError(Exception):
    """ingestion 失败（面向用户的中文错误消息，上传端点直接透出）"""


def _safe_filename(filename: str) -> str:
    """文件名净化（防路径穿越：只保留字母数字点横下划线）"""
    safe = re.sub(r"[^0-9A-Za-z.\-_]", "_", filename or "").strip(".")
    return safe or "unnamed"


def save_original(data: bytes, filename: str, upload_dir: Optional[str] = None) -> str:
    """WP5 原件落盘：uploads/{sha256[:16]}_{净文件名}

    返回落盘路径（绝对路径）。落盘失败抛异常（调用方捕获降级继续入库）。
    """
    directory = upload_dir or settings.upload_dir
    os.makedirs(directory, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()[:16]
    path = os.path.join(directory, f"{digest}_{_safe_filename(filename)}")
    with open(path, "wb") as f:
        f.write(data)
    logger.info("原件落盘: %s (%d bytes)", path, len(data))
    return path


async def _find_exact_duplicate(doc_content_hash: str, session=None) -> Optional[int]:
    """L1 内容哈希去重：查现有文档根父块（parent_id IS NULL）同 doc_content_hash"""
    def _query(conn):
        return conn.execute(
            select(Document.id).where(
                Document.doc_content_hash == doc_content_hash,
                Document.parent_id.is_(None),
            ).limit(1)
        )

    if session is not None:
        result = await _query(session)
    else:
        from src.database import async_session_factory
        async with async_session_factory() as sess:
            result = await _query(sess)
    row = result.first()
    return row[0] if row else None


async def ingest_document(
    data: bytes,
    filename: str = "",
    title: str = "",
    source: str = "",
    *,
    persist_original: bool = True,
) -> dict:
    """完整 ingestion 入口

    Args:
        data: 上传文件原始字节
        filename: 原始文件名（格式识别/标题推导/原件命名用）
        title: 显式标题（空 → 端点按文件名推导）
        source: 来源标识
        persist_original: 是否落盘原件（WP5；命名避开模块级 save_original 函数）

    Returns:
        {
            "id": int, "title": str, "chunks": int,
            "duplicate": bool, "dup_kind": "exact"/"semantic"/None,
            "page_count": int|None, "original_path": str,
            "duplicate_cluster_id": str|None, "canonical": bool,
        }

    Raises:
        IngestError: 解析失败（message 为面向用户的中文提示）或文档无有效文本
    """
    if not data:
        raise IngestError("上传文件为空")

    # 1. 解析层（WP1）：字节 + 文件名 → Markdown
    parsed = document_parser.parse_document(data, filename)
    text = parsed.text or ""

    # 2. PDF 内嵌图片三层（WP4）：默认关，仅 PDF/含图文档触发；任何层失败 fail-open
    try:
        text = image_pipeline.process_pdf_images(text, page_count=parsed.page_count)
    except Exception as e:
        logger.warning("图片处理失败降级（fail-open）: %s", e)

    # 3. 五步清洗（WP2）：失败降级原始 Markdown（fail-open，不阻断入库）
    try:
        cleaned = document_cleaner.clean(text, source_format=parsed.format)
    except Exception as e:
        logger.warning("清洗失败降级原始 Markdown（fail-open）: %s", e)
        cleaned = text

    # 4. 无损归一化（WP3）：失败降级清洗文本（fail-open）
    try:
        normalized = document_cleaner.normalize(cleaned)
    except Exception as e:
        logger.warning("归一化失败降级清洗文本（fail-open）: %s", e)
        normalized = cleaned

    if not normalized.strip():
        raise IngestError(
            f"文档解析后无有效文本内容（{parsed.format}，可能是扫描版/纯图片文档，"
            f"OCR 默认关闭）"
        )

    # 5. L1 内容哈希去重（WP6）：完全相同 → 直接丢弃（不落原件不入库）
    doc_hash = document_dedup.exact_hash(normalized)
    existing_id = await _find_exact_duplicate(doc_hash)
    if existing_id:
        logger.info("L1 内容哈希命中重复: doc_content_hash=%s, existing_id=%s",
                    doc_hash[:12], existing_id)
        return {"id": existing_id, "title": title, "chunks": 0,
                "duplicate": True, "dup_kind": "exact",
                "page_count": parsed.page_count, "original_path": ""}

    # 6. WP5 原件留存：落盘（失败降级，不阻断入库）
    original_path = ""
    if persist_original:
        try:
            original_path = save_original(data, filename)
        except Exception as e:
            logger.warning("原件落盘失败（继续入库）: %s", e)

    # 7. L2 文档级 embedding 语义去重（WP6）：≥0.95 → 不删，标簇 + 非 canonical
    cluster_id: Optional[str] = None
    is_canonical = True
    if settings.doc_dedup_semantic_enabled:
        sdup = await document_dedup.find_semantic_duplicate(normalized)
        if sdup:
            cluster_id = sdup["cluster_id"]
            is_canonical = False
            logger.info("L2 语义重复命中: 与文档 id=%s title=%r 余弦=%.4f，标簇 %s 抑制检索",
                        sdup["id"], sdup["title"], sdup["cosine"], cluster_id)

    # 8. 文档级 embedding（供后续上传语义去重比对；失败 fail-open 跳过）
    #    与 find_semantic_duplicate 查询侧同口径（ADR-0014 决策 6 坑①）：
    #    Boilerplate 先剥离再 embed——否则候选侧向量被共同页脚/免责声明污染，
    #    套话主导相似度致误判语义重复/漏判（Review 修复 1）。
    doc_embedding = None
    if settings.doc_dedup_semantic_enabled:
        doc_embedding = await document_dedup.compute_doc_embedding(
            document_dedup.strip_boilerplate(normalized)
        )

    # 9. 入库（分块→嵌入→落库）
    from rag.engine import rag_engine  # 延迟导入避免循环依赖
    result = await rag_engine.add_document(
        title, normalized, source,
        original_path=original_path,
        doc_content_hash=doc_hash,
        duplicate_cluster_id=cluster_id,
        is_canonical=is_canonical,
        doc_embedding=doc_embedding,
    )
    result["original_path"] = original_path
    result["page_count"] = parsed.page_count
    if result.get("duplicate") and not cluster_id:
        # add_document 内部 title/content_hash 命中（存量逻辑兜底）
        result["dup_kind"] = "exact"
    elif cluster_id:
        result["duplicate"] = True
        result["dup_kind"] = "semantic"
        result["duplicate_cluster_id"] = cluster_id
        result["canonical"] = False
    else:
        result["dup_kind"] = None
    return result
