"""SAG 检索器 — SQL join 动态连边检索（module-081）

查询时用 LLM 提取查询中的实体名 → SQL join 检索（实体匹配 ILIKE +
关系一跳）→ 输出对齐 HybridRetriever.retrieve() 格式。

设计决策：
  1. 复用 graph_extractor.extract_from_query 提取查询实体名（共享 LLM 范式）
  2. ILIKE 模糊匹配 + sag_relations 一跳（pg_trgm GIN 索引加速）
  3. 返回格式对齐 hybrid_retriever（{title, content, score, source, ...}）
  4. score 语义对齐：实体匹配直接命中文档给 1.0，一跳关系给 0.8（启发式）
"""
import logging
from typing import Optional

from sqlalchemy import text as sql_text

from src.database import async_session_factory
from rag.graph.graph_extractor import graph_extractor

logger = logging.getLogger(__name__)


async def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """SAG 检索：查询实体名 → SQL join → 相关文档

    Args:
        query: 用户查询
        top_k: 返回结果上限

    Returns:
        [{title, content, score, source, id, ...}]，对齐 HybridRetriever 输出
    """
    if not query or not query.strip():
        return []

    try:
        # Step 1: 提取查询实体名（复用 graph_extractor）
        entity_names = await graph_extractor.extract_from_query(query)
        if not entity_names:
            logger.info("SAG 检索: 未提取到实体，返回空")
            return []

        # Step 2: SQL 检索（实体匹配 ILIKE + 一跳关系）
        docs = await _sql_entity_search(entity_names, top_k)
        if len(docs) >= top_k:
            return docs[:top_k]

        # Step 3: 一跳关系补充
        hop_docs = await _sql_relation_search(entity_names, top_k - len(docs))
        seen_ids = {d["id"] for d in docs}
        for hd in hop_docs:
            if hd["id"] not in seen_ids:
                docs.append(hd)
                seen_ids.add(hd["id"])
                if len(docs) >= top_k:
                    break

        return docs[:top_k]
    except Exception as e:
        logger.warning("SAG 检索失败，返回空: %s", e)
        return []


async def _sql_entity_search(entity_names: list[str], top_k: int) -> list[dict]:
    """实体直接匹配检索：sag_entities.name ILIKE ANY → join documents"""
    if not entity_names:
        return []
    # 构造 ILIKE 模式
    patterns = [f"%{name}%" for name in entity_names[:10]]
    try:
        async with async_session_factory() as session:
            # pg_trgm GIN 索引加速 ILIKE；先查实体匹配的 doc_ids
            stmt = sql_text("""
                SELECT DISTINCT se.source_doc_ids
                FROM sag_entities se
                WHERE se.name ILIKE ANY(:patterns)
                LIMIT :limit
            """)
            result = await session.execute(stmt, {"patterns": patterns, "limit": top_k * 3})
            rows = result.fetchall()

            if not rows:
                return []

            # 展开 JSONB 数组，收集 doc_ids
            doc_ids = set()
            for row in rows:
                ids = row[0]  # JSONB array
                if isinstance(ids, list):
                    doc_ids.update(ids)
                elif isinstance(ids, str):
                    import json
                    try:
                        doc_ids.update(json.loads(ids))
                    except (json.JSONDecodeError, TypeError):
                        pass

            if not doc_ids:
                return []

            # 查 documents 表获取文档内容
            from rag.models import Document
            stmt2 = sql_text("""
                SELECT id, title, content, source, metadata
                FROM documents
                WHERE id = ANY(:doc_ids) AND parent_id IS NULL
                LIMIT :limit
            """)
            result2 = await session.execute(stmt2, {
                "doc_ids": list(doc_ids)[:top_k * 2],
                "limit": top_k,
            })
            docs = []
            for row in result2.fetchall():
                docs.append({
                    "id": row[0],
                    "title": row[1] or "",
                    "content": row[2] or "",
                    "source": row[3] or "",
                    "score": 1.0,  # 实体直接命中
                    "metadata": row[4] or {},
                })
            return docs
    except Exception as e:
        logger.warning("SAG 实体匹配检索失败: %s", e)
        return []


async def _sql_relation_search(entity_names: list[str], top_k: int) -> list[dict]:
    """一跳关系检索：通过 sag_relations 找关联实体的文档"""
    if not entity_names or top_k <= 0:
        return []
    patterns = [f"%{name}%" for name in entity_names[:10]]
    try:
        async with async_session_factory() as session:
            # 一跳：查询实体 → relations → 目标实体 → 文档
            stmt = sql_text("""
                SELECT DISTINCT sr.source_doc_id
                FROM sag_relations sr
                JOIN sag_entities se ON (se.id = sr.source_entity_id OR se.id = sr.target_entity_id)
                WHERE se.name ILIKE ANY(:patterns)
                LIMIT :limit
            """)
            result = await session.execute(stmt, {"patterns": patterns, "limit": top_k * 2})
            rows = result.fetchall()

            if not rows:
                return []

            doc_ids = [row[0] for row in rows]
            stmt2 = sql_text("""
                SELECT id, title, content, source, metadata
                FROM documents
                WHERE id = ANY(:doc_ids) AND parent_id IS NULL
                LIMIT :limit
            """)
            result2 = await session.execute(stmt2, {
                "doc_ids": doc_ids[:top_k * 2],
                "limit": top_k,
            })
            docs = []
            for row in result2.fetchall():
                docs.append({
                    "id": row[0],
                    "title": row[1] or "",
                    "content": row[2] or "",
                    "source": row[3] or "",
                    "score": 0.8,  # 一跳关系启发式分值
                    "metadata": row[4] or {},
                })
            return docs
    except Exception as e:
        logger.warning("SAG 一跳关系检索失败: %s", e)
        return []
