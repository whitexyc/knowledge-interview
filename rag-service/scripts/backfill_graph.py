"""
Graph RAG 补跑脚本 — 对已有文档提取实体/关系并写入 AGE 图

为什么需要：
  之前 ensure_graph() 吞异常导致 knowledge_graph 从未创建成功，
  文档入库时的实体提取/写入全部静默降级，图里 0 节点 0 边。
  这个脚本对现有父块文档重新跑一遍 LLM 实体提取，填充图。

用法：
  python backfill_graph.py            # 全部文档
  python backfill_graph.py --limit 5  # 只处理前 5 篇（试运行）
  python backfill_graph.py --id 87    # 只处理指定文档

幂等性：
  已有关联实体（doc_ids 含该文档）的文档会跳过。
  重复执行不会重复写入实体（MERGE 幂等）。
"""
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ai_service 根

from sqlalchemy import select, text

from src.database import async_session_factory
from rag.models import Document
from rag.graph.graph_store import graph_store, GRAPH_NAME
from rag.graph.graph_extractor import graph_extractor

logger = logging.getLogger("backfill_graph")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


async def _doc_has_entities(doc_id: int) -> bool:
    """检查图中是否已有该文档关联的实体（幂等跳过）"""
    try:
        async with async_session_factory() as session:
            await session.execute(text("LOAD 'age'"))
            await session.execute(text(
                "SET search_path = ag_catalog, \"$user\", public"
            ))
            rows = await session.execute(text(f"""
                SELECT * FROM cypher('{GRAPH_NAME}', $$
                    MATCH (e:Entity)
                    WHERE CAST(e.doc_ids AS TEXT) LIKE '%{doc_id}%'
                    RETURN count(e) AS cnt
                $$) AS (cnt agtype)
            """))
            row = rows.fetchone()
            return row is not None and int(str(row[0]).strip('"')) > 0
    except Exception:
        return False


async def main(limit: int = 0, only_id: int = 0):
    # 1. 确保图和扩展就绪
    ok = await graph_store.ensure_graph()
    if not ok:
        logger.error("AGE 图就绪失败，终止")
        sys.exit(1)
    logger.info("AGE 图就绪: %s", GRAPH_NAME)

    # 2. 取父块文档
    async with async_session_factory() as session:
        q = select(Document).where(Document.parent_id.is_(None))
        if only_id:
            q = q.where(Document.id == only_id)
        q = q.order_by(Document.id)
        if limit:
            q = q.limit(limit)
        docs = (await session.execute(q)).scalars().all()

    logger.info("待处理父块文档: %d 篇", len(docs))

    stats = {"processed": 0, "skipped": 0, "failed": 0, "entities": 0, "relations": 0}

    for doc in docs:
        # 幂等：已有实体则跳过
        if await _doc_has_entities(doc.id):
            logger.info("[跳过] doc_id=%d 已有实体", doc.id)
            stats["skipped"] += 1
            continue

        try:
            extraction = await graph_extractor.extract_from_document(doc.content)
            entities = extraction.get("entities", [])
            relations = extraction.get("relations", [])

            for ent in entities:
                name = ent.get("name", "").strip()
                ent_type = ent.get("type", "concept")
                if name:
                    await graph_store.upsert_entity(name, ent_type, int(doc.id))

            for rel in relations:
                src = rel.get("source", "").strip()
                tgt = rel.get("target", "").strip()
                if src and tgt:
                    await graph_store.upsert_relation(src, tgt)

            stats["processed"] += 1
            stats["entities"] += len(entities)
            stats["relations"] += len(relations)
            logger.info("[完成] doc_id=%d %s | entities=%d relations=%d",
                        doc.id, doc.title[:40], len(entities), len(relations))
        except Exception as e:
            stats["failed"] += 1
            logger.error("[失败] doc_id=%d: %s", doc.id, e)

    logger.info("=== 汇总: processed=%d, skipped=%d, failed=%d, entities=%d, relations=%d ===",
                stats["processed"], stats["skipped"], stats["failed"],
                stats["entities"], stats["relations"])


if __name__ == "__main__":
    setup_logging()
    args = sys.argv[1:]
    limit = 0
    only_id = 0
    for i, a in enumerate(args):
        if a == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
        if a == "--id" and i + 1 < len(args):
            only_id = int(args[i + 1])
    asyncio.run(main(limit=limit, only_id=only_id))
