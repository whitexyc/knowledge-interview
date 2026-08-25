"""
embedding 维度迁移脚本 — 384 → 1024（本地 bge-m3 GGUF）

流程：
1. 清空 documents.embedding（旧 384 维向量）
2. ALTER COLUMN embedding TYPE vector(1024)
3. 遍历所有子块文档（parent_id IS NOT NULL），用本地 bge-m3 重新向量化
4. 批量 UPDATE embedding

用法：
  python migrate_embedding_1024.py

注意：
- 破坏性：旧 384 维向量丢失（文档内容/标题/分块结构保留）
- 需要本地嵌入模型就绪（models/bge-m3-gguf/bge-m3-q8_0.gguf）
- 建议迁移前备份数据库
"""
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ai_service 根

from sqlalchemy import text

from src.database import async_session_factory
from rag.retrieval.embeddings import embedding_service

logger = logging.getLogger("migrate_embedding")


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


async def migrate():
    # 1. 清空旧向量 + 改列类型
    logger.info("=== Step 1: 清空旧 embedding + ALTER 列 ===")
    async with async_session_factory() as session:
        await session.execute(text("UPDATE documents SET embedding = NULL"))
        await session.execute(text("ALTER TABLE documents ALTER COLUMN embedding TYPE vector(1024)"))
        await session.commit()
        logger.info("列已改为 vector(1024)")

    # 2. 收集所有子块文档
    logger.info("=== Step 2: 收集子块文档 ===")
    async with async_session_factory() as session:
        rows = await session.execute(text("""
            SELECT id, content FROM documents
            WHERE parent_id IS NOT NULL
            ORDER BY id
        """))
        docs = rows.fetchall()
    logger.info("待向量化子块: %d 条", len(docs))

    # 3. 预热嵌入模型
    logger.info("=== Step 3: 预热嵌入模型 ===")
    await embedding_service.embed_text("预热")
    logger.info("模型就绪")

    # 4. 分批重新向量化
    logger.info("=== Step 4: 重新向量化 ===")
    batch_size = 20
    total = len(docs)
    for start in range(0, total, batch_size):
        batch = docs[start:start + batch_size]
        texts = [d[1] for d in batch]
        ids = [d[0] for d in batch]

        try:
            embs = await embedding_service.embed_documents(texts)
            async with async_session_factory() as session:
                for doc_id, emb in zip(ids, embs):
                    # pgvector 需要字符串格式 '[0.1,0.2,...]'
                    emb_str = "[" + ",".join(str(v) for v in emb) + "]"
                    await session.execute(
                        text("UPDATE documents SET embedding = :emb WHERE id = :id"),
                        {"emb": emb_str, "id": doc_id},
                    )
                await session.commit()
        except Exception as e:
            logger.error("批次失败 (start=%d): %s", start, e)
            continue

        done = min(start + batch_size, total)
        logger.info("  进度: %d/%d (%.0f%%)", done, total, done / total * 100)

    # 5. 验证
    logger.info("=== Step 5: 验证 ===")
    async with async_session_factory() as session:
        cnt = await session.execute(text("""
            SELECT COUNT(*) FROM documents
            WHERE embedding IS NOT NULL
        """))
        total_vec = cnt.scalar()
    logger.info("向量化完成: %d/%d 条有向量", total_vec, total)
    logger.info("=== MIGRATION DONE ===")


if __name__ == "__main__":
    setup_logging()
    try:
        asyncio.run(migrate())
    except Exception as e:
        logger.error("迁移失败: %s", e, exc_info=True)
        sys.exit(1)
