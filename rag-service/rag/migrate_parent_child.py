"""
M17 父子分块迁移脚本 — 将旧格式文档转为父子分块格式

用法：
  python -m rag.migrate_parent_child          # 正式执行迁移
  python -m rag.migrate_parent_child --dry-run  # 试运行（仅输出统计）

迁移逻辑：
  旧格式：parent_id IS NULL AND embedding IS NOT NULL
  → 原行置空 embedding 变为父块，新增一行作为子块（parent_id 指向父块，保留 embedding）

幂等性：
  重复执行无害：每次迁移前检查是否已有子块（WHERE parent_id = 原行.id），
  若已有则跳过该行。

注意：
  - 需要在数据库连接可用的环境下运行
  - 迁移前建议备份数据库
"""
import asyncio
import logging
import sys
from typing import Optional

from sqlalchemy import select, update, text

from src.database import async_session_factory
from rag.models import Document

logger = logging.getLogger(__name__)


def setup_logging():
    """配置日志输出到控制台"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def migrate(dry_run: bool = False) -> dict:
    """执行父子分块迁移

    Args:
        dry_run: True 时仅统计待迁移行数，不修改数据

    Returns:
        {"total_candidates": int, "migrated": int, "skipped": int}
    """
    stats = {"total_candidates": 0, "migrated": 0, "skipped": 0}

    async with async_session_factory() as session:
        # 1. 查找旧格式文档（有 embedding 但无 parent_id）
        result = await session.execute(
            select(Document).where(
                Document.parent_id.is_(None),
                Document.embedding.isnot(None),
            ).order_by(Document.id)
        )
        candidates = result.scalars().all()

        stats["total_candidates"] = len(candidates)
        logger.info("找到 %d 条旧格式文档待迁移", len(candidates))

        if not candidates:
            return stats

        for doc in candidates:
            # 2. 检查是否已有子块（幂等）
            existing_child = await session.execute(
                select(Document).where(Document.parent_id == doc.id).limit(1)
            )
            if existing_child.scalar_one_or_none() is not None:
                logger.debug("文档 id=%d 已有子块，跳过", doc.id)
                stats["skipped"] += 1
                continue

            if dry_run:
                stats["migrated"] += 1
                continue

            # 3. 保存原始 embedding
            old_embedding = doc.embedding

            # 4. 将原行 embedding 置 NULL → 变为父块
            await session.execute(
                update(Document)
                .where(Document.id == doc.id)
                .values(embedding=None)
            )

            # 5. 新增子块行（parent_id 指向原行，保留 embedding）
            child = Document(
                title=doc.title,
                content=doc.content,
                source=doc.source,
                page_num=doc.page_num,
                meta=doc.meta,
                content_hash=doc.content_hash,
                embedding=old_embedding,
                parent_id=doc.id,
            )
            session.add(child)

            stats["migrated"] += 1

        if not dry_run:
            await session.commit()
            logger.info("迁移完成: 共 %d 条，迁移 %d 条，跳过 %d 条",
                        stats["total_candidates"], stats["migrated"], stats["skipped"])
        else:
            logger.info("试运行完成: 共 %d 条，可迁移 %d 条，已迁移 %d 条",
                        stats["total_candidates"], stats["migrated"], stats["skipped"])

    return stats


def main():
    """命令行入口"""
    setup_logging()

    dry_run = "--dry-run" in sys.argv

    if dry_run:
        logger.info("=== 试运行模式：仅统计，不修改数据 ===")
    else:
        logger.info("=== 正式迁移模式 ===")

    try:
        stats = asyncio.run(migrate(dry_run=dry_run))
        logger.info("统计: total_candidates=%d, migrated=%d, skipped=%d",
                     stats["total_candidates"], stats["migrated"], stats["skipped"])
    except Exception as e:
        logger.error("迁移失败: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
