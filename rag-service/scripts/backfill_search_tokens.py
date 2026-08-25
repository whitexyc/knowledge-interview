"""
Module-020 backfill 脚本 — search_tokens 列迁移 + 已有子块文档 jieba 分词回填
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

为什么需要：
  module-020 引入 search_tokens 列后，只有新入库的子块才有分词；
  库中已有子块（parent_id IS NOT NULL）的 search_tokens 为 NULL，
  若不回填，FTS 查不到旧文档（module-019 基线正是基于旧文档）。

用法：
  python backfill_search_tokens.py            # 迁移 DDL + 全量回填
  python backfill_search_tokens.py --dry-run  # 仅统计待回填数量，不写库
  python backfill_search_tokens.py --limit 5  # 只回填前 5 条（试运行）

幂等性：
  迁移 DDL 使用 IF NOT EXISTS；回填只处理 search_tokens IS NULL 的行，
  重复执行安全，可随时重跑。

只回填子块（parent_id IS NOT NULL）：检索只查子块，父块无需分词。
"""
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ai_service 根

from sqlalchemy import select, update, text

from src.database import async_session_factory
from rag.models import Document
from rag.retrieval.text_tokenizer import tokenize

logger = logging.getLogger(__name__)

# 迁移 DDL（幂等，与 plan.md §3.2 一致）
MIGRATION_DDL = """
ALTER TABLE documents ADD COLUMN IF NOT EXISTS search_tokens TEXT;
COMMENT ON COLUMN documents.search_tokens IS 'jieba分词后的空格连接文本（用于中文FTS检索）';

CREATE INDEX IF NOT EXISTS idx_documents_search_tokens
    ON documents USING GIN (to_tsvector('simple', search_tokens));
"""


def setup_logging():
    """配置日志输出到控制台"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


async def apply_migration() -> None:
    """幂等执行 search_tokens 列迁移 + GIN 索引创建

    asyncpg 不允许单条 prepared statement 执行多条命令，
    因此按 ';' 拆分逐条执行（同 golden_retrieval.ensure_eval_runs_table）。
    """
    async with async_session_factory() as session:
        for stmt in [s.strip() for s in MIGRATION_DDL.split(";") if s.strip()]:
            await session.execute(text(stmt))
        await session.commit()
    logger.info("迁移完成: search_tokens 列 + GIN 索引已就绪")


async def backfill(dry_run: bool = False, limit: int = 0) -> dict:
    """对已有子块文档（parent_id IS NOT NULL）回填 search_tokens

    遍历 search_tokens 为 NULL 的子块，jieba 分词写入 search_tokens。
    单文档分词失败时记录 warning 并跳过（该文档 FTS 不可见），不影响其余；
    分词为空（内容为空/纯标点）的行保持 NULL 并计入 skipped_empty。

    Args:
        dry_run: True 时仅统计待回填数量，不写库
        limit: >0 时最多处理前 N 条（试运行）

    Returns:
        {"total": int, "updated": int, "skipped_empty": int, "failed": int}
    """
    async with async_session_factory() as session:
        q = select(Document.id, Document.content).where(
            Document.parent_id.isnot(None),
            Document.search_tokens.is_(None),
        ).order_by(Document.id)
        if limit:
            q = q.limit(limit)
        rows = (await session.execute(q)).all()

    stats = {"total": len(rows), "updated": 0, "skipped_empty": 0, "failed": 0}
    if dry_run:
        logger.info("[dry-run] 待回填子块: %d", stats["total"])
        return stats

    async with async_session_factory() as session:
        for row in rows:
            doc_id, content = int(row[0]), row[1] or ""
            try:
                tokens = tokenize(content)
            except Exception as e:
                stats["failed"] += 1
                logger.error("[失败] doc_id=%d: %s", doc_id, e)
                continue
            if not tokens:
                stats["skipped_empty"] += 1
                logger.debug("[跳过] doc_id=%d 分词为空", doc_id)
                continue
            await session.execute(
                update(Document).where(Document.id == doc_id)
                .values(search_tokens=tokens)
            )
            stats["updated"] += 1
        await session.commit()

    logger.info("回填完成: total=%d, updated=%d, skipped_empty=%d, failed=%d",
                stats["total"], stats["updated"], stats["skipped_empty"], stats["failed"])
    return stats


async def run(dry_run: bool, limit: int) -> None:
    """执行迁移 + 回填（单事件循环内完成，避免 Windows Proactor 双 loop 问题）

    Args:
        dry_run: True 时仅统计待回填数量，不写库
        limit: >0 时最多处理前 N 条（试运行）
    """
    # 迁移 DDL 幂等，dry-run 也先执行（否则查询 search_tokens 列会报错）
    await apply_migration()
    stats = await backfill(dry_run=dry_run, limit=limit)
    logger.info("统计: total=%d, updated=%d, skipped_empty=%d, failed=%d",
                stats["total"], stats["updated"], stats["skipped_empty"], stats["failed"])


def main():
    """命令行入口"""
    setup_logging()
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    limit = 0
    for i, a in enumerate(args):
        if a == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])

    try:
        asyncio.run(run(dry_run=dry_run, limit=limit))
    except Exception as e:
        logger.error("backfill 失败: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
