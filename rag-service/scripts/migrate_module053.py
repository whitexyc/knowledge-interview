"""
module-053 WP-0 DB 修复迁移脚本（幂等，可重复执行）

修复两处历史环境欠账（本地开发库从未跑过对应迁移/自愈）：
  1. documents 表补 last_mentioned_at / mention_count 两列
     —— module-046 记忆进化的 ORM 字段，本地库缺列导致
        graph_store.search_related 的 select(Document) 全列查询报
        "column documents.last_mentioned_at does not exist"
     —— 幂等：查 information_schema.columns 已存在则跳过
  2. feedback 表建表
     —— module-048 反馈飞轮的独立新表（FEEDBACK_DDL 自愈未跑过）
     —— 直接复用 src.database.ensure_feedback_table（CREATE TABLE
        IF NOT EXISTS + COMMENT，'；' 拆分逐条执行，与 eval_runs 同款幂等模式）

用法（ai_service 目录下）:
    python scripts/migrate_module053.py
"""
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ai_service 根

import asyncpg

from src.database import ensure_feedback_table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 与 .env PW_DATABASE_URL 同库（asyncpg 直连，脚本不经过 SQLAlchemy）
DSN = "postgresql://postgres:123456@localhost:5432/personal_website"

# 缺列名清单（module-046 记忆进化，ORM 见 rag/models.py）
MISSING_COLUMNS = [
    (
        "last_mentioned_at",
        "ALTER TABLE documents ADD COLUMN last_mentioned_at TIMESTAMP WITH TIME ZONE",
    ),
    (
        "mention_count",
        "ALTER TABLE documents ADD COLUMN mention_count INTEGER NOT NULL DEFAULT 0",
    ),
]


async def main() -> None:
    conn = await asyncpg.connect(DSN)

    # ── 1. documents 补两列（幂等：查 information_schema 已存在则跳过） ──
    for col, ddl in MISSING_COLUMNS:
        exists = await conn.fetchval(
            """
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'documents'
              AND column_name = $1
            """,
            col,
        )
        if exists:
            logger.info("列已存在，跳过: documents.%s", col)
            continue
        await conn.execute(ddl)
        logger.info("已补列: documents.%s", col)

    # ── 2. feedback 表建表（复用 module-048 FEEDBACK_DDL 幂等自愈） ──
    await ensure_feedback_table()
    logger.info("feedback 表已就绪（module-048 FEEDBACK_DDL）")

    # ── 3. 验证输出 ──
    cols = await conn.fetch(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'documents'
          AND column_name IN ('last_mentioned_at', 'mention_count')
        ORDER BY column_name
        """
    )
    print("documents 校验列:", [r["column_name"] for r in cols])
    feedback_exists = await conn.fetchval(
        """
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'feedback'
        """
    )
    print("feedback 表存在:", bool(feedback_exists))

    await conn.close()
    print("=== ALL DONE ===")


if __name__ == "__main__":
    asyncio.run(main())
