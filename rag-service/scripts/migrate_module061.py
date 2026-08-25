"""
module-061 P0 DB 迁移脚本（幂等，可重复执行）

documents 表补 superseded / updated_at 两列（memory-conflict 记忆纠错，ADR-0007
P0+P1）——ORM 字段（rag/models.py Document）已加，本地开发库 schema 未迁移先决
（module-046 经验）。幂等：查 information_schema.columns 已存在则跳过；
也可直接复用 src.database.ensure_memory_superseded_columns（ADD COLUMN IF NOT
EXISTS 自愈），本脚本走查列跳过 + init_db 同款语义双保险。

用法（ai_service 目录下）:
    python scripts/migrate_module061.py
"""
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ai_service 根

import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 与 .env PW_DATABASE_URL 同库（asyncpg 直连，脚本不经过 SQLAlchemy）
DSN = os.environ.get("PW_DATABASE_URL", "postgresql://postgres:123456@localhost:5432/personal_website") \
    .replace("postgresql+asyncpg://", "postgresql://")

# 缺列名清单（module-061 记忆纠错，ORM 见 rag/models.py）
MISSING_COLUMNS = [
    (
        "superseded",
        "ALTER TABLE documents ADD COLUMN superseded BOOLEAN NOT NULL DEFAULT FALSE",
    ),
    (
        "updated_at",
        "ALTER TABLE documents ADD COLUMN updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP",
    ),
]


async def main() -> None:
    conn = await asyncpg.connect(DSN)

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

    # 校验输出
    cols = await conn.fetch(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'documents'
          AND column_name IN ('superseded', 'updated_at')
        ORDER BY column_name
        """
    )
    print("documents 校验列:", [r["column_name"] for r in cols])

    await conn.close()
    print("=== ALL DONE ===")


if __name__ == "__main__":
    asyncio.run(main())
