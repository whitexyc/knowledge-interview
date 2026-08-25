"""
module-064 多格式解析/清洗/去重 DB 迁移脚本（幂等，可重复执行）

documents 表补四列（ADR-0014 WP5/WP6）：
  original_path        VARCHAR(512)   —— 上传原始文件落盘路径（WP5 原件留存）
  doc_content_hash     VARCHAR(64)    —— 文档级全文本 SHA256（WP6 L1 内容哈希去重）
  duplicate_cluster_id VARCHAR(64)    —— 语义重复簇 ID（WP6 L2）
  is_canonical         BOOLEAN 默认 TRUE —— 簇内 canonical（false=重复副本检索抑制）
+ 两个索引（doc_content_hash / duplicate_cluster_id）。

ORM 字段（rag/models.py Document）已加，本地开发库 schema 未迁移先决
（module-046/061/062 经验）。幂等：查 information_schema.columns 已存在则跳过；
也可直接复用 src.database.ensure_document_parsing_columns（ADD COLUMN IF NOT EXISTS
自愈），本脚本走查列跳过 + init_db 同款语义双保险。

用法（ai_service 目录下）:
    python scripts/migrate_module064.py
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

# 缺列名清单（module-064 多格式解析/清洗/去重，ORM 见 rag/models.py）
MISSING_COLUMNS = [
    (
        "original_path",
        "ALTER TABLE documents ADD COLUMN original_path VARCHAR(512)",
    ),
    (
        "doc_content_hash",
        "ALTER TABLE documents ADD COLUMN doc_content_hash VARCHAR(64)",
    ),
    (
        "duplicate_cluster_id",
        "ALTER TABLE documents ADD COLUMN duplicate_cluster_id VARCHAR(64)",
    ),
    (
        "is_canonical",
        "ALTER TABLE documents ADD COLUMN is_canonical BOOLEAN NOT NULL DEFAULT TRUE",
    ),
]

# 缺索引清单（幂等 CREATE INDEX IF NOT EXISTS）
MISSING_INDEXES = [
    ("idx_documents_doc_content_hash", "documents", "doc_content_hash"),
    ("idx_documents_duplicate_cluster_id", "documents", "duplicate_cluster_id"),
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

    for idx, table, col in MISSING_INDEXES:
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {idx} ON {table} ({col})"
        )
        logger.info("已补索引: %s (%s)", idx, col)

    # 校验输出
    cols = await conn.fetch(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'documents'
          AND column_name IN ('original_path', 'doc_content_hash',
                              'duplicate_cluster_id', 'is_canonical')
        ORDER BY column_name
        """
    )
    print("documents 校验列:", [r["column_name"] for r in cols])

    await conn.close()
    print("=== ALL DONE ===")


if __name__ == "__main__":
    asyncio.run(main())
