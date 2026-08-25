"""
创建 RAG 元数据表 + 填充初始配置 + 统计现有文档分块

表 1: rag_config             全局 RAG 配置（key-value）
表 2: document_chunk_stats   每篇文档的分块统计

用法：
  python create_metadata_tables.py

幂等：重复执行安全（CREATE TABLE IF NOT EXISTS + 配置按 key 更新）
"""
import asyncio
import asyncpg

DSN = "postgresql://postgres:123456@localhost:5432/personal_website"

# 初始配置：与当前代码实际参数保持一致
INITIAL_CONFIG = [
    ("embedding_model", "OllmOne/bge-m3-GGUF", "嵌入模型（ModelScope 云端）"),
    ("embedding_dim", "1024", "向量维度（bge-m3）"),
    ("chunk_size", "300", "子块目标字符数"),
    ("chunk_overlap", "50", "子块重叠字符数"),
    ("min_chars", "50", "父块最小字符数（低于则过滤）"),
    ("reranker_model", "BAAI/bge-reranker-v2-m3", "重排模型（本地分类式 CrossEncoder）"),
    ("rerank_top_k", "5", "重排后保留条数"),
    ("hybrid_search_alpha", "0.3", "混合检索 BM25 权重（向量权重为 1-alpha）"),
    ("graph_name", "knowledge_graph", "Graph RAG 图名（Apache AGE）"),
    ("llm_provider", "fallback", "LLM 供应商（降级链 qwen→zhipu→deepseek）"),
]

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS rag_config (
    id           BIGSERIAL    PRIMARY KEY,
    config_key   VARCHAR(100) NOT NULL UNIQUE,
    config_value TEXT         NOT NULL,
    description  VARCHAR(200) DEFAULT '',
    updated_at   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE rag_config IS 'RAG 全局配置表（key-value）';
COMMENT ON COLUMN rag_config.config_key IS '配置键';
COMMENT ON COLUMN rag_config.config_value IS '配置值';
COMMENT ON COLUMN rag_config.description IS '配置说明';

CREATE TABLE IF NOT EXISTS document_chunk_stats (
    id             BIGSERIAL   PRIMARY KEY,
    document_id    BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    title          VARCHAR(512) NOT NULL DEFAULT '',
    source         VARCHAR(256) NOT NULL DEFAULT '',
    parent_count   INTEGER     NOT NULL DEFAULT 0,
    child_count    INTEGER     NOT NULL DEFAULT 0,
    embedding_dim  INTEGER     NOT NULL DEFAULT 0,
    chunk_size     INTEGER     NOT NULL DEFAULT 0,
    chunk_overlap  INTEGER     NOT NULL DEFAULT 0,
    created_at     TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_id)
);
COMMENT ON TABLE document_chunk_stats IS '文档分块统计表（每篇文档一行）';
COMMENT ON COLUMN document_chunk_stats.parent_count IS '父块数';
COMMENT ON COLUMN document_chunk_stats.child_count IS '子块数';
"""

# 统计: 对每个父块文档，统计其子块数
STATS_SQL = """
INSERT INTO document_chunk_stats
    (document_id, title, source, parent_count, child_count, embedding_dim, chunk_size, chunk_overlap)
SELECT
    p.id,
    p.title,
    p.source,
    1,
    COUNT(c.id),
    1024,
    300,
    50
FROM documents p
LEFT JOIN documents c ON c.parent_id = p.id
WHERE p.parent_id IS NULL
GROUP BY p.id
ON CONFLICT (document_id) DO UPDATE SET
    child_count = EXCLUDED.child_count
"""


async def main():
    conn = await asyncpg.connect(DSN)

    # 1. 建表
    await conn.execute(CREATE_SQL)
    print("✅ 表创建完成: rag_config, document_chunk_stats")

    # 2. 填充初始配置
    for key, value, desc in INITIAL_CONFIG:
        await conn.execute("""
            INSERT INTO rag_config (config_key, config_value, description)
            VALUES ($1, $2, $3)
            ON CONFLICT (config_key) DO UPDATE SET
                config_value = EXCLUDED.config_value,
                description = EXCLUDED.description,
                updated_at = CURRENT_TIMESTAMP
        """, key, value, desc)
    print(f"✅ 配置填充完成: {len(INITIAL_CONFIG)} 条")

    # 3. 统计现有文档分块
    await conn.execute(STATS_SQL)
    cnt = await conn.fetchval("SELECT COUNT(*) FROM document_chunk_stats")
    print(f"✅ 文档分块统计完成: {cnt} 篇父块文档")

    # 4. 验证输出
    print("\n=== rag_config ===")
    rows = await conn.fetch("SELECT config_key, config_value, description FROM rag_config ORDER BY config_key")
    for r in rows:
        print(f"  {r['config_key']:<22} = {r['config_value']:<40} {r['description']}")

    print("\n=== document_chunk_stats (前10条) ===")
    rows = await conn.fetch("""
        SELECT id, document_id, title, source, parent_count, child_count
        FROM document_chunk_stats ORDER BY id LIMIT 10
    """)
    for r in rows:
        print(f"  id={r['id']:<4} doc={r['document_id']:<5} parents={r['parent_count']} children={r['child_count']:<3} {r['title'][:40]}")

    await conn.close()
    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    asyncio.run(main())
