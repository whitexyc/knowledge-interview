-- =====================================================================
-- RAG 元数据表（rag_config + document_chunk_stats）
-- 生成时间: 2026-08-01
--
-- 表 1: rag_config             全局 RAG 配置（key-value）
--   记录嵌入模型、向量维度、分块参数、重排模型等，便于运维/演示查询
-- 表 2: document_chunk_stats   每篇文档的分块统计
--   记录每篇父块文档的子块数、维度、分块参数快照
--
-- 用法（任选其一）：
--   1. psql -U postgres -d personal_website -f rag_metadata_tables.sql
--   2. 或在 psql 中 \i rag_metadata_tables.sql
--
-- 幂等性：CREATE TABLE IF NOT EXISTS + ON CONFLICT 更新，可重复执行
-- =====================================================================

-- ---------------------------------------------------------------------
-- 表 1: rag_config
-- ---------------------------------------------------------------------
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

-- 初始配置（与当前代码实际参数保持一致）
INSERT INTO rag_config (config_key, config_value, description) VALUES
    ('embedding_model',   'OllmOne/bge-m3-GGUF',     '嵌入模型（ModelScope 云端）'),
    ('embedding_dim',     '1024',                    '向量维度（bge-m3）'),
    ('chunk_size',        '300',                     '子块目标字符数'),
    ('chunk_overlap',     '50',                      '子块重叠字符数'),
    ('min_chars',         '50',                      '父块最小字符数（低于则过滤）'),
    ('reranker_model',    'BAAI/bge-reranker-v2-m3', '重排模型（本地分类式 CrossEncoder）'),
    ('rerank_top_k',      '5',                       '重排后保留条数'),
    ('hybrid_search_alpha','0.3',                    '混合检索 BM25 权重（向量权重为 1-alpha）'),
    ('graph_name',        'knowledge_graph',         'Graph RAG 图名（Apache AGE）'),
    ('llm_provider',      'fallback',                'LLM 供应商（降级链 qwen→zhipu→deepseek）')
ON CONFLICT (config_key) DO UPDATE SET
    config_value = EXCLUDED.config_value,
    description  = EXCLUDED.description,
    updated_at   = CURRENT_TIMESTAMP;

-- ---------------------------------------------------------------------
-- 表 2: document_chunk_stats
-- ---------------------------------------------------------------------
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
COMMENT ON COLUMN document_chunk_stats.embedding_dim IS '向量维度';
COMMENT ON COLUMN document_chunk_stats.chunk_size IS '分块大小快照';
COMMENT ON COLUMN document_chunk_stats.chunk_overlap IS '分块重叠快照';

-- 统计现有文档分块（每个父块文档一行，统计其子块数）
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
    child_count = EXCLUDED.child_count;

-- =====================================================================
-- 完成
-- =====================================================================
