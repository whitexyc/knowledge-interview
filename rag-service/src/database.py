"""
数据库连接管理
异步 SQLAlchemy + asyncpg + pgvector
"""
import logging
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text
from .config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=5,
    max_overflow=10,
)

async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# feedback 表 DDL（module-048 反馈飞轮）：与 eval_runs 同款模式——
# 独立建表脚本 + 启动 init_db 自愈建表（CREATE TABLE IF NOT EXISTS，幂等）
FEEDBACK_DDL = """
CREATE TABLE IF NOT EXISTS feedback (
    id           BIGSERIAL    PRIMARY KEY,
    message_id   INTEGER      NOT NULL,
    rating       INTEGER      NOT NULL,
    comment      TEXT,
    identity     VARCHAR(256) NOT NULL DEFAULT '',
    created_at   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE feedback IS '用户反馈表（👍👎，层 4 分类器再训练数据源）';
COMMENT ON COLUMN feedback.message_id IS '关联的消息 ID（飞轮回填用）';
COMMENT ON COLUMN feedback.rating IS '评分：1=赞，-1=踩';
COMMENT ON COLUMN feedback.comment IS '补充评论（可选，≤500）';
COMMENT ON COLUMN feedback.identity IS '反馈者身份（user_id 优先，client_ip 兜底）';
"""


async def ensure_feedback_table() -> None:
    """幂等创建 feedback 表（数据库不可用时抛异常，由调用方处理）

    DDL 含多条语句（CREATE TABLE + COMMENT），asyncpg 不允许单条
    prepared statement 执行多条命令，因此按 ';' 拆分逐条执行。
    """
    statements = [s.strip() for s in FEEDBACK_DDL.split(";") if s.strip()]
    async with async_session_factory() as session:
        for stmt in statements:
            await session.execute(text(stmt))
        await session.commit()


# request_logs 表 DDL（module-058 WP-C 可观测性）：与 feedback 表同款模式——
# 独立建表 + 启动 init_db 自愈建表（CREATE TABLE IF NOT EXISTS，幂等）。
# timings/usage 用 JSONB 存阶段耗时与 token 用量（按供应商），避免列爆炸；
# 缓存命中/错误标记单列，供聚合查询（P50/P95 延迟、单问题成本分布）。
REQUEST_LOGS_DDL = """
CREATE TABLE IF NOT EXISTS request_logs (
    id            BIGSERIAL    PRIMARY KEY,
    trace_id      VARCHAR(64)  NOT NULL,
    identity      VARCHAR(256) NOT NULL DEFAULT '',
    endpoint      VARCHAR(128) NOT NULL DEFAULT '',
    intent        VARCHAR(64)  NOT NULL DEFAULT '',
    timings       JSONB        NOT NULL DEFAULT '{}',
    usage         JSONB        NOT NULL DEFAULT '{}',
    cache_hits    INTEGER      NOT NULL DEFAULT 0,
    cache_misses  INTEGER      NOT NULL DEFAULT 0,
    error         BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE request_logs IS '请求观测日志（trace_id/阶段耗时/token用量/缓存命中/错误标记）';
COMMENT ON COLUMN request_logs.trace_id IS '请求追踪 ID（UUID hex，贯穿日志与落库）';
COMMENT ON COLUMN request_logs.identity IS '请求身份（user_id 优先，client_ip 兜底，对齐 048 口径）';
COMMENT ON COLUMN request_logs.timings IS '各阶段耗时（毫秒）：意图/分诊改写/检索FTS·向量·图谱/rerank/反思/生成/幻觉检测';
COMMENT ON COLUMN request_logs.usage IS 'token 用量（按供应商：{provider: {prompt, completion}}）';
COMMENT ON COLUMN request_logs.error IS '请求错误标记（主链路异常时置 true）';
"""


async def ensure_request_logs_table() -> None:
    """幂等创建 request_logs 表（与 feedback 表同款拆分执行模式）"""
    statements = [s.strip() for s in REQUEST_LOGS_DDL.split(";") if s.strip()]
    async with async_session_factory() as session:
        for stmt in statements:
            await session.execute(text(stmt))
        await session.commit()


# tool_call_logs 表 DDL（module-066 / ADR-0017 决策 2，一字不改）：与 feedback/
# request_logs 同款模式——独立建表 + 启动 init_db 自愈建表（CREATE TABLE IF NOT
# EXISTS，幂等）。补 request_logs 缺工具调用明细的核心缺口：Agent 每次实际执行
# 工具落一行（trace_id/工具名/参数/成败/预览/耗时）；args 用 JSONB 存参数，
# result_preview 截断 200 防大文档撑爆列，duration_ms 为单次工具执行耗时。
TOOL_CALL_LOGS_DDL = """
CREATE TABLE IF NOT EXISTS tool_call_logs (
    id             BIGSERIAL    PRIMARY KEY,
    trace_id       VARCHAR(64)  NOT NULL,
    tool_name      VARCHAR(64)  NOT NULL,
    args           JSONB        NOT NULL DEFAULT '{}',
    result_ok      BOOLEAN      NOT NULL DEFAULT TRUE,
    result_preview VARCHAR(200) NOT NULL DEFAULT '',
    duration_ms    INTEGER      NOT NULL DEFAULT 0,
    created_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE tool_call_logs IS '工具调用明细日志（Agent 每次实际执行的工具一行，ADR-0017）';
COMMENT ON COLUMN tool_call_logs.trace_id IS '请求追踪 ID（UUID hex，关联 request_logs）';
COMMENT ON COLUMN tool_call_logs.tool_name IS '工具名（ToolRegistry 注册名）';
COMMENT ON COLUMN tool_call_logs.args IS '工具参数（JSONB，LLM 传入的 args）';
COMMENT ON COLUMN tool_call_logs.result_ok IS '执行成功标记（工具不存在/执行异常才 false，空结果属正常）';
COMMENT ON COLUMN tool_call_logs.result_preview IS '结果预览（截断 200 字符，防大文档撑爆列）';
COMMENT ON COLUMN tool_call_logs.duration_ms IS '单次工具执行耗时（毫秒）';
"""


async def ensure_tool_call_logs_table() -> None:
    """幂等创建 tool_call_logs 表（与 feedback/request_logs 同款拆分执行模式）"""
    statements = [s.strip() for s in TOOL_CALL_LOGS_DDL.split(";") if s.strip()]
    async with async_session_factory() as session:
        for stmt in statements:
            await session.execute(text(stmt))
        await session.commit()


# verify_results 表 DDL（module-060 verify 异步化）：与 feedback/request_logs
# 同款模式——独立建表 + 启动 init_db 自愈建表（CREATE TABLE IF NOT EXISTS，幂等）。
# claims 用 JSONB 存逐句验证结果（claim/verdict/evidence），overall_confidence/
# supported/inferred/unsupported 单列供聚合；verified_in_ms 为 verify_answer
# 任务耗时（口径对齐 module-058 计时，异步化后由轮询接口返回）。done 结果
# 永久保留不清理（飞轮数据源——答案可信度/幻觉调优数据积累）。
VERIFY_RESULTS_DDL = """
CREATE TABLE IF NOT EXISTS verify_results (
    id                  BIGSERIAL   PRIMARY KEY,
    task_id             VARCHAR(64) NOT NULL UNIQUE,
    trace_id            VARCHAR(64) NOT NULL DEFAULT '',
    identity            VARCHAR(256) NOT NULL DEFAULT '',
    endpoint            VARCHAR(128) NOT NULL DEFAULT 'chat_stream',
    query               TEXT        NOT NULL DEFAULT '',
    status              VARCHAR(16) NOT NULL DEFAULT 'pending',
    claims              JSONB,
    overall_confidence  DOUBLE PRECISION,
    supported           INTEGER     NOT NULL DEFAULT 0,
    inferred            INTEGER     NOT NULL DEFAULT 0,
    unsupported         INTEGER     NOT NULL DEFAULT 0,
    error               TEXT,
    verified_in_ms      INTEGER,
    created_at          TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE verify_results IS '证据链验证任务与结果（异步 verify 落库，pending→done/failed）';
COMMENT ON COLUMN verify_results.task_id IS '验证任务 ID（UUID hex，前端轮询 key）';
COMMENT ON COLUMN verify_results.trace_id IS '请求追踪 ID（关联 request_logs）';
COMMENT ON COLUMN verify_results.identity IS '请求身份（user_id 优先，client_ip 兜底，对齐 048 口径）';
COMMENT ON COLUMN verify_results.status IS '任务状态：pending（进行中）/ done（完成）/ failed（失败）';
COMMENT ON COLUMN verify_results.claims IS '验证结果（claims 数组 JSONB：claim/verdict/evidence）';
COMMENT ON COLUMN verify_results.verified_in_ms IS 'verify_answer 任务耗时（毫秒，口径对齐 module-058 计时）';
"""


async def ensure_verify_results_table() -> None:
    """幂等创建 verify_results 表（与 feedback/request_logs 同款拆分执行模式）"""
    statements = [s.strip() for s in VERIFY_RESULTS_DDL.split(";") if s.strip()]
    async with async_session_factory() as session:
        for stmt in statements:
            await session.execute(text(stmt))
        await session.commit()


# documents 表加列 DDL（module-061 P0 记忆纠错）：与 feedback/request_logs 同款
# 模式——init_db 自愈幂等 ALTER（ADD COLUMN IF NOT EXISTS，重复启动不报错）。
# superseded/updated_at 为记忆纠错（ADR-0007 P0+P1）字段：升级留后悔药（长期
# 新条目 superseded=false + updated_at=now）+ 写路径冲突消解（矛盾 → 旧父块
# superseded=true）。默认值兜底存量行（superseded=false / updated_at=当前时间），
# 零迁移 fail-open（存量记忆不因加列受影响）。本地开发库 schema 未迁移先决：
# 手动跑 scripts/migrate_module061.py（module-046 经验）。
MEMORY_SUPERSEDED_DDL = """
ALTER TABLE documents ADD COLUMN IF NOT EXISTS superseded BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP;
COMMENT ON COLUMN documents.superseded IS '记忆是否已被新说法取代（true=SUPERSEDED，不删除可审计，Zep 模式）';
COMMENT ON COLUMN documents.updated_at IS '记忆最近更新（升级/冲突标记/去重追加时刷新）';
"""


async def ensure_memory_superseded_columns() -> None:
    """幂等补 documents 表 superseded/updated_at 两列（与 feedback 同款拆分执行模式）"""
    statements = [s.strip() for s in MEMORY_SUPERSEDED_DDL.split(";") if s.strip()]
    async with async_session_factory() as session:
        for stmt in statements:
            await session.execute(text(stmt))
        await session.commit()


# documents 表加列 DDL（module-062 P2/P3 记忆进化 2）：与 module-061 同款模式——
# init_db 自愈幂等 ALTER（ADD COLUMN IF NOT EXISTS，重复启动不报错）。
# type 记忆类型（preference/fact/event，P2 类型化衰减依据）；last_recalled_at
# 长期层最后召回时间（P3 冷记忆降权依据，久未召回降权不删除）。默认值兜底存量行
#（type='fact' / last_recalled_at=NULL 不降权）零迁移 fail-open。本地开发库
# schema 未迁移先决：手动跑 scripts/migrate_module062.py。
MEMORY_TYPE_COLUMNS_DDL = """
ALTER TABLE documents ADD COLUMN IF NOT EXISTS type VARCHAR(16) NOT NULL DEFAULT 'fact';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS last_recalled_at TIMESTAMP;
COMMENT ON COLUMN documents.type IS '记忆类型：preference（偏好，慢衰减）/ fact（事实，中衰减）/ event（事件，快衰减）——module-062 P2 类型化衰减';
COMMENT ON COLUMN documents.last_recalled_at IS '长期层最后召回时间——module-062 P3 冷记忆降权依据（久未召回降权不删除）';
"""


async def ensure_memory_type_columns() -> None:
    """幂等补 documents 表 type/last_recalled_at 两列（与 superseded 同款拆分执行模式）"""
    statements = [s.strip() for s in MEMORY_TYPE_COLUMNS_DDL.split(";") if s.strip()]
    async with async_session_factory() as session:
        for stmt in statements:
            await session.execute(text(stmt))
        await session.commit()


# documents 表加列 DDL（module-064 多格式解析/清洗/去重）：与 module-061/062 同款
# 模式——init_db 自愈幂等 ALTER（ADD COLUMN IF NOT EXISTS + CREATE INDEX IF NOT
# EXISTS，重复启动不报错）。四列对应 ADR-0014 WP5/WP6：
#   original_path —— WP5 原件留存（上传原始文件落盘路径，重灌依赖）；
#   doc_content_hash —— WP6 L1 文档级全文本 SHA256（完全相同直接丢弃）；
#   duplicate_cluster_id / is_canonical —— WP6 L2 语义重复簇 + canonical 选择
#   （不删，标簇 + 检索抑制只出 canonical）。
# 默认值兜底存量行（NULL / TRUE）零迁移 fail-open。本地开发库 schema 未迁移
# 先决：手动跑 scripts/migrate_module064.py（module-046/061/062 经验）。
DOCUMENT_PARSING_COLUMNS_DDL = """
ALTER TABLE documents ADD COLUMN IF NOT EXISTS original_path VARCHAR(512);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_content_hash VARCHAR(64);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS duplicate_cluster_id VARCHAR(64);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_canonical BOOLEAN NOT NULL DEFAULT TRUE;
CREATE INDEX IF NOT EXISTS idx_documents_doc_content_hash ON documents (doc_content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_duplicate_cluster_id ON documents (duplicate_cluster_id);
COMMENT ON COLUMN documents.original_path IS '上传原始文件落盘路径（module-064 WP5 原件留存，重灌依赖）';
COMMENT ON COLUMN documents.doc_content_hash IS '文档级全文本 SHA256（module-064 WP6 L1 内容哈希去重）';
COMMENT ON COLUMN documents.duplicate_cluster_id IS '语义重复簇 ID（module-064 WP6 L2，检索抑制只出 canonical）';
COMMENT ON COLUMN documents.is_canonical IS '簇内 canonical：true=检索可见，false=重复副本检索抑制（module-064）';
"""


async def ensure_document_parsing_columns() -> None:
    """幂等补 documents 表 original_path/doc_content_hash/duplicate_cluster_id/is_canonical
    四列（与 superseded 同款拆分执行模式）"""
    statements = [s.strip() for s in DOCUMENT_PARSING_COLUMNS_DDL.split(";") if s.strip()]
    async with async_session_factory() as session:
        for stmt in statements:
            await session.execute(text(stmt))
        await session.commit()


async def init_db():
    """初始化数据库：启用 pgvector 扩展 + 自愈建表/加列（feedback / request_logs / verify_results / superseded / type+last_recalled_at / tool_call_logs）"""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("pgvector extension 已就绪")
    await ensure_feedback_table()
    logger.info("feedback 表已就绪（module-048）")
    await ensure_request_logs_table()
    logger.info("request_logs 表已就绪（module-058）")
    await ensure_verify_results_table()
    logger.info("verify_results 表已就绪（module-060）")
    await ensure_memory_superseded_columns()
    logger.info("documents 表 superseded/updated_at 列已就绪（module-061）")
    await ensure_memory_type_columns()
    logger.info("documents 表 type/last_recalled_at 列已就绪（module-062）")
    await ensure_document_parsing_columns()
    logger.info("documents 表 original_path/doc_content_hash/duplicate_cluster_id/is_canonical 列已就绪（module-064）")
    await ensure_tool_call_logs_table()
    logger.info("tool_call_logs 表已就绪（module-066）")


async def get_db() -> AsyncSession:
    """FastAPI 依赖注入：获取数据库会话"""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
