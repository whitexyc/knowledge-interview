"""
RAG 知识库文档 ORM 模型
"""
import logging

from sqlalchemy import Boolean, Column, Integer, String, Text, DateTime, Float, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase
from pgvector.sqlalchemy import Vector


logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """统一 ORM 基类"""


class Document(Base):
    """文档模型 — 存储知识库文档及其向量嵌入"""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="文档 ID")
    title = Column(String(512), nullable=False, default="", comment="文档标题")
    content = Column(Text, nullable=False, comment="文档内容")
    source = Column(String(256), nullable=False, default="", comment="来源标识")
    page_num = Column(Integer, nullable=True, comment="页码")
    # 使用 meta 避免与 SQLAlchemy 保留属性 metadata 冲突
    meta = Column("metadata", JSONB, nullable=False, default=dict, comment="元数据")
    content_hash = Column(String(64), nullable=True, index=True, comment="内容 SHA256 哈希（去重用）")
    embedding = Column(Vector(1024), nullable=True, comment="向量嵌入")
    parent_id = Column(Integer, ForeignKey("documents.id"),
                       nullable=True, index=True,
                       comment="父块 ID（NULL=父块/根块，非NULL=子块指向其父块）")
    search_tokens = Column(Text, nullable=True,
                           comment="jieba分词后的空格连接文本（中文FTS检索用，仅子块写入）")
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), comment="创建时间"
    )
    # module-046 记忆进化：仅短期层使用（last_mentioned_at 提及刷新 / mention_count
    # 召回加权 + 升级阈值）。存量行字段为 NULL/0 时按 created_at 衰减、count=0 加权
    #（零迁移 fail-open，不写迁移脚本）
    last_mentioned_at = Column(
        DateTime(timezone=True), nullable=True, comment="最近提及时间（module-046 仅短期层使用）"
    )
    mention_count = Column(
        Integer, nullable=False, default=0, comment="提及次数（module-046 仅短期层使用）"
    )
    # module-061 记忆纠错（ADR-0007 P0+P1）：
    #   superseded —— 记忆是否已被新说法取代（true=SUPERSEDED，不删除可审计，
    #                  Zep 模式）。写路径冲突消解（_merge_duplicate NLI 判矛盾）
    #                  与 P0 升级留后悔药均可能标 true；召回侧过滤 superseded=true
    #                  （_expand_to_parents / _evolve_recall），旧说法不参与召回。
    #   updated_at —— 记忆最近更新（升级/冲突标记/去重追加时刷新），可审计时间线。
    # 存量行默认 FALSE / 当前时间（init_db 幂等 ALTER 兜底，零迁移 fail-open）。
    superseded = Column(
        Boolean, nullable=False, default=False, comment="是否已被新说法取代（SUPERSEDED，不删除可审计）"
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(),
        comment="记忆最近更新时间（升级/冲突标记/去重追加时刷新）"
    )
    # module-062 记忆进化 2（ADR-0007 P2/P3）：
    #   type —— 记忆类型（preference 偏好/ fact 事实/ event 事件），P2 类型化衰减
    #           依据（_evolve_recall 按 type 选半衰期：preference 30 天慢 / event 1 天
    #           快 / 其余按 memory_short_half_life=3）。类型来源由 memory_type_mode
    #           （clf/llm/none）决定；存量行默认 'fact'（DB DEFAULT，零迁移兜底）。
    #   last_recalled_at —— 长期层最后召回时间，P3 冷记忆降权依据（recall 命中后
    #                        fire-and-forget 刷新为 now；久未召回 → 分数 ×0.3-1.0 降权
    #                        不删除）。存量行 NULL → 按 created_at 计算且不降权（零回归）。
    type = Column(
        String(16), nullable=False, default="fact",
        comment="记忆类型：preference（偏好，慢衰减）/ fact（事实，中衰减）/ event（事件，快衰减）"
    )
    last_recalled_at = Column(
        DateTime(timezone=True), nullable=True,
        comment="长期层最后召回时间（P3 冷记忆降权依据：久未召回降权不删除）"
    )
    # module-064 多格式解析/清洗/去重（ADR-0014）：
    #   original_path —— 上传原始文件落盘路径（WP5 原件留存，重灌依赖）。仅文档
    #                    根父块（parent_id IS NULL）与子块写；存量行 NULL 兼容。
    #   doc_content_hash —— 文档级全文本 SHA256（WP6 L1 内容哈希去重：完全相同
    #                    直接丢弃，复用 content_hash 列的思想但粒度是整篇文档，
    #                    存全部行便于按文档定位）。存量行 NULL（跨格式精确去重
    #                    对存量靠 title 匹配，如实声明）。
    #   duplicate_cluster_id —— 语义重复簇 ID（WP6 L2：embedding 余弦≥0.95 不删，
    #                    标簇 + canonical 选择；检索抑制只出 canonical）。存量行
    #                    NULL（未参与语义去重）。
    #   is_canonical —— 簇内是否 canonical（true=检索可见，false=重复副本检索
    #                    抑制）。存量行默认 TRUE（零回归）。
    original_path = Column(
        String(512), nullable=True,
        comment="上传原始文件落盘路径（module-064 WP5 原件留存，重灌依赖）"
    )
    doc_content_hash = Column(
        String(64), nullable=True, index=True,
        comment="文档级全文本 SHA256（module-064 WP6 L1 内容哈希去重）"
    )
    duplicate_cluster_id = Column(
        String(64), nullable=True, index=True,
        comment="语义重复簇 ID（module-064 WP6 L2，检索抑制只出 canonical）"
    )
    is_canonical = Column(
        Boolean, nullable=False, default=True,
        comment="簇内 canonical：true=检索可见，false=重复副本检索抑制（module-064）"
    )

    # module-075 知识抓取流水线：review_status 审查状态标记（approved/rejected）
    # 抓取内容经 reflector + factcheck_judge 审查；rejected 仍入库（fail-open 不丢
    # 数据，仅标记供复核/后续人工处置）。存量行默认 'approved'（DB DEFAULT 兜底）。
    review_status = Column(
        String(16), nullable=False, default="approved",
        comment="审查状态：approved（通过）/ rejected（不通过，仍入库可复核）——module-075"
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} title={self.title!r} source={self.source!r}>"

    def to_dict(self) -> dict:
        """转为可序列化字典"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "page_num": self.page_num,
            "metadata": self.meta,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Feedback(Base):
    """用户反馈模型 — 层 4 分类器（intent/充分性）再训练数据源（module-048）

    👍👎 反馈飞轮：前端对每条 AI 回复点赞/点踩（可选评论），落 feedback 表
    累积标注数据。feedback 与 documents 表无关（独立新表），message_id 先
    落前端消息 ID，飞轮回填脚本再按需关联 query/answer（本模块不建外键）。
    """

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="反馈 ID")
    message_id = Column(Integer, nullable=False, index=True,
                        comment="关联的消息 ID（飞轮回填用）")
    rating = Column(Integer, nullable=False, comment="评分：1=赞，-1=踩")
    comment = Column(Text, nullable=True, comment="补充评论（可选，≤500）")
    identity = Column(String(256), nullable=False, default="",
                      comment="反馈者身份（user_id 优先，client_ip 兜底）")
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), comment="创建时间"
    )

    def __repr__(self) -> str:
        return (f"<Feedback id={self.id} message_id={self.message_id} "
                f"rating={self.rating}>")


class RequestLog(Base):
    """请求观测日志 — 线上可观测性（module-058 WP-C）

    trace_id 贯穿日志与落库；timings/usage 为 JSONB（阶段耗时按毫秒、
    token 用量按供应商），支撑"单问题成本分布 / P50-P95 延迟"聚合查询。
    identity 对齐 048 口径（user_id 优先，client_ip 兜底）；建表走
    init_db 自愈幂等 DDL（src/database.py REQUEST_LOGS_DDL），
    落库失败 fail-open 不阻塞主链路。
    """

    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="日志 ID")
    trace_id = Column(String(64), nullable=False, index=True, comment="请求追踪 ID")
    identity = Column(String(256), nullable=False, default="",
                      comment="请求身份（user_id 优先，client_ip 兜底）")
    endpoint = Column(String(128), nullable=False, default="",
                      comment="端点（chat/chat_stream/agent/agent-lg）")
    intent = Column(String(64), nullable=False, default="",
                      comment="意图（knowledge/casual_chat/realtime/agent）")
    timings = Column(JSONB, nullable=False, default=dict, comment="各阶段耗时（毫秒）")
    usage = Column(JSONB, nullable=False, default=dict, comment="token 用量（按供应商）")
    cache_hits = Column(Integer, nullable=False, default=0, comment="检索缓存命中次数")
    cache_misses = Column(Integer, nullable=False, default=0, comment="检索缓存未命中次数")
    error = Column(Boolean, nullable=False, default=False, comment="请求错误标记")
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), comment="创建时间"
    )

    def __repr__(self) -> str:
        return f"<RequestLog id={self.id} trace_id={self.trace_id!r}>"


class VerifyResult(Base):
    """证据链验证任务与结果 — verify 异步化（module-060）

    verify（幻觉检测）后台异步执行后的任务状态与结果，前端凭 task_id 轮询
    DB 为准（不读内存任务池）：pending（进行中）/ done（完成，含逐句 claims）/
    failed（异常）。done 结果永久保留不清理（飞轮数据源——答案可信度/幻觉
    调优数据积累）。建表走 init_db 自愈幂等 DDL（src/database.py
    VERIFY_RESULTS_DDL），字段与 DDL 对齐。
    """

    __tablename__ = "verify_results"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="记录 ID")
    task_id = Column(String(64), nullable=False, unique=True, index=True,
                     comment="验证任务 ID（UUID hex，前端轮询 key）")
    trace_id = Column(String(64), nullable=False, default="",
                      comment="请求追踪 ID（关联 request_logs）")
    identity = Column(String(256), nullable=False, default="",
                      comment="请求身份（user_id 优先，client_ip 兜底）")
    endpoint = Column(String(128), nullable=False, default="chat_stream",
                      comment="端点（当前仅 chat_stream 提交）")
    query = Column(Text, nullable=False, default="",
                   comment="用户问题（飞轮数据源可关联）")
    status = Column(String(16), nullable=False, default="pending",
                    comment="任务状态：pending/done/failed")
    claims = Column(JSONB, nullable=True, comment="验证结果（claims 数组 JSONB）")
    overall_confidence = Column(Float, nullable=True,
                                comment="整体置信度（0.0-1.0）")
    supported = Column(Integer, nullable=False, default=0, comment="supported 计数")
    inferred = Column(Integer, nullable=False, default=0, comment="inferred 计数")
    unsupported = Column(Integer, nullable=False, default=0, comment="unsupported 计数")
    error = Column(Text, nullable=True, comment="失败原因（status=failed 时）")
    verified_in_ms = Column(Integer, nullable=True, comment="verify 任务耗时（毫秒）")
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), comment="更新时间"
    )

    def __repr__(self) -> str:
        return f"<VerifyResult id={self.id} task_id={self.task_id!r} status={self.status!r}>"
