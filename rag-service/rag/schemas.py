"""
RAG 知识库请求/响应模型
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=2000)
    top_k: int = 5


class SearchResponse(BaseModel):
    results: list[dict] = []
    message: str = ""


class ChatRequest(BaseModel):
    query: str = Field(..., max_length=2000)
    history: list[dict] = Field(default_factory=list)

    @field_validator("history", mode="before")
    @classmethod
    def truncate_history(cls, v):
        """AC 1.4: 超条数截断 — 静默保留最近 20 条消息，不返回 422"""
        if isinstance(v, list) and len(v) > 20:
            return v[-20:]
        return v


class ChatSteps(BaseModel):
    """RAG 中间步骤数据，供前端管线面板展示

    每个步骤包含：
    - timing_ms: 该步骤耗时（毫秒）
    - intent: 意图识别结果（label, confidence）
    - retrieval: 检索结果统计（count, top_score, documents_preview, relevance）
    - rerank: 重排前后数量
    - reflection: 反思结果（sufficient, query_rewritten, rewritten_query）
    """
    intent: dict = {}         # {"label": str, "confidence": float, "timing_ms": int}
    retrieval: dict = {}      # {"count": int, "top_score": float, "timing_ms": int,
                              #  "documents_preview": [{"title": str, "snippet": str, "score": float}],
                              #  "relevance": {"qualified": int, "total": int, "min_score": float}}
    rerank: dict = {}         # {"before": int, "after": int, "timing_ms": int}
    reflection: dict = {}     # {"sufficient": bool, "query_rewritten": bool, "timing_ms": int}


class ChatResponse(BaseModel):
    answer: str = ""
    sources: list[dict] = []
    message: str = ""
    steps: Optional[ChatSteps] = None
    verified_claims: Optional[dict] = None  # module-039: 证据链验证结果


class MemorySaveRequest(BaseModel):
    """保存长期记忆请求体（module-023）"""
    content: str = Field(..., max_length=2000)
    ip: str = "unknown"


class MemoryRecallRequest(BaseModel):
    """检索长期记忆请求体（module-023）"""
    query: str = Field(..., max_length=2000)
    ip: str = "unknown"


class FeedbackRequest(BaseModel):
    """用户反馈请求体（module-048 反馈飞轮）

    rating 仅允许 1（赞）/ -1（踩）；comment 可选 ≤500 字符。
    非法 rating / 超长 comment → 422（前端按钮触发，防落库污染飞轮数据）。
    """
    message_id: int = Field(..., description="关联的消息 ID")
    rating: int = Field(..., description="评分：1=赞，-1=踩")
    comment: Optional[str] = Field(default=None, max_length=500,
                                   description="补充评论（可选，≤500）")

    @field_validator("rating")
    @classmethod
    def rating_must_be_like_dislike(cls, v: int) -> int:
        """rating ∈ {1, -1}：0 或 2 等非法值一律 422"""
        if v not in (1, -1):
            raise ValueError("rating 必须为 1（赞）或 -1（踩）")
        return v


class WeakTopicIngestRequest(BaseModel):
    """待学笔记录入请求体（module-080 反向闭环）

    topic 为弱题主题关键词（必填），context 为薄弱点描述（可选），
    identity 为身份标识（可选，默认从 request.state 取）。
    """
    topic: str = Field(..., min_length=1, max_length=200, description="弱题主题关键词")
    context: Optional[str] = Field(default=None, max_length=500, description="薄弱点描述（可选）")
    identity: Optional[str] = Field(default=None, max_length=128, description="身份标识（可选）")

