"""
AI 推理服务入口
FastAPI + pgvector + LangChain 多供应商 LLM
"""
import logging
import json
import time
import asyncio
import hmac
from collections import defaultdict
from typing import Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, Body, File, Form, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from src.config import settings
from src.database import init_db, async_session_factory
from src.ratelimit import check_rate_limit, get_client_ip
from src.cache import cache
from src.identity import parse_jwt, resolve_identity
from src import observability
from src.verify_tasks import submit_verify_task, get_verify_task
from mcp_server import mcp as mcp_server, mcp_http_lifespan
from rag.engine import rag_engine, resolve_tool_history
from rag.schemas import (
    SearchRequest, SearchResponse, ChatRequest, ChatResponse,
    MemorySaveRequest, MemoryRecallRequest, FeedbackRequest,
)
from rag.models import Document, Feedback
from rag.memory.memory import memory_service
from rag.retrieval.document_parser import SUPPORTED_EXTENSIONS, DocumentParseError
from rag.retrieval.document_ingest import ingest_document, IngestError
from llm.client import LLMFactory


# ─── IP 会话缓存（module-034：内存态降级为兜底缓存） ───
# 结构: {client_ip: [{"role": str, "content": str, "timestamp": float}, ...]}
# 每个 IP 最多保存 MAX_MESSAGES_PER_IP 条
# module-034 后会话持久化为主（session_memory 写库，供刷新/换设备恢复），
# 本内存 dict 保留为会话内即时兜底缓存（/ai/chat/sessions 等端点即时读取）。
IP_SESSION_MESSAGES: dict[str, list[dict]] = defaultdict(list)
MAX_MESSAGES_PER_IP = 50
MAX_ANSWER_LEN = 10000  # module-042: 答案最大长度，超出截断并附加提示

# module-055 minor 修复：持有 HHEM 后台预热任务引用（lifespan 内赋值），
# 防服务在预热期间关闭时任务被 GC 触发 "Task was destroyed but it is pending"
# 告警（fail-soft，仅持引用不 await/不取消，语义与无预热行为一致）
_HHEM_WARMUP_TASK: Optional[asyncio.Task] = None

logging.basicConfig(
    level=logging.INFO,
    # module-058（WP-C）Review 修复：日志格式含 %(trace_id)s（TraceIdFilter
    # 保证该字段恒存在，无请求上下文时为空串）——请求期间日志行可肉眼关联
    format="%(asctime)s [%(name)s] %(levelname)s [%(trace_id)s]: %(message)s",
)
logger = logging.getLogger("ai_service")

# module-058（WP-C）Review 修复（MAJOR-1）：trace_id 贯穿日志——过滤器从
# 请求上下文取 trace_id 注入 record.trace_id extra（幂等挂根 logger + handler）
observability.install_trace_id_filter()


class ChainUpdateRequest(BaseModel):
    """LLM 降级链调整请求体（module-029 动态调序）

    Attributes:
        chain: 供应商顺序列表（如 ["zhipu", "deepseek", "qwen"]）
    """
    chain: list[str]


async def load_fallback_chain_from_redis() -> None:
    """启动时从 Redis 加载持久化降级链（module-029）

    优先级：Redis 中用户调整过的顺序 > 配置默认（.env PW_FALLBACK_CHAIN）。
    Redis 不可用 / 无链 / 存储链不合法时静默降级为配置默认（不改任何状态），
    不阻塞服务启动。

    调用时机：lifespan 中 LLM 客户端预热之前，确保预热即用持久化链。
    """
    try:
        raw = await cache.get_str("llm:fallback_chain")
    except Exception as e:
        logger.warning("读取 Redis 降级链失败，使用配置默认: %s", e)
        return
    if not raw:
        return
    try:
        chain = LLMFactory.validate_chain(
            [p.strip() for p in raw.split(",") if p.strip()]
        )
    except ValueError as e:
        logger.warning("Redis 降级链不合法，使用配置默认: %s", e)
        return
    LLMFactory.set_fallback_chain(chain)
    logger.info("从 Redis 加载降级链: %s", " → ".join(chain))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("AI 服务启动中...")

    # module-032: JWT 共享密钥必须配置（与 Java 端一致，走 .env，不进仓库）。
    # 缺失时明确报错启动失败，不静默运行无认证状态（plan §3.4）。
    if not settings.jwt_secret:
        raise RuntimeError(
            "JWT_SECRET 未配置：请在 .env 设置 PW_JWT_SECRET（与 Java application.yml 同值）"
        )

    # module-067: MCP HTTP 模式 fail-closed —— PW_MCP_TOKEN 未配置拒绝启动
    #（宁可不用不能裸奔；stdio 本地模式零认证是设计，不受影响）。
    # 放 lifespan 不放 import 期（存量测试全量 import main，import 期 raise 全炸；
    # 测试用 ASGITransport 不触发 lifespan，零影响）。
    if not settings.mcp_token:
        raise RuntimeError(
            "PW_MCP_TOKEN 未设置：MCP HTTP 模式 fail-closed 拒绝启动（请在 .env 配置）"
        )

    await init_db()

    # 预热 embedding 模型 + LLM 客户端，避免首次请求卡顿
    from rag.retrieval.embeddings import embedding_service
    logger.info("预热 embedding 模型中...")
    await embedding_service.embed_text("warmup")
    logger.info("embedding 模型已就绪")

    from llm.client import LLMFactory
    # 先加载 Redis 中持久化的降级链（module-029），无则用配置默认，
    # 确保后续 LLM 预热/调用都使用最新顺序
    await load_fallback_chain_from_redis()
    logger.info("预热 LLM 客户端...")
    try:
        LLMFactory.get_client()  # 触发默认 provider（fallback 降级链）
        logger.info("LLM 客户端已预热 (default/fallback)")
    except Exception as e:
        logger.warning("LLM 客户端预热失败（可接受）: %s", e)

    # 预热 Qwen + Zhipu（ModelScope 降级链的前两环），避免首次调用冷启动
    for label, provider in [("Qwen", "qwen"), ("ZhipuAI GLM", "zhipu")]:
        try:
            LLMFactory.get_client(provider)
            logger.info("%s 客户端已预热", label)
        except Exception as e:
            logger.warning("%s 预热失败（可接受）: %s", label, e)

    # module-055: 后台预热 HHEM 裁判模型（fail-soft，不阻塞启动）。
    # 依据（changelog 实测）：冷加载独立进程 ≈9s、服务进程（CPU 争用）≈17-19s，
    # 首请求 verify 的 20s 预算内"加载+推理"超时 → verified_claims=0（E2E 复现）；
    # 预热后 predict 纯推理（实测 0.11-0.5s/对）。后台任务通常先于首个验证请求
    # 完成；失败仅告警（首次验证请求退回冷加载路径，与无预热行为一致）。
    import asyncio as _asyncio

    async def _warmup_hhem() -> None:
        try:
            from rag.retrieval.factcheck_judge import hhem_judge
            scores = await hhem_judge.predict(["warmup"], ["warmup"])
            if scores is not None:
                logger.info("HHEM 裁判模型已预热")
        except Exception as e:
            logger.warning("HHEM 预热失败（可接受，首个验证请求将含冷加载）: %s", e)

    global _HHEM_WARMUP_TASK
    _HHEM_WARMUP_TASK = _asyncio.create_task(_warmup_hhem())

    # P3 性能优化：预热 reranker（int8 量化后首次加载约 20-30s），首个请求不再
    # 冷加载 20s（TTFT 最大头之一）。与 HHEM 后台 fail-soft 不同，reranker 在
    # 首个 chat/search 请求的同步关键路径上，故**阻塞启动**等待就绪；失败
    # fail-open（首个请求退回冷加载路径，与无预热行为一致）。CPU 密集加载挪到
    # 线程池，不阻塞事件循环。
    try:
        from rag.retrieval.reranker import reranker as _reranker
        logger.info("预热 reranker 模型中...")
        await _asyncio.to_thread(_reranker._lazy_load)
        logger.info("reranker 模型已预热")
    except Exception as e:
        logger.warning("reranker 预热失败（可接受，首个请求将含冷加载）: %s", e)

    # module-067: MCP Streamable HTTP 会话任务组——Starlette Mount 不转发
    # lifespan scope 给挂载子应用，手动进入（等价 FastMCP 独立 uvicorn 运行
    # 的 lifespan；不初始化则每个 /ai/mcp 请求抛 "Task group is not
    # initialized"，见 mcp_server.mcp_http_lifespan）
    mcp_http_ctx = mcp_http_lifespan()
    await mcp_http_ctx.__aenter__()
    try:
        yield
    finally:
        await mcp_http_ctx.__aexit__(None, None, None)
    logger.info("AI 服务关闭")


app = FastAPI(
    title=settings.app_name,
    description="Agentic RAG 知识库推理服务",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── IP 限流中间件（除 health 外所有请求） ───
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """基于 IP 的请求频率限制

    在请求进入路由之前检查限流，超出阈值返回 429。
    同时提取客户端 IP 注入 request.state.client_ip；
    并解析 JWT 注入 request.state.user_id（无/非法/过期 token 为 ""，module-032）。

    module-058（WP-C）：请求入口生成 trace_id 挂 request.state + 观测上下文
    （contextvar），引擎/重排/LLM 客户端在请求任务内读取；request_logs_enabled
    =false 时跳过（零埋点零落库）。
    """
    # 可观测性：trace_id 初始化（关闭时零埋点）
    if settings.request_logs_enabled:
        trace_id = observability.make_trace_id()
        observability.init_request(trace_id)
        request.state.trace_id = trace_id

    # 健康检查不限制
    if request.url.path == "/ai/health":
        return await call_next(request)

    # 提取客户端 IP
    forwarded = request.headers.get("X-Forwarded-For")
    client_ip = get_client_ip(forwarded, request.client.host if request.client else None)
    request.state.client_ip = client_ip

    # module-032: JWT 身份解析（Authorization: Bearer <token> → user_id）
    # 成功注入 request.state.user_id；无/非法/过期 token → ""（降级 client_ip，零回归）
    request.state.user_id = parse_jwt(request.headers.get("Authorization"))

    # 限流检查
    allowed, retry_after = check_rate_limit(client_ip)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"message": f"请求过于频繁，请 {retry_after} 秒后重试", "retry_after": retry_after},
            headers={"Retry-After": str(retry_after)},
        )

    return await call_next(request)


# ─── MCP Server 挂载（module-067 / ADR-0018） ───
def _mcp_auth_middleware(mcp_app):
    """/ai/mcp 认证包装（ASGI）：Authorization: Bearer <PW_MCP_TOKEN>

    fail-closed：token 为空恒 401（双保险——lifespan 已拒绝空 token 启动）；
    每次请求实时读 settings.mcp_token（不缓存，改 token 立即生效）；
    比较用 hmac.compare_digest（常量时间，防时序侧信道）。
    """
    async def auth_wrapper(scope, receive, send):
        if scope["type"] != "http":
            await mcp_app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        token = settings.mcp_token
        expected = f"Bearer {token}".encode()
        auth = headers.get(b"authorization", b"")
        if not token or not hmac.compare_digest(auth, expected):
            await JSONResponse(
                status_code=401,
                content={"message": "未授权：MCP token 缺失或错误（fail-closed）"},
            )(scope, receive, send)
            return
        await mcp_app(scope, receive, send)
    return auth_wrapper


app.mount("/ai/mcp", _mcp_auth_middleware(mcp_server.streamable_http_app()))


# ─── 保存消息到 IP 会话缓存 ───
def save_messages_to_session(client_ip: str, user_msg: str, assistant_msg: str, assistant_sources: list):
    """把一次问答追加到 IP 会话记录中，超出上限时丢弃最旧消息"""
    records = IP_SESSION_MESSAGES[client_ip]
    now = time.time()
    records.append({"role": "user", "content": user_msg, "timestamp": now})
    records.append({"role": "assistant", "content": assistant_msg, "sources": assistant_sources, "timestamp": now})
    # 裁剪超出部分
    if len(records) > MAX_MESSAGES_PER_IP:
        IP_SESSION_MESSAGES[client_ip] = records[-MAX_MESSAGES_PER_IP:]


# ─── 请求观测落库（module-058 WP-C 可观测性） ───
def persist_request_log(fastapi_req: Request, endpoint: str, intent: str = "",
                        error: bool = False) -> None:
    """请求结束异步落库 request_logs（fire-and-forget，fail-open 不阻塞响应）

    观测数据来自请求上下文（中间件初始化的 trace_id + 引擎/重排/LLM 客户端
    累积的阶段耗时/token 用量/缓存命中）；identity 对齐 048 口径（user_id
    优先，client_ip 兜底）。开关关闭时零埋点零落库。

    Args:
        fastapi_req: 当前请求（取 identity / trace_id）
        endpoint: 端点标识（chat/chat_stream/agent/agent-lg）
        intent: 意图（knowledge/casual_chat/realtime/agent）
        error: 请求错误标记（主链路异常置 true）
    """
    if not settings.request_logs_enabled:
        return
    stats = observability.get_request_stats()
    record = {
        "trace_id": stats.get("trace_id") or getattr(fastapi_req.state, "trace_id", ""),
        "identity": resolve_identity(fastapi_req),
        "endpoint": endpoint,
        "intent": intent,
        "timings": stats.get("timings", {}),
        "usage": stats.get("usage", {}),
        "cache_hits": stats.get("cache_hits", 0),
        "cache_misses": stats.get("cache_misses", 0),
        "error": error,
    }
    asyncio.create_task(observability.save_request_log(record))


def schedule_stream_persist(intent: str, query: str, answer: str,
                            identity: str, history: list) -> None:
    """chat_stream 生成结束后异步触发长期记忆自动写入（module-033，fire-and-forget）

    仅 intent=knowledge 且 answer 非空时触发（闲聊/实时不提取，省成本避免存垃圾）。
    asyncio.create_task 只调度不 await，写入后台进行不阻塞 SSE 响应；后台任务
    异常全部在 rag_engine._persist_memory 内降级捕获，绝不抛回响应（零回归）。

    Args:
        intent: 意图识别结果（knowledge / casual_chat / realtime）
        query: 用户问题
        answer: 生成的完整答案文本（非空才提取）
        identity: 请求身份（user_id 优先，否则 client_ip）
        history: 最近对话历史
    """
    if intent == "knowledge" and answer and answer.strip():
        asyncio.create_task(rag_engine._persist_memory(query, answer, identity, history))


@app.get("/ai/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "ai-service"}


@app.get("/ai/config")
async def get_config():
    """返回当前配置（不含密钥）"""
    return {
        "provider": settings.llm_provider,
        "claude_model": settings.claude_model,
        "deepseek_model": settings.deepseek_model,
        "debug": settings.debug,
    }


# ─── LLM 降级链动态调序 API（module-029） ───


@app.get("/ai/llm/chain")
async def get_llm_chain():
    """获取当前 LLM 降级链顺序

    返回运行时链（Redis 持久化的用户配置优先），否则配置默认
    （.env PW_FALLBACK_CHAIN）。

    返回格式: {"code": 0, "data": {"chain": ["qwen", "zhipu", "deepseek"]}}
    """
    return {"code": 0, "data": {"chain": LLMFactory.get_fallback_chain()}}


@app.put("/ai/llm/chain")
async def put_llm_chain(request: ChainUpdateRequest):
    """调整 LLM 降级链顺序（校验 → 存 Redis → 清缓存即时生效）

    流程：
      1. 校验链合法（非空、全为支持供应商、无重复）
      2. 写入 Redis（key: llm:fallback_chain，无 TTL 跨重启持久）
      3. 更新运行时链 + clear_cache → 下次 get_client("fallback") 按新链重建

    Redis 写入失败时返回 code 2 且不修改运行时链（调序不生效但服务正常）。

    Args:
        request: {chain: ["zhipu", "deepseek", "qwen"]}

    Returns:
        code=0: {"code": 0, "data": {"chain": [...]}}
        code=1: 校验失败（非法供应商/重复/空链）
        code=2: Redis 持久化失败
    """
    try:
        validated = LLMFactory.validate_chain(request.chain)
    except ValueError as e:
        return {"code": 1, "message": str(e)}

    saved = await cache.set_str("llm:fallback_chain", ",".join(validated))
    if not saved:
        return {"code": 2, "message": "降级链保存失败（Redis 不可用），顺序未修改"}

    LLMFactory.set_fallback_chain(validated)
    LLMFactory.clear_cache()
    logger.info("降级链已更新并持久化: %s", " → ".join(validated))
    return {"code": 0, "data": {"chain": validated}}


# ─── IP 会话管理 API ───


@app.get("/ai/chat/sessions")
async def get_chat_sessions():
    """获取当前活跃的 IP 会话列表

    只返回元信息（IP、消息数、最后活跃时间），不返回消息内容。
    """
    sessions = []
    for ip, messages in IP_SESSION_MESSAGES.items():
        if messages:
            sessions.append({
                "id": ip,
                "message_count": len(messages),
                "last_active": messages[-1].get("timestamp", 0),
            })
    # 按最后活跃时间降序排列
    sessions.sort(key=lambda s: s["last_active"], reverse=True)
    return {"data": sessions}


@app.get("/ai/chat/sessions/{ip}/messages")
async def get_session_messages(ip: str):
    """获取指定 IP 的会话消息列表

    返回的消息不带 timestamp（前端不需要），带 sources。
    """
    messages = IP_SESSION_MESSAGES.get(ip, [])
    # 去掉 timestamp 字段，前端不需要
    clean = []
    for msg in messages:
        entry = {"role": msg["role"], "content": msg["content"]}
        if msg.get("sources"):
            entry["sources"] = msg["sources"]
        clean.append(entry)
    return {"data": clean, "count": len(clean)}


@app.post("/ai/rag/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """RAG 知识库检索"""
    return await rag_engine.search(request)


@app.post("/ai/rag/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, fastapi_req: Request):
    """RAG 知识库问答

    将请求身份（user_id 优先，否则 client_ip）传给 rag_engine.chat，
    用于按身份隔离检索长期记忆（module-032；匿名降级 client_ip，零回归）。
    """
    client_ip = getattr(fastapi_req.state, "client_ip", "unknown")
    identity = resolve_identity(fastapi_req)
    result = await rag_engine.chat(request, identity=identity)
    # module-042: 答案截断保护（不影响 sources）
    if len(result.answer) > MAX_ANSWER_LEN:
        result.answer = result.answer[:MAX_ANSWER_LEN] + "\n\n[答案过长，已截断]"
    # 保存消息到 IP 会话缓存（仅知识库路径保存；内存态，module-034 降级为兜底缓存）
    if result.message not in ("casual_chat", "realtime_not_implemented") and result.answer:
        save_messages_to_session(client_ip, request.query, result.answer, result.sources)
        # 注：会话持久化（_schedule_session_persist）已由 engine.chat 内部在 no-docs/docs
        # 两个 return 点自包含调度（module-034），此处不再重复调用——此前双重调度导致
        # 每轮会话消息确定性重复落库 4 行/轮（Reviewer 阻塞 #1，content_hash 无唯一约束）。
    # module-058：请求观测落库（intent 取 ChatSteps/消息语义；error 取 internal_error）
    intent = ""
    if result.message == "casual_chat":
        intent = "casual_chat"
    elif result.message == "realtime_not_implemented":
        intent = "realtime"
    elif getattr(result, "steps", None) and result.steps.intent:
        intent = result.steps.intent.get("label", "")
    persist_request_log(fastapi_req, "chat", intent=intent,
                        error=result.message == "internal_error")
    return result


@app.post("/ai/rag/chat/stream")
async def chat_stream(request: ChatRequest, fastapi_req: Request):
    """RAG 知识库问答（流式输出）

    先完成前置步骤（意图→检索→Rerank→反思），每步结果通过 SSE step 事件推送，
    LLM 生成部分通过 token 事件逐字输出。

    长期记忆（module-025）：流式路径在 Step 5 生成前调用
    rag_engine._recall_memory 召回跨会话记忆（5s 超时 + 失败降级返回空串），
    无记忆时 memory 为空串，行为与之前完全一致（零回归）。

    SSE 事件：
      event: step   data: {"step":str, "data":dict, "timing_ms":int}
      event: token  data: "文本片段"
      event: done   data: {"sources":[...]}
      event: error  data: {"message":str}
    """
    # 身份由限流中间件注入 request.state（module-032：user_id 优先，否则 client_ip）
    identity = resolve_identity(fastapi_req)

    async def event_stream():
        import time
        _t = time.monotonic
        intent = ""
        failed = False
        try:
            # ====== Step 1: 意图识别 ======
            t0 = _t()
            from agent.router import router_agent
            # module-063（WP-A，纪律 §八.2）：流式检索链也接 history——漏一个
            # 就是"chat 正常、stream 回归"（空 history 零回归）
            # module-072（WP-B）：流式路径 classify 补传 tool_history（持久化
            # 工具轨迹，查询不可得/失败 → None fail-open）
            intent_result = await router_agent.classify(
                request.query, history=request.history,
                tool_history=await resolve_tool_history(identity))
            intent = intent_result.get("intent", "knowledge")
            observability.timing("intent", _t() - t0)
            intent_labels = {"knowledge": "知识库", "casual_chat": "闲聊", "realtime": "实时数据"}
            step_data = json.dumps({
                "step": "intent",
                "data": {"label": intent_labels.get(intent, intent), "confidence": intent_result.get("confidence", 0)},
                "timing_ms": int((_t() - t0) * 1000),
            })
            yield f"event: step\ndata: {step_data}\n\n"

            if intent == "casual_chat":
                from llm.client import LLMFactory
                client = LLMFactory.get_client()
                async for token in client.generate_stream(
                    f"你是知识库问答系统的 AI 助手。\n用户: {request.query}"
                ):
                    yield f"event: token\ndata: {json.dumps(token)}\n\n"
                yield "event: done\ndata: {}\n\n"
                return

            # ====== Step 2: 检索 ======
            t0 = _t()
            # module-072（WP-A）：流式路径透传对话历史给上下文改写
            #（contextual_rewrite_enabled 关闭时 history 参数零影响）
            docs = await rag_engine._retrieve(request.query, top_k=20,
                                              history=request.history)
            retrieval_count = len(docs)
            observability.timing("retrieve", _t() - t0)
            # module-045 WP3: L3 标记接入流式路径（对齐非流式 engine.chat）——
            # 检索 top-1 绝对余弦 < 0.3 → suspected_misclassify（先度量后干预，
            # 只写入 step 事件可观测）。_retrieve 已做父块映射，abs_cosine 经
            # WP2b 透传（子块最大值），流式路径不再恒 0.0 恒标记
            suspected_misclassify, top1_abs = rag_engine._check_suspected_misclassify(docs)
            # 预览文档（前5条标题+摘要）
            previews = []
            # module-035 (P2)：移除失真阈值——hybrid_score 是 min-max 相对分
            #（跨查询不可比），旧 MIN_SCORE=0.3 套相对分当绝对阈值语义失真。
            # relevant 仅供 UI 展示统计（不影响回答正确性），检索步骤本身即
            # 相关性门控，故直接统计检索召回数，不做虚假的绝对质量判断。
            relevant_count = retrieval_count
            for d in docs:
                score = d.get("hybrid_score", 0)
                if len(previews) < 5:
                    previews.append({
                        "title": d.get("title", ""),
                        "snippet": d.get("content", "")[:80],
                        "score": round(score, 3),
                    })
            step_data = json.dumps({
                "step": "retrieval",
                "data": {"count": retrieval_count, "relevant": relevant_count,
                         "top_abs_cosine": round(top1_abs, 4) if docs else None,
                         "suspected_misclassify": suspected_misclassify,
                         "previews": previews},
                "timing_ms": int((_t() - t0) * 1000),
            })
            yield f"event: step\ndata: {step_data}\n\n"

            if not docs:
                from llm.client import LLMFactory
                client = LLMFactory.get_client()
                answer_parts = []
                async for token in client.generate_stream(
                    f"用户问：{request.query}\n\n知识库暂无相关信息。"
                ):
                    answer_parts.append(token)
                    yield f"event: token\ndata: {json.dumps(token)}\n\n"
                # module-033：knowledge 路径生成结束后异步触发长期记忆自动写入
                schedule_stream_persist(intent, request.query, "".join(answer_parts), identity, request.history)
                # module-034：会话持久化为主（异步写库，不阻塞 SSE 响应）
                rag_engine._schedule_session_persist(identity, request.query, "".join(answer_parts))
                yield "event: done\ndata: {}\n\n"
                return

            # ====== Step 3: Rerank ======
            t0 = _t()
            rerank_before = len(docs)
            docs = await rag_engine._rerank(request.query, docs)
            observability.timing("rerank", _t() - t0)
            step_data = json.dumps({
                "step": "rerank",
                "data": {"before": rerank_before, "after": len(docs)},
                "timing_ms": int((_t() - t0) * 1000),
            })
            yield f"event: step\ndata: {step_data}\n\n"

            # ====== Step 4: 反思 ======
            t0 = _t()
            from agent.reflector import reflector
            check = await reflector.check_sufficiency(request.query, docs)
            observability.timing("reflection", _t() - t0)
            reflection_data = {
                "sufficient": check.get("sufficient", True),
                "reason": check.get("reason", ""),
            }
            if not check.get("sufficient", True) and check.get("rewritten_query"):
                reflection_data["rewritten_query"] = check["rewritten_query"]
            step_data = json.dumps({
                "step": "reflection", "data": reflection_data,
                "timing_ms": int((_t() - t0) * 1000),
            })
            yield f"event: step\ndata: {step_data}\n\n"

            # ====== Step 5: 流式生成 ======
            # module-025: 流式路径接入记忆（复用 engine._recall_memory，
            # 5s 超时 + 失败降级返回空串；无记忆时 memory 为空串，零回归）
            # module-032: 记忆按身份隔离（user_id 优先，否则 client_ip）
            # module-034: 会话恢复优先持久化（刷新/换设备不丢）；无则用当前请求
            memory = await rag_engine._recall_memory(request.query, identity)
            history = await rag_engine._resolve_session_history(identity, request.history)
            answer_parts = []
            gen_t0 = _t()
            total_len = 0
            async for token in reflector.generate_answer_stream(request.query, docs, history=history, memory=memory):
                answer_parts.append(token)
                total_len += len(token)
                yield f"event: token\ndata: {json.dumps(token)}\n\n"
                # module-042: 答案长度保护 — 超出上限停止流式输出并追加截断提示
                if total_len >= MAX_ANSWER_LEN:
                    truncation_note = "\n\n[答案过长，已截断]"
                    answer_parts.append(truncation_note)
                    yield f"event: token\ndata: {json.dumps(truncation_note)}\n\n"
                    break

            # ====== Step 6: 引用溯源 ======
            sources = []
            for i, doc in enumerate(docs[:5]):
                sources.append({
                    "id": doc.get("id"),
                    "title": doc.get("title", ""),
                    "content": doc.get("content", "")[:300],
                    "source": doc.get("source", ""),
                    "ref_index": i + 1,
                })
            # module-033：knowledge 路径流式生成结束后异步触发长期记忆自动写入
            #（fire-and-forget；casual_chat 已提前返回、realtime 由 intent 检查跳过）
            schedule_stream_persist(intent, request.query, "".join(answer_parts), identity, request.history)
            # module-034：会话持久化为主（异步写库，不阻塞 SSE 响应）
            rag_engine._schedule_session_persist(identity, request.query, "".join(answer_parts))

            # ====== Step 7: 证据链验证（module-039；module-060 异步后置） ======
            observability.timing("generate", _t() - gen_t0)
            full_answer = "".join(answer_parts)
            # module-042: 剥离截断标记后验证，避免标记文本误导置信度评估
            clean_answer = full_answer.replace("\n\n[答案过长，已截断]", "")
            vf_t0 = _t()
            if settings.verify_async_enabled:
                # module-060：异步 verify——答案先交付（done 带 verify_task_id、
                # verified=False、不再发 verified 事件），验证后台跑、前端轮询
                # GET /ai/rag/chat/verify/{task_id} 补结果，结果落 verify_results
                # 表持久化。提交失败（DB 写失败）→ done 无 task_id，前端
                # fail-open 不显示面板（与现状空 claims 不显示一致）。
                verify_task_id = await submit_verify_task(
                    clean_answer, docs, identity=identity,
                    query=request.query,
                    trace_id=getattr(fastapi_req.state, "trace_id", ""),
                )
                observability.timing("verify_submit", _t() - vf_t0)
                if verify_task_id:
                    yield f"event: done\ndata: {json.dumps({'sources': sources, 'verified': False, 'verify_task_id': verify_task_id})}\n\n"
                else:
                    yield f"event: done\ndata: {json.dumps({'sources': sources, 'verified': False})}\n\n"
            else:
                # module-060 开关 false：现状同步路径（verified→done 顺序逐字一致，逃生口）
                verified = await reflector.verify_answer(clean_answer, docs)
                observability.timing("verify", _t() - vf_t0)
                if verified.get("claims"):
                    yield f"event: verified\ndata: {json.dumps({'claims': verified['claims'], 'overall_confidence': verified['overall_confidence'], 'total_claims': verified['total_claims'], 'supported': verified['supported'], 'inferred': verified['inferred'], 'unsupported': verified['unsupported']}, ensure_ascii=False)}\n\n"
                    yield f"event: done\ndata: {json.dumps({'sources': sources, 'verified': True, 'overall_confidence': verified['overall_confidence']})}\n\n"
                else:
                    yield f"event: done\ndata: {json.dumps({'sources': sources, 'verified': False})}\n\n"

        except Exception as e:
            failed = True
            logger.error("流式问答失败: %s", e, exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': '服务暂时不可用'})}\n\n"
        finally:
            # module-058：请求观测落库（流式结束/断开均触发，fail-open）
            persist_request_log(fastapi_req, "chat_stream", intent=intent,
                                error=failed)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/ai/rag/chat/verify/{task_id}")
async def get_verify_result(task_id: str):
    """轮询 verify 后台任务结果（module-060 verify 异步化，**DB 为准**）

    前端对 chat_stream done 事件返回的 verify_task_id 每 ~2s 轮询本端点，
    答案先交付、验证后到；结果落 verify_results 表持久化（done 不因重启丢失）。

    Returns:
        200 pending → {"status": "pending"}
        200 done   → {"status": "done", "claims", "overall_confidence",
                      "total_claims", "supported", "inferred", "unsupported",
                      "verified_in_ms"}
        200 failed → {"status": "failed", "error"}
        404        → {"detail": "task not found"}（重启丢未完成任务/过期 →
                     前端停止轮询 fail-open，与现状空 claims 不显示一致）
    """
    try:
        result = await get_verify_task(task_id)
    except Exception as e:
        logger.warning("verify 结果查询失败（按 404 处理）: %s", e)
        return JSONResponse(status_code=404, content={"detail": "task not found"})
    if result is None:
        return JSONResponse(status_code=404, content={"detail": "task not found"})
    if result["status"] == "pending":
        return {"status": "pending"}
    if result["status"] == "done":
        claims = result.get("claims") or []
        return {
            "status": "done",
            "claims": claims,
            "overall_confidence": result.get("overall_confidence"),
            "total_claims": len(claims),
            "supported": result.get("supported", 0),
            "inferred": result.get("inferred", 0),
            "unsupported": result.get("unsupported", 0),
            "verified_in_ms": result.get("verified_in_ms"),
        }
    return {"status": "failed", "error": result.get("error", "verify failed")}


@app.post("/ai/rag/chat/agent")
async def chat_agent(request: ChatRequest, fastapi_req: Request):
    """Agent 工具化问答（ReAct 循环，SSE，module-028）

    把固定流水线升级为 Agentic ReAct 循环：LLM 自主决定调用哪些工具、以什么
    顺序，直到信息足够直接回答，或达到工具总调用次数预算（settings.max_agent_tools）。
    与现有 /ai/rag/chat、/ai/rag/chat/stream 并存（A/B 对比）。

    SSE 事件：
      event: tool_call    data: {"name", "args", "tool_count"}
      event: tool_result  data: {"name", "args", "result", "tool_count"}
      event: token        data: "推理/回答文本片段"
      event: done         data: {"answer", "sources", "tool_count", "budget"}
      event: error        data: {"message"}
    """
    identity = resolve_identity(fastapi_req)

    async def event_stream():
        from agent.react import ReactContext, _build_messages, react_loop
        failed = False
        try:
            # module-036：会话恢复优先持久化（刷新/换设备不丢）；无持久化会话
            # 则回退当前请求 history（零回归），与 chat_stream Step 5 一致
            effective_history = await rag_engine._resolve_session_history(identity, request.history)
            ctx = ReactContext(request.query, identity, effective_history)
            budget = settings.max_agent_tools
            answer = ""
            tool_count = 0
            async for evt in react_loop(ctx, _build_messages(ctx), budget,
                                        max_answer_len=MAX_ANSWER_LEN):
                t = evt["type"]
                if t == "tool_call":
                    yield f"event: tool_call\ndata: {json.dumps({'name': evt['name'], 'args': evt['args'], 'tool_count': evt['tool_count']}, ensure_ascii=False)}\n\n"
                elif t == "tool_result":
                    yield f"event: tool_result\ndata: {json.dumps({'name': evt['name'], 'args': evt['args'], 'result': evt['result'][:500], 'tool_count': evt['tool_count']}, ensure_ascii=False)}\n\n"
                elif t == "token":
                    if evt["content"]:
                        yield f"event: token\ndata: {json.dumps(evt['content'], ensure_ascii=False)}\n\n"
                elif t == "done":
                    answer = evt.get("answer", "")
                    tool_count = evt.get("tool_count", 0)

            # 引用溯源：基于循环累积的已检索文档
            sources = []
            for i, doc in enumerate(ctx.docs[:5]):
                sources.append({
                    "id": doc.get("id"),
                    "title": doc.get("title", ""),
                    "content": doc.get("content", "")[:300],
                    "source": doc.get("source", ""),
                    "ref_index": i + 1,
                })
            # module-036：Agent 对话完成后异步持久化会话轮次（fire-and-forget，
            # 不阻塞 SSE；内部 guard 空 answer 不写，与 chat_stream 一致）
            rag_engine._schedule_session_persist(identity, request.query, answer)
            yield f"event: done\ndata: {json.dumps({'answer': answer, 'sources': sources, 'tool_count': tool_count, 'budget': budget}, ensure_ascii=False)}\n\n"
        except Exception as e:
            failed = True
            logger.error("Agent 问答失败: %s", e, exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': '服务暂时不可用'})}\n\n"
        finally:
            # module-058：请求观测落库（agent 端点无独立意图分类，intent="agent"）
            persist_request_log(fastapi_req, "agent", intent="agent", error=failed)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/ai/rag/chat/agent-lg")
async def chat_agent_langgraph(request: ChatRequest, fastapi_req: Request):
    """LangGraph 实验端点（SSE，module-030）

    与 /ai/rag/chat/agent 并存：用 LangGraph StateGraph 编排 ReAct 循环
    （见 agent/langgraph_react.py），行为与手写版对齐（预算/工具/上下文），
    不动现有 react.py（零回归）。实验端点，非生产主路径。

    SSE 事件（与 agent 一致）：
      event: tool_call    data: {"name", "args", "tool_count"}
      event: tool_result  data: {"name", "args", "result", "tool_count"}
      event: token        data: "推理/回答文本片段"
      event: done         data: {"answer", "sources", "tool_count", "budget"}
      event: error        data: {"message"}
    """
    identity = resolve_identity(fastapi_req)

    async def event_stream():
        from agent.langgraph_react import (
            ReactContext, _build_messages, langgraph_react_loop,
        )
        failed = False
        try:
            # module-036：会话恢复优先持久化（刷新/换设备不丢）；无持久化会话
            # 则回退当前请求 history（零回归），与 chat_stream Step 5 一致
            effective_history = await rag_engine._resolve_session_history(identity, request.history)
            ctx = ReactContext(request.query, identity, effective_history)
            budget = settings.max_agent_tools
            answer = ""
            tool_count = 0
            async for evt in langgraph_react_loop(ctx, _build_messages(ctx), budget,
                                                  max_answer_len=MAX_ANSWER_LEN):
                t = evt["type"]
                if t == "tool_call":
                    yield f"event: tool_call\ndata: {json.dumps({'name': evt['name'], 'args': evt['args'], 'tool_count': evt['tool_count']}, ensure_ascii=False)}\n\n"
                elif t == "tool_result":
                    yield f"event: tool_result\ndata: {json.dumps({'name': evt['name'], 'args': evt['args'], 'result': evt['result'][:500], 'tool_count': evt['tool_count']}, ensure_ascii=False)}\n\n"
                elif t == "token":
                    if evt["content"]:
                        yield f"event: token\ndata: {json.dumps(evt['content'], ensure_ascii=False)}\n\n"
                elif t == "done":
                    answer = evt.get("answer", "")
                    tool_count = evt.get("tool_count", 0)

            # 引用溯源：基于循环累积的已检索文档（与 /ai/rag/chat/agent 一致）
            sources = []
            for i, doc in enumerate(ctx.docs[:5]):
                sources.append({
                    "id": doc.get("id"),
                    "title": doc.get("title", ""),
                    "content": doc.get("content", "")[:300],
                    "source": doc.get("source", ""),
                    "ref_index": i + 1,
                })
            # module-036：Agent 对话完成后异步持久化会话轮次（fire-and-forget，
            # 不阻塞 SSE；内部 guard 空 answer 不写，与 chat_stream 一致）
            rag_engine._schedule_session_persist(identity, request.query, answer)
            yield f"event: done\ndata: {json.dumps({'answer': answer, 'sources': sources, 'tool_count': tool_count, 'budget': budget}, ensure_ascii=False)}\n\n"
        except Exception as e:
            failed = True
            logger.error("LangGraph Agent 问答失败: %s", e, exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': '服务暂时不可用'})}\n\n"
        finally:
            # module-058：请求观测落库（agent-lg 端点无独立意图分类，intent="agent"）
            persist_request_log(fastapi_req, "agent-lg", intent="agent", error=failed)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ─── 长期记忆 API（module-023；复用 documents 表，source='memory:<identity>:' 区分，module-032 身份化） ───


@app.post("/ai/memory/save")
async def memory_save(request: MemorySaveRequest, fastapi_req: Request):
    """保存长期记忆（按身份隔离写入 documents，source='memory:<identity>:'）

    identity = user_id（JWT.sub）优先，否则 client_ip（匿名降级，零回归）；
    client_ip 取不到时兼容旧调用方 body ip（module-023）。
    分块 → 本地 bge-m3 向量化 → 写 documents（父块 + 子块）。
    content 为空返回错误；embedding 不可用返回错误码（不崩）。
    """
    try:
        identity = resolve_identity(fastapi_req)
        if identity == "unknown" and request.ip:
            identity = request.ip
        result = await memory_service.save(request.content, identity)
        return {"code": 0, "data": result}
    except ValueError as e:
        return {"code": 1, "message": str(e)}
    except Exception as e:
        logger.error("记忆保存失败: %s", e, exc_info=True)
        return {"code": 2, "message": "记忆保存失败"}


@app.post("/ai/memory/recall")
async def memory_recall(request: MemoryRecallRequest, fastapi_req: Request):
    """检索与 query 相关的长期记忆（按身份隔离，source 过滤）

    identity = user_id（JWT.sub）优先，否则 client_ip（匿名降级，零回归）。
    """
    try:
        identity = resolve_identity(fastapi_req)
        if identity == "unknown" and request.ip:
            identity = request.ip
        memories = await memory_service.recall(request.query, identity)
        return {"code": 0, "data": {"memories": memories}}
    except Exception as e:
        logger.error("记忆检索失败: %s", e, exc_info=True)
        return {"code": 1, "data": {"memories": []}, "message": "记忆检索失败"}


# ─── 用户反馈 API（module-048 反馈飞轮）───


@app.post("/ai/feedback")
async def submit_feedback(request: FeedbackRequest, fastapi_req: Request):
    """提交用户反馈（👍/👎，module-048 反馈飞轮）

    feedback 表是层 4 分类器（intent/充分性）再训练的数据源：前端每条
    AI 回复可点赞/点踩 + 可选评论，落库累积标注数据。

    identity 从 request.state 取（user_id 优先 client_ip 兜底，对齐现有
    中间件注入与 /ai/rag/chat 口径）。Pydantic 已校验 rating ∈ {1,-1}、
    comment ≤500（非法值 422 拦截，防落库污染）；落库失败返回 500，
    前端降级 Toast 提示，不阻塞聊天（降级验收 §6.1）。
    """
    try:
        identity = resolve_identity(fastapi_req)
        async with async_session_factory() as session:
            session.add(Feedback(
                message_id=request.message_id,
                rating=request.rating,
                comment=request.comment,
                identity=identity,
            ))
            await session.commit()
        logger.info("反馈落库: message_id=%d, rating=%d, identity=%s",
                    request.message_id, request.rating, identity)
        return {"status": "ok"}
    except Exception as e:
        logger.error("反馈落库失败: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"message": "反馈保存失败"})


@app.post("/ai/rag/documents")
async def add_document(
    title: str = Body(...),
    content: str = Body(...),
    source: str = Body(default=""),
):
    """添加文档到知识库（向量化后自动入库）"""
    result = await rag_engine.add_document(title, content, source)
    return {"code": 0, "data": result}


@app.post("/ai/rag/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    source: str = Form(default=""),
):
    """多格式文档上传：解析 → 图片 → 清洗 → 归一化 → 去重 → 分块 → 嵌入 → 入库（module-064）

    支持 .md/.txt/.pdf/.docx/.xlsx/.pptx/.epub/.csv（前端 accept 同源，见
    document_parser.SUPPORTED_EXTENSIONS）。四层 ingestion：
      解析层 document_parser（格式识别读字节魔数，AnyDoc 主引擎 + PyMuPDF/
      轻量回退）→ 图片三层（默认关，WP4）→ 五步清洗 + 无损归一化（WP2/WP3）
      → 三级去重（WP6）→ 原件留存（WP5）→ add_document 分块嵌入落库。
    错误变体映射中文提示（Unsupported/Malformed/Encrypted → DocumentParseError/
    IngestError 消息直接透出）。
    """
    if not file.filename:
        return {"code": 1, "message": "未获取到上传文件名"}

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if not ext or f".{ext}" not in SUPPORTED_EXTENSIONS:
        return {"code": 1, "message":
                f"不支持的文件格式（.{ext or '未知'}），请上传 {'/'.join(SUPPORTED_EXTENSIONS)}"}

    # 读取文件内容
    content_bytes = await file.read()
    if not content_bytes:
        return {"code": 2, "message": "上传文件为空"}

    # 确定标题
    if not title:
        stem = file.filename.rsplit(".", 1)[0]
        title = stem.replace("_", " ").replace("-", " ").strip()

    # 确定来源
    if not source:
        source = f"{ext}_upload:{file.filename}"

    # 统一 ingestion 管线（解析→清洗→归一化→去重→原件留存→入库）
    try:
        result = await ingest_document(content_bytes, file.filename, title, source)
    except (DocumentParseError, IngestError) as e:
        logger.warning("文档上传失败: %s", e)
        return {"code": 3, "message": str(e)}
    except Exception as e:
        logger.error("文档处理失败: %s", e, exc_info=True)
        return {"code": 3, "message": f"文档处理失败: {e}"}

    return {"code": 0, "data": result}


@app.get("/ai/documents")
async def list_documents(page: int = 1, page_size: int = 20):
    """查看知识库文档列表（分页，按原始标题聚类去重）"""
    from sqlalchemy import func, or_, select

    async with async_session_factory() as session:
        # 按原始标题分组取最旧 id 作为代表；
        # 排除记忆文档（source='memory:%'，module-023 复用 documents 表），
        # 避免记忆行污染知识库管理面板（review #7）
        subq = (
            select(
                Document.title,
                func.min(Document.id).label("min_id"),
                func.count(Document.id).label("chunk_count"),
            )
            .where(or_(Document.source.is_(None), Document.source.not_like("memory:%")))
            .group_by(Document.title)
            .subquery()
        )
        q = (
            select(Document, subq.c.chunk_count)
            .join(subq, Document.id == subq.c.min_id)
            .order_by(Document.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        total = (await session.execute(
            select(func.count()).select_from(subq)
        )).scalar() or 0
        rows = await session.execute(q)
        docs = []
        for doc, chunk_count in rows:
            docs.append({
                "id": doc.id,
                "title": doc.title,
                "source": doc.source or "",
                "content_preview": doc.content[:120] if doc.content else "",
                "chunk_count": chunk_count,
                "created_at": doc.created_at.isoformat() if doc.created_at else "",
            })

    return {"code": 0, "data": {"documents": docs, "total": total, "page": page, "page_size": page_size}}


@app.delete("/ai/documents/{doc_id}")
async def delete_document(doc_id: int):
    """删除文档及其所有相关分块"""
    from sqlalchemy import select as sel

    async with async_session_factory() as session:
        doc = await session.get(Document, doc_id)
        if not doc:
            return {"code": 1, "message": "文档不存在"}

        title = doc.title
        stmt = sel(Document).where(
            (Document.title == title) | (Document.title.like(f"{title} > %"))
        )
        rows = await session.execute(stmt)
        to_delete = rows.scalars().all()
        for d in to_delete:
            await session.delete(d)
        await session.commit()

    # 检索缓存失效：删除文档后结果可能变化，全量清空
    # 缓存是优化层，失效失败降级（delete_by_prefix 内部 catch，返回 False）
    await cache.delete_by_prefix("rag:retrieve:")

    logger.info("删除文档: id=%d, title=%s, chunks=%d", doc_id, title, len(to_delete))
    return {"code": 0, "message": f"已删除 {len(to_delete)} 条记录"}


if __name__ == "__main__":
    import uvicorn
    # 端口 8001：项目服务统一 +1（前端 3001 / Java 8081 / AI 8001），与 vite 代理 /ai→8001 对齐
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
