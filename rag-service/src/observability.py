"""
请求可观测性 — trace_id + 阶段计时 + token 用量 + 缓存命中 + request_logs 落库
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

module-058（WP-C）：线上运行时追踪（此前只有离线评测 eval_runs，无 trace）。

设计要点：
  1. 全部状态挂在 contextvar 上（每请求上下文，非全局状态）——中间件在
     请求入口初始化，引擎/重排/LLM 客户端在请求任务内读取，多会话并发隔离。
     日志侧由 TraceIdFilter（install_trace_id_filter 挂根 logger 及 handler）
     注入 record.trace_id，请求期间日志行可跨模块关联。
  2. 阶段计时（time.perf_counter → 毫秒落 timings dict）：意图路由 /
     分诊改写 / 检索（FTS·向量·图谱各自，retriever 内采集）/ rerank /
     反思 / 生成 / 幻觉检测。
  3. token 用量（usage dict 按供应商累积 prompt/completion）：各供应商
     LLM 客户端在响应返回处采集（无 usage 静默跳过，不中断主链路）。
  4. 缓存命中计数（cache_hits / cache_misses）：engine._retrieve 缓存
     检查处记录。
  5. request_logs 落库：请求结束（含流式结束/断开）后台写入，fail-open
     不阻塞主链路；建表走 init_db 自愈幂等 DDL（对齐 module-048 feedback
     表模式）。开关 PW_REQUEST_LOGS 关闭时零埋点零落库。
  6. 不引入新依赖（复用现有日志 + SQLAlchemy，无重型 tracing 框架）。
"""
import contextvars
import logging
import time
import uuid

from src.config import settings

logger = logging.getLogger(__name__)

# 每请求观测上下文（contextvar，非全局状态；中间件 init_request 初始化，
# 缺失时 helper 惰性创建——直接调用引擎的测试/脚本路径也安全）
_obs_var: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "request_obs", default={}
)


def _obs() -> dict:
    """取当前请求的观测字典（惰性初始化默认结构）"""
    d = _obs_var.get()
    if not d:
        d = {
            "trace_id": "",
            "timings": {},     # {stage: 毫秒}
            "usage": {},       # {provider: {"prompt": int, "completion": int}}
            "cache_hits": 0,   # _retrieve_cache_key 处命中计数
            "cache_misses": 0, # _retrieve_cache_key 处未命中计数
        }
        _obs_var.set(d)
    return d


def init_request(trace_id: str) -> None:
    """请求入口初始化观测上下文（中间件调用；关闭时不调用=零埋点）"""
    d = _obs()
    d["trace_id"] = trace_id
    d["timings"] = {}
    d["usage"] = {}
    d["cache_hits"] = 0
    d["cache_misses"] = 0


def get_trace_id() -> str:
    """取当前请求 trace_id（无则空串；TraceIdFilter 日志 extra 用）

    只读不惰性初始化（避免无请求上下文的日志触发 contextvar 写入）。
    """
    d = _obs_var.get()
    if not d:
        return ""
    return d.get("trace_id", "")


class TraceIdFilter(logging.Filter):
    """日志过滤器：从请求上下文取 trace_id 注入 record.trace_id extra

    module-058（WP-C）Review 修复（MAJOR-1）：trace_id 贯穿日志——请求期间
    所有服务日志行都带 trace_id，可用同一 trace_id 关联一次请求的日志行；
    无请求上下文（或开关关闭未初始化）时为空串，不影响其他日志消费方。

    注意挂载位置：Python logging 中祖先 logger 的 filter 不作用于子 logger
    传播上来的 record（callHandlers 只经 handler.filter），故 install 同时
    挂到根 logger 的 handler 上，才能覆盖模块级 logger（getLogger(__name__)）
    输出的日志行。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id() or ""
        return True


def install_trace_id_filter() -> TraceIdFilter:
    """挂 trace_id 过滤器到根 logger 及其 handler（幂等，重复调用不重复挂）

    由 main.py 在 logging.basicConfig 之后调用（根 handler 已创建），
    服务日志格式含 %(trace_id)s 时每次请求日志行可直接肉眼关联。
    """
    root = logging.getLogger()
    for f in root.filters:
        if isinstance(f, TraceIdFilter):
            return f
    filt = TraceIdFilter()
    root.addFilter(filt)      # 根 logger 直发记录（logging.info 等）
    for h in root.handlers:
        h.addFilter(filt)     # 子 logger 传播记录（模块级 logger）
    return filt


def timing(stage: str, seconds: float) -> None:
    """累积一段阶段耗时到当前请求上下文（毫秒；开关关闭时零埋点）"""
    if not settings.request_logs_enabled:
        return
    if seconds is None or seconds <= 0:
        return
    _obs()["timings"][stage] = round(seconds * 1000, 1)


def record_usage(provider: str, prompt_tokens: int, completion_tokens: int) -> None:
    """按供应商累积 token 用量（开关关闭时零埋点；无 usage 由调用方跳过）"""
    if not settings.request_logs_enabled:
        return
    usage = _obs()["usage"]
    entry = usage.setdefault(provider, {"prompt": 0, "completion": 0})
    entry["prompt"] += int(prompt_tokens or 0)
    entry["completion"] += int(completion_tokens or 0)


def record_cache(hit: bool) -> None:
    """记录检索缓存命中/未命中（开关关闭时零埋点）"""
    if not settings.request_logs_enabled:
        return
    d = _obs()
    if hit:
        d["cache_hits"] += 1
    else:
        d["cache_misses"] += 1


def get_request_stats() -> dict:
    """取当前请求观测快照（供端点落库 request_logs）"""
    return dict(_obs())


async def save_request_log(record: dict) -> None:
    """异步落库 request_logs（fail-open：失败仅日志告警，不阻塞主链路）

    由端点请求结束处 fire-and-forget 调用（含流式结束/断开）；建表走
    init_db 自愈幂等 DDL（ensure_request_logs_table），本函数不建表。
    开关关闭时直接返回（零落库）。

    Args:
        record: 观测记录（trace_id/identity/endpoint/intent/timings/usage/
            cache_hits/cache_misses/error）
    """
    if not settings.request_logs_enabled:
        return
    try:
        from rag.models import RequestLog
        from src.database import async_session_factory

        async with async_session_factory() as session:
            session.add(RequestLog(**record))
            await session.commit()
    except Exception as e:
        logger.warning("request_logs 落库失败（fail-open，不影响主链路）: %s", e)


def make_trace_id() -> str:
    """生成一次请求的 trace_id（UUID hex）"""
    return uuid.uuid4().hex


def elapsed(start: float) -> float:
    """perf_counter 差分（语义化，配合 timing 使用）"""
    return time.perf_counter() - start
