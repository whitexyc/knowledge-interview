"""
verify 后台任务池 + verify_results 表读写（module-060 verify 异步化）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

module-060（verify 后置推送 P2，落库持久化）：chat_stream 流式生成完不再同步
await verify（15-50s 阻塞主链路尾部，loading 空转），改为：
  1. submit_verify_task：先插 verify_results 表一条 pending 记录（DB 为准，
     重启后未完成任务丢失属 fail-open 边界，已 done 结果持久可查）→
     asyncio.create_task(_run_verify) fire-and-forget（不 await，对齐
     engine._schedule_persist 成熟模式）→ 返回 task_id。
  2. _run_verify：time.perf_counter() 计时 → await reflector.verify_answer
     （内部 15/20/15s 超时，不会无限 hang）→ 成功 UPDATE status=done +
     claims/overall_confidence/supported/inferred/unsupported/verified_in_ms；
     异常 UPDATE status=failed + error。任务内全捕获，绝不抛回主链路。
  3. get_verify_task：**读 DB 为准**（轮询端点用）——pending/done/failed/
     不存在返回 None。

内存池只持执行期中间态（answer+docs+task 句柄），任务完成即释放（done
callback）；DB 结果永久保留（不清理，飞轮数据源——verify 结果含逐句 verdict
可支撑答案可信度/幻觉调优数据积累）。开关 verify_async_enabled 关闭时 submit
直接返回 None（不产生后台任务）。
"""
import asyncio
import logging
import time
from typing import Optional

from src import observability
from src.config import settings

logger = logging.getLogger(__name__)

# 内存任务池：task_id -> {answer, docs, identity, query, trace_id, task}
# 只持执行期中间态；任务完成/异常后经 done callback 释放（DB 结果不清理）。
_pool: dict[str, dict] = {}


async def submit_verify_task(
    answer: str,
    docs: list[dict],
    *,
    identity: str,
    query: str,
    trace_id: str,
) -> Optional[str]:
    """提交一个后台 verify 任务（fire-and-forget，不 await）

    流程：
      1. 开关关闭 → 返回 None（调用方不发 task_id，前端不轮询，fail-open）
      2. 生成 task_id（uuid hex，复用 observability.make_trace_id()）
      3. 先插 verify_results 表 pending 记录（DB 写失败 → 返回 None，fail-open，
         不影响主链路答案交付；不抛回响应）
      4. asyncio.create_task(_run_verify) 调度后台执行（只调度不 await，
         任务引用存入 _pool 防 GC——与 engine._schedule_persist 同款防坑）
      5. 返回 task_id 供前端轮询

    Args:
        answer: LLM 生成的答案文本（已剥离截断标记）
        docs: 检索到的文档列表（供 verify_answer 判分）
        identity: 请求身份（user_id 优先，client_ip 兜底）
        query: 用户问题（落库，飞轮数据源可关联）
        trace_id: 请求追踪 ID（关联 request_logs）

    Returns:
        task_id（成功）；None（开关关闭 / pending 落库失败 → 调用方 fail-open）
    """
    if not settings.verify_async_enabled:
        return None

    task_id = observability.make_trace_id()
    try:
        await _insert_pending(task_id=task_id, trace_id=trace_id,
                              identity=identity, query=query)
    except Exception as e:
        logger.warning("verify 任务 pending 落库失败，跳过后台验证（fail-open）: %s", e)
        return None

    task = asyncio.create_task(_run_verify(task_id, answer, docs))
    # 任务引用存入池防 GC（asyncio.create_task 只调度，若无引用可能在完成前被回收）；
    # done callback 释放池项（DB 结果永久保留）
    _pool[task_id] = {
        "answer": answer,
        "docs": docs,
        "identity": identity,
        "query": query,
        "trace_id": trace_id,
        "task": task,
    }
    task.add_done_callback(lambda _t: _pool.pop(task_id, None))
    return task_id


async def _run_verify(task_id: str, answer: str, docs: list[dict]) -> None:
    """后台执行 verify_answer 并落库结果（任务内全捕获，绝不抛回）

    成功 → UPDATE status=done + claims/overall_confidence/counts/verified_in_ms；
    异常 → UPDATE status=failed + error。verify_answer 内部已有 15/20/15s 超时
    降级（返回空 claims 而非异常），本函数不再新增无限任务风险。
    """
    t0 = time.perf_counter()
    try:
        from agent.reflector import reflector
        verified = await reflector.verify_answer(answer, docs)
        verified_in_ms = int((time.perf_counter() - t0) * 1000)
        await _update_done(task_id, verified, verified_in_ms)
        logger.info("verify 后台任务完成: task_id=%s, total=%s, verified_in_ms=%s",
                    task_id, verified.get("total_claims", 0), verified_in_ms)
    except Exception as e:
        logger.warning("verify 后台任务异常: task_id=%s, %s", task_id, e)
        try:
            await _update_failed(task_id, str(e))
        except Exception as e2:
            logger.warning("verify 失败状态落库失败（fail-open）: %s", e2)


async def get_verify_task(task_id: str) -> Optional[dict]:
    """按 task_id 查 verify_results 表（**DB 为准**，轮询端点用）

    Returns:
        {"task_id", "status", "claims", "overall_confidence", "supported",
         "inferred", "unsupported", "error", "verified_in_ms"}
        不存在 → None
    """
    from sqlalchemy import select
    from rag.models import VerifyResult
    from src.database import async_session_factory

    async with async_session_factory() as session:
        row = (await session.execute(
            select(VerifyResult).where(VerifyResult.task_id == task_id)
        )).scalar_one_or_none()
    if row is None:
        return None
    return {
        "task_id": row.task_id,
        "status": row.status,
        "claims": row.claims or [],
        "overall_confidence": row.overall_confidence,
        "supported": row.supported,
        "inferred": row.inferred,
        "unsupported": row.unsupported,
        "error": row.error,
        "verified_in_ms": row.verified_in_ms,
    }


async def _insert_pending(*, task_id: str, trace_id: str, identity: str,
                          query: str) -> None:
    """插一条 pending 记录（DB 为准的起点）"""
    from rag.models import VerifyResult
    from src.database import async_session_factory

    async with async_session_factory() as session:
        session.add(VerifyResult(
            task_id=task_id,
            trace_id=trace_id,
            identity=identity,
            endpoint="chat_stream",
            query=query,
            status="pending",
        ))
        await session.commit()


async def _update_done(task_id: str, verified: dict, verified_in_ms: int) -> None:
    """verify 成功后 UPDATE 为 done + 结果字段"""
    from sqlalchemy import update
    from rag.models import VerifyResult
    from src.database import async_session_factory

    async with async_session_factory() as session:
        await session.execute(
            update(VerifyResult)
            .where(VerifyResult.task_id == task_id)
            .values(
                status="done",
                claims=verified.get("claims") or [],
                overall_confidence=verified.get("overall_confidence"),
                supported=verified.get("supported", 0),
                inferred=verified.get("inferred", 0),
                unsupported=verified.get("unsupported", 0),
                verified_in_ms=verified_in_ms,
            )
        )
        await session.commit()


async def _update_failed(task_id: str, error: str) -> None:
    """verify 异常后 UPDATE 为 failed + error"""
    from sqlalchemy import update
    from rag.models import VerifyResult
    from src.database import async_session_factory

    async with async_session_factory() as session:
        await session.execute(
            update(VerifyResult)
            .where(VerifyResult.task_id == task_id)
            .values(status="failed", error=error[:2000])
        )
        await session.commit()
