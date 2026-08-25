"""module-058 WP-C：真实请求 trace 样例 + request_logs 落库验证（临时探测脚本）

流程：
  1. init_db() 幂等建表（含 request_logs，对齐 module-048 feedback 模式）
  2. 真实 rag_engine.chat（真实 PG 知识库 + 本地 bge-m3 + deepseek 降级链，
     verify 走 LLM 判分避免 HHEM 冷加载干扰）→ 观测上下文收集各阶段耗时 /
     token 用量 / 缓存命中
  3. save_request_log 落库 → 查回打印完整 trace 记录
  4. 清理探测身份的记忆/会话行（保留 request_logs 样例行）

运行：python scripts/probe_request_trace.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src import observability

IDENTITY = "probe-trace-058"


async def main() -> None:
    settings.request_logs_enabled = True
    settings.verify_judge_model = "llm"  # 探测用：避免 HHEM 冷加载

    from src.database import init_db, async_session_factory
    from rag.engine import rag_engine
    from rag.schemas import ChatRequest
    from sqlalchemy import text

    await init_db()
    print("[1/4] init_db 完成（request_logs 表幂等建表）")

    from agent.router import router_agent

    async def _classify(q):
        return await router_agent.classify(q)

    observability.init_request("trace-sample-058")
    resp = await rag_engine.chat(
        ChatRequest(query="Java 线程池的核心参数有哪些？", history=[]),
        identity=IDENTITY,
    )
    print(f"[2/4] 真实 chat 完成: message={resp.message}, answer_len={len(resp.answer)}")

    stats = observability.get_request_stats()
    print("观测统计（阶段耗时 ms / token 用量 / 缓存命中）:")
    for k, v in stats["timings"].items():
        print(f"  timing[{k}] = {v}ms")
    print(f"  usage = {stats['usage']}")
    print(f"  cache_hits={stats['cache_hits']} cache_misses={stats['cache_misses']}")

    record = {
        "trace_id": stats["trace_id"],
        "identity": IDENTITY,
        "endpoint": "probe-engine.chat",
        "intent": "knowledge",
        "timings": stats["timings"],
        "usage": stats["usage"],
        "cache_hits": stats["cache_hits"],
        "cache_misses": stats["cache_misses"],
        "error": False,
    }
    await observability.save_request_log(record)
    print(f"[3/4] request_logs 落库完成: trace_id={stats['trace_id']}")

    async with async_session_factory() as session:
        row = (await session.execute(text(
            "SELECT trace_id, identity, endpoint, intent, timings, usage, "
            "cache_hits, cache_misses, error, created_at FROM request_logs "
            "WHERE trace_id = :t ORDER BY id DESC LIMIT 1"
        ), {"t": stats["trace_id"]})).first()
        print("[4/4] 查回记录:")
        for name, val in zip(
                ["trace_id", "identity", "endpoint", "intent", "timings",
                 "usage", "cache_hits", "cache_misses", "error", "created_at"],
                row):
            print(f"  {name} = {val}")

    # 清理探测身份的记忆/会话行（保留 request_logs 样例行）
    async with async_session_factory() as session:
        await session.execute(text(
            "DELETE FROM documents WHERE source LIKE :pat"),
            {"pat": f"memory:{IDENTITY}:%"})
        await session.commit()
    print("已清理探测身份记忆/会话行（request_logs 样例行保留）")


if __name__ == "__main__":
    asyncio.run(main())
