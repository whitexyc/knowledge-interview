"""
飞轮冒烟脚本（module-057 WP-A5）— 真实 HTTP chat + 👍👎 反馈 + 落库验证 + 防重复
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

背景:
    module-048 建了 feedback 表 + POST /ai/feedback + 前端 👍👎 按钮
    （localStorage rag_feedback_rated 已评态防重复），但 feedback 表 0 条——
    真实链路（真实 chat 回答 → 模拟点击 → 落库 → 查表）从未端到端验证过。
    本脚本按用户指示"自己造一些对话然后根据回答点击"，把链路跑通并把冒烟
    数据保留为飞轮种子。

用法（在 ai_service 目录下）:
    python -m eval.flywheel_smoke              # 自动拉起 uvicorn 8001 → 冒烟 → 停服
    python -m eval.flywheel_smoke --port 8001  # 指定端口
    python -m eval.flywheel_smoke --no-verify-db  # 跳过 DB 落库校验（仅链路冒烟）

流程:
    ① 服务就绪：端口无服务则子进程拉起 uvicorn main:app（轮询 /docs 至多 60s），
       已有服务则直接复用（结束时不停别人起的服务）
    ② 自造 5 条知识库问题（G1/Kafka/volatile/Redis/HashMap）→ POST /ai/rag/chat
       真实 HTTP 获取回答（X-Forwarded-For 注入测试身份 IP）
    ③ message_id 声明：AI 层直连 chat 无 Java 后端消息主键（message_id 来自
       Java 消息表主键），按规划降级用**构造标识** 990000+i（Integer 主键范围外，
       与真实 Java id 不冲突），如实声明
    ④ 模拟点击：POST /ai/feedback，rating 交替 +1/-1，≥1 条带 comment，
       +1 条非法 rating=0 验证 422 拦截
    ⑤ 查 feedback 表：确认 message_id/rating/identity/created_at 落库正确
    ⑥ 防重复验证：同一 message_id 二次提交（换 rating 模拟"改主意"）→
       后端无幂等时如实记录（会落库两行），前端已评态 localStorage 是唯一
       防重机制（grep 前端 rag_feedback_rated 确认）
    ⑦ 冒烟数据保留为飞轮种子（不清理；changelog 注明身份 IP 与构造 id 范围）

诚实边界:
    1. 冒烟数据为自造（非真实用户点击），验证链路 + 作为种子数据；
       真实飞轮仍靠用户点击积累（module-048 待办）。
    2. message_id 为构造标识（990000+i），非 Java 消息主键；Java 侧回填
       口径待真实链路（经 Java 后端 chat）验证。
    3. 后端 /ai/feedback 无幂等（module-048 设计如此：防重在前端已评态），
       重复提交会落库多行——如实记录，不改生产行为。
"""
import argparse
import asyncio
import json
import subprocess
import sys
import time
import urllib.request

import httpx

# 测试身份 IP（X-Forwarded-For 注入，避免污染真实用户）
SMOKE_IP = "203.0.113.66"
# 构造 message_id 起点（AI 层直连 chat 无 Java 消息主键；990000+ 远离真实主键范围）
MESSAGE_ID_BASE = 990000

QUESTIONS = [
    "什么是G1垃圾收集器的核心创新？",
    "Kafka 如何保证消息不丢失？",
    "volatile 关键字能保证原子性吗？",
    "Redis 的 RDB 和 AOF 持久化有什么区别？",
    "HashMap 在什么情况下会树化？",
]


async def _http_post(client: httpx.AsyncClient, url: str, payload: dict) -> dict:
    resp = await client.post(url, json=payload)
    return {"status_code": resp.status_code, "body": resp.json()}


def server_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/docs", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _wait_ready(port: int, timeout_s: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server_ready(port):
            return True
        time.sleep(2)
    return False


def ensure_server(port: int, start: bool) -> subprocess.Popen | None:
    """端口无服务则拉起 uvicorn 子进程；已有服务返回 None（不接管生命周期）"""
    if server_ready(port):
        print(f"[flywheel] 已有服务在 127.0.0.1:{port}，直接复用（结束时不停）")
        return None
    if not start:
        raise RuntimeError(
            f"127.0.0.1:{port} 无服务且未允许拉起（--start-server），请先启动 uvicorn")
    print(f"[flywheel] 拉起 uvicorn main:app (127.0.0.1:{port}) ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=sys.path[0] if sys.path[0] else None,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not _wait_ready(port):
        proc.terminate()
        raise RuntimeError("uvicorn 90s 内未就绪（检查 .env/模型加载），已终止")
    print("[flywheel] 服务就绪")
    return proc


async def query_feedback_rows(identity: str) -> list[dict]:
    """查 feedback 表（按 identity），返回行列表"""
    from sqlalchemy import text
    from src.database import async_session_factory

    async with async_session_factory() as session:
        rows = await session.execute(text(
            "SELECT id, message_id, rating, comment, identity, created_at "
            "FROM feedback WHERE identity = :identity ORDER BY id"),
            {"identity": identity})
        return [{"id": r[0], "message_id": r[1], "rating": r[2],
                 "comment": r[3], "identity": r[4], "created_at": str(r[5])}
                for r in rows]


async def run_smoke(port: int, verify_db: bool) -> None:
    base = f"http://127.0.0.1:{port}"
    headers = {"X-Forwarded-For": SMOKE_IP}
    async with httpx.AsyncClient(timeout=180.0, headers=headers) as client:

        print("\n== ① 自造对话 → 真实 HTTP chat ==")
        message_ids: list[int] = []
        for i, q in enumerate(QUESTIONS, start=1):
            result = await _http_post(client, f"{base}/ai/rag/chat",
                                      {"query": q, "history": []})
            code = result["status_code"]
            body = result["body"]
            answer = (body or {}).get("answer", "")
            if code != 200 or not answer:
                print(f"  [{i}] chat 失败 code={code}: {str(body)[:120]}")
                continue
            # 构造标识声明：AI 层直连无 Java 消息主键（见模块 docstring）
            mid = MESSAGE_ID_BASE + i
            message_ids.append(mid)
            print(f"  [{i}] mid={mid} 回答({len(answer)}字): {answer[:60]}...")

        if not message_ids:
            raise RuntimeError("全部 chat 请求失败，链路不可用（见上方错误）")

        print("\n== ② 模拟点击 👍👎（rating 交替 ±1，≥1 条带 comment）==")
        comments = {2: "回答不完整，缺少 ISR 的 min.insync.replicas 细节",
                    4: "RDB 和 AOF 的区别讲清楚了"}
        for idx, mid in enumerate(message_ids):
            rating = 1 if idx % 2 == 0 else -1
            result = await _http_post(client, f"{base}/ai/feedback", {
                "message_id": mid,
                "rating": rating,
                "comment": comments.get(idx + 1),
            })
            print(f"  mid={mid} rating={rating:+d} comment={comments.get(idx + 1) is not None} "
                  f"→ HTTP {result['status_code']} {result['body']}")
            if result["status_code"] != 200:
                print("    [警告] 反馈提交非 200（链路异常，继续验证其余）")

        print("\n== ③ 非法输入拦截（rating=0 → 422）==")
        bad = await _http_post(client, f"{base}/ai/feedback", {
            "message_id": MESSAGE_ID_BASE + 999, "rating": 0})
        print(f"  rating=0 → HTTP {bad['status_code']}"
              f"（期望 422：Pydantic 校验拦截，防落库污染）")

        print("\n== ④ 防重复验证（同一 message_id 二次提交，rating 翻转模拟改主意）==")
        dup_mid = message_ids[0]
        dup = await _http_post(client, f"{base}/ai/feedback", {
            "message_id": dup_mid, "rating": -1, "comment": "改主意了，这条其实不对"})
        print(f"  二次提交 mid={dup_mid} rating=-1 → HTTP {dup['status_code']} {dup['body']}")

    # ── DB 落库验证 ──
    if verify_db:
        print("\n== ⑤ feedback 表落库验证 ==")
        rows = await query_feedback_rows(SMOKE_IP)
        print(f"  identity={SMOKE_IP} 共 {len(rows)} 行:")
        for r in rows:
            print(f"    id={r['id']} message_id={r['message_id']} "
                  f"rating={r['rating']:+d} comment={r['comment'] is not None} "
                  f"created_at={r['created_at'][:19]}")
        mid_rows = [r for r in rows if r["message_id"] == dup_mid]
        print(f"\n  mid={dup_mid} 共 {len(mid_rows)} 行"
              + ("（首次+二次提交=2 行 → 后端无幂等，如实记录：防重在前端已评态 "
                 "localStorage rag_feedback_rated，后端重复提交会落库多行）"
                 if len(mid_rows) >= 2 else "（仅 1 行 → 存在幂等，需复查）"))
        n_expected = len(message_ids) + 1  # 5 次提交 + 1 次重复
        print(f"  预期 {n_expected} 行（{len(message_ids)} 次点击 + 1 次重复提交），"
              f"实际 {len(rows)} 行 -> {'一致' if len(rows) == n_expected else '不一致（见上）'}")
    else:
        print("\n== ⑤ DB 落库验证已跳过（--no-verify-db）==")

    print("\n" + "=" * 60)
    print(f"飞轮冒烟完成: {len(message_ids)} 条真实回答 + 反馈落库验证"
          + (" + 防重复记录" if verify_db else ""))
    print(f"冒烟数据保留为飞轮种子: identity={SMOKE_IP}，"
          f"message_id ∈ [{MESSAGE_ID_BASE + 1}, {MESSAGE_ID_BASE + len(message_ids)}]"
          f"（构造标识，非 Java 消息主键）")
    print("=" * 60)


async def main() -> None:
    parser = argparse.ArgumentParser(description="飞轮冒烟：真实 chat → 👍👎 → 落库验证")
    parser.add_argument("--port", type=int, default=8001, help="AI 服务端口（默认 8001）")
    parser.add_argument("--no-verify-db", action="store_true",
                        help="跳过 DB 落库校验（仅链路冒烟）")
    args = parser.parse_args()

    proc = ensure_server(args.port, start=True)
    try:
        await run_smoke(args.port, verify_db=not args.no_verify_db)
    finally:
        if proc is not None:
            print(f"[flywheel] 停止自拉起服务 (pid={proc.pid})")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
