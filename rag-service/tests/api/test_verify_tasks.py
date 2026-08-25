"""module-060 verify 异步化：后台任务池 + 轮询端点 + chat_stream 开关行为

覆盖（验收 §1/§2/§4/§5）：
- submit_verify_task：返回 task_id（uuid hex）/ 先插 pending 落库 / 调度后台任务 /
  池项完成后释放；开关关闭返回 None 不产生后台任务；pending 落库失败 fail-open
- _run_verify：成功 → _update_done（claims/confidence/verified_in_ms）；异常 → _update_failed
- get_verify_task：读 DB 为准（pending/done/failed/不存在 None）
- 轮询端点 GET /ai/rag/chat/verify/{task_id}：pending/done/failed/404 状态机
- chat_stream：开关 true → done 事件带 verify_task_id 且无 verified 事件；
  开关 false → verified→done 顺序（与现状逐字一致）
- DDL 幂等：ensure_verify_results_table 重复调用不报错

实现说明：假 session 打桩 async_session_factory（对齐 test_observability.py /
test_feedback.py 模式），不依赖真实 PG；conftest autouse 钉住 verify_async_enabled
=False，本文件用例显式 setattr True。同步用例内 asyncio.run（不依赖 pytest-asyncio）。
"""
import asyncio
import json
from unittest import mock

import httpx
import pytest

import main as main_module
from agent import reflector as _reflector  # noqa: F401  确保 patch 目标类已导入
from agent import router as _router  # noqa: F401
from rag import engine as _engine  # noqa: F401
from src import verify_tasks
from src.config import settings
from src.database import ensure_verify_results_table
from src.verify_tasks import submit_verify_task, get_verify_task


def _run(coro):
    return asyncio.run(coro)


def _parse_sse(body: bytes) -> list[dict]:
    """把 SSE 响应体解析成事件列表 [{event, data}, ...]"""
    events = []
    for block in body.decode("utf-8").split("\n\n"):
        block = block.strip()
        if not block:
            continue
        evt = {}
        for line in block.split("\n"):
            if line.startswith("event: "):
                evt["event"] = line[len("event: "):]
            elif line.startswith("data: "):
                evt["data"] = line[len("data: "):]
        if evt:
            events.append(evt)
    return events


def _doc(doc_id: int = 1) -> dict:
    return {
        "id": doc_id,
        "title": "测试文档",
        "content": "这是一段测试内容，用于检索与生成。",
        "source": "test",
        "hybrid_score": 0.95,
    }


class _FakeSession:
    """假 AsyncSession：add 记录对象；execute 对 select 返回可配置行、其余 no-op

    - select（get_verify_task）：scalar_one_or_none 返回 rows[0] / None
    - text/update（DDL / _update_done / _update_failed）：no-op MagicMock
    - commit 可配置抛异常（落库失败 fail-open 用例）
    """

    def __init__(self, rows: list | None = None, commit_error: bool = False):
        self.added: list = []
        self.rows = rows or []
        self._commit_error = commit_error

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        if self._commit_error:
            raise RuntimeError("数据库不可用")

    async def execute(self, stmt):
        from sqlalchemy.sql.selectable import Select

        if isinstance(stmt, Select):
            result = mock.MagicMock()
            result.scalar_one_or_none.return_value = self.rows[0] if self.rows else None
            return result
        return mock.MagicMock()


def _fake_factory(session):
    """把 async_session_factory 打桩成返回 session 的异步上下文管理器"""
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=session)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    return mock.MagicMock(return_value=cm)


@pytest.fixture(autouse=True)
def _clean_pool():
    """清理 verify_tasks 内存任务池（防跨用例残留已关闭 loop 的 task 引用）"""
    verify_tasks._pool.clear()
    yield
    verify_tasks._pool.clear()


def _enable_async(monkeypatch) -> None:
    """显式开启 verify 异步开关（conftest autouse 默认钉住 false）"""
    monkeypatch.setattr(settings, "verify_async_enabled", True)


VERIFIED_OK = {
    "claims": [{"claim": "测试", "verdict": "supported", "evidence": "[1]"}],
    "overall_confidence": 1.0,
    "total_claims": 1,
    "supported": 1,
    "inferred": 0,
    "unsupported": 0,
}


class TestSubmitVerifyTask:
    """submit_verify_task：pending 落库 + 调度后台任务 + 开关/失败降级"""

    def test_submit_returns_task_id_and_inserts_pending(self, monkeypatch):
        """返回 uuid hex；先插 pending 落库；池持有任务句柄；完成后释放"""
        _enable_async(monkeypatch)
        session = _FakeSession()
        done = mock.AsyncMock()

        async def run():
            with mock.patch("src.database.async_session_factory", _fake_factory(session)):
                with mock.patch("agent.reflector.reflector.verify_answer",
                                new=mock.AsyncMock(return_value=VERIFIED_OK)):
                    with mock.patch("src.verify_tasks._update_done", done):
                        task_id = await submit_verify_task(
                            "答案", [_doc()], identity="u1", query="q", trace_id="tr")
                        assert task_id is not None
                        assert len(task_id) == 32  # uuid hex
                        # pending 记录已落库（先插 DB 再调度）
                        assert len(session.added) == 1
                        pend = session.added[0]
                        assert pend.status == "pending"
                        assert pend.task_id == task_id
                        assert pend.endpoint == "chat_stream"
                        assert pend.query == "q"
                        # 池持有执行期中间态
                        assert task_id in verify_tasks._pool
                        # 等待后台任务完成（done callback 释放池项）
                        await verify_tasks._pool[task_id]["task"]
                        return task_id

        task_id = _run(run())
        # 完成后释放内存池中间态（DB 结果保留）
        assert task_id not in verify_tasks._pool
        # 成功路径：_update_done 被调用（task_id + verified + verified_in_ms 毫秒）
        assert done.called
        args, _kwargs = done.call_args
        assert args[0] == task_id
        assert args[1]["total_claims"] == 1
        assert isinstance(args[2], int) and args[2] >= 0

    def test_run_verify_failure_marks_failed(self, monkeypatch):
        """verify_answer 异常 → _update_failed（task_id + error）"""
        _enable_async(monkeypatch)
        session = _FakeSession()
        failed = mock.AsyncMock()

        async def run():
            with mock.patch("src.database.async_session_factory", _fake_factory(session)):
                with mock.patch("agent.reflector.reflector.verify_answer",
                                new=mock.AsyncMock(side_effect=RuntimeError("boom"))):
                    with mock.patch("src.verify_tasks._update_failed", failed):
                        task_id = await submit_verify_task(
                            "答案", [_doc()], identity="u1", query="q", trace_id="tr")
                        assert task_id
                        await verify_tasks._pool[task_id]["task"]
                        return task_id

        task_id = _run(run())
        assert task_id not in verify_tasks._pool  # 异常路径同样释放
        assert failed.called
        args, _kwargs = failed.call_args
        assert args[0] == task_id
        assert "boom" in args[1]

    def test_submit_disabled_returns_none_no_task(self, monkeypatch):
        """开关关闭（conftest 默认）→ submit 返回 None、不产生 pending 落库与后台任务"""
        session = _FakeSession()

        async def run():
            with mock.patch("src.database.async_session_factory", _fake_factory(session)):
                task_id = await submit_verify_task(
                    "答案", [_doc()], identity="u1", query="q", trace_id="tr")
                return task_id

        assert _run(run()) is None
        assert session.added == []
        assert verify_tasks._pool == {}

    def test_submit_pending_db_failure_fail_open(self, monkeypatch):
        """pending 落库失败 → 返回 None、不调度后台任务（fail-open 不影响主链路）"""
        _enable_async(monkeypatch)
        session = _FakeSession(commit_error=True)

        async def run():
            with mock.patch("src.database.async_session_factory", _fake_factory(session)):
                task_id = await submit_verify_task(
                    "答案", [_doc()], identity="u1", query="q", trace_id="tr")
                return task_id

        assert _run(run()) is None
        assert verify_tasks._pool == {}  # 不创建后台任务


class TestGetVerifyTask:
    """get_verify_task：读 DB 为准（pending/done/failed/不存在 None）"""

    @staticmethod
    def _row(status: str, **over):
        row = mock.MagicMock()
        row.task_id = "task-1"
        row.status = status
        row.claims = [{"claim": "x", "verdict": "supported", "evidence": "[1]"}]
        row.overall_confidence = 1.0
        row.supported = 1
        row.inferred = 0
        row.unsupported = 0
        row.error = None
        row.verified_in_ms = 500
        for k, v in over.items():
            setattr(row, k, v)
        return row

    def test_get_done_row(self):
        """done → claims/confidence/counts/verified_in_ms 透传"""
        session = _FakeSession(rows=[self._row("done")])

        async def run():
            with mock.patch("src.database.async_session_factory", _fake_factory(session)):
                return await get_verify_task("task-1")

        result = _run(run())
        assert result["status"] == "done"
        assert result["claims"] == [{"claim": "x", "verdict": "supported", "evidence": "[1]"}]
        assert result["overall_confidence"] == 1.0
        assert result["verified_in_ms"] == 500

    def test_get_pending_row(self):
        """pending → status=pending（轮询端点继续轮询）"""
        session = _FakeSession(rows=[self._row("pending")])

        async def run():
            with mock.patch("src.database.async_session_factory", _fake_factory(session)):
                return await get_verify_task("task-1")

        assert _run(run())["status"] == "pending"

    def test_get_failed_row(self):
        """failed → error 透传（轮询端点 → 前端停止 fail-open）"""
        session = _FakeSession(rows=[self._row("failed", error="verify timeout")])

        async def run():
            with mock.patch("src.database.async_session_factory", _fake_factory(session)):
                return await get_verify_task("task-1")

        result = _run(run())
        assert result["status"] == "failed"
        assert result["error"] == "verify timeout"

    def test_get_missing_returns_none(self):
        """不存在（重启丢任务/过期）→ None → 轮询 404"""
        session = _FakeSession(rows=[])

        async def run():
            with mock.patch("src.database.async_session_factory", _fake_factory(session)):
                return await get_verify_task("missing")

        assert _run(run()) is None


class TestVerifyResultsDDL:
    """verify_results 建表幂等（对齐 feedback/request_logs DDL 模式）"""

    def test_ddl_idempotent(self):
        """ensure_verify_results_table 重复调用不报错（CREATE TABLE IF NOT EXISTS）"""
        session = _FakeSession()

        async def run():
            with mock.patch("src.database.async_session_factory", _fake_factory(session)):
                await ensure_verify_results_table()
                await ensure_verify_results_table()

        _run(run())  # 不抛异常即通过


class TestVerifyPollingEndpoint:
    """轮询端点 GET /ai/rag/chat/verify/{task_id} 状态机"""

    def _hit(self, get_result):
        async def run():
            with mock.patch("main.get_verify_task",
                            new=mock.AsyncMock(return_value=get_result)):
                transport = httpx.ASGITransport(app=main_module.app, raise_app_exceptions=True)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/ai/rag/chat/verify/task-1")
                    return resp

        return _run(run())

    def test_pending(self):
        """pending → 200 {"status": "pending"}"""
        resp = self._hit({"task_id": "task-1", "status": "pending",
                          "claims": [], "overall_confidence": None,
                          "supported": 0, "inferred": 0, "unsupported": 0,
                          "error": None, "verified_in_ms": None})
        assert resp.status_code == 200
        assert resp.json() == {"status": "pending"}

    def test_done(self):
        """done → 200 含 claims/confidence/total_claims/counts/verified_in_ms"""
        resp = self._hit({
            "task_id": "task-1", "status": "done",
            "claims": [{"claim": "测试", "verdict": "supported", "evidence": "[1]"}],
            "overall_confidence": 1.0,
            "supported": 1, "inferred": 0, "unsupported": 0,
            "error": None, "verified_in_ms": 1200,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "done"
        assert body["total_claims"] == 1
        assert body["claims"][0]["verdict"] == "supported"
        assert body["overall_confidence"] == 1.0
        assert body["verified_in_ms"] == 1200

    def test_failed(self):
        """failed → 200 {"status": "failed", "error"}"""
        resp = self._hit({"task_id": "task-1", "status": "failed",
                          "claims": [], "overall_confidence": None,
                          "supported": 0, "inferred": 0, "unsupported": 0,
                          "error": "verify timeout", "verified_in_ms": None})
        assert resp.status_code == 200
        assert resp.json() == {"status": "failed", "error": "verify timeout"}

    def test_missing_404(self):
        """不存在 → 404 {"detail": "task not found"}（前端 fail-open）"""
        resp = self._hit(None)
        assert resp.status_code == 404
        assert resp.json() == {"detail": "task not found"}

    def test_db_error_404(self):
        """查询异常 → 404（fail-open，不 500）"""
        async def run():
            with mock.patch("main.get_verify_task",
                            new=mock.AsyncMock(side_effect=RuntimeError("db down"))):
                transport = httpx.ASGITransport(app=main_module.app, raise_app_exceptions=True)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    return await client.get("/ai/rag/chat/verify/task-1")

        resp = _run(run())
        assert resp.status_code == 404


class _GenCapture:
    """捕获 generate_answer_stream 调用参数，并用假 token 流式产出"""

    def __init__(self, tokens: list[str] | None = None):
        self.tokens = tokens or ["好", "的"]

    def make_gen(self):
        async def fake_generate_answer_stream(query, documents, history=None, memory=""):
            for tok in self.tokens:
                yield tok
        return fake_generate_answer_stream


def _hit_stream(*, submit_task_id: str | None = None,
                verify_result: dict | None = None):
    """发起一次 chat_stream 请求（mock 全链路），按当前开关分支

    开关由测试体内显式 setattr settings.verify_async_enabled 决定（conftest
    autouse 默认钉住 false）。开关 true 时 patch main.submit_verify_task（返回
    submit_task_id，异步路径不调用 verify_answer）；开关 false 时 patch
    verify_answer（返回 verify_result，submit 不调用）。返回
    (sse_events, status, submit_calls, verify_calls)。
    """
    events: list[dict] = []
    status = 0
    submit_calls: list = []
    verify_calls: list = []

    async def run():
        nonlocal status
        with mock.patch("agent.router.router_agent.classify",
                        new=mock.AsyncMock(return_value={"intent": "knowledge"})):
            with mock.patch("rag.engine.rag_engine._retrieve",
                            new=mock.AsyncMock(return_value=[_doc()])):
                with mock.patch("rag.engine.rag_engine._rerank",
                                new=mock.AsyncMock(side_effect=lambda q, docs: docs)):
                    with mock.patch("rag.engine.rag_engine._recall_memory",
                                    new=mock.AsyncMock(return_value="")):
                        with mock.patch("agent.reflector.reflector.check_sufficiency",
                                        new=mock.AsyncMock(
                                            return_value={"sufficient": True, "reason": ""})):
                            with mock.patch("agent.reflector.reflector.generate_answer_stream",
                                            new=_GenCapture().make_gen()):
                                with mock.patch("rag.engine.rag_engine._resolve_session_history",
                                                new=mock.AsyncMock(side_effect=lambda i, h: h)):
                                    with mock.patch("rag.engine.rag_engine._schedule_session_persist",
                                                    new=mock.MagicMock()):
                                        with mock.patch("main.submit_verify_task",
                                                        new=mock.AsyncMock(
                                                            return_value=submit_task_id)) as sub:
                                            with mock.patch("agent.reflector.reflector.verify_answer",
                                                            new=mock.AsyncMock(
                                                                return_value=verify_result)) as ver:
                                                transport = httpx.ASGITransport(
                                                    app=main_module.app, raise_app_exceptions=True)
                                                async with httpx.AsyncClient(
                                                        transport=transport,
                                                        base_url="http://test") as client:
                                                    resp = await client.post(
                                                        "/ai/rag/chat/stream",
                                                        headers={"X-Forwarded-For": "9.9.9.9"},
                                                        json={"query": "回答风格", "history": []},
                                                    )
                                                events.extend(_parse_sse(resp.content))
                                                status = resp.status_code
                                            submit_calls.extend(sub.call_args_list)
                                            verify_calls.extend(ver.call_args_list)

    with mock.patch("rag.engine.rag_engine._persist_memory", new=mock.AsyncMock()):
        _run(run())
    return events, status, submit_calls, verify_calls


class TestChatStreamVerifyAsync:
    """chat_stream 开关行为：true 异步后置 / false 现状同步（逃生口）"""

    def test_async_done_carries_task_id_no_verified_event(self, monkeypatch):
        """开关 true：done 事件带 verify_task_id + verified=False；无 verified 事件；
        不调用同步 verify_answer（连接早于 verify 完成）"""
        _enable_async(monkeypatch)
        events, status, submit_calls, verify_calls = _hit_stream(
            submit_task_id="task-xyz")
        assert status == 200
        evt_names = [e["event"] for e in events]
        assert "verified" not in evt_names
        assert evt_names[-1] == "done"
        done = json.loads(events[-1]["data"])
        assert done["verified"] is False
        assert done["verify_task_id"] == "task-xyz"
        assert done["sources"]  # 引用溯源保留
        # 主链路不再同步 await verify
        assert verify_calls == []
        # submit 收到 clean_answer / docs / identity / query / trace_id
        assert len(submit_calls) == 1
        args, kwargs = submit_calls[0]
        assert kwargs["identity"] == "9.9.9.9"  # XFF → client_ip 兜底
        assert kwargs["query"] == "回答风格"
        assert len(args) == 2  # (answer, docs)
        assert "好" in args[0]

    def test_async_submit_fail_done_no_task_id(self, monkeypatch):
        """开关 true 但提交失败（DB 写失败）→ done 无 verify_task_id（前端 fail-open）"""
        _enable_async(monkeypatch)
        events, status, _submit, _verify = _hit_stream(
            submit_task_id=None)
        assert status == 200
        done = json.loads(events[-1]["data"])
        assert done["verified"] is False
        assert "verify_task_id" not in done

    def test_sync_verified_then_done_unchanged(self, monkeypatch):
        """开关 false（conftest 默认）：verified→done 顺序逐字一致，submit 不调用"""
        events, status, submit_calls, verify_calls = _hit_stream(
            verify_result=VERIFIED_OK)
        assert status == 200
        assert [e["event"] for e in events] == [
            "step", "step", "step", "step", "token", "token", "verified", "done"]
        # 开关 false 走现状同步路径，不提交后台任务
        assert submit_calls == []
        assert len(verify_calls) == 1
