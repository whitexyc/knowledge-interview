"""module-066 WP-A：tool_call_logs 落库单测（ADR-0017 决策 2）

覆盖（验收 §1.1 功能 + §2 边界 + §3 异常）：
- DDL 幂等建表（ensure 拆分逐条执行）
- record_tool_call：成功落库（trace_id/工具名/args JSONB/result_ok/预览/耗时）
- result_preview 截断 200
- args 非 JSON 序列化防御（兜底 {}）
- 开关 false 零落库（不构造记录）
- fail-open（DB 断连不阻断工具执行）
- execute_tool_with_log：成功 / 工具不存在 / run 抛异常 → result_ok 语义
- react_loop 接线：只记实际执行的 tool_calls（预算截断不落库）、事件流不变
- langgraph_react_loop 同构接线

实现说明：
- conftest autouse 钉住 tool_call_logs_enabled=false（hermetic）；本文件显式
  开启 + mock src.database.async_session_factory（对齐 test_observability 模式）
- 断言 INSERT 的 SQL 文本与绑定参数（含 CAST(:args AS jsonb) 与预览截断）
"""
import asyncio
import json
from unittest import mock

from agent.react import (
    ReactContext, _build_messages, execute_tool_with_log,
    react_loop, record_tool_call,
)
from agent.langgraph_react import langgraph_react_loop
from agent.tool_registry import ToolRegistry
from src import observability
from src.config import settings
from src.database import ensure_tool_call_logs_table


class _FakeSession:
    """假 AsyncSession：记录 execute 的 (SQL, 参数)；可配置 commit 抛异常"""

    def __init__(self, commit_error: bool = False):
        self.executed: list = []
        self._commit_error = commit_error

    async def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params or {}))

    async def commit(self):
        if self._commit_error:
            raise RuntimeError("数据库不可用")


def _fake_factory(session):
    """把 async_session_factory 打桩成返回 session 的异步上下文管理器"""
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=session)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    return mock.MagicMock(return_value=cm)


def _enable_logs(monkeypatch) -> None:
    """显式开启 tool_call_logs（conftest autouse 默认钉住 false）"""
    monkeypatch.setattr(settings, "tool_call_logs_enabled", True)


class _StubTool:
    """假工具：可配置结果或抛异常"""

    def __init__(self, result="stub 结果", error: Exception | None = None):
        self.result = result
        self.error = error

    async def run(self, args, ctx):
        if self.error:
            raise self.error
        return self.result


def _stub_registry() -> ToolRegistry:
    """两个 stub 工具注册表（真实注册表会触发 DB 检索，测试不用）"""

    async def _f(ctx, args):
        return "stub 结果"

    reg = ToolRegistry()
    reg.register("search_knowledge", "混合检索",
                 {"type": "object",
                  "properties": {"query": {"type": "string"}},
                  "required": ["query"]},
                 _f, group=["retrieval"])
    reg.register("generate_answer", "生成答案",
                 {"type": "object",
                  "properties": {"query": {"type": "string"}},
                  "required": ["query"]},
                 _f, group=["generation"])
    return reg


class TestDDL:
    """tool_call_logs 表 DDL 幂等建表"""

    def test_ensure_table_idempotent(self, monkeypatch):
        """ensure_tool_call_logs_table 拆分逐条执行 DDL（CREATE + COMMENT 全执行）"""
        session = _FakeSession()
        monkeypatch.setattr("src.database.async_session_factory", _fake_factory(session))

        asyncio.run(ensure_tool_call_logs_table())

        sqls = [sql for sql, _ in session.executed]
        assert len(sqls) == 8  # CREATE TABLE + 7 条 COMMENT
        assert any("CREATE TABLE IF NOT EXISTS tool_call_logs" in s for s in sqls)
        assert any("COMMENT ON TABLE tool_call_logs" in s for s in sqls)


class TestRecordToolCall:
    """record_tool_call：落库字段 / 截断 / 防御 / fail-open / 开关"""

    def test_record_success_persists(self, monkeypatch):
        """成功落库一行：trace_id/工具名/args JSONB/result_ok/预览/耗时"""
        _enable_logs(monkeypatch)
        session = _FakeSession()
        monkeypatch.setattr("src.database.async_session_factory", _fake_factory(session))

        async def run():
            observability.init_request("trace-abc")
            await record_tool_call("search_knowledge", {"query": "RRF"}, True,
                                   "检索结果", 12)

        asyncio.run(run())
        sql, params = session.executed[0]
        assert "INSERT INTO tool_call_logs" in sql
        assert "CAST(:args AS jsonb)" in sql
        assert params["trace_id"] == "trace-abc"
        assert params["tool_name"] == "search_knowledge"
        assert json.loads(params["args"]) == {"query": "RRF"}
        assert params["result_ok"] is True
        assert params["result_preview"] == "检索结果"
        assert params["duration_ms"] == 12

    def test_preview_truncated_to_200(self, monkeypatch):
        """超长结果预览截断 200（大文档不撑爆列）"""
        _enable_logs(monkeypatch)
        session = _FakeSession()
        monkeypatch.setattr("src.database.async_session_factory", _fake_factory(session))

        async def run():
            await record_tool_call("search_knowledge", {}, True, "x" * 500, 1)

        asyncio.run(run())
        assert session.executed[0][1]["result_preview"] == "x" * 200

    def test_args_not_serializable_falls_back(self, monkeypatch):
        """args 含非 JSON 序列化对象（供应商防御路径）→ 兜底 {} 不崩"""
        _enable_logs(monkeypatch)
        session = _FakeSession()
        monkeypatch.setattr("src.database.async_session_factory", _fake_factory(session))

        async def run():
            await record_tool_call("search_knowledge", {"bad": {1, 2}}, True, "", 1)

        asyncio.run(run())  # 不抛异常
        assert json.loads(session.executed[0][1]["args"]) == {}

    def test_disabled_skips_zero_overhead(self, monkeypatch):
        """开关 false（conftest 默认钉住）→ 不构造记录不落库"""
        session = _FakeSession()
        monkeypatch.setattr("src.database.async_session_factory", _fake_factory(session))

        async def run():
            await record_tool_call("search_knowledge", {}, True, "", 1)

        asyncio.run(run())
        assert session.executed == []  # 工厂未被调用

    def test_fail_open_on_commit_error(self, monkeypatch):
        """DB 断连（commit 抛异常）→ 不阻断，仅吞掉"""
        _enable_logs(monkeypatch)
        session = _FakeSession(commit_error=True)
        monkeypatch.setattr("src.database.async_session_factory", _fake_factory(session))

        async def run():
            await record_tool_call("search_knowledge", {}, True, "", 1)

        asyncio.run(run())  # 不抛异常（fail-open）


class TestExecuteToolWithLog:
    """execute_tool_with_log：result_ok 语义 + 计时 + 落库"""

    def test_success_result_ok_true(self, monkeypatch):
        """工具正常执行 → result_ok=true，返回结果，落库 duration_ms≥0"""
        _enable_logs(monkeypatch)
        session = _FakeSession()
        monkeypatch.setattr("src.database.async_session_factory", _fake_factory(session))
        ctx = ReactContext("q")

        async def run():
            return await execute_tool_with_log("search_knowledge", {}, _StubTool(), ctx)

        assert asyncio.run(run()) == "stub 结果"
        params = session.executed[0][1]
        assert params["result_ok"] is True
        assert params["duration_ms"] >= 0

    def test_missing_tool_result_ok_false(self, monkeypatch):
        """工具不存在（tools.get 返回 None）→ result_ok=false、结果空串、循环继续"""
        _enable_logs(monkeypatch)
        session = _FakeSession()
        monkeypatch.setattr("src.database.async_session_factory", _fake_factory(session))
        ctx = ReactContext("q")

        async def run():
            return await execute_tool_with_log("no_such_tool", {}, None, ctx)

        assert asyncio.run(run()) == ""
        assert session.executed[0][1]["result_ok"] is False

    def test_run_raises_result_ok_false(self, monkeypatch):
        """run 抛异常（防御路径）→ result_ok=false、不向上抛"""
        _enable_logs(monkeypatch)
        session = _FakeSession()
        monkeypatch.setattr("src.database.async_session_factory", _fake_factory(session))
        ctx = ReactContext("q")

        async def run():
            return await execute_tool_with_log(
                "search_knowledge", {}, _StubTool(error=RuntimeError("boom")), ctx)

        assert asyncio.run(run()) == ""
        assert session.executed[0][1]["result_ok"] is False


class TestLoopWiring:
    """两条 ReAct 循环接线：只记实际执行、事件流不变"""

    @staticmethod
    def _tool_call(name: str, args: dict, cid: str = "c1") -> dict:
        return {
            "content": "",
            "tool_calls": [{"id": cid, "name": name, "args": args}],
            "message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": cid, "type": "function",
                                "function": {"name": name,
                                             "arguments": json.dumps(args, ensure_ascii=False)}}],
            },
        }

    @staticmethod
    def _answer(content: str) -> dict:
        return {"content": content, "tool_calls": [],
                "message": {"role": "assistant", "content": content}}

    class _FakeLLM:
        def __init__(self, responses):
            self.responses = list(responses)

        async def chat_with_tools(self, messages, tools):
            return self.responses.pop(0)

        async def chat(self, messages):
            return "直接回答"

    def test_react_loop_logs_only_executed(self, monkeypatch):
        """预算截断的 LLM 提议不落库：budget=1 只执行/记录 1 个工具"""
        _enable_logs(monkeypatch)
        session = _FakeSession()
        monkeypatch.setattr("src.database.async_session_factory", _fake_factory(session))
        monkeypatch.setattr(
            "agent.react.LLMFactory.get_client",
            mock.Mock(return_value=self._FakeLLM([
                self._tool_call("search_knowledge", {"query": "RRF"}, "c1"),
                self._tool_call("generate_answer", {"query": "RRF"}, "c2"),
            ])))
        with mock.patch("agent.react.reflector.generate_answer",
                        mock.AsyncMock(return_value="兜底答案")):
            ctx = ReactContext("什么是RRF")
            events = [e["type"] for e in asyncio.run(
                _collect(react_loop(ctx, _build_messages(ctx), budget=1,
                                    tools=_stub_registry())))]

        # 事件流不变：tool_call → tool_result → 预算耗尽兜底 token → done
        assert events == ["tool_call", "tool_result", "token", "done"]
        # 只落 1 行（截断掉的 generate_answer 不记）
        assert len(session.executed) == 1
        assert session.executed[0][1]["tool_name"] == "search_knowledge"
        assert session.executed[0][1]["result_ok"] is True

    def test_langgraph_loop_logs_tools(self, monkeypatch):
        """agent-lg 同构落库：执行工具即记录"""
        _enable_logs(monkeypatch)
        session = _FakeSession()
        monkeypatch.setattr("src.database.async_session_factory", _fake_factory(session))
        monkeypatch.setattr(
            "agent.langgraph_react.LLMFactory.get_client",
            mock.Mock(return_value=self._FakeLLM([
                self._tool_call("search_knowledge", {"query": "RRF"}, "c1"),
                self._answer("最终答案"),
            ])))
        ctx = ReactContext("什么是RRF")
        events = [e["type"] for e in asyncio.run(
            _collect(langgraph_react_loop(ctx, _build_messages(ctx), budget=2,
                                          tools=_stub_registry())))]

        assert events == ["tool_call", "tool_result", "token", "done"]
        assert len(session.executed) == 1
        assert session.executed[0][1]["tool_name"] == "search_knowledge"

    def test_trace_id_from_observability_context(self, monkeypatch):
        """落库 trace_id 取自观测上下文（不改 ReactContext/循环签名）"""
        _enable_logs(monkeypatch)
        session = _FakeSession()
        monkeypatch.setattr("src.database.async_session_factory", _fake_factory(session))
        monkeypatch.setattr(
            "agent.react.LLMFactory.get_client",
            mock.Mock(return_value=self._FakeLLM([
                self._tool_call("search_knowledge", {"query": "RRF"}, "c1"),
                self._answer("答案"),
            ])))
        ctx = ReactContext("什么是RRF")

        async def run():
            observability.init_request("eval-abc-1")
            await _collect(react_loop(ctx, _build_messages(ctx), budget=2,
                                      tools=_stub_registry()))

        asyncio.run(run())
        assert session.executed[0][1]["trace_id"] == "eval-abc-1"


async def _collect(gen):
    """把异步生成器事件收进 list（测试辅助）"""
    return [evt async for evt in gen]
