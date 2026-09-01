"""
Module-083 工具治理单元测试（schema 校验 / 幂等 / 工具级超时 / 高风险审批 / Agent 级最小权限）

覆盖（WP-A~E，对齐验收 AC-1~AC-44）：
- WP-A 参数校验：非法类型不执行（wait_for 0 次）/ 合法执行 / 缺省回退契约 /
  run({}, None) 容忍 / 非 dict 提示 / jsonschema 内部异常 fail-open
- WP-B 幂等：同参拦截 / 键序无关 / 异参放行 / 失败不记可重放 / 超时不记可重放 /
  排除清单照常 / 轻量 ctx 与 ctx=None 零拦截 / 073 重试成功记 1 次 / 每请求独立 / 循环自愈
- WP-C 超时：timeout 参数透传 wait_for / 默认 15.0 来源 config / 精确文案一字不改
- WP-D 审批：auto 零 DB / required 拦截+插申请 / pending 去重 / approved 放行 /
  DB 异常 fail-closed / GET+POST 端点 / DDL 幂等 / init_db 挂接
- WP-E 权限：None 全量 / 白名单外拒绝 result_ok=false / 白名单内放行 /
  react_agent→react_loop 透传 / 阶段与权限两维独立

实现说明：
- 全 mock hermetic：不依赖真实 DB / LLM / Redis（对齐 test_tool_retry_dedup 手法）
- 同步用例内 asyncio.run 执行（沿用既有模式）；DB 交互经 src.database.async_session_factory
  打桩假 session（对齐 test_tool_call_logs._fake_factory 模式）
- conftest autouse 已钉 tool_phase_split=False / tool_auto_retry=False / tool_call_logs=False；
  相关用例体内显式 monkeypatch 覆盖
"""
import asyncio
import datetime
import inspect
import json
import logging
from types import SimpleNamespace
from unittest import mock

from src.config import settings
from src.database import ensure_approval_requests_table, init_db
from agent.tool_registry import (
    ToolRegistry,
    register_builtin_tools,
    _SEARCH_SCHEMA,
    _IDEMPOTENT_TOOLS,
    _fingerprint,

)
from agent.react import (
    ReactContext,
    react_agent,
    react_loop,
    execute_tool_with_log,
)


def _make_tool(name: str, func, schema: dict | None = None, *,
               timeout=None, approval: str = "auto") -> ToolRegistry:
    """注册单工具注册表（隔离测试，不动全局 registry）"""
    reg = ToolRegistry()
    reg.register(name, f"工具 {name}", schema or {"type": "object"}, func,
                 timeout=timeout, approval=approval)
    return reg


class _FakeLLM:
    """脚本化的假 LLM：按序返回 chat_with_tools 响应（react_loop 集成用）"""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)

    async def chat_with_tools(self, messages, tools):
        return self.responses.pop(0)


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


def _answer(content: str) -> dict:
    return {"content": content, "tool_calls": [],
            "message": {"role": "assistant", "content": content}}


# ─── 假 DB（审批表交互打桩，对齐 test_tool_call_logs._fake_factory 模式） ───


class _FakeSession:
    """假 AsyncSession：记录 execute 的 (SQL, 参数)；SELECT 可配 first() 结果"""

    def __init__(self, first_value=None, rows=None):
        self.first_value = first_value
        self.rows = rows
        self.executed: list = []
        self.commits = 0

    async def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params or {}))
        result = mock.Mock()
        if "SELECT" in str(stmt):
            result.first.return_value = self.first_value
            if self.rows is not None:
                result.fetchall.return_value = self.rows
        return result

    async def commit(self):
        self.commits += 1


def _fake_factory(session):
    """把 async_session_factory 打桩成返回 session 的异步上下文管理器"""
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=session)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    return mock.MagicMock(return_value=cm)


class _BoomFactory:
    """async with 表达式内即抛异常（模拟 DB 完全不可用）"""

    def __call__(self):
        raise RuntimeError("数据库不可用")


# ══════════════════════════════════════════════════════════════════
# WP-A：args schema 校验（AC-1~AC-6 / AC-40）
# ══════════════════════════════════════════════════════════════════


class TestSchemaValidation:
    """AC-1~AC-6：校验失败不执行 / 合法执行 / 缺省回退 / ctx=None / 非 dict / fail-open"""

    def test_invalid_type_blocked_no_execute(self, monkeypatch):
        # AC-1：top_k="abc" → 参数错误提示 + func 不执行 + 不进重试（wait_for 0 次）
        calls = {"n": 0}

        async def f(ctx, args):
            calls["n"] += 1
            return "不应执行"

        reg = _make_tool("search_knowledge", f, _SEARCH_SCHEMA)
        tool = reg.get("search_knowledge")
        wait_for = mock.Mock()
        monkeypatch.setattr("agent.tool_registry.asyncio.wait_for", wait_for)
        result = asyncio.run(tool.run({"top_k": "abc", "query": "q"}, ReactContext("q")))
        assert "(工具 search_knowledge 参数错误" in result
        assert calls["n"] == 0
        assert wait_for.call_count == 0  # 校验失败不进重试分支

    def test_valid_args_executes(self):
        # AC-2：合法参数正常执行、结果原样返回
        async def f(ctx, args):
            return f"结果:{args.get('query')}"

        tool = _make_tool("search_knowledge", f, _SEARCH_SCHEMA).get("search_knowledge")
        result = asyncio.run(tool.run({"query": "q", "top_k": 5}, ReactContext("q")))
        assert result == "结果:q"

    def test_empty_args_fallback_contract(self):
        # AC-3：schema required 含 query，run({}, ctx) 不被判参数错误 → 走工具内缺省回退
        async def f(ctx, args):
            return f"回退:{ctx.query}"

        tool = _make_tool("search_knowledge", f, _SEARCH_SCHEMA).get("search_knowledge")
        result = asyncio.run(tool.run({}, ReactContext("原始问题")))
        assert result == "回退:原始问题"

    def test_ctx_none_no_crash(self):
        # AC-4：run({}, None)（存量测试形态）不抛 AttributeError，校验照常、幂等短路
        async def f(ctx, args):
            return "ok"

        tool = _make_tool("search_knowledge", f, _SEARCH_SCHEMA).get("search_knowledge")
        result = asyncio.run(tool.run({}, None))
        assert result == "ok"

    def test_non_dict_args_rejected(self):
        # AC-5：非 dict args → "参数应为 object" 提示，不执行
        async def f(ctx, args):
            raise AssertionError("不应执行")

        tool = _make_tool("search_knowledge", f, _SEARCH_SCHEMA).get("search_knowledge")
        result = asyncio.run(tool.run(["a", "b"], ReactContext("q")))
        assert result == "(工具 search_knowledge 参数错误: 参数应为 object)"

    def test_jsonschema_internal_error_fail_open(self, monkeypatch):
        # AC-40：jsonschema 非 ValidationError 异常 → fail-open 放行执行 + warning
        def boom(instance, schema):
            raise RuntimeError("版本差异")

        monkeypatch.setattr("agent.tool_registry._js_validate", boom)
        calls = {"n": 0}

        async def f(ctx, args):
            calls["n"] += 1
            return "执行了"

        tool = _make_tool("search_knowledge", f, _SEARCH_SCHEMA).get("search_knowledge")
        result = asyncio.run(tool.run({"top_k": "abc"}, ReactContext("q")))
        assert result == "执行了"
        assert calls["n"] == 1

    def test_fingerprint_ignores_key_order(self):
        # AC-9 支撑：_fingerprint 纯函数键序无关
        fp1 = _fingerprint("t", {"query": "q", "top_k": 5})
        fp2 = _fingerprint("t", {"top_k": 5, "query": "q"})
        assert fp1 == fp2
        assert len(fp1) == 64


# ══════════════════════════════════════════════════════════════════
# WP-B：幂等（AC-7~AC-15 / AC-36）
# ══════════════════════════════════════════════════════════════════


class TestIdempotency:
    """AC-7~AC-15：同参只读检索二次拦截、失败/超时/排除清单/轻量 ctx"""

    def test_same_args_second_blocked(self):
        # AC-8：同参二次 → 拦截提示，func 仅执行 1 次
        calls = {"n": 0}

        async def f(ctx, args):
            calls["n"] += 1
            return "检索结果"

        tool = _make_tool("search_knowledge", f, _SEARCH_SCHEMA).get("search_knowledge")
        ctx = ReactContext("q")
        r1 = asyncio.run(tool.run({"query": "q"}, ctx))
        r2 = asyncio.run(tool.run({"query": "q"}, ctx))
        assert r1 == "检索结果"
        assert r2 == "(该调用已执行过，结果见上文)"
        assert calls["n"] == 1

    def test_key_order_insensitive_blocked(self):
        # AC-9：指纹键序无关 → 换键序同参也拦截（func 仅 1 次）
        calls = {"n": 0}

        async def f(ctx, args):
            calls["n"] += 1
            return "检索结果"

        tool = _make_tool("search_knowledge", f, _SEARCH_SCHEMA).get("search_knowledge")
        ctx = ReactContext("q")
        asyncio.run(tool.run({"query": "q", "top_k": 5}, ctx))
        r2 = asyncio.run(tool.run({"top_k": 5, "query": "q"}, ctx))
        assert r2 == "(该调用已执行过，结果见上文)"
        assert calls["n"] == 1

    def test_different_args_executes(self):
        # AC-10：参数变化（query/top_k）→ 不拦截正常执行
        calls = {"n": 0}

        async def f(ctx, args):
            calls["n"] += 1
            return f"结果:{args.get('query')}"

        tool = _make_tool("search_knowledge", f, _SEARCH_SCHEMA).get("search_knowledge")
        ctx = ReactContext("q")
        asyncio.run(tool.run({"query": "q"}, ctx))
        r2 = asyncio.run(tool.run({"query": "q2"}, ctx))
        assert r2 == "结果:q2"
        assert calls["n"] == 2

    def test_failure_not_recorded_replayable(self):
        # AC-11：首次抛异常（返回空串）不记指纹 → 同参再次真实执行
        calls = {"n": 0}

        async def flaky(ctx, args):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("首次失败")
            return "检索结果 ok"

        tool = _make_tool("search_knowledge", flaky, _SEARCH_SCHEMA).get("search_knowledge")
        ctx = ReactContext("q")
        r1 = asyncio.run(tool.run({"query": "q"}, ctx))
        assert r1 == ""  # conftest 钉住 tool_auto_retry=False → 失败即空
        assert len(ctx.executed_fingerprints) == 0  # 失败不记
        r2 = asyncio.run(tool.run({"query": "q"}, ctx))
        assert r2 == "检索结果 ok"
        assert calls["n"] == 2

    def test_timeout_not_recorded_replayable(self, monkeypatch):
        # AC-12：超时（返回精确文案）不记指纹 → 同参再次真实执行
        async def f(ctx, args):
            return "结果"

        tool = _make_tool("search_knowledge", f, _SEARCH_SCHEMA).get("search_knowledge")
        ctx = ReactContext("q")

        def fake_wait_for(awaitable, timeout=None):
            awaitable.close()
            raise asyncio.TimeoutError

        monkeypatch.setattr("agent.tool_registry.asyncio.wait_for", fake_wait_for)
        r1 = asyncio.run(tool.run({"query": "q"}, ctx))
        assert r1 == "(工具 search_knowledge 执行超时)"
        assert len(ctx.executed_fingerprints) == 0  # 超时不记
        monkeypatch.undo()
        r2 = asyncio.run(tool.run({"query": "q"}, ctx))
        assert r2 == "结果"
        assert len(ctx.executed_fingerprints) == 1  # 真实执行后才记

    def test_excluded_tools_not_intercepted(self):
        # AC-13：generate_answer / verify_answer / note_to_self 同参二次仍执行
        async def f(ctx, args):
            return "结果"

        for name in ("generate_answer", "verify_answer", "note_to_self"):
            tool = _make_tool(name, f).get(name)
            ctx = ReactContext("q")
            r1 = asyncio.run(tool.run({"query": "q"}, ctx))
            r2 = asyncio.run(tool.run({"query": "q"}, ctx))
            assert r1 == r2 == "结果"
            assert len(ctx.executed_fingerprints) == 0
        assert "generate_answer" not in _IDEMPOTENT_TOOLS
        assert "verify_answer" not in _IDEMPOTENT_TOOLS
        assert "note_to_self" not in _IDEMPOTENT_TOOLS

    def test_lightweight_ctx_and_none_no_interception(self):
        # AC-14：MCP 轻量 ctx（无 executed_fingerprints）/ ctx=None → getattr 短路零拦截
        async def f(ctx, args):
            return "结果"

        tool = _make_tool("search_knowledge", f, _SEARCH_SCHEMA).get("search_knowledge")
        mcp_ctx = SimpleNamespace(query="q", identity="mcp", docs=[], memory="",
                                  scratchpad=[], add_docs=lambda d: None,
                                  add_note=lambda n: None)
        r1 = asyncio.run(tool.run({"query": "q"}, mcp_ctx))
        r2 = asyncio.run(tool.run({"query": "q"}, mcp_ctx))
        assert r1 == r2 == "结果"  # 无双写拦截
        assert asyncio.run(tool.run({"query": "q"}, None)) == "结果"
        assert asyncio.run(tool.run({"query": "q"}, None)) == "结果"

    def test_retry_success_records_once(self, monkeypatch):
        # AC-15：073 重试成功 → 指纹只记 1 次（func 计数 2，同参三次调用被拦）
        monkeypatch.setattr(settings, "tool_auto_retry", True)
        calls = {"n": 0}

        async def flaky(ctx, args):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("瞬时抖动")
            return "检索结果 ok"

        tool = _make_tool("search_knowledge", flaky, _SEARCH_SCHEMA).get("search_knowledge")
        ctx = ReactContext("q")
        r1 = asyncio.run(tool.run({"query": "q"}, ctx))
        assert r1 == "检索结果 ok"  # 首败 + 重试成功
        assert calls["n"] == 2
        assert len(ctx.executed_fingerprints) == 1  # 只记 1 次
        r2 = asyncio.run(tool.run({"query": "q"}, ctx))
        assert r2 == "(该调用已执行过，结果见上文)"
        assert calls["n"] == 2  # 不新增执行

    def test_per_request_isolated(self):
        # AC-36 前半：指纹集合每请求独立（两个 ReactContext 不共享拦截）
        async def f(ctx, args):
            return "结果"

        tool = _make_tool("search_knowledge", f, _SEARCH_SCHEMA).get("search_knowledge")
        ctx1, ctx2 = ReactContext("q"), ReactContext("q")
        asyncio.run(tool.run({"query": "q"}, ctx1))
        r = asyncio.run(tool.run({"query": "q"}, ctx2))
        assert r == "结果"  # 跨请求不共享拦截

    def test_schema_error_records_no_fingerprint(self):
        # 校验失败不真执行 → 不记指纹（同参错误参数两次都报参数错误，func 0 次）
        calls = {"n": 0}

        async def f(ctx, args):
            calls["n"] += 1
            return "X"

        tool = _make_tool("search_knowledge", f, _SEARCH_SCHEMA).get("search_knowledge")
        ctx = ReactContext("q")
        r1 = asyncio.run(tool.run({"top_k": "abc"}, ctx))
        r2 = asyncio.run(tool.run({"top_k": "abc"}, ctx))
        assert "参数错误" in r1 and "参数错误" in r2
        assert calls["n"] == 0
        assert len(ctx.executed_fingerprints) == 0

    def test_idempotent_hit_loop_continues(self):
        # AC-36 后半 / AC-44：幂等命中提示作为 tool 结果喂回 LLM，循环正常结束（无死锁）
        async def f(ctx, args):
            return "检索结果"

        reg = ToolRegistry()
        reg.register("search_knowledge", "混合检索", _SEARCH_SCHEMA, f)
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "问题"}, "c1"),
            _tool_call("search_knowledge", {"query": "问题"}, "c2"),
            _answer("最终答案"),
        ])
        with mock.patch("agent.react.LLMFactory.get_client", return_value=fake), \
             mock.patch("agent.react.record_tool_call", new=mock.AsyncMock()):
            ctx = ReactContext("问题")
            messages = [{"role": "user", "content": "问题"}]

            async def run():
                return [evt async for evt in react_loop(ctx, messages, budget=4, tools=reg)]

            events = asyncio.run(run())
        tool_results = [e for e in events if e["type"] == "tool_result"]
        done = [e for e in events if e["type"] == "done"][-1]
        assert tool_results[0]["result"] == "检索结果"
        assert tool_results[1]["result"] == "(该调用已执行过，结果见上文)"
        assert done["answer"] == "最终答案"  # 循环继续，无死锁
        assert "（该调用已执行过，结果见上文）" in messages[-1]["content"] or \
               any(m.get("role") == "tool" and "已执行过" in m.get("content", "")
                   for m in messages)


# ══════════════════════════════════════════════════════════════════
# WP-C：工具级超时（AC-16~AC-19）
# ══════════════════════════════════════════════════════════════════


class TestToolTimeout:
    """AC-16~AC-19：timeout 参数透传 wait_for / 默认 15.0 / 精确文案"""

    def test_timeout_param_forwarded(self, monkeypatch):
        # AC-16：AgentTool(timeout=0.01) → wait_for 收到 timeout=0.01；返回精确文案
        async def slow(ctx, args):
            await asyncio.sleep(999)
            return "不会到达"

        tool = _make_tool("slow_tool", slow, timeout=0.01).get("slow_tool")
        captured = {}

        def fake_wait_for(awaitable, timeout=None):
            captured["timeout"] = timeout
            awaitable.close()
            raise asyncio.TimeoutError

        monkeypatch.setattr("agent.tool_registry.asyncio.wait_for", fake_wait_for)
        result = asyncio.run(tool.run({}, None))
        assert result == "(工具 slow_tool 执行超时)"  # 精确文案一字不改
        assert captured["timeout"] == 0.01

    def test_default_timeout_from_settings_and_builtin_attr(self):
        # AC-17/AC-18/AC-20/AC-48：默认 15.0 + 10 内置工具 timeout 全 15.0、approval 全 auto
        assert settings.tool_default_timeout == 15.0
        reg = ToolRegistry()
        register_builtin_tools(reg)
        for t in reg.list_tools():
            assert t.timeout == settings.tool_default_timeout == 15.0
            assert t.approval == "auto"

    def test_timeout_marker_exact_string(self, monkeypatch):
        # AC-19 支撑：超时提示不含秒数、精确等于 "(工具 X 执行超时)"
        async def slow(ctx, args):
            await asyncio.sleep(999)
            return "X"

        tool = _make_tool("t", slow).get("t")

        def fake_wait_for(awaitable, timeout=None):
            awaitable.close()
            raise asyncio.TimeoutError

        monkeypatch.setattr("agent.tool_registry.asyncio.wait_for", fake_wait_for)
        result = asyncio.run(tool.run({}, None))
        assert result == "(工具 t 执行超时)"
        assert "15" not in result  # 超时提示不含秒数（存量断言兼容）


# ══════════════════════════════════════════════════════════════════
# WP-D：高风险审批（AC-20~AC-28 / AC-37 / AC-42）
# ══════════════════════════════════════════════════════════════════


class TestApproval:
    """AC-20~AC-28：auto 短路 / required 拦截 / pending 去重 / 放行 / fail-closed"""

    def test_auto_tool_zero_approval_db_access(self, monkeypatch):
        # AC-21：auto 工具 _approval_allowed 断言 0 次调用（不查库）
        approval = mock.AsyncMock(return_value=False)
        monkeypatch.setattr("agent.tool_registry._approval_allowed", approval)

        async def f(ctx, args):
            return "执行了"

        tool = _make_tool("search_knowledge", f).get("search_knowledge")
        result = asyncio.run(tool.run({"query": "q"}, ReactContext("q")))
        assert result == "执行了"
        assert approval.await_count == 0

    def test_required_no_approval_blocks_and_submits(self, monkeypatch):
        # AC-22：required + 无 approved → 返回审批提示 + func 未执行 + 插 pending
        approval = mock.AsyncMock(return_value=False)
        request = mock.AsyncMock(return_value=None)
        monkeypatch.setattr("agent.tool_registry._approval_allowed", approval)
        monkeypatch.setattr("agent.tool_registry._request_approval", request)
        calls = {"n": 0}

        async def f(ctx, args):
            calls["n"] += 1
            return "不应执行"

        tool = _make_tool("evil_tool", f, approval="required").get("evil_tool")
        ctx = ReactContext("q", identity="user-1")
        result = asyncio.run(tool.run({"cmd": "x"}, ctx))
        assert result == "(工具 evil_tool 需人工审批，调用申请已提交)"
        assert calls["n"] == 0
        assert request.await_count == 1
        name, args, requester = request.await_args.args
        assert name == "evil_tool" and args == {"cmd": "x"} and requester == "user-1"

    def test_request_approval_dedup_pending(self, monkeypatch):
        # AC-23：同 tool_name 已有 pending → 不重复插入
        from agent.tool_registry import _request_approval
        session = _FakeSession(first_value=(1,))
        monkeypatch.setattr("src.database.async_session_factory", _fake_factory(session))
        asyncio.run(_request_approval("evil_tool", {"cmd": "x"}, "u1"))
        sqls = [sql for sql, _ in session.executed]
        assert len(sqls) == 1 and "SELECT" in sqls[0]  # 只查不插
        assert not any("INSERT" in s for s in sqls)
        assert session.commits == 0

    def test_request_approval_inserts_when_no_pending(self, monkeypatch):
        # AC-23 反例：无 pending → 参数化 INSERT 一次
        from agent.tool_registry import _request_approval
        session = _FakeSession(first_value=None)
        monkeypatch.setattr("src.database.async_session_factory", _fake_factory(session))
        asyncio.run(_request_approval("evil_tool", {"cmd": "x"}, "u1"))
        sqls = [sql for sql, _ in session.executed]
        insert = [s for s in sqls if "INSERT" in s]
        assert len(insert) == 1
        assert "approval_requests" in insert[0]
        assert session.commits == 1
        params = session.executed[-1][1]
        assert params["n"] == "evil_tool" and params["r"] == "u1"
        assert "pending" in insert[0]

    def test_required_approved_allows(self, monkeypatch):
        # AC-24：已有 approved → 放行（工具级，不校验 args 差异）
        approval = mock.AsyncMock(return_value=True)
        request = mock.AsyncMock()
        monkeypatch.setattr("agent.tool_registry._approval_allowed", approval)
        monkeypatch.setattr("agent.tool_registry._request_approval", request)

        async def f(ctx, args):
            return "执行了"

        tool = _make_tool("evil_tool", f, approval="required").get("evil_tool")
        result = asyncio.run(tool.run({"cmd": "y"}, ReactContext("q")))
        assert result == "执行了"
        assert request.await_count == 0  # 放行不提交新申请

    def test_approval_db_error_fail_closed(self, monkeypatch, caplog):
        # AC-25：_approval_allowed DB 异常 → logger.warning + fail-closed 拒绝
        from agent.tool_registry import _approval_allowed
        monkeypatch.setattr("src.database.async_session_factory", _BoomFactory())
        caplog.set_level(logging.WARNING)
        assert asyncio.run(_approval_allowed("evil_tool")) is False
        assert any("fail-closed" in r.message for r in caplog.records)

    def test_required_db_down_blocks_execution(self, monkeypatch, caplog):
        # AC-25 run 级：DB 不可用 → 审批闸拒绝执行（宁拒勿放安全侧）
        monkeypatch.setattr("src.database.async_session_factory", _BoomFactory())
        calls = {"n": 0}

        async def f(ctx, args):
            calls["n"] += 1
            return "X"

        tool = _make_tool("evil_tool", f, approval="required").get("evil_tool")
        caplog.set_level(logging.WARNING)
        result = asyncio.run(tool.run({"cmd": "x"}, ReactContext("q")))
        assert result == "(工具 evil_tool 需人工审批，调用申请已提交)"
        assert calls["n"] == 0

    def test_approval_gate_precedes_schema_validation(self, monkeypatch):
        # AC-37：审批闸先于 schema 校验（required + 非法参数 → 返回审批提示而非参数错误）
        approval = mock.AsyncMock(return_value=False)
        request = mock.AsyncMock()
        monkeypatch.setattr("agent.tool_registry._approval_allowed", approval)
        monkeypatch.setattr("agent.tool_registry._request_approval", request)

        async def f(ctx, args):
            return "X"

        tool = _make_tool("evil_tool", f, _SEARCH_SCHEMA, approval="required").get("evil_tool")
        result = asyncio.run(tool.run({"top_k": "abc"}, ReactContext("q")))
        assert "(工具 evil_tool 需人工审批" in result
        assert "参数错误" not in result  # 审批先拦截，未到校验

    def test_ensure_ddl_idempotent(self, monkeypatch):
        # AC-28：ensure_approval_requests_table 二次执行不报错（拆 DDL 逐条断言）
        session = _FakeSession()
        monkeypatch.setattr("src.database.async_session_factory", _fake_factory(session))
        asyncio.run(ensure_approval_requests_table())
        asyncio.run(ensure_approval_requests_table())
        sqls = [sql for sql, _ in session.executed]
        assert len(sqls) == 4  # 2 次 × (CREATE + COMMENT)
        assert any("CREATE TABLE IF NOT EXISTS approval_requests" in s for s in sqls)
        assert any("COMMENT ON TABLE approval_requests" in s for s in sqls)

    def test_init_db_hooks_approval_table(self):
        # AC-28：init_db 挂接点存在（自愈建表链）
        src = inspect.getsource(init_db)
        assert "ensure_approval_requests_table" in src


# ─── 端点（直调函数断言响应 dict，hermetic 不走 HTTP） ───


class TestApprovalEndpoints:
    """AC-26 / AC-27 / AC-42：GET 列表 + status 过滤 / POST approve|reject / 非法输入"""

    def _patch_db(self, monkeypatch, session):
        import main as main_module
        monkeypatch.setattr(main_module, "async_session_factory", _fake_factory(session))
        return main_module

    def test_get_lists_pending_fields(self, monkeypatch):
        # AC-26：GET 返回 {code,msg,data:{approvals:[...]}}，含全字段
        import main as main_module
        now = datetime.datetime.now()
        session = _FakeSession(rows=[
            (1, "evil_tool", {"cmd": "x"}, "pending", "user-1", now, None),
        ])
        self._patch_db(monkeypatch, session)
        resp = asyncio.run(main_module.list_tool_approvals())
        assert resp["code"] == 0
        item = resp["data"]["approvals"][0]
        assert item["id"] == 1 and item["tool_name"] == "evil_tool"
        assert item["args"] == {"cmd": "x"} and item["status"] == "pending"
        assert item["requester"] == "user-1"
        assert item["requested_at"] == now.isoformat()
        assert item["decided_at"] is None
        params = session.executed[0][1]
        assert params == {"s": "pending"}  # 缺省 status=pending

    def test_get_status_filter(self, monkeypatch):
        # AC-26：?status=approved 过滤参数透传
        import main as main_module
        session = _FakeSession(rows=[])
        self._patch_db(monkeypatch, session)
        resp = asyncio.run(main_module.list_tool_approvals(status="approved"))
        assert resp["code"] == 0 and resp["data"]["approvals"] == []
        assert session.executed[0][1] == {"s": "approved"}

    def test_post_approve(self, monkeypatch):
        # AC-27：approve → approved + decided_at 置位（UPDATE）+ commit
        import main as main_module
        session = _FakeSession(first_value=(1, "pending"))
        self._patch_db(monkeypatch, session)
        resp = asyncio.run(main_module.decide_tool_approval(
            main_module.ApprovalDecisionRequest(id=1, action="approve")))
        assert resp["code"] == 0
        assert resp["data"] == {"id": 1, "status": "approved"}
        update = [s for s, _ in session.executed if "UPDATE" in s]
        assert len(update) == 1 and "decided_at=CURRENT_TIMESTAMP" in update[0]
        assert session.commits == 1

    def test_post_reject(self, monkeypatch):
        # AC-27：reject → rejected + decided_at
        import main as main_module
        session = _FakeSession(first_value=(1, "pending"))
        self._patch_db(monkeypatch, session)
        resp = asyncio.run(main_module.decide_tool_approval(
            main_module.ApprovalDecisionRequest(id=1, action="reject")))
        assert resp["code"] == 0
        assert resp["data"] == {"id": 1, "status": "rejected"}

    def test_post_invalid_action(self, monkeypatch):
        # AC-27：非法 action → code 1 提示不崩（Pydantic 不枚举，端点显式校验）
        import main as main_module
        self._patch_db(monkeypatch, _FakeSession())
        resp = asyncio.run(main_module.decide_tool_approval(
            main_module.ApprovalDecisionRequest(id=1, action="banana")))
        assert resp["code"] == 1 and "非法 action" in resp["msg"]

    def test_post_id_missing(self, monkeypatch):
        # AC-27：id 不存在 → code 1
        import main as main_module
        session = _FakeSession(first_value=None)
        self._patch_db(monkeypatch, session)
        resp = asyncio.run(main_module.decide_tool_approval(
            main_module.ApprovalDecisionRequest(id=999, action="approve")))
        assert resp["code"] == 1 and "不存在" in resp["msg"]

    def test_post_already_decided(self, monkeypatch):
        # AC-42：已处理（非 pending）→ code 1，不再 UPDATE
        import main as main_module
        session = _FakeSession(first_value=(1, "approved"))
        self._patch_db(monkeypatch, session)
        resp = asyncio.run(main_module.decide_tool_approval(
            main_module.ApprovalDecisionRequest(id=1, action="approve")))
        assert resp["code"] == 1 and "已处理" in resp["msg"]
        assert not any("UPDATE" in s for s, _ in session.executed)


# ══════════════════════════════════════════════════════════════════
# WP-E：Agent 级最小权限（AC-29~AC-34）
# ══════════════════════════════════════════════════════════════════


class TestAllowedTools:
    """AC-29~AC-34：allowed_tools 白名单守门 + 透传链路 + 与阶段守门两维独立"""

    def test_none_allows_all(self, monkeypatch):
        # AC-32：None = 全量放行（存量调用形态零回归）
        async def f(ctx, args):
            return "执行了"

        tool = _make_tool("search_fts", f, _SEARCH_SCHEMA).get("search_fts")
        ctx = ReactContext("q")
        result = asyncio.run(execute_tool_with_log("search_fts", {}, tool, ctx))
        assert result == "执行了"

    def test_outside_whitelist_denied(self, monkeypatch):
        # AC-30：白名单外 → 拒绝 + func 未执行 + result_ok=false + 提示含"权限白名单"
        calls = {"n": 0}

        async def f(ctx, args):
            calls["n"] += 1
            return "不应执行"

        reg = _make_tool("search_fts", f, _SEARCH_SCHEMA)
        tool = reg.get("search_fts")
        rec = mock.AsyncMock()
        monkeypatch.setattr("agent.react.record_tool_call", rec)
        ctx = ReactContext("q")
        result = asyncio.run(execute_tool_with_log(
            "search_fts", {}, tool, ctx, allowed_tools={"search_knowledge"}))
        assert calls["n"] == 0
        assert "权限白名单" in result
        assert rec.await_count == 1
        _, _, result_ok, _, _ = rec.await_args.args
        assert result_ok is False  # 审计可见

    def test_inside_whitelist_allows(self):
        # AC-31：白名单内正常执行
        async def f(ctx, args):
            return "执行了"

        reg = _make_tool("search_fts", f, _SEARCH_SCHEMA)
        tool = reg.get("search_fts")
        ctx = ReactContext("q")
        result = asyncio.run(execute_tool_with_log(
            "search_fts", {}, tool, ctx, allowed_tools={"search_fts"}))
        assert result == "执行了"

    def test_phase_and_allowed_independent(self, monkeypatch):
        # 两维独立判因：generation 工具在检索阶段 + 白名单内 → 阶段提示（非白名单提示）；
        # 检索工具 + 白名单外 → 白名单提示
        monkeypatch.setattr(settings, "tool_phase_split", True)
        monkeypatch.setattr("agent.react.record_tool_call", mock.AsyncMock())

        async def f(ctx, args):
            return "X"

        reg_gen = _make_tool("generate_answer", f)
        tool_gen = reg_gen.get("generate_answer")
        ctx = ReactContext("q")  # phase=retrieval
        r1 = asyncio.run(execute_tool_with_log(
            "generate_answer", {}, tool_gen, ctx, allowed_tools={"generate_answer"}))
        assert "当前阶段不可用" in r1 and "权限白名单" not in r1

        reg_fts = _make_tool("search_fts", f)
        tool_fts = reg_fts.get("search_fts")
        r2 = asyncio.run(execute_tool_with_log(
            "search_fts", {}, tool_fts, ctx, allowed_tools={"search_knowledge"}))
        assert "权限白名单" in r2 and "当前阶段不可用" not in r2

    def test_transparent_chain_through_react_agent(self, monkeypatch):
        # AC-34：react_agent(allowed_tools=...) → react_loop → execute_tool_with_log 生效
        async def f(ctx, args):
            return "检索结果"

        reg = ToolRegistry()
        reg.register("search_fts", "全文检索", _SEARCH_SCHEMA, f,
                     group=["retrieval"])
        fake = _FakeLLM([
            _tool_call("search_fts", {"query": "问题"}, "c1"),
            _answer("最终答案"),
        ])
        with mock.patch("agent.react.LLMFactory.get_client", return_value=fake), \
             mock.patch("agent.react.record_tool_call", new=mock.AsyncMock()):
            result = asyncio.run(react_agent(
                "问题", budget=4, tools=reg, allowed_tools={"search_knowledge"}))
        assert result["answer"] == "最终答案"
        assert "search_fts" in [t["name"] for t in result["tool_trace"]]
        denied = [t for t in result["tool_trace"] if t["name"] == "search_fts"][0]
        assert "权限白名单" in denied["result"]