"""Module-073 工具防重复 + 失败自动重试单元测试

覆盖（WP-A / WP-B，对齐验收 AC-1~AC-11 / AC-16~AC-21）：
- WP-A 防重复：add_note 完全一致去重返回 bool / note_to_self 重复提示 /
  re_search 同改写 query 守卫（拦截"重检索 + 文档格式化"大头）
- WP-B 自动重试：检索工具异常重试 1 次成功 / 重试仍失败返回 "" /
  超时不重试 / generate_answer + verify_answer 排除不重试 /
  开关 false 不重试 / 重试内超时单独处理 / 预算锁定（react_loop 集成）

实现说明：
- conftest autouse 钉住 tool_auto_retry=False（hermetic）；重试用例体内显式
  monkeypatch 置 True
- 重试用例全用 AsyncMock + monkeypatch asyncio.wait_for，测试瞬时完成不 sleep
- 同步用例内 asyncio.run 执行（沿用 test_agent_tools.py 既有模式）
"""
import asyncio
import json
import logging
from unittest import mock

from src.config import settings
from agent.tool_registry import ToolRegistry, _note_to_self, _re_search
from agent.react import ReactContext, react_loop


def _doc(doc_id: int = 1) -> dict:
    return {
        "id": doc_id,
        "title": f"文档{doc_id}",
        "content": f"这是文档{doc_id}的内容，涉及 Java 线程池。",
        "source": "test",
        "hybrid_score": 0.9,
    }


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


# ─── WP-A：note_to_self / add_note 防重复 ───


class TestNoteDedup:
    """AC-1/AC-2/AC-16：add_note 完全一致去重 + note_to_self 重复提示"""

    def test_add_note_exact_duplicate_returns_false(self):
        # 完全一致（strip 后逐字）：含首尾空白差异也判重复，不追加
        ctx = ReactContext("q")
        assert ctx.add_note("重要发现") is True
        assert ctx.add_note("  重要发现  ") is False
        assert ctx.scratchpad == ["重要发现"]

    def test_add_note_different_appends(self):
        # 不同 note 正常追加返回 True（措辞变体是正常产出，不做近似去重）
        ctx = ReactContext("q")
        assert ctx.add_note("笔记一") is True
        assert ctx.add_note("笔记二") is True
        assert ctx.scratchpad == ["笔记一", "笔记二"]

    def test_note_to_self_duplicate_returns_hint(self):
        # 重复 note → "笔记已存在（未重复记录）"，scratchpad 长度不变
        ctx = ReactContext("q")
        r1 = asyncio.run(_note_to_self(ctx, {"note": "记住 G1 是分区垃圾收集器"}))
        r2 = asyncio.run(_note_to_self(ctx, {"note": "记住 G1 是分区垃圾收集器"}))
        assert r1.startswith("已记录笔记")
        assert r2 == "笔记已存在（未重复记录）"
        assert len(ctx.scratchpad) == 1

    def test_note_to_self_whitespace_variant_duplicate(self):
        # AC-16：带首尾空白 note 与 strip 后等价文本判重复
        ctx = ReactContext("q")
        asyncio.run(_note_to_self(ctx, {"note": "笔记"}))
        r = asyncio.run(_note_to_self(ctx, {"note": "  笔记  "}))
        assert r == "笔记已存在（未重复记录）"
        assert len(ctx.scratchpad) == 1

    def test_note_to_self_empty_returns_hint(self):
        # AC-16：空/纯空白 note 仍返回"未提供笔记内容"
        ctx = ReactContext("q")
        assert asyncio.run(_note_to_self(ctx, {"note": ""})) == "（未提供笔记内容）"
        assert asyncio.run(_note_to_self(ctx, {"note": "   "})) == "（未提供笔记内容）"
        assert ctx.scratchpad == []

    def test_note_to_self_long_note_truncated_dedup(self):
        # AC-16：>500 字 note 截断后判重（两次相同超长 note 截断结果一致 → 判重复）
        ctx = ReactContext("q")
        note = "长" * 600
        r1 = asyncio.run(_note_to_self(ctx, {"note": note}))
        r2 = asyncio.run(_note_to_self(ctx, {"note": note}))
        assert "已记录笔记" in r1
        assert r2 == "笔记已存在（未重复记录）"
        assert len(ctx.scratchpad) == 1


# ─── WP-A：re_search 同改写 query 守卫 ───


class TestReSearchGuard:
    """AC-4/AC-5/AC-19：同改写 query 连续调用拦截，防 LLM 空转"""

    def _ctx_with_docs(self):
        ctx = ReactContext("问题")
        ctx.add_docs([_doc(1)])
        return ctx

    def test_same_rewritten_blocks_second(self):
        # 连续两次相同改写 query → 第二次拦截，retrieve 仅首次被调
        ctx = self._ctx_with_docs()
        sufficiency = mock.AsyncMock(
            return_value={"sufficient": False, "rewritten_query": "改写A"})
        retrieve = mock.AsyncMock(return_value=[_doc(2)])
        with mock.patch("agent.tool_registry.reflector.check_sufficiency", sufficiency), \
             mock.patch("agent.tool_registry.hybrid_retriever.retrieve", retrieve):
            r1 = asyncio.run(_re_search(ctx, {"query": "问题"}))
            r2 = asyncio.run(_re_search(ctx, {"query": "问题"}))
        assert "改写A" in r1 and "检索到" in r1
        assert r2 == "已按该改写重检过，无新结果"
        assert retrieve.await_count == 1
        # 守卫在 check_sufficiency 之后：第二次仍重新评估充分性（如实标注）
        assert sufficiency.await_count == 2
        assert ctx.last_research_query == "改写A"

    def test_different_rewritten_ok(self):
        # 不同改写 query 正常执行（换 query 可继续重检）
        ctx = self._ctx_with_docs()
        sufficiency = mock.AsyncMock(side_effect=[
            {"sufficient": False, "rewritten_query": "改写A"},
            {"sufficient": False, "rewritten_query": "改写B"},
        ])
        retrieve = mock.AsyncMock(return_value=[_doc(2)])
        with mock.patch("agent.tool_registry.reflector.check_sufficiency", sufficiency), \
             mock.patch("agent.tool_registry.hybrid_retriever.retrieve", retrieve):
            r1 = asyncio.run(_re_search(ctx, {"query": "问题"}))
            r2 = asyncio.run(_re_search(ctx, {"query": "问题"}))
        assert "改写A" in r1 and "改写B" in r2
        assert retrieve.await_count == 2
        assert ctx.last_research_query == "改写B"

    def test_sufficient_does_not_update_guard(self):
        # sufficient → 提前返回，不更新守卫字段（AC-5）
        ctx = self._ctx_with_docs()
        sufficiency = mock.AsyncMock(return_value={"sufficient": True})
        retrieve = mock.AsyncMock()
        with mock.patch("agent.tool_registry.reflector.check_sufficiency", sufficiency), \
             mock.patch("agent.tool_registry.hybrid_retriever.retrieve", retrieve):
            r = asyncio.run(_re_search(ctx, {"query": "问题"}))
        assert "已充分" in r
        assert ctx.last_research_query == ""
        assert retrieve.await_count == 0

    def test_same_raw_query_blocks(self):
        # 空改写（rewritten_query 缺失 → rewritten=原 query）同输入二次调用拦截
        ctx = self._ctx_with_docs()
        sufficiency = mock.AsyncMock(return_value={"sufficient": False})
        retrieve = mock.AsyncMock(return_value=[])
        with mock.patch("agent.tool_registry.reflector.check_sufficiency", sufficiency), \
             mock.patch("agent.tool_registry.hybrid_retriever.retrieve", retrieve):
            r1 = asyncio.run(_re_search(ctx, {"query": "问题"}))
            r2 = asyncio.run(_re_search(ctx, {"query": "问题"}))
        assert "仍无结果" in r1
        assert r2 == "已按该改写重检过，无新结果"
        assert retrieve.await_count == 1

    def test_first_call_records_guard(self):
        # AC-19：守卫字段初始 ""，首次调用正常执行并记录
        ctx = self._ctx_with_docs()
        assert ctx.last_research_query == ""
        sufficiency = mock.AsyncMock(
            return_value={"sufficient": False, "rewritten_query": "改写A"})
        retrieve = mock.AsyncMock(return_value=[_doc(2)])
        with mock.patch("agent.tool_registry.reflector.check_sufficiency", sufficiency), \
             mock.patch("agent.tool_registry.hybrid_retriever.retrieve", retrieve):
            r = asyncio.run(_re_search(ctx, {"query": "问题"}))
        assert "改写A" in r
        assert ctx.last_research_query == "改写A"
        assert retrieve.await_count == 1


# ─── WP-B：AgentTool.run 失败自动重试 ───


class TestToolAutoRetry:
    """AC-6~AC-11 / AC-17 / AC-20~AC-21：异常重试 1 次，超时不重试"""

    def _tool(self, name: str, func):
        reg = ToolRegistry()
        reg.register(name, "desc", {"type": "object"}, func)
        return reg.get(name)

    def test_retry_recovers_after_first_failure(self, monkeypatch):
        # AC-6/AC-21：首次异常（瞬时抖动）→ 自动重试同一 func 同参数 → 成功
        monkeypatch.setattr(settings, "tool_auto_retry", True)
        calls = {"n": 0}

        async def flaky(ctx, args):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("瞬时 429")
            return "检索结果 ok"

        tool = self._tool("search_knowledge", flaky)
        result = asyncio.run(tool.run({}, None))
        assert calls["n"] == 2
        assert result == "检索结果 ok"

    def test_retry_still_fails_returns_empty(self, monkeypatch):
        # AC-7/AC-20：重试仍失败 → 返回 ""（LLM 判断继续/放弃，降级哲学不变）
        monkeypatch.setattr(settings, "tool_auto_retry", True)
        f = mock.AsyncMock(side_effect=RuntimeError("恒失败"))
        tool = self._tool("search_knowledge", f)
        assert asyncio.run(tool.run({}, None)) == ""
        assert f.await_count == 2

    def test_timeout_no_retry(self, monkeypatch):
        # AC-8/AC-17：超时（15s）不重试 → 精确文案不变，wait_for 仅调 1 次
        monkeypatch.setattr(settings, "tool_auto_retry", True)
        f = mock.AsyncMock()
        tool = self._tool("slow_tool", f)

        def fake_wait_for(awaitable, timeout=None):
            awaitable.close()  # 关闭未 await 的协程，防 RuntimeWarning
            raise asyncio.TimeoutError

        wait_for = mock.Mock(side_effect=fake_wait_for)
        monkeypatch.setattr("agent.tool_registry.asyncio.wait_for", wait_for)
        result = asyncio.run(tool.run({}, None))
        assert result == "(工具 slow_tool 执行超时)"
        assert wait_for.call_count == 1  # 超时不进入重试分支
        assert f.await_count == 0

    def test_generate_verify_not_retried(self, monkeypatch):
        # AC-9：generate_answer / verify_answer 异常不重试（func 仅执行 1 次）
        monkeypatch.setattr(settings, "tool_auto_retry", True)
        for name in ("generate_answer", "verify_answer"):
            f = mock.AsyncMock(side_effect=RuntimeError("失败"))
            tool = self._tool(name, f)
            assert asyncio.run(tool.run({}, None)) == ""
            assert f.await_count == 1

    def test_switch_off_no_retry(self):
        # AC-10/AC-17：开关 false（conftest autouse 钉住）→ 全工具不重试，存量行为零回归
        f = mock.AsyncMock(side_effect=RuntimeError("失败"))
        tool = self._tool("search_knowledge", f)
        assert asyncio.run(tool.run({}, None)) == ""
        assert f.await_count == 1

    def test_retry_warning_logs(self, monkeypatch, caplog):
        # AC-7：重试两次失败各有 warning（"首次失败，自动重试"/"重试仍失败，返回空"）
        monkeypatch.setattr(settings, "tool_auto_retry", True)
        caplog.set_level(logging.WARNING)
        f = mock.AsyncMock(side_effect=RuntimeError("失败"))
        tool = self._tool("search_knowledge", f)
        asyncio.run(tool.run({}, None))
        msgs = [r.message for r in caplog.records]
        assert any("首次失败，自动重试" in m for m in msgs)
        assert any("重试仍失败，返回空" in m for m in msgs)

    def test_retry_timeout_inside_retry(self, monkeypatch):
        # AC-17：重试内 TimeoutError 单独处理 → 返回超时提示（非空串）
        monkeypatch.setattr(settings, "tool_auto_retry", True)
        f = mock.AsyncMock()
        tool = self._tool("search_knowledge", f)
        states = iter([RuntimeError("首次失败"), asyncio.TimeoutError])

        def fake_wait_for(awaitable, timeout=None):
            awaitable.close()  # 关闭未 await 的协程，防 RuntimeWarning
            raise next(states)

        monkeypatch.setattr("agent.tool_registry.asyncio.wait_for", fake_wait_for)
        result = asyncio.run(tool.run({}, None))
        assert result == "(工具 search_knowledge 执行超时)"

    def test_retry_budget_locked_in_react_loop(self, monkeypatch):
        # AC-11 预算锁定：重试发生在 AgentTool.run 内部，对 react_loop 完全不可见——
        # 工具首败后重试成功 → tool_count==1 / 1 个 tool_call 事件 / 消息历史 1 条
        # tool 结果 / record_tool_call 只调 1 次（tool_call_logs 只记最终结果）
        monkeypatch.setattr(settings, "tool_auto_retry", True)
        calls = {"n": 0}

        async def flaky(ctx, args):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("瞬时抖动")
            return "检索结果 ok"

        reg = ToolRegistry()
        reg.register("search_knowledge", "混合检索", {"type": "object"}, flaky)
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "问题"}),
            _answer("最终答案"),
        ])
        with mock.patch("agent.react.LLMFactory.get_client", return_value=fake), \
             mock.patch("agent.react.record_tool_call",
                        new=mock.AsyncMock()) as rec:
            ctx = ReactContext("问题")
            messages = [{"role": "user", "content": "问题"}]

            async def run():
                return [evt async for evt in react_loop(ctx, messages, budget=2, tools=reg)]

            events = asyncio.run(run())
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        done = [e for e in events if e["type"] == "done"][-1]
        assert calls["n"] == 2                       # 首败 + 重试成功
        assert len(tool_calls) == 1                  # 无第二个 tool_call 事件
        assert done["tool_count"] == 1               # 预算不因重试增加
        assert len([m for m in messages if m.get("role") == "tool"]) == 1
        assert rec.await_count == 1                  # tool_call_logs 只记 1 次

# ══════════════════════════════════════════════════════════════════
# 2026-08-20 追加：执行层 schema 守门 + 去个人化 + _CHECK_PROMPT 静态前置
# ══════════════════════════════════════════════════════════════════

class TestPhaseGate:
    """执行层 schema 守门（066 实测"执行层不校验 schema 暴露"漏洞闭环）"""

    @staticmethod
    def _tool(name: str, f) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(name, f"工具 {name}", {"type": "object"}, f)
        return reg

    def test_phase_gate_rejects_generation_tool_in_retrieval(self, monkeypatch):
        # 检索阶段调 generate_answer（schema 外）→ 拒绝 + 可读提示
        from agent.react import execute_tool_with_log
        monkeypatch.setattr(settings, "tool_phase_split", True)
        called = {"n": 0}

        async def fake_gen(ctx, args):
            called["n"] += 1
            return "不应执行"

        reg = ToolRegistry()
        reg.register("generate_answer", "生成答案", {"type": "object"}, fake_gen)
        tool = reg.get("generate_answer")
        ctx = ReactContext("问题")
        result = asyncio.run(execute_tool_with_log("generate_answer", {}, tool, ctx))
        assert called["n"] == 0                      # 未执行
        assert "当前阶段不可用" in result            # 可读提示

    def test_phase_gate_allows_retrieval_tool(self, monkeypatch):
        # 检索阶段调 search_knowledge（schema 内）→ 正常执行
        from agent.react import execute_tool_with_log
        monkeypatch.setattr(settings, "tool_phase_split", True)
        called = {"n": 0}

        async def fake_search(ctx, args):
            called["n"] += 1
            return "检索结果"

        reg = self._tool("search_knowledge", fake_search)
        tool = reg.get("search_knowledge")
        ctx = ReactContext("问题")
        result = asyncio.run(execute_tool_with_log("search_knowledge", {}, tool, ctx))
        assert called["n"] == 1
        assert result == "检索结果"

    def test_phase_gate_disabled_allows_all(self, monkeypatch):
        # tool_phase_split=false → 全放行（零回归逃生口）
        from agent.react import execute_tool_with_log
        monkeypatch.setattr(settings, "tool_phase_split", False)
        called = {"n": 0}

        async def fake_gen(ctx, args):
            called["n"] += 1
            return "执行了"

        reg = ToolRegistry()
        reg.register("generate_answer", "生成答案", {"type": "object"}, fake_gen)
        tool = reg.get("generate_answer")
        ctx = ReactContext("问题")
        result = asyncio.run(execute_tool_with_log("generate_answer", {}, tool, ctx))
        assert called["n"] == 1
        assert result == "执行了"


class TestPromptHygiene:
    """去个人化 + 提示词结构（缓存友好静态前置）"""

    def test_no_personal_info_in_prompts(self):
        # 生产 prompt 不含姓名/个人网站标识（2026-08-20 去个人化）
        from agent.react import _SYSTEM_PROMPT
        from agent.reflector import _CHECK_PROMPT, _GENERATE_PROMPT
        from rag.engine import _HYDE_PROMPT
        for p in (_SYSTEM_PROMPT, _CHECK_PROMPT, _GENERATE_PROMPT, _HYDE_PROMPT):
            assert "熊艺诚" not in p
            assert "个人网站" not in p

    def test_check_prompt_static_before_dynamic(self):
        # 缓存友好结构：静态段（角色/步骤/规则/示例）全在动态变量之前
        from agent.reflector import _CHECK_PROMPT
        q_pos = _CHECK_PROMPT.index("{query}")
        d_pos = _CHECK_PROMPT.index("{docs_summary}")
        assert q_pos < d_pos                          # query 在 docs 前
        assert _CHECK_PROMPT.index("判断步骤") < q_pos   # 静态步骤在动态前
        assert _CHECK_PROMPT.index("规则（严格遵守）") < q_pos
        assert _CHECK_PROMPT.index("示例 1（充分）") < q_pos
        assert _CHECK_PROMPT.index("只返回 JSON") < q_pos
