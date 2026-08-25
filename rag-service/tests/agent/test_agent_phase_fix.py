"""module-068：Agent 阶段推进死锁修复单测

覆盖（plan WP-A ①-⑥ + WP-B ①-⑤ + AC-8/9/10/12）：
WP-A 死锁修复：
- 检索命中（非空真实结果）→ 下一轮 schema 含 generate_answer（生成组 4）
- 3 轮未命中（空结果标记）→ 第 4 轮 schema 强制为生成组（防空转兜底）
- generation 内 re_search 不回退（带 results 参数版本）
- 原条件仍生效（旧签名单列表调用 advance_phase 直接断言，存量行为）
- _retrieval_hit 纯函数边界（空串/空结果标记/JSON 空实体/非检索工具/re_search 排除）
- langgraph 同构冒烟（检索命中 → 下一轮生成组；阶段预算截断 → fallback 路由）
WP-B 预算按阶段：
- 检索阶段累计 3 次后即使总预算剩 2 也不放检索工具（第 4 次截断 → 兜底）
- 生成阶段累计 2 次后截断（生成组内第 3 个工具被截断）
- 总预算兜底仍生效（budget=2 收紧场景，阶段预算让位）
- 开关 false 阶段预算失效（纯总预算，存量行为逐字）
- phase_count 按执行时阶段计数正确（命中切 generation 后新执行计生成）

实现说明：
- 测试内显式 setattr settings.tool_phase_split=True（conftest autouse fixture
  默认钉住 false，对齐 test_tool_phase_split 模式）；阶段预算相关测试显式传
  budget 参数（conftest 钉住 max_agent_tools=4 不影响显式传参）
- mock 打桩 LLMFactory.get_client（脚本化假 LLM，记录每轮 tools schema），
  asyncio.run 执行，不依赖 pytest-asyncio
"""
import asyncio
import json
from unittest import mock

from agent.react import (
    ReactContext, _retrieval_hit, advance_phase, react_agent, react_loop,
    _build_messages,
)
from agent.langgraph_react import langgraph_react_agent
from src.config import settings

RETRIEVAL_7 = {
    "search_knowledge", "search_fts", "search_vector", "search_graph",
    "extract_entities", "recall_memory", "re_search",
}
GENERATION_4 = {"generate_answer", "verify_answer", "note_to_self", "re_search"}


def _tool_call(name: str, args: dict, cid: str = "c1") -> dict:
    """脚本化的单工具调用响应（含 assistant message，供循环追加回传）"""
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


def _multi_tool_call(names: list[str]) -> dict:
    """脚本化的多工具调用响应（同一轮 LLM 提议多个工具，args 均为 {}）"""
    calls = [{"id": f"c{i}", "name": name, "args": {}}
             for i, name in enumerate(names)]
    return {
        "content": "",
        "tool_calls": calls,
        "message": {
            "role": "assistant", "content": "",
            "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": name, "arguments": "{}"}}
                for i, name in enumerate(names)
            ],
        },
    }


def _answer(content: str) -> dict:
    return {"content": content, "tool_calls": [],
            "message": {"role": "assistant", "content": content}}


class _FakeLLM:
    """脚本化假 LLM：按序返回 chat_with_tools 响应，并记录每轮收到的工具名"""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.tools_seen: list[list[str]] = []

    async def chat_with_tools(self, messages, tools):
        self.tools_seen.append([t["function"]["name"] for t in tools])
        return self.responses.pop(0)

    async def chat(self, messages):
        return "预算为0直接回答"


def _doc(doc_id: int = 1) -> dict:
    return {
        "id": doc_id,
        "title": f"文档{doc_id}",
        "content": f"这是文档{doc_id}的内容，涉及 Java 线程池。",
        "source": "test",
        "hybrid_score": 0.9,
    }


def _patch_retriever(docs):
    """patch hybrid_retriever.retrieve 返回固定 docs（[] → "（无检索结果）"）"""
    return mock.patch(
        "agent.tool_registry.hybrid_retriever.retrieve",
        new=mock.AsyncMock(return_value=docs),
    )


class TestRetrievalHitSwitch:
    """WP-A：检索命中即切 generation（死锁修复）"""

    def test_retrieval_hit_switches_to_generation(self):
        """① 检索命中（非空 docs）→ 下一轮 schema 含 generate_answer（生成组 4）"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),
            _answer("最终答案"),
        ])
        with mock.patch("agent.react.settings.tool_phase_split", True):
            with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                with _patch_retriever([_doc(1)]):
                    result = asyncio.run(react_agent("q", budget=5))

        assert result["tool_count"] == 1
        assert result["answer"] == "最终答案"
        assert set(fake.tools_seen[0]) == RETRIEVAL_7   # 第 1 轮：检索阶段 7 个
        assert set(fake.tools_seen[1]) == GENERATION_4  # 命中 → 下一轮生成组 4

    def test_retrieval_max_rounds_force_switch(self):
        """② 3 轮未命中（空结果标记）→ 第 4 轮 schema 强制为生成组（兜底生效）"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),
            _tool_call("search_fts", {"query": "q"}),
            _tool_call("search_vector", {"query": "q"}),
            _answer("答案"),
        ])
        with mock.patch("agent.react.settings.tool_phase_split", True):
            with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                with _patch_retriever([]):  # 空检索结果 → "（无检索结果）" 不算命中
                    result = asyncio.run(react_agent("q", budget=5))

        assert result["tool_count"] == 3
        assert set(fake.tools_seen[0]) == RETRIEVAL_7
        assert set(fake.tools_seen[1]) == RETRIEVAL_7
        assert set(fake.tools_seen[2]) == RETRIEVAL_7
        assert set(fake.tools_seen[3]) == GENERATION_4  # 第 4 轮强制生成组

    def test_retrieval_max_rounds_parameterized(self):
        """② 阈值参数化：agent_retrieval_max_rounds 可覆盖（1 轮未命中即强制切）"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),
            _answer("答案"),
        ])
        with mock.patch("agent.react.settings.tool_phase_split", True):
            with mock.patch("agent.react.settings.agent_retrieval_max_rounds", 1):
                with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                    with _patch_retriever([]):
                        result = asyncio.run(react_agent("q", budget=5))

        assert result["tool_count"] == 1
        assert set(fake.tools_seen[0]) == RETRIEVAL_7
        assert set(fake.tools_seen[1]) == GENERATION_4  # 1 轮未命中即强制切

    def test_re_search_in_generation_no_regression_with_results(self):
        """③ generation 内调 re_search 不回退（带 results 参数版本）"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),
            _tool_call("generate_answer", {"query": "q"}),
            _tool_call("re_search", {"query": "q"}),  # 生成后补检
            _answer("答案"),
        ])
        check = mock.AsyncMock(return_value={
            "sufficient": False, "rewritten_query": "q2"})
        with mock.patch("agent.react.settings.tool_phase_split", True):
            with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                with _patch_retriever([_doc(1)]):
                    with mock.patch("agent.tool_registry.reflector.generate_answer",
                                    new=mock.AsyncMock(return_value="生成答案")):
                        with mock.patch("agent.tool_registry.reflector.check_sufficiency",
                                        new=check):
                            result = asyncio.run(react_agent("q", budget=5))

        assert result["tool_count"] == 3
        assert set(fake.tools_seen[0]) == RETRIEVAL_7   # 第 1 轮：检索
        assert set(fake.tools_seen[1]) == GENERATION_4  # 第 1 轮命中已切
        assert set(fake.tools_seen[2]) == GENERATION_4
        assert set(fake.tools_seen[3]) == GENERATION_4  # 补检后仍 generation 不回退
        assert check.called

    def test_advance_phase_old_signature_backward_compat(self):
        """④ 旧签名单列表调用 = 旧行为（仅生成工具判定，results=None 不判命中）"""
        ctx = ReactContext("q", "u1", [])
        advance_phase(ctx, ["search_knowledge"])
        assert ctx.phase == "retrieval"    # 检索工具不触发（且无 results 不判命中）
        assert ctx.retrieval_rounds == 1   # 兜底计数递增（< 默认 3 不强制切）
        advance_phase(ctx, ["re_search"])
        assert ctx.phase == "retrieval"    # re_search 不触发
        advance_phase(ctx, ["generate_answer"])
        assert ctx.phase == "generation"   # 原条件（生成工具）仍生效
        advance_phase(ctx, ["re_search"])
        assert ctx.phase == "generation"   # 不回退

    def test_advance_phase_hit_condition_with_results(self):
        """④ 提供 results 时：检索命中 → 切 generation（新分支）"""
        ctx = ReactContext("q", "u1", [])
        advance_phase(ctx, ["search_knowledge"],
                      ["[1] 文档1 (score=0.9)\n这是文档1的内容。"])
        assert ctx.phase == "generation"

    def test_retrieval_hit_pure_function_boundaries(self):
        """⑤ _retrieval_hit 边界：空串/空结果标记/JSON 空实体/非检索工具/re_search 排除"""
        # 空结果标记均为非空字符串——bool(result) 误判命中坑必须排除
        assert _retrieval_hit("search_knowledge", "（无检索结果）") is False
        assert _retrieval_hit("recall_memory", "（无相关历史记忆）") is False
        # 工具执行失败空串不算命中
        assert _retrieval_hit("search_knowledge", "") is False
        # extract_entities JSON entities 空 → 不算命中
        assert _retrieval_hit("extract_entities", '{"entities": []}') is False
        # extract_entities JSON entities 非空 → 命中
        assert _retrieval_hit("extract_entities", '{"entities": ["Java"]}') is True
        # extract_entities 解析失败按非空文本判定
        assert _retrieval_hit("extract_entities", "不是JSON") is True
        # re_search（双组补检工具）不参与命中判定
        assert _retrieval_hit("re_search", "有结果文本") is False
        # 非检索工具名不参与
        assert _retrieval_hit("note_to_self", "x") is False
        assert _retrieval_hit("generate_answer", "x") is False
        # 正常非空结果 → 命中
        assert _retrieval_hit("search_knowledge",
                              "[1] 文档1 (score=0.9)\n这是文档1的内容。") is True

    def test_langgraph_retrieval_hit_switches(self):
        """⑥ langgraph 同构：检索命中 → 下一轮 schema 为生成组"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),
            _answer("LG答案"),
        ])
        with mock.patch("agent.langgraph_react.settings.tool_phase_split", True):
            with mock.patch("agent.langgraph_react.LLMFactory.get_client",
                            return_value=fake):
                with _patch_retriever([_doc(1)]):
                    result = asyncio.run(langgraph_react_agent("q", budget=5))

        assert result["answer"] == "LG答案"
        assert set(fake.tools_seen[0]) == RETRIEVAL_7
        assert set(fake.tools_seen[1]) == GENERATION_4


class TestPhaseBudget:
    """WP-B：预算按阶段（仅 tool_phase_split=true 生效）"""

    def test_retrieval_phase_budget_truncates(self):
        """① 检索阶段累计 3 次后即使总预算剩 2 也不放检索工具（第 4 次截断 → 兜底）"""
        fake = _FakeLLM([
            _multi_tool_call(["search_knowledge", "search_fts", "search_vector"]),
            _tool_call("search_knowledge", {"query": "q"}),  # 第 4 次检索：阶段截断
        ])
        with mock.patch("agent.react.settings.tool_phase_split", True):
            with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                with _patch_retriever([]):  # 空结果防命中切 generation
                    with mock.patch("agent.react.reflector.generate_answer",
                                    new=mock.AsyncMock(return_value="兜底答案")):
                        result = asyncio.run(react_agent("q", budget=5))

        assert result["tool_count"] == 3          # 检索阶段预算 3 次用尽
        assert result["answer"] == "兜底答案"       # 截断 → 兜底生成
        assert [t["name"] for t in result["tool_trace"]] == [
            "search_knowledge", "search_fts", "search_vector"]

    def test_generation_phase_budget_truncates(self):
        """② 生成阶段累计 2 次后截断（生成组内第 3 个工具被截断 → 兜底）"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),  # 命中 → 切 generation
            _multi_tool_call(["generate_answer", "verify_answer", "note_to_self"]),
            _tool_call("note_to_self", {"note": "n"}),        # 第 3 个生成工具：截断
        ])
        with mock.patch("agent.react.settings.tool_phase_split", True):
            with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                with _patch_retriever([_doc(1)]):
                    with mock.patch("agent.tool_registry.reflector.generate_answer",
                                    new=mock.AsyncMock(return_value="生成答案")):
                        with mock.patch("agent.tool_registry.reflector.verify_answer",
                                        new=mock.AsyncMock(return_value={
                                            "claims": [], "overall_confidence": 0.0})):
                            with mock.patch("agent.react.reflector.generate_answer",
                                            new=mock.AsyncMock(return_value="兜底答案")):
                                result = asyncio.run(react_agent("q", budget=5))

        assert result["tool_count"] == 3   # 检索 1 + 生成 2（第 3 个生成工具截断）
        assert result["answer"] == "兜底答案"
        assert [t["name"] for t in result["tool_trace"]] == [
            "search_knowledge", "generate_answer", "verify_answer"]

    def test_total_budget_overrides_phase_budget(self):
        """③ 总预算兜底仍生效：budget=2 收紧场景，阶段预算让位（总调用 ≤2）"""
        fake = _FakeLLM([
            _multi_tool_call(["search_knowledge", "search_fts", "search_vector"]),
            _tool_call("search_graph", {"query": "q"}),  # 第 3 次：总预算截断
        ])
        with mock.patch("agent.react.settings.tool_phase_split", True):
            with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                with _patch_retriever([]):
                    with mock.patch("agent.react.reflector.generate_answer",
                                    new=mock.AsyncMock(return_value="兜底答案")):
                        result = asyncio.run(react_agent("q", budget=2))

        assert result["tool_count"] == 2   # 总预算硬上限
        assert result["answer"] == "兜底答案"
        assert [t["name"] for t in result["tool_trace"]] == [
            "search_knowledge", "search_fts"]

    def test_phase_budget_off_when_switch_false(self):
        """④ 开关 false：阶段预算失效，纯总预算（存量行为逐字）"""
        fake = _FakeLLM([
            _multi_tool_call(["search_knowledge", "search_fts", "search_vector"]),
            _tool_call("search_knowledge", {"query": "q"}),  # 第 4 次检索：仍可执行
            _answer("答案"),
        ])
        # 不显式开 true：conftest autouse fixture 已钉住 false
        with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
            with _patch_retriever([]):
                result = asyncio.run(react_agent("q", budget=5))

        assert result["tool_count"] == 4   # 纯总预算 5，4 次检索全执行
        assert result["answer"] == "答案"
        assert len(fake.tools_seen[0]) == 10  # 全量 10 个 schema

    def test_phase_count_counts_by_execution_phase(self):
        """⑤ phase_count 按执行时阶段计数（命中切 generation 后新执行计生成）"""
        ctx = ReactContext("q", "u1", [])
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),
            _tool_call("generate_answer", {"query": "q"}),
            _tool_call("note_to_self", {"note": "n"}),
            _answer("答案"),
        ])
        events: list[dict] = []

        async def run():
            with mock.patch("agent.react.settings.tool_phase_split", True):
                with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                    with _patch_retriever([_doc(1)]):
                        with mock.patch("agent.tool_registry.reflector.generate_answer",
                                        new=mock.AsyncMock(return_value="生成答案")):
                            async for evt in react_loop(
                                    ctx, _build_messages(ctx), 5):
                                events.append(evt)

        asyncio.run(run())
        assert ctx.phase_count == {"retrieval": 1, "generation": 2}
        assert ctx.phase == "generation"
        assert events[-1]["type"] == "done"

    def test_langgraph_retrieval_phase_budget_truncates(self):
        """AC-12 两条循环同构：langgraph 检索阶段 3 次后第 4 次截断 → fallback"""
        fake = _FakeLLM([
            _multi_tool_call(["search_knowledge", "search_fts", "search_vector"]),
            _tool_call("search_knowledge", {"query": "q"}),  # 第 4 次检索：阶段截断
        ])
        with mock.patch("agent.langgraph_react.settings.tool_phase_split", True):
            with mock.patch("agent.langgraph_react.LLMFactory.get_client",
                            return_value=fake):
                with _patch_retriever([]):
                    with mock.patch("agent.langgraph_react.reflector.generate_answer",
                                    new=mock.AsyncMock(return_value="兜底答案")):
                        result = asyncio.run(langgraph_react_agent("q", budget=5))

        assert result["tool_count"] == 3
        assert result["answer"] == "兜底答案"  # phase_exhausted 路由 → fallback 不空转
