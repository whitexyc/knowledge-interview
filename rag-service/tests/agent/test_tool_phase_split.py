"""module-058 WP-E：工具阶段切分状态机测试（ADR-0012 方案 A，原 module-059）

覆盖（验收 §3 功能验收 + §7 测试验收）：
- 检索组 schema 恰好 7 个且不含 generate_answer/verify_answer
- 生成组 schema 恰好 4 个（generate_answer/verify_answer/note_to_self + re_search 双组）
- 调 generate_answer 后下一轮 schema = 生成组 4（react + langgraph 两条循环）
- generation 内调 re_search 后仍 generation（单向前进，不回退）
- 调 verify_answer 同样切 generation
- 开关 false 回退全量 10（零回归逃生口）
- 预算=0 / 预算耗尽兜底路径行为不变
- to_llm_schemas(group=None) 默认全量 10（test_agent_tools.py:94 不挂）

实现说明：
- 测试内显式 setattr settings.tool_phase_split=True（conftest autouse fixture
  默认钉住 false，存量测试零漂移；本文件显式开启验证切分）
- mock 打桩 LLMFactory.get_client（脚本化假 LLM，记录每轮 tools schema），
  asyncio.run 执行，不依赖 pytest-asyncio
"""
import asyncio
import json
from unittest import mock

from agent.tool_registry import registry, ToolRegistry
from agent.react import ReactContext, advance_phase, schemas_for_phase, react_agent
from agent.langgraph_react import langgraph_react_agent
from src.config import settings

RETRIEVAL_7 = {
    "search_knowledge", "search_fts", "search_vector", "search_graph",
    "extract_entities", "recall_memory", "re_search",
}
GENERATION_4 = {"generate_answer", "verify_answer", "note_to_self", "re_search"}
ALL_10 = RETRIEVAL_7 | GENERATION_4


def _tool_call(name: str, args: dict, cid: str = "c1") -> dict:
    """脚本化的 tool_call 响应（含 assistant message，供循环追加回传）"""
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


class TestToolGroups:
    """ToolRegistry 阶段归组 + to_llm_schemas(group=...) 过滤"""

    def test_retrieval_group_exactly_7(self):
        """检索阶段 schema 恰好 7 个且不含 generate_answer/verify_answer"""
        schemas = registry.to_llm_schemas(group="retrieval")
        assert len(schemas) == 7
        names = {s["function"]["name"] for s in schemas}
        assert names == RETRIEVAL_7

    def test_generation_group_exactly_4(self):
        """生成阶段 schema 恰好 4 个（含 re_search 双组）"""
        schemas = registry.to_llm_schemas(group="generation")
        assert len(schemas) == 4
        names = {s["function"]["name"] for s in schemas}
        assert names == GENERATION_4

    def test_group_none_full_10(self):
        """to_llm_schemas() 无参调用仍全量 10（存量 test_agent_tools 不挂）"""
        assert len(registry.to_llm_schemas()) == 10

    def test_builtin_tool_group_metadata(self):
        """10 个工具的 name/description/args_schema 一字不改（只新增 group）"""
        for t in registry.list_tools():
            assert t.name in ALL_10
            assert t.description
            assert isinstance(t.args_schema, dict)
            assert t.group  # 内置工具均已归组

    def test_ungrouped_tool_visible_in_both_phases(self):
        """未分组工具（测试自定义）恒全阶段可见（向后兼容）"""
        reg = ToolRegistry()

        async def f(ctx, args):
            return "x"

        reg.register("custom", "自定义", {"type": "object"}, f)
        assert len(reg.to_llm_schemas(group="retrieval")) == 1
        assert len(reg.to_llm_schemas(group="generation")) == 1


class TestPhaseStateMachine:
    """ctx.phase 状态机（react_loop 手写循环）"""

    def _patch_retriever(self, docs):
        return mock.patch(
            "agent.tool_registry.hybrid_retriever.retrieve",
            new=mock.AsyncMock(return_value=docs),
        )

    def test_initial_phase_retrieval(self):
        """ctx.phase 初始为 retrieval"""
        ctx = ReactContext("q", "u1", [])
        assert ctx.phase == "retrieval"

    def test_advance_phase_unit(self):
        """advance_phase：仅生成工具触发切换；re_search 不触发；不回退"""
        ctx = ReactContext("q", "u1", [])
        advance_phase(ctx, ["search_knowledge"])
        assert ctx.phase == "retrieval"
        advance_phase(ctx, ["re_search"])  # 检索阶段 re_search 不触发
        assert ctx.phase == "retrieval"
        advance_phase(ctx, ["generate_answer"])
        assert ctx.phase == "generation"
        advance_phase(ctx, ["re_search"])  # generation 内 re_search 不回退
        assert ctx.phase == "generation"

    def test_schemas_for_phase_switch_true(self):
        """schemas_for_phase：开关 true 时按阶段过滤"""
        ctx = ReactContext("q", "u1", [])
        with mock.patch("agent.react.settings.tool_phase_split", True):
            names = {s["function"]["name"] for s in schemas_for_phase(registry, ctx)}
        assert names == RETRIEVAL_7

    def test_react_loop_retrieval_schema_7_then_generation_4(self):
        """调 generate_answer 后**下一轮** schema = 生成组 4（阶段按"本轮调用前"确定）"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),
            _tool_call("generate_answer", {"query": "q"}),
            _answer("最终答案"),
        ])
        with mock.patch("agent.react.settings.tool_phase_split", True):
            with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                with self._patch_retriever([_doc(1)]):
                    with mock.patch("agent.tool_registry.reflector.generate_answer",
                                    new=mock.AsyncMock(return_value="生成答案")):
                        result = asyncio.run(react_agent("q", budget=4))

        assert result["tool_count"] == 2
        assert result["answer"] == "最终答案"
        assert set(fake.tools_seen[0]) == RETRIEVAL_7   # 第 1 轮：检索阶段 7 个
        # 第 2 轮已切 generation（module-068：第 1 轮 search_knowledge 检索命中
        # → 确定性推进规则，不再等调生成工具；原"先检后生"等待语义被命中即切取代）
        assert set(fake.tools_seen[1]) == GENERATION_4
        assert set(fake.tools_seen[2]) == GENERATION_4  # 下一轮仍 generation

    def test_verify_answer_switches_to_generation(self):
        """调 verify_answer 同样切 generation（下一轮 schema = 生成组 4）"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),
            _tool_call("verify_answer", {"answer": "答案文本"}),
            _answer("答案"),
        ])
        with mock.patch("agent.react.settings.tool_phase_split", True):
            with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                with self._patch_retriever([_doc(1)]):
                    with mock.patch("agent.tool_registry.reflector.verify_answer",
                                    new=mock.AsyncMock(return_value={
                                        "claims": [], "overall_confidence": 0.0})):
                        result = asyncio.run(react_agent("q", budget=4))

        assert set(fake.tools_seen[0]) == RETRIEVAL_7
        # module-068：第 1 轮 search_knowledge 检索命中 → 调用轮 schema 已为生成组
        assert set(fake.tools_seen[1]) == GENERATION_4
        assert set(fake.tools_seen[2]) == GENERATION_4  # 下一轮仍 generation

    def test_re_search_in_generation_no_regression(self):
        """generation 内调 re_search → 不回退（补检口仍在生成组）"""
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
                with self._patch_retriever([_doc(1)]):
                    with mock.patch("agent.tool_registry.reflector.generate_answer",
                                    new=mock.AsyncMock(return_value="生成答案")):
                        with mock.patch("agent.tool_registry.reflector.check_sufficiency",
                                        new=check):
                            result = asyncio.run(react_agent("q", budget=4))

        assert result["tool_count"] == 3
        # module-068：第 1 轮 search_knowledge 检索命中 → 第 2 轮已切 generation；
        # re_search 在生成组可见；补检后仍 generation（不回退）
        assert set(fake.tools_seen[1]) == GENERATION_4
        assert set(fake.tools_seen[2]) == GENERATION_4
        assert set(fake.tools_seen[3]) == GENERATION_4
        assert check.called

    def test_switch_false_full_10_zero_regression(self):
        """开关 false（conftest 默认钉住）→ 全量 10 个 schema"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),
            _answer("答案"),
        ])
        # 不显式开 true：conftest autouse fixture 已钉住 false
        with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
            with self._patch_retriever([_doc(1)]):
                result = asyncio.run(react_agent("q", budget=4))

        assert result["answer"] == "答案"
        assert len(fake.tools_seen[0]) == 10

    def test_budget_zero_direct_answer_no_tools(self):
        """预算=0：不调用工具、无 schema（阶段切分不改变预算路径）"""
        fake = _FakeLLM([])
        with mock.patch("agent.react.settings.tool_phase_split", True):
            with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                result = asyncio.run(react_agent("q", budget=0))

        assert result["tool_count"] == 0
        assert result["answer"] == "预算为0直接回答"
        assert len(fake.tools_seen) == 0  # chat_with_tools 从未被调用

    def test_budget_exhausted_fallback_unchanged(self):
        """预算耗尽兜底路径不变：检索阶段循环到预算耗尽 → reflector 兜底生成"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),
            _tool_call("search_fts", {"query": "q"}),
        ])
        with mock.patch("agent.react.settings.tool_phase_split", True):
            with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                with self._patch_retriever([_doc(1)]):
                    with mock.patch("agent.react.reflector.generate_answer",
                                    new=mock.AsyncMock(return_value="兜底答案")):
                        result = asyncio.run(react_agent("q", budget=2))

        assert result["tool_count"] == 2
        assert result["answer"] == "兜底答案"
        # module-068：第 1 轮 search_knowledge 检索命中 → 第 2 轮已切 generation
        #（search_fts 在生成阶段执行，执行层不校验 schema 暴露——预算耗尽兜底不变）
        assert set(fake.tools_seen[0]) == RETRIEVAL_7
        assert set(fake.tools_seen[1]) == GENERATION_4


class TestLangGraphPhaseSplit:
    """LangGraph 版循环（langgraph_react_agent）阶段切分同步改造"""

    def _patch_retriever(self, docs):
        return mock.patch(
            "agent.tool_registry.hybrid_retriever.retrieve",
            new=mock.AsyncMock(return_value=docs),
        )

    def test_langgraph_phase_switch(self):
        """langgraph_react_loop：检索阶段 7 个 → 调 generate_answer 下一轮切 generation 4"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),
            _tool_call("generate_answer", {"query": "q"}),
            _answer("LG答案"),
        ])
        with mock.patch("agent.langgraph_react.settings.tool_phase_split", True):
            with mock.patch("agent.langgraph_react.LLMFactory.get_client",
                            return_value=fake):
                with self._patch_retriever([_doc(1)]):
                    with mock.patch("agent.tool_registry.reflector.generate_answer",
                                    new=mock.AsyncMock(return_value="生成答案")):
                        result = asyncio.run(langgraph_react_agent("q", budget=4))

        assert result["answer"] == "LG答案"
        assert set(fake.tools_seen[0]) == RETRIEVAL_7
        # module-068：第 1 轮 search_knowledge 检索命中 → 调用轮 schema 已为生成组
        assert set(fake.tools_seen[1]) == GENERATION_4
        assert set(fake.tools_seen[2]) == GENERATION_4  # 下一轮仍 generation

    def test_langgraph_re_search_in_generation_no_regression(self):
        """langgraph 版 generation 内调 re_search 不回退（与手写版一致）"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),
            _tool_call("generate_answer", {"query": "q"}),
            _tool_call("re_search", {"query": "q"}),
            _answer("LG答案"),
        ])
        with mock.patch("agent.langgraph_react.settings.tool_phase_split", True):
            with mock.patch("agent.langgraph_react.LLMFactory.get_client",
                            return_value=fake):
                with self._patch_retriever([_doc(1)]):
                    with mock.patch("agent.tool_registry.reflector.generate_answer",
                                    new=mock.AsyncMock(return_value="生成答案")):
                        with mock.patch("agent.tool_registry.reflector.check_sufficiency",
                                        new=mock.AsyncMock(return_value={
                                            "sufficient": False,
                                            "rewritten_query": "q2"})):
                            result = asyncio.run(langgraph_react_agent("q", budget=4))

        assert result["tool_count"] == 3
        # module-068：第 1 轮 search_knowledge 检索命中 → 第 2 轮已切 generation；
        # re_search 在生成组可见；补检后仍 generation（不回退）
        assert set(fake.tools_seen[1]) == GENERATION_4
        assert set(fake.tools_seen[2]) == GENERATION_4
        assert set(fake.tools_seen[3]) == GENERATION_4

    def test_langgraph_switch_false_full_10(self):
        """langgraph 版开关 false → 全量 10（逃生口）"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),
            _answer("答案"),
        ])
        with mock.patch("agent.langgraph_react.LLMFactory.get_client",
                        return_value=fake):
            with self._patch_retriever([_doc(1)]):
                result = asyncio.run(langgraph_react_agent("q", budget=4))

        assert result["answer"] == "答案"
        assert len(fake.tools_seen[0]) == 10


class TestSystemPromptUnchanged:
    """阶段切分不改系统提示词（工具清单一字不改，避免存量断言漂移）"""

    def test_system_prompt_lists_all_10(self):
        from agent.react import _SYSTEM_PROMPT
        for name in sorted(ALL_10):
            assert name in _SYSTEM_PROMPT
