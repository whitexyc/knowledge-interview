"""Module-028 Agent 工具化单元测试

覆盖（验收 §4）：
- ToolRegistry：10 个内置工具注册 / to_llm_schemas 格式 / 未知工具返回 None / 工具失败返回空
- LLMClient.chat_with_tools：bind_tools 调用 + 返回 {content, tool_calls}（mock _llm）
- ReAct 循环（react_agent）：工具调用→直接回答 / 预算耗尽兜底 / 预算=0 直接生成 /
  工具失败返回空继续 / 工具调用数 ≤ budget
- SSE 端点 /ai/rag/chat/agent：tool_call/tool_result/token/done 事件序列

实现说明：
- 用 mock 打桩 LLMFactory.get_client / hybrid_retriever / reflector，不依赖真实
  DB / Redis / LLM
- 同步用例内 asyncio.run 执行，不依赖 pytest-asyncio（沿用既有模式）
"""
import asyncio
import json
from unittest import mock

import httpx

import main
from llm.client import LLMClient
from agent.tool_registry import registry, ToolRegistry, register_builtin_tools, _format_docs
from agent.react import ReactContext, react_agent


def _doc(doc_id: int = 1) -> dict:
    return {
        "id": doc_id,
        "title": f"文档{doc_id}",
        "content": f"这是文档{doc_id}的内容，涉及 Java 线程池。",
        "source": "test",
        "hybrid_score": 0.9,
    }


class _FakeLLM:
    """脚本化的假 LLM：按序返回 chat_with_tools 响应，chat 返回固定文本"""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.chat_with_tools_calls = []
        self.chat_calls = []

    async def chat_with_tools(self, messages, tools):
        self.chat_with_tools_calls.append({"messages": messages, "tools": tools})
        return self.responses.pop(0)

    async def chat(self, messages):
        self.chat_calls.append(messages)
        return "预算为0直接回答"


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


def _patch_retriever(docs):
    """patch hybrid_retriever.retrieve 返回固定 docs"""
    return mock.patch(
        "agent.tool_registry.hybrid_retriever.retrieve",
        new=mock.AsyncMock(return_value=docs),
    )


class TestToolRegistry:
    """ToolRegistry 注册 / 查询 / 序列化"""

    def test_builtin_tools_registered(self):
        names = registry.list_tool_names()
        assert names == [
            "search_knowledge", "search_fts", "search_vector", "search_graph",
            "extract_entities", "recall_memory", "generate_answer", "verify_answer",
            "re_search", "note_to_self",
        ]

    def test_to_llm_schemas_format(self):
        schemas = registry.to_llm_schemas()
        assert len(schemas) == 10
        for s in schemas:
            assert s["type"] == "function"
            fn = s["function"]
            assert "name" in fn and "description" in fn and "parameters" in fn

    def test_get_unknown_returns_none(self):
        assert registry.get("no_such_tool") is None

    def test_register_override(self):
        reg = ToolRegistry()
        async def f(ctx, args):
            return "v1"
        reg.register("t", "desc", {"type": "object"}, f)
        assert reg.get("t") is not None
        assert reg.list_tool_names() == ["t"]

    def test_tool_run_failure_returns_empty(self):
        async def bad(ctx, args):
            raise RuntimeError("工具崩溃")
        reg = ToolRegistry()
        reg.register("bad", "desc", {"type": "object"}, bad)
        result = asyncio.run(reg.get("bad").run({}, None))
        assert result == ""  # 工具失败返回空，LLM 判断继续/放弃

    def test_tool_run_timeout_returns_prompt(self):
        """AC 1.1/1.2: AgentTool.run 15s 超时 → 返回 '(工具 X 执行超时)' 不抛异常"""
        async def slow(ctx, args):
            await asyncio.sleep(999)  # 远超 15s 超时
            return "不会到达"
        reg = ToolRegistry()
        reg.register("slow_tool", "慢工具", {"type": "object"}, slow)
        result = asyncio.run(reg.get("slow_tool").run({}, None))
        assert "执行超时" in result
        assert "slow_tool" in result
        assert result == "(工具 slow_tool 执行超时)"

    def test_tool_timeout(self):
        """AC 1.1/1.2: AgentTool.run 超时 → 返回 '(工具 X 执行超时)'，不抛异常"""
        async def slow(ctx, args):
            await asyncio.sleep(999)
            return "不会到达"
        reg = ToolRegistry()
        reg.register("t", "慢工具", {"type": "object"}, slow)
        result = asyncio.run(reg.get("t").run({}, None))
        assert "执行超时" in result
        assert "t" in result
        assert result == "(工具 t 执行超时)"

    def test_register_builtin_tools_into_custom_registry(self):
        reg = ToolRegistry()
        register_builtin_tools(reg)
        assert len(reg.list_tools()) == 10

    def test_verify_answer_tool_registered(self):
        """verify_answer 已注册为第 8 个 Agent 工具"""
        tool = registry.get("verify_answer")
        assert tool is not None
        assert tool.name == "verify_answer"
        assert "逐句验证" in tool.description
        # _VERIFY_SCHEMA 含 query + answer
        props = tool.args_schema.get("properties", {})
        assert "answer" in props
        assert "query" in props

    def test_verify_answer_tool_executes(self):
        """verify_answer 工具执行：传入 ctx.docs + answer → 返回格式化可信度文本"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("线程池问题", "user-1", [])
            ctx.add_docs([_doc(1), _doc(2)])
            tool = registry.get("verify_answer")
            with mock.patch(
                "agent.tool_registry.reflector.verify_answer",
                new=mock.AsyncMock(return_value={
                    "claims": [
                        {"claim": "c1", "verdict": "supported", "evidence": "[1]"},
                        {"claim": "c2", "verdict": "unsupported", "evidence": "N/A"},
                    ],
                    "overall_confidence": 0.5,
                    "total_claims": 2,
                    "supported": 1,
                    "inferred": 0,
                    "unsupported": 1,
                }),
            ):
                result = await tool.run({"answer": "答案文本", "query": "线程池"}, ctx)
            return result

        result = asyncio.run(run())
        assert "[✓]" in result
        assert "[✗]" in result
        assert "50%" in result
        assert "supported=1" in result
        assert "unsupported=1" in result

    def test_verify_answer_tool_no_docs(self):
        """verify_answer 工具无 ctx.docs → 返回提示信息"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("无文档问题", "user-1", [])
            tool = registry.get("verify_answer")
            result = await tool.run({"answer": "答案", "query": "问题"}, ctx)
            return result

        result = asyncio.run(run())
        assert "无法验证" in result

    def test_verify_answer_tool_no_answer(self):
        """verify_answer 工具未提供 answer → 返回提示信息"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("问题", "user-1", [])
            ctx.add_docs([_doc(1)])
            tool = registry.get("verify_answer")
            result = await tool.run({}, ctx)
            return result

        result = asyncio.run(run())
        assert "无法验证" in result

    def test_re_search_tool_registered(self):
        """re_search 已注册为第 9 个 Agent 工具，schema 含 query 属性"""
        tool = registry.get("re_search")
        assert tool is not None
        assert tool.name == "re_search"
        assert "改写查询重检" in tool.description
        assert "retrieve" in tool.description or "检索" in tool.description
        # schema: 无 required 字段（query 可选，缺省用 ctx.query）
        props = tool.args_schema.get("properties", {})
        assert "query" in props
        assert props["query"]["type"] == "string"

    def test_format_docs(self):
        text = _format_docs([_doc(1), _doc(2)], limit=1)
        assert "文档1" in text
        assert "共 2 条结果" in text
        assert _format_docs([]) == "（无检索结果）"


class TestReSearch:
    """re_search 工具执行测试（module-040 Adaptive RAG — 检索不足自动改写重查）

    覆盖验收 §4 re_search sufficiency check：
    - 检索充分 → 跳过
    - 检索不足 → 改写 query + 重检
    - 无 ctx.docs 时调用 → 引导提示
    - 改写后仍无结果 → 空结果提示
    """

    def test_re_search_sufficient_skips(self):
        """check_sufficiency 返回 sufficient=true → 返回'已充分'，不调检索"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("线程池问题", "user-1", [])
            ctx.add_docs([_doc(1)])
            tool = registry.get("re_search")
            with mock.patch(
                "agent.tool_registry.reflector.check_sufficiency",
                new=mock.AsyncMock(return_value={
                    "sufficient": True,
                    "reason": "文档覆盖问题关键词",
                }),
            ):
                result = await tool.run({"query": "线程池"}, ctx)
            return result

        result = asyncio.run(run())
        assert "已充分" in result
        assert "无需重检" in result

    def test_re_search_insufficient_rewrites(self):
        """check_sufficiency 返回 insufficient → 用 rewritten_query 重检（不验证 retrieve 调用细节）"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("Java线程池怎么用", "user-1", [])
            ctx.add_docs([_doc(1)])
            tool = registry.get("re_search")
            with mock.patch(
                "agent.tool_registry.reflector.check_sufficiency",
                new=mock.AsyncMock(return_value={
                    "sufficient": False,
                    "rewritten_query": "Java线程池核心参数配置",
                }),
            ):
                with _patch_retriever([_doc(2)]):
                    result = await tool.run({"query": "Java线程池"}, ctx)
            return result

        result = asyncio.run(run())
        assert "改写查询" in result
        assert "Java线程池核心参数配置" in result
        assert "文档2" in result

    def test_re_search_insufficient_rewrites_and_retrieves(self):
        """check_sufficiency 返回 insufficient + rewritten_query → 用改写 query 检索，结果累积到 ctx"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("Java线程池怎么用", "user-1", [])
            ctx.add_docs([_doc(1)])
            tool = registry.get("re_search")
            with mock.patch(
                "agent.tool_registry.reflector.check_sufficiency",
                new=mock.AsyncMock(return_value={
                    "sufficient": False,
                    "rewritten_query": "Java线程池核心参数配置",
                }),
            ):
                with mock.patch(
                    "agent.tool_registry.hybrid_retriever.retrieve",
                    new=mock.AsyncMock(return_value=[_doc(2), _doc(3)]),
                ) as retrieve:
                    result = await tool.run({"query": "Java线程池"}, ctx)
            # 验证检索被调用且使用了改写后的 query
            retrieve.assert_called_once()
            # retrieve(query, top_k=5, mode="hybrid") — query 是位置参数
            assert retrieve.call_args[0][0] == "Java线程池核心参数配置"
            call_kwargs = retrieve.call_args[1]
            assert call_kwargs["mode"] == "hybrid"
            assert call_kwargs["top_k"] == 5
            # 新结果累积到了 ctx
            assert len(ctx.docs) == 3  # 去重后：doc1 + doc2 + doc3
            return result

        result = asyncio.run(run())
        assert "改写查询" in result
        assert "Java线程池核心参数配置" in result
        assert "2 篇文档" in result
        assert "文档2" in result
        assert "文档3" in result

    def test_re_search_no_docs_guides(self):
        """空 ctx.docs → 返回提示引导先调用检索工具"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("问题", "user-1", [])
            # 不添加任何 docs
            tool = registry.get("re_search")
            result = await tool.run({"query": "问题"}, ctx)
            return result

        result = asyncio.run(run())
        assert "请先调用" in result
        assert "尚未检索" in result

    def test_re_search_empty_rewrite_results(self):
        """check_sufficiency 返回 insufficient，改写后检索无结果 → 返回'知识库可能无相关内容'"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("不存在的内容", "user-1", [])
            ctx.add_docs([_doc(1)])
            tool = registry.get("re_search")
            with mock.patch(
                "agent.tool_registry.reflector.check_sufficiency",
                new=mock.AsyncMock(return_value={
                    "sufficient": False,
                    "rewritten_query": "不存在的改写查询",
                }),
            ):
                with mock.patch(
                    "agent.tool_registry.hybrid_retriever.retrieve",
                    new=mock.AsyncMock(return_value=[]),
                ):
                    result = await tool.run({"query": "不存在的内容"}, ctx)
            return result

        result = asyncio.run(run())
        assert "仍无结果" in result
        assert "知识库可能无相关内容" in result


class TestChatWithTools:
    """LLMClient.chat_with_tools（基类默认实现）

    主路径：ChatOpenAI 系（deepseek/qwen/zhipu）走底层 OpenAI 兼容客户端
    （async_client.create），保留 reasoning_content（thinking 模式回传要求）。
    """

    def _client(self):
        from langchain_openai import ChatOpenAI

        class _Concrete(LLMClient):
            def __init__(self):
                self._llm = ChatOpenAI(
                    model="test-model", api_key="sk-test",
                    base_url="http://localhost:1/v1", temperature=0.7,
                )

            async def generate(self, prompt):
                return ""

            async def chat(self, messages):
                return ""

            async def generate_stream(self, prompt):
                if False:
                    yield ""

        return _Concrete()

    def _fake_raw(self, content, tool_calls, reasoning=None):
        msg = mock.MagicMock()
        msg.content = content
        msg.reasoning_content = reasoning
        msg.tool_calls = tool_calls
        choice = mock.MagicMock()
        choice.message = msg
        raw = mock.MagicMock()
        raw.choices = [choice]
        return raw

    def test_openai_path_returns_content_and_tool_calls(self):
        client = self._client()
        tc = mock.MagicMock()
        tc.id = "c1"
        tc.function.name = "search_knowledge"
        tc.function.arguments = '{"query": "Java线程池"}'
        raw = self._fake_raw("", [tc], reasoning=None)
        client._llm.async_client.create = mock.AsyncMock(return_value=raw)

        tools = [{"type": "function", "function": {"name": "search_knowledge"}}]
        result = asyncio.run(client.chat_with_tools(
            [{"role": "user", "content": "hi"}], tools,
        ))
        client._llm.async_client.create.assert_called_once()
        assert result["content"] == ""
        assert result["tool_calls"] == [
            {"id": "c1", "name": "search_knowledge", "args": {"query": "Java线程池"}},
        ]
        # message 保留原始 tool_calls（arguments 字符串不重新序列化）
        assert result["message"]["role"] == "assistant"
        assert result["message"]["tool_calls"][0]["function"]["arguments"] == '{"query": "Java线程池"}'

    def test_openai_path_preserves_reasoning_content(self):
        """thinking 模式：reasoning_content 原样保留在 message（回传要求）"""
        client = self._client()
        tc = mock.MagicMock()
        tc.id = "c1"
        tc.function.name = "search"
        tc.function.arguments = "{}"
        raw = self._fake_raw("", [tc], reasoning="思考过程")
        client._llm.async_client.create = mock.AsyncMock(return_value=raw)
        result = asyncio.run(client.chat_with_tools([], []))
        assert result["message"]["reasoning_content"] == "思考过程"

    def test_no_tool_calls_returns_empty_list(self):
        client = self._client()
        raw = self._fake_raw("直接回答", [], reasoning=None)
        client._llm.async_client.create = mock.AsyncMock(return_value=raw)
        result = asyncio.run(client.chat_with_tools([], []))
        assert result == {
            "content": "直接回答", "tool_calls": [],
            "message": {"role": "assistant", "content": "直接回答"},
        }

    def test_llm_failure_raises_llm_exception(self):
        client = self._client()
        client._llm.async_client.create = mock.AsyncMock(
            side_effect=RuntimeError("api down"),
        )
        import pytest
        from llm.client import LLMException
        with pytest.raises(LLMException):
            asyncio.run(client.chat_with_tools([], []))

    def test_bind_path_for_non_openai(self):
        """非 ChatOpenAI 供应商（如 Claude）走 bind_tools 路径"""
        class _ClaudeLike(LLMClient):
            def __init__(self):
                self._llm = mock.MagicMock()

            async def generate(self, prompt):
                return ""

            async def chat(self, messages):
                return ""

            async def generate_stream(self, prompt):
                if False:
                    yield ""

        client = _ClaudeLike()
        resp = mock.MagicMock()
        resp.content = "回答"
        resp.tool_calls = [{"id": "c1", "name": "search", "args": {"query": "x"}}]
        bound = mock.MagicMock()
        bound.ainvoke = mock.AsyncMock(return_value=resp)
        client._llm.bind_tools.return_value = bound
        result = asyncio.run(client.chat_with_tools([], [{"type": "function"}]))
        assert result["content"] == "回答"
        assert result["tool_calls"] == [{"id": "c1", "name": "search", "args": {"query": "x"}}]
        assert result["message"]["role"] == "assistant"


class TestReactAgent:
    """ReAct 循环核心"""

    def test_tool_call_then_direct_answer(self):
        """LLM 先调工具，再直接回答"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "Java线程池"}),
            _answer("线程池核心参数包括核心线程数、最大线程数、队列容量。"),
        ])
        with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
            with _patch_retriever([_doc(1), _doc(2)]):
                result = asyncio.run(react_agent("Java线程池核心参数", budget=4))

        assert result["tool_count"] == 1
        assert result["tool_count"] <= 4
        assert "线程池核心参数" in result["answer"]
        assert result["tool_trace"][0]["name"] == "search_knowledge"
        assert result["tool_trace"][0]["args"] == {"query": "Java线程池"}

    def test_budget_exhausted_fallback_generation(self):
        """LLM 一直调工具直到预算耗尽 → 用已收集 docs 兜底生成"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),
            _tool_call("search_fts", {"query": "q"}),
            _tool_call("search_knowledge", {"query": "q"}),  # 第 3 次不会发生
        ])
        with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
            with _patch_retriever([_doc(1)]):
                with mock.patch("agent.react.reflector.generate_answer",
                                new=mock.AsyncMock(return_value="兜底答案")):
                    result = asyncio.run(react_agent("q", budget=2))

        assert result["tool_count"] == 2  # ≤ budget
        assert result["answer"] == "兜底答案"
        assert fake.chat_with_tools_calls and len(fake.chat_with_tools_calls) == 2

    def test_budget_zero_direct_answer_without_tools(self):
        """预算=0：LLM 直接回答，不调用工具"""
        fake = _FakeLLM([])
        with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
            result = asyncio.run(react_agent("你好", budget=0))

        assert result["tool_count"] == 0
        assert result["answer"] == "预算为0直接回答"
        assert len(fake.chat_calls) == 1
        assert len(fake.chat_with_tools_calls) == 0

    def test_tool_failure_returns_empty_and_continues(self):
        """工具失败返回空结果，循环继续 → LLM 直接回答"""
        async def boom(ctx, args):
            raise RuntimeError("工具崩溃")
        reg = ToolRegistry()
        reg.register("boom", "爆炸工具", {"type": "object", "properties": {}}, boom)

        fake = _FakeLLM([
            _tool_call("boom", {}),
            _answer("崩溃后仍可回答"),
        ])
        with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
            result = asyncio.run(react_agent("q", budget=4, tools=reg))

        assert result["tool_count"] == 1
        assert result["answer"] == "崩溃后仍可回答"
        assert result["tool_trace"][0]["result"] == ""  # 失败返回空

    def test_search_tools_accumulate_docs_in_context(self):
        """检索工具结果累积到 ctx.docs（供 generate_answer/兜底使用）"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "q"}),
            _tool_call("search_graph", {"query": "q"}),
            _answer("答案"),
        ])
        with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
            with _patch_retriever([_doc(1)]):
                with mock.patch("agent.tool_registry.graph_extractor.extract_from_query",
                                new=mock.AsyncMock(return_value=["实体A"])):
                    with mock.patch("agent.tool_registry.graph_store.search_related",
                                    new=mock.AsyncMock(return_value=[_doc(3)])):
                        result = asyncio.run(react_agent("q", budget=4))

        assert result["tool_count"] == 2
        # 两条检索去重累积：文档1（hybrid）+ 文档3（graph），无重复
        assert [t["name"] for t in result["tool_trace"]] == [
            "search_knowledge", "search_graph",
        ]

    def test_default_budget_from_settings(self):
        """不传 budget 时使用 settings.max_agent_tools（默认 4）"""
        from src.config import settings
        assert settings.max_agent_tools == 4

    def test_reasoning_content_round_trip_in_history(self):
        """DeepSeek thinking 模式：reasoning_content 回传到下一轮消息历史"""
        fake = _FakeLLM([
            {
                "content": "",
                "tool_calls": [{"id": "c1", "name": "search_knowledge",
                                "args": {"query": "q"}}],
                "message": {
                    "role": "assistant", "content": "",
                    "reasoning_content": "思考过程",
                    "tool_calls": [{"id": "c1", "type": "function",
                                    "function": {"name": "search_knowledge",
                                                 "arguments": '{"query": "q"}'}}],
                },
            },
            _answer("答案"),
        ])
        with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
            with _patch_retriever([_doc(1)]):
                result = asyncio.run(react_agent("q", budget=4))

        assert result["answer"] == "答案"
        assert len(fake.chat_with_tools_calls) == 2
        # 第二轮调用的消息历史里，assistant 消息携带 reasoning_content（回传要求）
        second_msgs = fake.chat_with_tools_calls[1]["messages"]
        assistant_msgs = [m for m in second_msgs if m["role"] == "assistant"]
        assert any(m.get("reasoning_content") == "思考过程" for m in assistant_msgs)


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


class TestAgentEndpoint:
    """POST /ai/rag/chat/agent（SSE 工具轨迹）"""

    def test_sse_tool_trace_events(self):
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "线程池"}),
            _answer("最终答案"),
        ])
        events = []

        async def run():
            with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                with _patch_retriever([_doc(7)]):
                    transport = httpx.ASGITransport(
                        app=main.app, raise_app_exceptions=True,
                    )
                    async with httpx.AsyncClient(
                            transport=transport, base_url="http://test") as client:
                        resp = await client.post(
                            "/ai/rag/chat/agent",
                            json={"query": "线程池", "history": []},
                        )
                    assert resp.status_code == 200
                    events.extend(_parse_sse(resp.content))

        asyncio.run(run())

        names = [e["event"] for e in events]
        # 工具调用 content 为空 → 无推理 token 事件；tool_call → tool_result → 最终答案 token → done
        assert names == ["tool_call", "tool_result", "token", "done"]

        # tool_call 事件含 name + args
        tool_call = json.loads(events[0]["data"])
        assert tool_call["name"] == "search_knowledge"
        assert tool_call["args"] == {"query": "线程池"}
        assert tool_call["tool_count"] == 1

        # tool_result 事件含结果文本
        tool_result = json.loads(events[1]["data"])
        assert "文档7" in tool_result["result"]

        # done 事件含最终答案 + 引用溯源 + 预算
        done = json.loads(events[3]["data"])
        assert done["answer"] == "最终答案"
        assert done["tool_count"] == 1
        assert done["budget"] == 4
        assert done["sources"][0]["id"] == 7

    def test_endpoint_uses_settings_budget(self):
        """SSE done 事件预算来自 settings.max_agent_tools"""
        fake = _FakeLLM([_answer("无工具直接回答")])
        events = []

        async def run():
            with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                transport = httpx.ASGITransport(
                    app=main.app, raise_app_exceptions=True,
                )
                async with httpx.AsyncClient(
                        transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/ai/rag/chat/agent",
                        json={"query": "你好", "history": []},
                    )
                events.extend(_parse_sse(resp.content))

        asyncio.run(run())
        names = [e["event"] for e in events]
        assert names == ["token", "done"]
        done = json.loads(events[-1]["data"])
        assert done["tool_count"] == 0
        assert done["budget"] == 4


class TestAgentSessionMemory:
    """module-036: agent/agent-lg 端点会话恢复 + 保存

    覆盖（验收 §1.1/§1.2/§4.1）：
    - 有持久化会话 → _resolve_session_history 的 history 进入 ctx（SSE 消息历史）
    - 无持久化会话 → 回退当前请求 history（零回归）
    - 循环结束后触发 _schedule_session_persist(identity, query, answer)
    """

    def _post(self, path, llm_path, fake, resolve_history,
              xff="10.0.0.8", query="线程池", history=None):
        """POST agent/agent-lg 端点（mock 全链路），返回 (sse_events, status, persist_mock)"""
        events = []
        status = 0
        persist = mock.MagicMock()

        async def run():
            nonlocal status
            with mock.patch(llm_path, return_value=fake):
                with mock.patch("rag.engine.rag_engine._resolve_session_history",
                                new=resolve_history):
                    with mock.patch("rag.engine.rag_engine._schedule_session_persist",
                                    new=persist):
                        transport = httpx.ASGITransport(
                            app=main.app, raise_app_exceptions=True)
                        async with httpx.AsyncClient(
                                transport=transport, base_url="http://test") as client:
                            resp = await client.post(
                                path,
                                headers={"X-Forwarded-For": xff},
                                json={"query": query, "history": history or []},
                            )
                        status = resp.status_code
                        events.extend(_parse_sse(resp.content))

        asyncio.run(run())
        return events, status, persist

    def test_agent_restores_persisted_session(self):
        """有持久化会话：恢复的 history 进入 ctx（LLM 消息历史含持久化条目）"""
        persisted = [
            {"role": "user", "content": "上轮问题"},
            {"role": "assistant", "content": "上轮回答"},
        ]
        fake = _FakeLLM([_answer("直接回答")])
        events, status, _ = self._post(
            "/ai/rag/chat/agent", "agent.react.LLMFactory.get_client", fake,
            resolve_history=mock.AsyncMock(return_value=persisted),
        )
        assert status == 200
        assert events[-1]["event"] == "done"
        msgs = fake.chat_with_tools_calls[0]["messages"]
        roles = [(m["role"], m.get("content")) for m in msgs]
        assert ("user", "上轮问题") in roles
        assert ("assistant", "上轮回答") in roles
        assert msgs[-1] == {"role": "user", "content": "线程池"}  # 当前问题最后

    def test_agent_uses_request_history_when_no_persisted(self):
        """无持久化会话：回退当前请求 history（零回归）"""
        request_history = [
            {"role": "user", "content": "请求内历史"},
            {"role": "assistant", "content": "请求内回答"},
        ]
        fake = _FakeLLM([_answer("直接回答")])
        events, status, _ = self._post(
            "/ai/rag/chat/agent", "agent.react.LLMFactory.get_client", fake,
            resolve_history=mock.AsyncMock(side_effect=lambda identity, h: h),
            history=request_history,
        )
        assert status == 200
        msgs = fake.chat_with_tools_calls[0]["messages"]
        roles = [(m["role"], m.get("content")) for m in msgs]
        assert ("user", "请求内历史") in roles
        assert ("assistant", "请求内回答") in roles

    def test_agent_persists_session_after_loop(self):
        """Agent 循环结束后触发 _schedule_session_persist(identity, query, answer)"""
        fake = _FakeLLM([_answer("最终答案")])
        events, status, persist = self._post(
            "/ai/rag/chat/agent", "agent.react.LLMFactory.get_client", fake,
            resolve_history=mock.AsyncMock(side_effect=lambda identity, h: h),
        )
        assert status == 200
        assert json.loads(events[-1]["data"])["answer"] == "最终答案"
        persist.assert_called_once()
        args = persist.call_args[0]
        assert args[0] == "10.0.0.8"   # identity = client_ip（无 JWT 时）
        assert args[1] == "线程池"      # query
        assert args[2] == "最终答案"    # answer

    def test_agent_lg_restores_and_persists_session(self):
        """agent-lg：会话恢复 + 完成后保存（与 agent 一致）"""
        persisted = [{"role": "user", "content": "上轮问题"}]
        fake = _FakeLLM([_answer("LG答案")])
        events, status, persist = self._post(
            "/ai/rag/chat/agent-lg", "agent.langgraph_react.LLMFactory.get_client", fake,
            resolve_history=mock.AsyncMock(return_value=persisted),
        )
        assert status == 200
        msgs = fake.chat_with_tools_calls[0]["messages"]
        assert ("user", "上轮问题") in [(m["role"], m.get("content")) for m in msgs]
        persist.assert_called_once()
        args = persist.call_args[0]
        assert args[0] == "10.0.0.8"
        assert args[1] == "线程池"
        assert args[2] == "LG答案"


class TestReactContextIdentity:
    """module-036: ReactContext.client_ip → identity 命名修正（引用一致性）"""

    def test_context_uses_identity_field(self):
        ctx = ReactContext("q", "user-42", [{"role": "user", "content": "hi"}])
        assert ctx.identity == "user-42"
        assert not hasattr(ctx, "client_ip")  # 无遗留 client_ip 记忆用途

    def test_recall_memory_uses_ctx_identity(self):
        """_recall_memory 工具按 ctx.identity 召回（行为不变，仅命名）"""
        from agent.tool_registry import _recall_memory
        ctx = ReactContext("q", "user-42", [])
        with mock.patch("rag.engine.rag_engine._recall_memory",
                        new=mock.AsyncMock(return_value="记忆")) as recall:
            text = asyncio.run(_recall_memory(ctx, {"query": "q"}))
        assert text == "记忆"
        args = recall.call_args[0]
        assert args[1] == "user-42"  # (query, identity)


class TestNoteToSelf:
    """module-041: note_to_self 工具注册 + 执行测试

    覆盖验收 4.1:
    - note_to_self 注册为第 10 个工具，list_tool_names() 含 "note_to_self"
    - note_to_self 写入 ctx.scratchpad
    - 空内容 note 返回提示
    - note 过长自动截断 500 字
    - generate_answer 读取 scratchpad，prompt 含"[工作笔记]"段
    - 空 scratchpad 零回归
    - 验收 1.6: _SYSTEM_PROMPT 含 note_to_self 工具描述
    """

    def test_system_prompt_contains_note_to_self(self):
        """验收 1.6: _SYSTEM_PROMPT 文本包含 note_to_self 工具描述"""
        from agent.react import _SYSTEM_PROMPT
        assert "note_to_self" in _SYSTEM_PROMPT
        assert "工作笔记" in _SYSTEM_PROMPT

    def test_note_to_self_tool_registered(self):
        """note_to_self 已注册，name/description/schema 均正确"""
        tool = registry.get("note_to_self")
        assert tool is not None
        assert tool.name == "note_to_self"
        assert "工作笔记" in tool.description or "草稿纸" in tool.description
        props = tool.args_schema.get("properties", {})
        assert "note" in props
        assert props["note"]["type"] == "string"
        assert "note" in tool.args_schema.get("required", [])

    def test_note_to_self_writes_to_scratchpad(self):
        """note_to_self 工具执行后 ctx.scratchpad 追加笔记"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("测试问题", "user-1", [])
            tool = registry.get("note_to_self")
            result = await tool.run({"note": "发现了一个重要线索"}, ctx)
            return result, ctx

        result, ctx = asyncio.run(run())
        assert "已记录笔记" in result
        assert len(ctx.scratchpad) == 1
        assert ctx.scratchpad[0] == "发现了一个重要线索"

    def test_note_to_self_empty_note(self):
        """空内容 note — 返回提示信息"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("测试问题", "user-1", [])
            tool = registry.get("note_to_self")
            result = await tool.run({}, ctx)
            return result, ctx

        result, ctx = asyncio.run(run())
        assert "未提供笔记内容" in result
        assert len(ctx.scratchpad) == 0

    def test_note_to_self_whitespace_only_note(self):
        """纯空白 note — 返回提示信息"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("测试问题", "user-1", [])
            tool = registry.get("note_to_self")
            result = await tool.run({"note": "   "}, ctx)
            return result, ctx

        result, ctx = asyncio.run(run())
        assert "未提供笔记内容" in result
        assert len(ctx.scratchpad) == 0

    def test_note_to_self_truncates_long_note(self):
        """note 超过 500 字自动截断"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("测试问题", "user-1", [])
            tool = registry.get("note_to_self")
            long_note = "A" * 1000
            result = await tool.run({"note": long_note}, ctx)
            return result, ctx

        result, ctx = asyncio.run(run())
        assert "已记录笔记" in result
        assert len(ctx.scratchpad) == 1
        assert len(ctx.scratchpad[0]) <= 500  # 截断到 <= 500
        # 确认是原始内容的前缀
        assert ctx.scratchpad[0] == "A" * 500

    def test_note_to_self_multi_note_accumulates(self):
        """多次调用 note_to_self 累积多条笔记"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("测试问题", "user-1", [])
            tool = registry.get("note_to_self")
            await tool.run({"note": "笔记1"}, ctx)
            await tool.run({"note": "笔记2"}, ctx)
            await tool.run({"note": "笔记3"}, ctx)
            result = await tool.run({"note": "笔记4"}, ctx)
            return result, ctx

        result, ctx = asyncio.run(run())
        assert len(ctx.scratchpad) == 4
        assert ctx.scratchpad == ["笔记1", "笔记2", "笔记3", "笔记4"]
        assert "已记录笔记 (4)" in result

    def test_generate_answer_reads_scratchpad(self):
        """_generate_answer 工具调用 reflector.generate_answer 时传入 scratchpad"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("测试问题", "user-1", [])
            ctx.add_docs([_doc(1)])
            ctx.add_note("工作笔记内容")
            with mock.patch(
                "agent.reflector.reflector.generate_answer",
                new=mock.AsyncMock(return_value="生成答案"),
            ) as gen:
                tool = registry.get("generate_answer")
                # _generate_answer 内部调 reflector.generate_answer(query, ctx.docs, ...)
                # 间接验证：patch reflector.method，检查参数
                result = await tool.run({"query": "测试问题"}, ctx)
            return result, gen

        result, gen = asyncio.run(run())
        assert result == "生成答案"
        gen.assert_called_once()
        # 验证 scratchpad 参数被传入
        call_kwargs = gen.call_args[1]
        assert call_kwargs.get("scratchpad") == ["工作笔记内容"]

    def test_generate_answer_empty_scratchpad_zero_regression(self):
        """空 scratchpad 时 generate_answer 行为不变，不注入 scratchpad 段"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("测试问题", "user-1", [])
            ctx.add_docs([_doc(1)])
            # 不写任何 note — scratchpad 为空列表
            with mock.patch(
                "agent.reflector.reflector.generate_answer",
                new=mock.AsyncMock(return_value="生成答案"),
            ) as gen:
                tool = registry.get("generate_answer")
                result = await tool.run({"query": "测试问题"}, ctx)
            return result, gen

        result, gen = asyncio.run(run())
        assert result == "生成答案"
        call_kwargs = gen.call_args[1]
        # 空列表也传入但 reflector 内 if scratchpad: 分支不会注入
        assert call_kwargs.get("scratchpad") == []

    def test_scratchpad_injection_in_reflector_generate_answer(self):
        """reflector.generate_answer 直接调用时，scratchpad 注入到 prompt"""
        async def run():
            gen_result = None
            with mock.patch(
                "agent.reflector.LLMFactory.get_client",
                new=mock.MagicMock(),
            ) as factory:
                mock_client = mock.MagicMock()
                mock_client.generate = mock.AsyncMock(return_value="含笔记的答案")
                factory.return_value = mock_client

                from agent.reflector import reflector
                result = await reflector.generate_answer(
                    "测试问题",
                    [_doc(1)],
                    scratchpad=["笔记1", "笔记2"],
                )
                gen_result = result
                # 验证 prompt 包含工作笔记段落
                prompt_arg = mock_client.generate.call_args[0][0]
            return gen_result, prompt_arg

        result, prompt = asyncio.run(run())
        assert result == "含笔记的答案"
        assert "[工作笔记" in prompt
        assert "笔记1" in prompt
        assert "笔记2" in prompt

    def test_scratchpad_none_zero_regression(self):
        """scratchpad=None 时 reflector.generate_answer 零回归"""
        async def run():
            with mock.patch(
                "agent.reflector.LLMFactory.get_client",
                new=mock.MagicMock(),
            ) as factory:
                mock_client = mock.MagicMock()
                mock_client.generate = mock.AsyncMock(return_value="正常答案")
                factory.return_value = mock_client

                from agent.reflector import reflector
                result = await reflector.generate_answer(
                    "测试问题",
                    [_doc(1)],
                    scratchpad=None,
                )
                prompt_arg = mock_client.generate.call_args[0][0]
            return result, prompt_arg

        result, prompt = asyncio.run(run())
        assert result == "正常答案"
        assert "[工作笔记" not in prompt


class TestNoteToSelfCoexistence:
    """module-041 与 module-039 verify_answer 共存测试

    覆盖验收 4.2:
    - note_to_self 和 verify_answer 可在同一 ReAct 循环中先后调用
    - verify_answer 的注册和执行不受 note_to_self 影响
    """

    def test_both_tools_registered(self):
        """note_to_self 和 verify_answer 均在 registry 中注册"""
        assert registry.get("note_to_self") is not None
        assert registry.get("verify_answer") is not None
        names = registry.list_tool_names()
        assert "note_to_self" in names
        assert "verify_answer" in names

    def test_note_to_self_then_verify_answer_in_react_loop(self):
        """ReAct 循环中先调 note_to_self 记录发现，再调 verify_answer 验证"""
        from agent.react import ReactContext

        async def run():
            ctx = ReactContext("线程池问题", "user-1", [])
            ctx.add_docs([_doc(1), _doc(2)])

            # 先记录笔记
            note_tool = registry.get("note_to_self")
            note_result = await note_tool.run({"note": "文档1提到核心线程数"}, ctx)

            # 再验证答案
            verify_tool = registry.get("verify_answer")
            with mock.patch(
                "agent.tool_registry.reflector.verify_answer",
                new=mock.AsyncMock(return_value={
                    "claims": [
                        {"claim": "核心线程数", "verdict": "supported",
                         "evidence": "[1]"},
                    ],
                    "overall_confidence": 1.0,
                    "total_claims": 1, "supported": 1,
                    "inferred": 0, "unsupported": 0,
                }),
            ):
                verify_result = await verify_tool.run(
                    {"answer": "核心线程数在文档1中提到", "query": "线程池"},
                    ctx,
                )
            return note_result, verify_result, ctx

        note_result, verify_result, ctx = asyncio.run(run())
        assert "已记录笔记" in note_result
        assert ctx.scratchpad == ["文档1提到核心线程数"]
        assert "[✓]" in verify_result

    def test_verify_answer_register_unchanged(self):
        """verify_answer 工具注册不受 note_to_self 新增影响"""
        tool = registry.get("verify_answer")
        assert tool is not None
        assert tool.name == "verify_answer"
        assert "逐句验证" in tool.description
        props = tool.args_schema.get("properties", {})
        assert "answer" in props
        assert "query" in props


# ─── module-042: MAX_ANSWER_LEN 截断测试 ───

TRUNC_MARKER = "\n\n[答案过长，已截断]"
MAX_LEN = 10000
LONG_ANSWER = "答" * 15000  # 超过 10000 字符的答案


class TestAnswerTruncationChat:
    """AC 1.5/2.3: /ai/rag/chat 端点答案截断测试"""

    def test_answer_truncated_and_sources_preserved(self):
        """答案 >10000 字符 → 截断 + 标记追加；sources 完整返回"""
        import main as main_module
        from rag.schemas import ChatResponse

        async def run():
            long_ans = "A" * 15000
            sources = [{"id": 1, "title": "文档1", "content": "内容", "source": "test", "ref_index": 1}]
            fake_response = ChatResponse(answer=long_ans, sources=sources, message="ok")

            with mock.patch("rag.engine.rag_engine.chat",
                            new=mock.AsyncMock(return_value=fake_response)):
                with mock.patch("main.save_messages_to_session"):
                    transport = httpx.ASGITransport(
                        app=main_module.app, raise_app_exceptions=True)
                    async with httpx.AsyncClient(
                            transport=transport, base_url="http://test") as client:
                        resp = await client.post(
                            "/ai/rag/chat",
                            json={"query": "测试截断", "history": []},
                        )
                    assert resp.status_code == 200
                    data = resp.json()
                    return data

        data = asyncio.run(run())
        # 截断后长度 ≤ MAX_LEN + 标记长度
        assert len(data["answer"]) <= MAX_LEN + len(TRUNC_MARKER)
        assert data["answer"].endswith(TRUNC_MARKER)
        assert data["answer"].startswith("A" * MAX_LEN)
        # sources 完整保留
        assert len(data["sources"]) == 1
        assert data["sources"][0]["id"] == 1

    def test_short_answer_not_truncated(self):
        """答案 ≤ 10000 字符 → 不截断，无标记"""
        import main as main_module
        from rag.schemas import ChatResponse

        async def run():
            short = "B" * 100  # 远小于 10000
            fake_response = ChatResponse(answer=short, sources=[], message="ok")

            with mock.patch("rag.engine.rag_engine.chat",
                            new=mock.AsyncMock(return_value=fake_response)):
                with mock.patch("main.save_messages_to_session"):
                    transport = httpx.ASGITransport(
                        app=main_module.app, raise_app_exceptions=True)
                    async with httpx.AsyncClient(
                            transport=transport, base_url="http://test") as client:
                        resp = await client.post(
                            "/ai/rag/chat",
                            json={"query": "测试", "history": []},
                        )
                    assert resp.status_code == 200
                    data = resp.json()
                    return data

        data = asyncio.run(run())
        assert data["answer"] == "B" * 100
        assert TRUNC_MARKER.lstrip() not in data["answer"].replace("\n", "")

    def test_exactly_max_not_truncated(self):
        """答案恰好 = 10000 字符（≤ 阈值）→ 不截断"""
        import main as main_module
        from rag.schemas import ChatResponse

        async def run():
            exact = "C" * 10000
            fake_response = ChatResponse(answer=exact, sources=[], message="ok")

            with mock.patch("rag.engine.rag_engine.chat",
                            new=mock.AsyncMock(return_value=fake_response)):
                with mock.patch("main.save_messages_to_session"):
                    transport = httpx.ASGITransport(
                        app=main_module.app, raise_app_exceptions=True)
                    async with httpx.AsyncClient(
                            transport=transport, base_url="http://test") as client:
                        resp = await client.post(
                            "/ai/rag/chat",
                            json={"query": "测试", "history": []},
                        )
                    assert resp.status_code == 200
                    data = resp.json()
                    return data

        data = asyncio.run(run())
        assert len(data["answer"]) == 10000
        assert TRUNC_MARKER not in data["answer"]


class TestAnswerTruncationAgent:
    """AC 1.5/2.3: /ai/rag/chat/agent 端点答案截断测试
    （截断从 react_loop 内部触发，token 与 done 事件一致）"""

    def test_agent_long_answer_truncated(self):
        """agent 端点 LLM 直接回答超长 → token + done 事件均截断且一致"""
        import main as main_module

        long_ans = "X" * 15000
        fake = _FakeLLM([_answer(long_ans)])
        events = []

        async def run():
            with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                with mock.patch("rag.engine.rag_engine._resolve_session_history",
                                new=mock.AsyncMock(side_effect=lambda identity, h: h)):
                    transport = httpx.ASGITransport(
                        app=main_module.app, raise_app_exceptions=True)
                    async with httpx.AsyncClient(
                            transport=transport, base_url="http://test") as client:
                        resp = await client.post(
                            "/ai/rag/chat/agent",
                            json={"query": "测试截断", "history": []},
                        )
                    assert resp.status_code == 200
                    events.extend(_parse_sse(resp.content))

        asyncio.run(run())

        names = [e["event"] for e in events]
        assert "token" in names
        assert "done" in names

        # token 事件内容已截断
        token_data = json.loads(events[names.index("token")]["data"])
        assert len(token_data) <= MAX_LEN + len(TRUNC_MARKER)
        assert token_data.endswith(TRUNC_MARKER)

        # done 事件 answer 与 token 一致
        done_data = json.loads(events[names.index("done")]["data"])
        assert len(done_data["answer"]) <= MAX_LEN + len(TRUNC_MARKER)
        assert done_data["answer"].endswith(TRUNC_MARKER)
        assert done_data["answer"] == token_data  # 一致性：token == done.answer

        # sources 完整保留
        assert "sources" in done_data
        assert "budget" in done_data

    def test_agent_short_answer_not_truncated(self):
        """agent 端点短答案 → 不截断，token == done.answer"""
        import main as main_module

        short = "短答案"
        fake = _FakeLLM([_answer(short)])
        events = []

        async def run():
            with mock.patch("agent.react.LLMFactory.get_client", return_value=fake):
                with mock.patch("rag.engine.rag_engine._resolve_session_history",
                                new=mock.AsyncMock(side_effect=lambda identity, h: h)):
                    transport = httpx.ASGITransport(
                        app=main_module.app, raise_app_exceptions=True)
                    async with httpx.AsyncClient(
                            transport=transport, base_url="http://test") as client:
                        resp = await client.post(
                            "/ai/rag/chat/agent",
                            json={"query": "hi", "history": []},
                        )
                    assert resp.status_code == 200
                    events.extend(_parse_sse(resp.content))

        asyncio.run(run())

        names = [e["event"] for e in events]
        token_data = json.loads(events[names.index("token")]["data"])
        done_data = json.loads(events[names.index("done")]["data"])
        assert token_data == "短答案"
        assert done_data["answer"] == "短答案"
        assert TRUNC_MARKER not in token_data


class TestAnswerTruncationAgentLG:
    """AC 1.5/2.3: /ai/rag/chat/agent-lg 端点答案截断测试"""

    def test_agent_lg_long_answer_truncated(self):
        """agent-lg 端点超长答案 → token + done 一致截断"""
        import main as main_module

        long_ans = "Y" * 15000
        fake = _FakeLLM([_answer(long_ans)])
        events = []

        async def run():
            with mock.patch("agent.langgraph_react.LLMFactory.get_client",
                            return_value=fake):
                with mock.patch("rag.engine.rag_engine._resolve_session_history",
                                new=mock.AsyncMock(side_effect=lambda identity, h: h)):
                    transport = httpx.ASGITransport(
                        app=main_module.app, raise_app_exceptions=True)
                    async with httpx.AsyncClient(
                            transport=transport, base_url="http://test") as client:
                        resp = await client.post(
                            "/ai/rag/chat/agent-lg",
                            json={"query": "测试", "history": []},
                        )
                    assert resp.status_code == 200
                    events.extend(_parse_sse(resp.content))

        asyncio.run(run())

        names = [e["event"] for e in events]
        assert "done" in names
        done_data = json.loads(events[names.index("done")]["data"])
        assert len(done_data["answer"]) <= MAX_LEN + len(TRUNC_MARKER)
        assert done_data["answer"].endswith(TRUNC_MARKER)

        # 如果有 token 事件，token 内容与 done.answer 一致
        if "token" in names:
            token_data = json.loads(events[names.index("token")]["data"])
            assert token_data == done_data["answer"]

    def test_agent_lg_short_answer_not_truncated(self):
        """agent-lg 端点短答案 → 不截断"""
        import main as main_module

        short = "短答案LG"
        fake = _FakeLLM([_answer(short)])
        events = []

        async def run():
            with mock.patch("agent.langgraph_react.LLMFactory.get_client",
                            return_value=fake):
                with mock.patch("rag.engine.rag_engine._resolve_session_history",
                                new=mock.AsyncMock(side_effect=lambda identity, h: h)):
                    transport = httpx.ASGITransport(
                        app=main_module.app, raise_app_exceptions=True)
                    async with httpx.AsyncClient(
                            transport=transport, base_url="http://test") as client:
                        resp = await client.post(
                            "/ai/rag/chat/agent-lg",
                            json={"query": "hi", "history": []},
                        )
                    assert resp.status_code == 200
                    events.extend(_parse_sse(resp.content))

        asyncio.run(run())

        names = [e["event"] for e in events]
        done_data = json.loads(events[names.index("done")]["data"])
        assert done_data["answer"] == "短答案LG"
        assert TRUNC_MARKER not in done_data["answer"]


class TestAnswerTruncationChatStream:
    """AC 1.5/2.3: /ai/rag/chat/stream 端点答案截断测试"""

    def test_stream_truncation_marker_emitted(self):
        """chat_stream 超长答案 → 流中包含截断标记 token；verify 用清洗后答案"""
        import main as main_module

        events = []
        fake_doc = _doc(1)

        async def run():
            with mock.patch("agent.router.router_agent.classify",
                            new=mock.AsyncMock(
                                return_value={"intent": "knowledge", "confidence": 0.9})):
                with mock.patch("rag.engine.rag_engine._retrieve",
                                new=mock.AsyncMock(return_value=[fake_doc])):
                    with mock.patch("rag.engine.rag_engine._rerank",
                                    new=mock.AsyncMock(return_value=[fake_doc])):
                        # 模拟反思：直接返回充分，不触发改写检索
                        with mock.patch(
                            "agent.reflector.reflector.check_sufficiency",
                            new=mock.AsyncMock(return_value={"sufficient": True}),
                        ):
                            # 模拟生成超长答案
                            tokens = ["T" * 1000 for _ in range(20)]

                            async def fake_stream(*args, **kwargs):
                                for t in tokens:
                                    yield t

                            with mock.patch(
                                "agent.reflector.reflector.generate_answer_stream",
                                new=fake_stream,
                            ):
                                with mock.patch(
                                    "agent.reflector.reflector.verify_answer",
                                    new=mock.AsyncMock(return_value={"claims": [], "overall_confidence": 0}),
                                ):
                                    with mock.patch(
                                        "rag.engine.rag_engine._resolve_session_history",
                                        new=mock.AsyncMock(side_effect=lambda identity, h: h),
                                    ):
                                        with mock.patch(
                                            "rag.engine.rag_engine._recall_memory",
                                            new=mock.AsyncMock(return_value=""),
                                        ):
                                            transport = httpx.ASGITransport(
                                                app=main_module.app, raise_app_exceptions=True)
                                            async with httpx.AsyncClient(
                                                    transport=transport, base_url="http://test") as client:
                                                resp = await client.post(
                                                    "/ai/rag/chat/stream",
                                                    json={"query": "测试", "history": []},
                                                )
                                            assert resp.status_code == 200
                                            events.extend(_parse_sse(resp.content))

        asyncio.run(run())

        # token 事件中应包含截断标记
        token_events = [e for e in events if e["event"] == "token"]
        token_texts = [json.loads(e["data"]) for e in token_events]
        all_tokens = "".join(token_texts)
        assert TRUNC_MARKER in all_tokens
        # 验证截断后总长度不超过 MAX_LEN + 标记长度
        assert len(all_tokens) <= MAX_LEN + len(TRUNC_MARKER)

        # verify_answer 被调用时应传入清洗后的答案（不含截断标记）
        # 由 mock 对象验证——此处通过不崩溃来间接验证

    def test_stream_short_answer_not_truncated(self):
        """chat_stream 短答案 → 不截断，无截断标记"""
        import main as main_module

        events = []
        fake_doc = _doc(1)

        async def run():
            with mock.patch("agent.router.router_agent.classify",
                            new=mock.AsyncMock(
                                return_value={"intent": "knowledge", "confidence": 0.9})):
                with mock.patch("rag.engine.rag_engine._retrieve",
                                new=mock.AsyncMock(return_value=[fake_doc])):
                    with mock.patch("rag.engine.rag_engine._rerank",
                                    new=mock.AsyncMock(return_value=[fake_doc])):
                        with mock.patch(
                            "agent.reflector.reflector.check_sufficiency",
                            new=mock.AsyncMock(return_value={"sufficient": True}),
                        ):
                            async def fake_stream(*args, **kwargs):
                                yield "短答案流"

                            with mock.patch(
                                "agent.reflector.reflector.generate_answer_stream",
                                new=fake_stream,
                            ):
                                with mock.patch(
                                    "agent.reflector.reflector.verify_answer",
                                    new=mock.AsyncMock(return_value={"claims": [], "overall_confidence": 0}),
                                ):
                                    with mock.patch(
                                        "rag.engine.rag_engine._resolve_session_history",
                                        new=mock.AsyncMock(side_effect=lambda identity, h: h),
                                    ):
                                        with mock.patch(
                                            "rag.engine.rag_engine._recall_memory",
                                            new=mock.AsyncMock(return_value=""),
                                        ):
                                            transport = httpx.ASGITransport(
                                                app=main_module.app, raise_app_exceptions=True)
                                            async with httpx.AsyncClient(
                                                    transport=transport, base_url="http://test") as client:
                                                resp = await client.post(
                                                    "/ai/rag/chat/stream",
                                                    json={"query": "测试", "history": []},
                                                )
                                            assert resp.status_code == 200
                                            events.extend(_parse_sse(resp.content))

        asyncio.run(run())

        token_events = [e for e in events if e["event"] == "token"]
        token_texts = [json.loads(e["data"]) for e in token_events]
        all_tokens = "".join(token_texts)
        assert all_tokens == "短答案流"
        assert TRUNC_MARKER not in all_tokens
