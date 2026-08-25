"""Module-030 重排切换 + LangGraph 实验端点单元测试

覆盖（验收 §4）：
- bge-reranker-v2-m3：默认模型路径 / 缺权重明确报错 / predict 传裸 pair
  （无 chat template 适配）/ 排序降序 + top_k / 空文档 / 缺 content 不崩
- LangGraph 版 ReAct：工具调用→直接回答 / 预算耗尽兜底 / 预算=0 直接生成 /
  工具失败返回空继续 / docs 累积 / reasoning_content 回传 / 预算截断
- SSE 端点 /ai/rag/chat/agent-lg：tool_call/tool_result/token/done 事件序列

实现说明：
- 用 mock 打桩 CrossEncoder / LLMFactory.get_client / hybrid_retriever /
  reflector，不依赖真实 DB / Redis / LLM / 真实模型权重
- 同步用例内 asyncio.run 执行，不依赖 pytest-asyncio（沿用既有模式）
"""
import asyncio
import json
import os
import tempfile
from contextlib import contextmanager
from unittest import mock

import httpx

import main
from rag.reranker import CrossEncoderReranker, RerankerException, _LOCAL_MODEL_DIR
from agent.langgraph_react import langgraph_react_agent
from agent.tool_registry import ToolRegistry


# ==================== 重排测试辅助 ====================


class _FakeCrossEncoder:
    """假 CrossEncoder：记录 predict 输入并返回固定分数"""

    instances = []

    def __init__(self, model_dir):
        self.model_dir = model_dir
        _FakeCrossEncoder.instances.append(self)

    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        self.kwargs = kwargs
        return [0.9, 0.1, 0.5]


@contextmanager
def _fake_model_dir():
    """创建含假权重文件（model.safetensors）的临时模型目录，验证后清理"""
    tmp = tempfile.TemporaryDirectory()
    try:
        with open(os.path.join(tmp.name, "model.safetensors"), "w") as f:
            f.write("dummy")
        yield tmp.name
    finally:
        tmp.cleanup()


# ==================== LangGraph 测试辅助（沿用 test_agent_tools 范式） ====================


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


# ==================== 重排器单测（module-030 部分 1） ====================


class TestRerankerModel:
    """bge-reranker-v2-m3 模型路径 / 缺权重校验"""

    def test_default_model_is_bge_reranker_v2_m3(self):
        assert os.path.basename(_LOCAL_MODEL_DIR) == "bge-reranker-v2-m3"

    def test_missing_dir_raises(self):
        bad = CrossEncoderReranker(model_name="/nonexistent/model/path")
        try:
            asyncio.run(bad.rerank("q", [{"id": 1, "content": "x"}]))
            assert False, "应抛 RerankerException"
        except RerankerException as e:
            # rerank 对外层包 "重排服务暂时不可用"，具体原因在 cause 链（module-018 行为）
            assert isinstance(e.__cause__, RerankerException)
            assert "目录不存在" in str(e.__cause__)

    def test_missing_weights_raises(self):
        """目录存在但缺权重文件 → 明确报错（module-018 缺权重策略保留）"""
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "tokenizer.json"), "w") as f:
                f.write("x")  # 只放无关文件，无 model.safetensors
            no_weight = CrossEncoderReranker(model_name=tmp)
            try:
                asyncio.run(no_weight.rerank("q", [{"id": 1, "content": "x"}]))
                assert False, "应抛 RerankerException"
            except RerankerException as e:
                assert isinstance(e.__cause__, RerankerException)
                assert "缺少权重文件" in str(e.__cause__)


class TestRerankerPredict:
    """bge 重排排序逻辑（mock CrossEncoder）"""

    def _rerank(self, docs, top_k=5, query="Java线程池"):
        _FakeCrossEncoder.instances.clear()
        with _fake_model_dir() as model_dir:
            with mock.patch("rag.reranker.CrossEncoder", _FakeCrossEncoder):
                rr = CrossEncoderReranker(model_name=model_dir)
                return asyncio.run(rr.rerank(query, docs, top_k=top_k))

    def test_predict_uses_bare_pairs_no_chat_template(self):
        """bge 是标准分类式 CrossEncoder：predict 收 (query, doc) 裸 pair，无 chat 适配"""
        _FakeCrossEncoder.instances.clear()
        docs = [
            {"id": 1, "content": "Java 线程池核心参数"},
            {"id": 2, "content": "Redis 缓存"},
            {"id": 3, "content": "线程池拒绝策略"},
        ]
        with _fake_model_dir() as model_dir:
            with mock.patch("rag.reranker.CrossEncoder", _FakeCrossEncoder):
                rr = CrossEncoderReranker(model_name=model_dir)
                asyncio.run(rr.rerank("Java线程池", docs, top_k=3))

        inst = _FakeCrossEncoder.instances[-1]
        assert inst.pairs == [
            ("Java线程池", "Java 线程池核心参数"),
            ("Java线程池", "Redis 缓存"),
            ("Java线程池", "线程池拒绝策略"),
        ]
        # 不再有 Qwen3 的 chat message / add_generation_prompt 适配
        assert "processing_kwargs" not in inst.kwargs

    def test_sorted_desc_and_top_k(self):
        result = self._rerank(
            [{"id": 1, "content": "a"}, {"id": 2, "content": "b"}, {"id": 3, "content": "c"}],
            top_k=2,
        )
        assert [d["id"] for d in result] == [1, 3]  # scores [0.9, 0.1, 0.5] 降序取前2
        assert result[0]["rerank_score"] == 0.9
        assert result[1]["rerank_score"] == 0.5

    def test_empty_docs_returns_empty(self):
        _FakeCrossEncoder.instances.clear()
        with _fake_model_dir() as model_dir:
            with mock.patch("rag.reranker.CrossEncoder", _FakeCrossEncoder):
                rr = CrossEncoderReranker(model_name=model_dir)
                assert asyncio.run(rr.rerank("q", [])) == []

    def test_missing_content_no_crash(self):
        result = self._rerank([{"id": 7}, {"id": 8, "content": "有内容"}])
        assert len(result) == 2  # 缺 content 不抛异常

    def test_long_content_truncated_to_max_pair_chars(self):
        """超长文档内容截断到 _MAX_PAIR_CHARS，避免 CrossEncoder 处理满长上下文

        回归：知识库父块可达数万字符，未截断时 fp32 CPU 下单次 rerank 实测
        ~200s（卡死链路）；截断后约 3.4s。
        """
        from rag.reranker import _MAX_PAIR_CHARS

        long_content = "长" * 5000  # 远超 _MAX_PAIR_CHARS
        _FakeCrossEncoder.instances.clear()
        with _fake_model_dir() as model_dir:
            with mock.patch("rag.reranker.CrossEncoder", _FakeCrossEncoder):
                rr = CrossEncoderReranker(model_name=model_dir)
                asyncio.run(rr.rerank("q", [{"id": 1, "content": long_content}], top_k=1))

        inst = _FakeCrossEncoder.instances[-1]
        assert len(inst.pairs) == 1
        assert len(inst.pairs[0][1]) == _MAX_PAIR_CHARS
        assert inst.pairs[0][1] == "长" * _MAX_PAIR_CHARS


# ==================== LangGraph 版 ReAct 单测（module-030 部分 2） ====================


class TestLangGraphReactAgent:
    """LangGraph 版 ReAct 循环核心（行为与手写 react_loop 对齐）"""

    def test_tool_call_then_direct_answer(self):
        """LLM 先调工具，再直接回答"""
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "Java线程池"}),
            _answer("线程池核心参数包括核心线程数、最大线程数、队列容量。"),
        ])
        with mock.patch("agent.langgraph_react.LLMFactory.get_client", return_value=fake):
            with _patch_retriever([_doc(1), _doc(2)]):
                result = asyncio.run(langgraph_react_agent("Java线程池核心参数", budget=4))

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
        with mock.patch("agent.langgraph_react.LLMFactory.get_client", return_value=fake):
            with _patch_retriever([_doc(1)]):
                with mock.patch("agent.langgraph_react.reflector.generate_answer",
                                new=mock.AsyncMock(return_value="兜底答案")):
                    result = asyncio.run(langgraph_react_agent("q", budget=2))

        assert result["tool_count"] == 2  # ≤ budget
        assert result["answer"] == "兜底答案"
        assert len(fake.chat_with_tools_calls) == 2

    def test_budget_zero_direct_answer_without_tools(self):
        """预算=0：LLM 直接回答，不调用工具"""
        fake = _FakeLLM([])
        with mock.patch("agent.langgraph_react.LLMFactory.get_client", return_value=fake):
            result = asyncio.run(langgraph_react_agent("你好", budget=0))

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
        with mock.patch("agent.langgraph_react.LLMFactory.get_client", return_value=fake):
            result = asyncio.run(langgraph_react_agent("q", budget=4, tools=reg))

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
        with mock.patch("agent.langgraph_react.LLMFactory.get_client", return_value=fake):
            with _patch_retriever([_doc(1)]):
                with mock.patch("agent.tool_registry.graph_extractor.extract_from_query",
                                new=mock.AsyncMock(return_value=["实体A"])):
                    with mock.patch("agent.tool_registry.graph_store.search_related",
                                    new=mock.AsyncMock(return_value=[_doc(3)])):
                        result = asyncio.run(langgraph_react_agent("q", budget=4))

        assert result["tool_count"] == 2
        assert [t["name"] for t in result["tool_trace"]] == [
            "search_knowledge", "search_graph",
        ]

    def test_budget_truncation_executes_only_allowed_tools(self):
        """预算内本轮只执行前 N 个工具调用（条件路由 + 预算检查）"""
        fake = _FakeLLM([{
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "search_knowledge", "args": {"query": "q"}},
                {"id": "c2", "name": "search_fts", "args": {"query": "q"}},
            ],
            "message": {
                "role": "assistant", "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "search_knowledge",
                                  "arguments": '{"query": "q"}'}},
                    {"id": "c2", "type": "function",
                     "function": {"name": "search_fts",
                                  "arguments": '{"query": "q"}'}},
                ],
            },
        }])
        with mock.patch("agent.langgraph_react.LLMFactory.get_client", return_value=fake):
            with _patch_retriever([_doc(1)]):
                with mock.patch("agent.langgraph_react.reflector.generate_answer",
                                new=mock.AsyncMock(return_value="兜底答案")):
                    result = asyncio.run(langgraph_react_agent("q", budget=1))

        assert result["tool_count"] == 1  # 只执行预算内 1 个
        assert result["answer"] == "兜底答案"
        assert len(result["tool_trace"]) == 1

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
        with mock.patch("agent.langgraph_react.LLMFactory.get_client", return_value=fake):
            with _patch_retriever([_doc(1)]):
                result = asyncio.run(langgraph_react_agent("q", budget=4))

        assert result["answer"] == "答案"
        assert len(fake.chat_with_tools_calls) == 2
        # 第二轮调用的消息历史里，assistant 消息携带 reasoning_content（回传要求）
        second_msgs = fake.chat_with_tools_calls[1]["messages"]
        assistant_msgs = [m for m in second_msgs if m["role"] == "assistant"]
        assert any(m.get("reasoning_content") == "思考过程" for m in assistant_msgs)

    def test_event_ordering(self):
        """SSE 事件顺序：token（推理）→ tool_call → tool_result → done"""
        fake = _FakeLLM([
            {"content": "思考中", "tool_calls": [{"id": "c1", "name": "search_knowledge", "args": {"query": "q"}}],
             "message": {"role": "assistant", "content": "思考中",
                         "tool_calls": [{"id": "c1", "type": "function",
                                         "function": {"name": "search_knowledge",
                                                      "arguments": '{"query": "q"}'}}]}},
            _answer("最终答案"),
        ])
        from agent.langgraph_react import ReactContext, _build_messages, langgraph_react_loop
        events = []
        async def run():
            with mock.patch("agent.langgraph_react.LLMFactory.get_client", return_value=fake):
                with _patch_retriever([_doc(1)]):
                    ctx = ReactContext("q", "unknown", [])
                    async for evt in langgraph_react_loop(ctx, _build_messages(ctx), 4):
                        events.append(evt)
        asyncio.run(run())

        assert [e["type"] for e in events] == ["token", "tool_call", "tool_result", "token", "done"]
        assert events[0]["content"] == "思考中"


# ==================== LangGraph 端点单测 ====================


class TestLangGraphEndpoint:
    """POST /ai/rag/chat/agent-lg（SSE 工具轨迹，实验端点）"""

    def test_sse_tool_trace_events(self):
        fake = _FakeLLM([
            _tool_call("search_knowledge", {"query": "线程池"}),
            _answer("最终答案"),
        ])
        events = []

        async def run():
            with mock.patch("agent.langgraph_react.LLMFactory.get_client", return_value=fake):
                with _patch_retriever([_doc(7)]):
                    transport = httpx.ASGITransport(
                        app=main.app, raise_app_exceptions=True,
                    )
                    async with httpx.AsyncClient(
                            transport=transport, base_url="http://test") as client:
                        resp = await client.post(
                            "/ai/rag/chat/agent-lg",
                            json={"query": "线程池", "history": []},
                        )
                    assert resp.status_code == 200
                    events.extend(_parse_sse(resp.content))

        asyncio.run(run())

        names = [e["event"] for e in events]
        # 工具调用 content 为空 → 无推理 token；tool_call → tool_result → 最终答案 token → done
        assert names == ["tool_call", "tool_result", "token", "done"]

        tool_call = json.loads(events[0]["data"])
        assert tool_call["name"] == "search_knowledge"
        assert tool_call["args"] == {"query": "线程池"}
        assert tool_call["tool_count"] == 1

        tool_result = json.loads(events[1]["data"])
        assert "文档7" in tool_result["result"]

        done = json.loads(events[3]["data"])
        assert done["answer"] == "最终答案"
        assert done["tool_count"] == 1
        assert done["budget"] == 4
        assert done["sources"][0]["id"] == 7

    def test_budget_zero_endpoint_direct_answer(self):
        """预算=0：端点直接回答（验收 §1.3）"""
        fake = _FakeLLM([])
        events = []

        async def run():
            with mock.patch.object(main.settings, "max_agent_tools", 0):
                with mock.patch("agent.langgraph_react.LLMFactory.get_client", return_value=fake):
                    transport = httpx.ASGITransport(
                        app=main.app, raise_app_exceptions=True,
                    )
                    async with httpx.AsyncClient(
                            transport=transport, base_url="http://test") as client:
                        resp = await client.post(
                            "/ai/rag/chat/agent-lg",
                            json={"query": "你好", "history": []},
                        )
                    events.extend(_parse_sse(resp.content))

        asyncio.run(run())

        names = [e["event"] for e in events]
        assert names == ["done"]
        done = json.loads(events[-1]["data"])
        assert done["tool_count"] == 0
        assert done["budget"] == 0
        assert len(fake.chat_calls) == 1
        assert len(fake.chat_with_tools_calls) == 0
