"""module-058 WP-C：可观测性测试（trace_id / 阶段计时 / token 用量 / 缓存命中 / request_logs）

覆盖（验收 §2 功能验收 + §5 降级验收）：
- trace_id：中间件生成挂 request.state，贯穿落库记录
- 阶段计时：engine.chat 各阶段（intent/retrieve/rerank/reflection/generate/verify）
- token 用量：LLM 客户端按供应商累积 prompt/completion（无 usage 静默跳过）
- 缓存命中：engine._retrieve 缓存检查处计数命中/未命中
- request_logs：落库字段完整；落库失败 fail-open 不抛异常
- 端点接线：chat 请求结束触发 persist_request_log（含 trace_id/intent）
- 开关关闭时零埋点（persist_request_log 直接返回、不落库）

实现说明：
- 测试内显式 setattr settings.request_logs_enabled=True（conftest 默认钉住
  false，存量测试不污染落库）；端点用例 patch save_request_log 防真实 DB
- 假 session 打桩 async_session_factory（对齐 test_feedback.py 模式）
"""
import asyncio
import logging
from unittest import mock

import httpx

import main as main_module
from rag.engine import rag_engine
from src import observability
from src.config import settings


class _FakeSession:
    """假 AsyncSession：记录 add 的对象；可配置 commit 抛异常（fail-open 用例）"""

    def __init__(self, commit_error: bool = False):
        self.added: list = []
        self._commit_error = commit_error

    def add(self, obj):
        self.added.append(obj)

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
    """显式开启 request_logs（conftest autouse 默认钉住 false）"""
    monkeypatch.setattr(settings, "request_logs_enabled", True)


class TestTraceAndStats:
    """观测上下文：timing / usage / cache 计数累积"""

    def test_timing_and_usage_accumulate(self, monkeypatch):
        """timing 分阶段毫秒记录（同阶段后写覆盖）+ usage 按供应商累积（求和）"""
        _enable_logs(monkeypatch)

        async def run():
            observability.init_request("trace-1")
            observability.timing("intent", 0.1)
            observability.timing("intent", 0.2)  # 同阶段第二次测量覆盖（单次测量语义）
            observability.timing("generate", 0.5)
            observability.record_usage("deepseek", 100, 20)
            observability.record_usage("deepseek", 50, 10)
            return observability.get_request_stats()

        stats = asyncio.run(run())
        assert stats["trace_id"] == "trace-1"
        assert stats["timings"]["intent"] == 200.0   # 后写覆盖
        assert stats["timings"]["generate"] == 500.0
        assert stats["usage"]["deepseek"] == {"prompt": 150, "completion": 30}

    def test_trace_id_generated_unique(self):
        """make_trace_id 生成非空且互异的 UUID hex"""
        t1 = observability.make_trace_id()
        t2 = observability.make_trace_id()
        assert t1 and t2
        assert t1 != t2
        assert len(t1) == 32

    def test_disabled_zero_instrumentation(self):
        """开关关闭（conftest 默认）→ helper 零埋点（不写观测上下文）"""
        async def run():
            observability.init_request("trace-x")
            observability.timing("intent", 0.5)
            observability.record_usage("deepseek", 1, 1)
            observability.record_cache(hit=True)
            return observability.get_request_stats()

        stats = asyncio.run(run())
        # request_logs_enabled=False（conftest 钉住）→ timings/usage/cache 保持初始值
        assert stats["timings"] == {}
        assert stats["usage"] == {}
        assert stats["cache_hits"] == 0


class _RecordCapture(logging.Handler):
    """收集经本 handler 处理的日志 record（TraceIdFilter 注入验证用）"""

    def __init__(self):
        super().__init__()
        self.records: list = []

    def emit(self, record):
        self.records.append(record)


class TestTraceIdInLogs:
    """trace_id 贯穿日志（Review 修复 MAJOR-1）：TraceIdFilter 注入 record"""

    def test_log_record_carries_trace_id(self):
        """请求上下文存在 → 日志 record.trace_id = 当前 trace_id（可跨日志行关联）"""
        handler = _RecordCapture()
        handler.addFilter(observability.TraceIdFilter())  # 模拟生产 wiring（根 handler 挂过滤器）
        log = logging.getLogger("obs.trace.test")
        log.setLevel(logging.INFO)  # 根 logger 默认 WARNING 会滤掉 INFO，显式放行
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            async def run():
                observability.init_request("trace-log-abc")
                log.info("观测日志行")

            asyncio.run(run())
        finally:
            root.removeHandler(handler)
        assert len(handler.records) == 1
        assert handler.records[0].trace_id == "trace-log-abc"

    def test_log_record_trace_id_empty_without_request(self):
        """无请求上下文（新 context）→ record.trace_id 空串（不干扰其他日志消费方）"""
        handler = _RecordCapture()
        handler.addFilter(observability.TraceIdFilter())
        log = logging.getLogger("obs.trace.empty")
        log.setLevel(logging.INFO)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            async def run():
                # 新 asyncio 上下文，未 init_request → 缺省为空串
                log.info("无请求日志行")

            asyncio.run(run())
        finally:
            root.removeHandler(handler)
        assert len(handler.records) == 1
        assert handler.records[0].trace_id == ""

    def test_install_trace_id_filter_idempotent(self):
        """install_trace_id_filter 幂等：重复调用不重复挂（同实例）"""
        f1 = observability.install_trace_id_filter()
        f2 = observability.install_trace_id_filter()
        assert f1 is f2
        assert isinstance(f1, observability.TraceIdFilter)


class TestCacheCounting:
    """engine._retrieve 缓存命中/未命中计数"""

    def test_retrieve_counts_hit_and_miss(self, monkeypatch):
        """缓存未命中 +1、命中 +1（同请求上下文可区分）"""
        _enable_logs(monkeypatch)
        fake_doc = {"id": 1, "title": "t", "content": "c",
                    "parent_id": None, "hybrid_score": 0.9}

        async def run():
            observability.init_request("trace-retrieve")
            # 未命中路径：cache.get 返回 None → 走完整检索 → cache.set
            with mock.patch("rag.engine.cache.get",
                            new=mock.AsyncMock(return_value=None)):
                with mock.patch("rag.engine.cache.set",
                                new=mock.AsyncMock(return_value=True)):
                    with mock.patch("rag.engine.hybrid_retriever.retrieve",
                                    new=mock.AsyncMock(return_value=[fake_doc])):
                        with mock.patch.object(rag_engine, "_hyde_expand",
                                               mock.AsyncMock(return_value="q")):
                            with mock.patch("agent.reflector.reflector.check_sufficiency",
                                            new=mock.AsyncMock(
                                                return_value={"sufficient": True})):
                                docs = await rag_engine._retrieve("查询", top_k=5)
            assert len(docs) == 1
            # 命中路径：cache.get 直接返回
            with mock.patch("rag.engine.cache.get",
                            new=mock.AsyncMock(return_value=[fake_doc])):
                docs2 = await rag_engine._retrieve("查询", top_k=5)
            assert len(docs2) == 1
            return observability.get_request_stats()

        stats = asyncio.run(run())
        assert stats["cache_misses"] == 1
        assert stats["cache_hits"] == 1
        # 阶段计时同样被记录（hyde / reflection）
        assert "hyde" in stats["timings"]
        assert "reflection" in stats["timings"]


class TestEngineChatTimings:
    """engine.chat 各阶段计时落观测上下文"""

    def test_chat_records_all_stages(self, monkeypatch):
        _enable_logs(monkeypatch)
        fake_doc = {"id": 1, "title": "t", "content": "c", "parent_id": None,
                    "hybrid_score": 0.9, "abs_cosine": 0.7}

        async def run():
            observability.init_request("trace-chat")
            req = mock.MagicMock()
            req.query = "测试问题"
            req.history = []
            with mock.patch("rag.engine.router_agent.classify",
                            new=mock.AsyncMock(
                                return_value={"intent": "knowledge",
                                              "confidence": 0.9})):
                with mock.patch("rag.engine.rag_engine._recall_memory",
                                new=mock.AsyncMock(return_value="")):
                    with mock.patch("rag.engine.hybrid_retriever.retrieve",
                                    new=mock.AsyncMock(return_value=[fake_doc])):
                        with mock.patch("rag.engine.reranker.rerank",
                                        new=mock.AsyncMock(return_value=[fake_doc])):
                            with mock.patch(
                                    "agent.reflector.reflector.check_sufficiency",
                                    new=mock.AsyncMock(
                                        return_value={"sufficient": True})):
                                with mock.patch(
                                        "rag.engine.rag_engine._expand_to_parents",
                                        new=mock.AsyncMock(return_value=[fake_doc])):
                                    with mock.patch(
                                            "rag.engine.rag_engine._resolve_session_history",
                                            new=mock.AsyncMock(
                                                side_effect=lambda i, h: h)):
                                        with mock.patch(
                                                "rag.engine.rag_engine._schedule_persist"):
                                            with mock.patch(
                                                    "rag.engine.rag_engine."
                                                    "_schedule_session_persist"):
                                                with mock.patch(
                                                        "agent.reflector.reflector."
                                                        "generate_answer",
                                                        new=mock.AsyncMock(
                                                            return_value="答案")):
                                                    with mock.patch(
                                                            "agent.reflector.reflector."
                                                            "verify_answer",
                                                            new=mock.AsyncMock(
                                                                return_value={
                                                                    "claims": []})):
                                                        resp = await rag_engine.chat(
                                                            req, identity="u1")
            return resp, observability.get_request_stats()

        resp, stats = asyncio.run(run())
        assert resp.message == "ok"
        for stage in ("intent", "retrieve", "rerank", "reflection",
                      "generate", "verify"):
            assert stage in stats["timings"], f"缺阶段计时: {stage}"


class TestRequestLogs:
    """request_logs 落库 + fail-open"""

    RECORD = {
        "trace_id": "trace-abc",
        "identity": "10.0.0.9",
        "endpoint": "chat",
        "intent": "knowledge",
        "timings": {"intent": 5.0, "generate": 200.0},
        "usage": {"deepseek": {"prompt": 100, "completion": 20}},
        "cache_hits": 0,
        "cache_misses": 1,
        "error": False,
    }

    def test_save_request_log_persists(self, monkeypatch):
        """落库：RequestLog 记录字段完整（timings/usage JSONB 结构透传）"""
        _enable_logs(monkeypatch)
        session = _FakeSession()

        async def run():
            with mock.patch("src.database.async_session_factory",
                            _fake_factory(session)):
                await observability.save_request_log(dict(self.RECORD))

        asyncio.run(run())
        assert len(session.added) == 1
        rl = session.added[0]
        assert rl.trace_id == "trace-abc"
        assert rl.identity == "10.0.0.9"
        assert rl.endpoint == "chat"
        assert rl.intent == "knowledge"
        assert rl.timings["generate"] == 200.0
        assert rl.usage["deepseek"] == {"prompt": 100, "completion": 20}
        assert rl.cache_misses == 1
        assert rl.error is False

    def test_save_request_log_fail_open(self, monkeypatch):
        """落库失败 → fail-open：不抛异常（日志告警，不阻塞主链路）"""
        _enable_logs(monkeypatch)
        session = _FakeSession(commit_error=True)

        async def run():
            with mock.patch("src.database.async_session_factory",
                            _fake_factory(session)):
                await observability.save_request_log(dict(self.RECORD))

        asyncio.run(run())  # 不抛异常即通过


class TestEndpointWiring:
    """端点接线：中间件 trace_id + chat 请求结束触发落库"""

    class _State:
        user_id = ""
        client_ip = "10.0.0.9"
        trace_id = ""

    def test_persist_request_log_builds_record(self, monkeypatch):
        """persist_request_log：从观测上下文构建完整 record 并异步落库"""
        _enable_logs(monkeypatch)

        async def run():
            observability.init_request("trace-endpoint")
            observability.timing("intent", 0.05)
            observability.record_usage("deepseek", 10, 5)
            observability.record_cache(hit=True)
            save_mock = mock.AsyncMock()
            with mock.patch.object(observability, "save_request_log", save_mock):
                req = mock.MagicMock()
                req.state = self._State()
                main_module.persist_request_log(req, "chat", intent="knowledge",
                                                error=False)
                await asyncio.sleep(0.05)  # 让后台任务完成
            return save_mock

        save_mock = asyncio.run(run())
        assert save_mock.called
        record = save_mock.call_args[0][0]
        assert record["trace_id"] == "trace-endpoint"
        assert record["identity"] == "10.0.0.9"
        assert record["endpoint"] == "chat"
        assert record["intent"] == "knowledge"
        assert record["timings"]["intent"] == 50.0
        assert record["usage"]["deepseek"] == {"prompt": 10, "completion": 5}
        assert record["cache_hits"] == 1
        assert record["error"] is False

    def test_persist_request_log_disabled_zero_write(self):
        """开关关闭（conftest 默认钉住）→ persist_request_log 零落库"""
        async def run():
            save_mock = mock.AsyncMock()
            with mock.patch.object(observability, "save_request_log", save_mock):
                req = mock.MagicMock()
                req.state = self._State()
                main_module.persist_request_log(req, "chat")
            return save_mock

        save_mock = asyncio.run(run())
        assert not save_mock.called  # 零埋点零落库

    def test_chat_endpoint_wires_trace_id(self, monkeypatch):
        """真实端点：POST /ai/rag/chat → 中间件生成 trace_id → persist_request_log 携带"""
        _enable_logs(monkeypatch)
        from rag.schemas import ChatResponse

        async def run():
            save_mock = mock.AsyncMock()
            with mock.patch.object(observability, "save_request_log", save_mock):
                with mock.patch("rag.engine.rag_engine.chat",
                                new=mock.AsyncMock(return_value=ChatResponse(
                                    answer="答案", sources=[], message="ok"))):
                    with mock.patch("main.save_messages_to_session"):
                        transport = httpx.ASGITransport(
                            app=main_module.app, raise_app_exceptions=True)
                        async with httpx.AsyncClient(
                                transport=transport, base_url="http://test") as client:
                            resp = await client.post(
                                "/ai/rag/chat",
                                json={"query": "线程池", "history": []},
                            )
                        assert resp.status_code == 200
                        await asyncio.sleep(0.05)  # 让后台落库任务完成
            return save_mock

        save_mock = asyncio.run(run())
        assert save_mock.called
        record = save_mock.call_args[0][0]
        assert record["endpoint"] == "chat"
        assert len(record["trace_id"]) == 32      # 中间件生成的 UUID hex
        assert record["identity"] == "127.0.0.1"  # 匿名 → client_ip 兜底
        assert record["error"] is False


class TestChatWithToolsUsageLabel:
    """chat_with_tools token 用量按供应商标签（Review 修复 MAJOR-2）

    旧实现恒标 "llm"（基类不感知具体供应商），工具调用轮次用量无法按
    供应商归属、fallback 链切换混在同一桶；修复后按 _provider_label 归属。
    """

    @staticmethod
    def _openai_raw(prompt: int = 100, completion: int = 20):
        """构造 OpenAI SDK 风格响应（usage 字段 + 无工具调用的 assistant 消息）"""
        raw = mock.MagicMock()
        raw.usage.prompt_tokens = prompt
        raw.usage.completion_tokens = completion
        msg = mock.MagicMock()
        msg.content = "答案"
        msg.reasoning_content = None
        msg.tool_calls = []
        raw.choices = [mock.MagicMock(message=msg)]
        return raw

    def test_openai_path_labeled_deepseek(self, monkeypatch):
        """DeepSeekClient（OpenAI 兼容路径）：usage 标签 = deepseek 而非 "llm" """
        _enable_logs(monkeypatch)
        from llm.client import DeepSeekClient

        client = object.__new__(DeepSeekClient)
        client._llm = mock.MagicMock()
        client._llm.async_client.create = mock.AsyncMock(
            return_value=self._openai_raw())

        async def run():
            observability.init_request("trace-usage-deepseek")
            await client._chat_with_tools_openai([], [])
            return observability.get_request_stats()

        stats = asyncio.run(run())
        assert stats["usage"] == {"deepseek": {"prompt": 100, "completion": 20}}

    def test_openai_path_labeled_modelscope_label(self, monkeypatch):
        """QwenClient（ModelScope 系）：usage 标签 = self._label（qwen）"""
        _enable_logs(monkeypatch)
        from llm.client import QwenClient

        client = object.__new__(QwenClient)
        client._label = "qwen"
        client._llm = mock.MagicMock()
        client._llm.async_client.create = mock.AsyncMock(
            return_value=self._openai_raw())

        async def run():
            observability.init_request("trace-usage-qwen")
            await client._chat_with_tools_openai([], [])
            return observability.get_request_stats()

        stats = asyncio.run(run())
        assert stats["usage"] == {"qwen": {"prompt": 100, "completion": 20}}

    def test_bind_path_labeled_claude(self, monkeypatch):
        """ClaudeClient（bind_tools 路径）：usage 标签 = claude"""
        _enable_logs(monkeypatch)
        from llm.client import ClaudeClient

        client = object.__new__(ClaudeClient)
        response = mock.MagicMock()
        response.usage = None  # 走 langchain response_metadata 形态
        response.content = "答案"
        response.tool_calls = []
        response.response_metadata = {
            "token_usage": {"prompt_tokens": 50, "completion_tokens": 10}}
        client._llm = mock.MagicMock()
        client._llm.bind_tools.return_value.ainvoke = mock.AsyncMock(
            return_value=response)

        async def run():
            observability.init_request("trace-usage-claude")
            await client._chat_with_tools_bind([], [])
            return observability.get_request_stats()

        stats = asyncio.run(run())
        assert stats["usage"] == {"claude": {"prompt": 50, "completion": 10}}
