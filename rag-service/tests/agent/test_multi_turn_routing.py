"""module-063 多轮对话意图路由单元测试（ADR-0015）

覆盖 WP-A~D（task-brief §三~六 + 验收 §1-4）：
- WP-A 会话级路由：classify(query, history) 空历史零回归 / LLM prompt 上下文 /
  L4 分类器 prev_user_query 拼接 2048 维
- WP-B 短句意图继承（规则层零 LLM）：去语气词 / 长度<6 无特征继承 /
  有特征正常路由（防话题漂移） / 单轮不继承
- WP-C 改写喂路由：engine.chat 改写后 query 用于路由 / precise 短路 /
  失败回退原 query / 默认关零回归
- WP-D 工具历史信号：上轮 search_knowledge → 短 query 强制 knowledge

实现说明：
- 同步 def + 函数内 asyncio.run 执行，不依赖 pytest-asyncio（套件同款模式）
- mock LLM/分类器/_deterministic_confirm，不依赖真实 DB/LLM/模型
"""
import asyncio
from unittest import mock

from agent.router import RouterAgent
from agent.intent_classifier import IntentClassifier
from rag.engine import rag_engine
from rag.schemas import ChatRequest

KNOWLEDGE_PAYLOAD = '{"intent": "knowledge", "confidence": 0.9, "reason": "知识库问题"}'
CASUAL_PAYLOAD = '{"intent": "casual_chat", "confidence": 0.9, "reason": "闲聊"}'
REALTIME_PAYLOAD = '{"intent": "realtime", "confidence": 0.9, "reason": "实时"}'

# 构造 history：上一轮知识库问题 + 助手回答（不含当前 query——当前 query 在
# request.query，history 是前序对话）
KB_HISTORY = [
    {"role": "user", "content": "什么是Java线程池？核心参数有哪些？"},
    {"role": "assistant", "content": "Java 线程池的核心参数包括核心线程数、最大线程数等。"},
]


class FakeLLMByQuery:
    """按 prompt 子串匹配返回不同 JSON payload（分类 prompt 含"用户问题: {query}"）"""

    def __init__(self, mapping=None, default=CASUAL_PAYLOAD):
        self._mapping = mapping or {}
        self._default = default

    async def generate(self, prompt: str) -> str:
        for sub, payload in self._mapping.items():
            if sub in prompt:
                return payload
        return self._default


class AsyncConfirm:
    """async 确认桩：返回固定 (confirmed, signal)"""

    def __init__(self, confirmed: bool, signal: str):
        self._confirmed = confirmed
        self._signal = signal

    async def __call__(self, query: str) -> tuple[bool, str]:
        return self._confirmed, self._signal


# ─── WP-B：去语气词 + 短句判定 ───


class TestStripParticles:
    """去语气词规则（哦/呢/呀/啦/请问/那个/嘛/吧）"""

    def test_strip_common_particles(self):
        assert RouterAgent._strip_particles("为什么呀") == "为什么"
        assert RouterAgent._strip_particles("那图谱呢") == "那图谱"
        assert RouterAgent._strip_particles("为什么呀呢") == "为什么"
        assert RouterAgent._strip_particles("那它呢") == "那它"

    def test_strip_keeps_normal_query(self):
        assert RouterAgent._strip_particles("今天天气怎么样") == "今天天气怎么样"
        assert RouterAgent._strip_particles("G1垃圾收集器是什么") == "G1垃圾收集器是什么"

    def test_last_user_turn(self):
        content, prev = RouterAgent._last_user_turn(KB_HISTORY)
        assert content == "什么是Java线程池？核心参数有哪些？"
        assert prev == []  # 最近一条 user 消息之前的历史

    def test_last_user_turn_skips_trailing_assistant(self):
        history = [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"},
                   {"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}]
        content, prev = RouterAgent._last_user_turn(history)
        assert content == "C"
        assert prev == [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]

    def test_last_user_turn_empty_history(self):
        assert RouterAgent._last_user_turn(None) == (None, None)
        assert RouterAgent._last_user_turn([]) == (None, None)


# ─── WP-A：会话级路由（空历史零回归 + LLM 上下文） ───


class TestMultiTurnRoutingWPAB:
    """WP-A/WP-B 核心路由行为"""

    @staticmethod
    def _classify(query, history=None, tool_history=None,
                  llm=None, confirm=None):
        async def run():
            agent = RouterAgent()
            if confirm is not None:
                agent._deterministic_confirm = confirm
            with mock.patch("llm.client.LLMFactory.get_client",
                            return_value=llm or FakeLLMByQuery()):
                return await agent.classify(query, history=history,
                                            tool_history=tool_history)
        return asyncio.run(run())

    def test_empty_history_zero_regression(self):
        """WP-A 通过标准①：空/None history 行为与改动前逐字一致"""
        result_none = self._classify("哈哈", history=None)
        result_empty = self._classify("哈哈", history=[])
        result_plain = self._classify("哈哈")
        for r in (result_none, result_empty, result_plain):
            assert r["intent"] == "casual_chat"
            assert r["confidence"] == 0.9
        # 三条路径结果完全一致（零回归）
        assert result_none == result_empty == result_plain

    def test_llm_prompt_includes_history_context_when_given(self):
        """WP-A：有 history 时 LLM prompt 拼对话上下文；空 history 用原模板"""
        captured = {}

        class CaptureLLM:
            async def generate(self, prompt):
                captured["prompt"] = prompt
                return CASUAL_PAYLOAD

        # 长 query（len≥6）不触发短句继承 → 走 LLM 且 history 拼上下文块
        self._classify("线程池为什么这样设计", history=KB_HISTORY,
                       llm=CaptureLLM(), confirm=AsyncConfirm(False, "no_signal"))
        assert "对话历史" in captured["prompt"]
        assert "省略句" in captured["prompt"]
        assert "什么是Java线程池" in captured["prompt"]

        captured.clear()
        self._classify("哈哈", history=None, llm=CaptureLLM(),
                       confirm=AsyncConfirm(False, "rule_veto"))
        assert "对话历史" not in captured["prompt"]  # 空 history 无上下文块

    def test_knowledge_followup_why_inherits(self):
        """[知识库, "为什么"] → knowledge（短句继承上一轮 intent）"""
        llm = FakeLLMByQuery({KB_HISTORY[0]["content"]: KNOWLEDGE_PAYLOAD})
        result = self._classify(
            "为什么", history=KB_HISTORY,
            confirm=AsyncConfirm(False, "no_signal"), llm=llm)
        assert result["intent"] == "knowledge"
        assert "短句意图继承" in result["reason"]

    def test_knowledge_followup_na_tu_pu_inherits(self):
        """[知识库, "那图谱呢"] → knowledge（去语气词"呢"后触发继承）"""
        llm = FakeLLMByQuery({KB_HISTORY[0]["content"]: KNOWLEDGE_PAYLOAD})
        result = self._classify(
            "那图谱呢", history=KB_HISTORY,
            confirm=AsyncConfirm(False, "no_signal"), llm=llm)
        assert result["intent"] == "knowledge"
        assert "短句意图继承" in result["reason"]

    def test_knowledge_followup_why_particle_inherits(self):
        """[知识库, "为什么呀"] → knowledge（去语气词"呀"后触发继承）"""
        llm = FakeLLMByQuery({KB_HISTORY[0]["content"]: KNOWLEDGE_PAYLOAD})
        result = self._classify(
            "为什么呀", history=KB_HISTORY,
            confirm=AsyncConfirm(False, "no_signal"), llm=llm)
        assert result["intent"] == "knowledge"
        assert "短句意图继承" in result["reason"]

    def test_chitchat_haha_not_inherited(self):
        """[闲聊, "哈哈"] → casual_chat（规则表 rule_veto → 正常路由不继承）"""
        result = self._classify(
            "哈哈", history=KB_HISTORY,
            confirm=AsyncConfirm(False, "rule_veto"))
        assert result["intent"] == "casual_chat"
        assert "短句意图继承" not in result["reason"]

    def test_topic_drift_weather_normal_routing(self):
        """[知识库, "今天天气怎么样"] → realtime（正常长度 → 正常路由，不继承）"""
        result = self._classify(
            "今天天气怎么样", history=KB_HISTORY,
            llm=FakeLLMByQuery(default=REALTIME_PAYLOAD),
            confirm=AsyncConfirm(False, "rule_veto"))
        assert result["intent"] == "realtime"
        assert "短句意图继承" not in result["reason"]

    def test_single_turn_short_query_no_inherit(self):
        """单轮（无 history）短 query → 正常路由（无上轮可继承）"""
        result = self._classify("为什么", history=None,
                                confirm=AsyncConfirm(False, "no_signal"))
        assert result["intent"] == "casual_chat"  # 走 LLM（mock 默认闲聊）
        assert "短句意图继承" not in result["reason"]

    def test_has_feature_forces_normal_routing(self):
        """有特征（FTS 术语命中）→ 必须正常路由，不继承（防话题漂移）"""
        llm = FakeLLMByQuery(default=KNOWLEDGE_PAYLOAD)
        result = self._classify(
            "那图谱呢", history=KB_HISTORY,
            confirm=AsyncConfirm(True, "fts_term"), llm=llm)
        assert result["intent"] == "knowledge"
        assert "短句意图继承" not in result["reason"]  # 正常路由非继承

    def test_short_query_with_long_prev_chain(self):
        """省略句链式继承：上一轮也是省略句时逐层回退到最近完整问题"""
        history = [
            {"role": "user", "content": "G1垃圾收集器是什么"},
            {"role": "assistant", "content": "G1 是 Garbage First 收集器。"},
            {"role": "user", "content": "为什么"},
            {"role": "assistant", "content": "因为它的停顿可预测。"},
        ]
        llm = FakeLLMByQuery({"G1垃圾收集器是什么": KNOWLEDGE_PAYLOAD})
        # 当前"为什么" → 继承 → 上一轮"为什么"（省略句）→ 继承 → "G1垃圾收集器是什么" → knowledge
        result = self._classify(
            "为什么", history=history, confirm=AsyncConfirm(False, "no_signal"),
            llm=llm)
        assert result["intent"] == "knowledge"
        assert "短句意图继承" in result["reason"]

    def test_history_capped_to_six(self):
        """路由只用最近 4-6 轮（task-brief §八.5：历史不全塞）"""
        long_history = []
        for i in range(20):
            long_history.append({"role": "user", "content": f"问题{i}"})
            long_history.append({"role": "assistant", "content": f"回答{i}"})
        content, prev = RouterAgent._last_user_turn(long_history[-6:])
        assert content == "问题19"
        assert len(prev) <= 4


# ─── WP-A：L4 分类器 prev_user_query 拼接（2048 维） ───


class _FakeEmbedding1024:
    """mock 特征：确定性 1024 维向量（模块内联，简单可判定）"""

    def __init__(self):
        self._calls = []

    async def embed_text(self, text: str) -> list:
        self._calls.append(text)
        return [1.0 if "为什么" in text else 0.0] * 1024

    async def embed_documents(self, texts: list) -> list:
        return [await self.embed_text(t) for t in texts]


class _RecordingModel:
    """记录 predict_proba 输入特征（验证维度拼接）"""

    def __init__(self):
        self.features = None
        self.classes_ = ["knowledge", "casual_chat", "realtime"]

    def predict_proba(self, X):
        self.features = X
        return [[0.9, 0.05, 0.05]] * len(X)


class TestIntentClassifierConcat:
    """WP-A：predict_proba 支持 prev_user_query 拼接（2048 维）"""

    def test_concat_feature_2048_dim(self):
        model = _RecordingModel()
        clf = IntentClassifier(model_path="x.joblib",
                               embedding_service=_FakeEmbedding1024())
        clf._model = model
        probs = asyncio.run(clf.predict_proba(
            "为什么", prev_user_query="什么是Java线程池"))
        assert model.features is not None
        assert len(model.features[0]) == 2048  # 1024 + 1024
        assert set(probs) == {"knowledge", "casual_chat", "realtime"}

    def test_no_prev_single_dim(self):
        model = _RecordingModel()
        clf = IntentClassifier(model_path="x.joblib",
                               embedding_service=_FakeEmbedding1024())
        clf._model = model
        asyncio.run(clf.predict_proba("什么是Java线程池"))
        assert len(model.features[0]) == 1024  # 单 query 存量契约

    def test_prev_blank_falls_back_single(self):
        model = _RecordingModel()
        clf = IntentClassifier(model_path="x.joblib",
                               embedding_service=_FakeEmbedding1024())
        clf._model = model
        asyncio.run(clf.predict_proba("为什么", prev_user_query="   "))
        assert len(model.features[0]) == 1024

    def test_router_passes_prev_when_flag_on(self, monkeypatch):
        """intent_classifier_multi_turn=True 时 router 给 L4 传最近一轮 user query"""
        from src.config import settings
        monkeypatch.setattr(settings, "intent_classifier_multi_turn", True)

        class Recorder:
            def __init__(self):
                self.prev = None
                self.calls = 0

            async def load(self):
                return True

            async def predict_proba(self, query, prev_user_query=None):
                self.calls += 1
                self.prev = prev_user_query
                return {"knowledge": 0.8, "casual_chat": 0.1, "realtime": 0.1}

        rec = Recorder()
        agent = RouterAgent(intent_classifier=rec)

        async def run():
            # 长 query 不触发短句继承 → 走 L4 分类器路径
            return await agent.classify(
                "线程池为什么这样设计", history=KB_HISTORY)
        result = asyncio.run(run())
        assert rec.calls == 1
        assert rec.prev == "什么是Java线程池？核心参数有哪些？"
        assert result["intent"] == "knowledge"

    def test_router_single_turn_keeps_no_prev(self, monkeypatch):
        """intent_classifier_multi_turn=True 但无 history → 不传 prev（单 query）"""
        from src.config import settings
        monkeypatch.setattr(settings, "intent_classifier_multi_turn", True)

        class Recorder:
            def __init__(self):
                self.prev = "sentinel"
                self.calls = 0

            async def load(self):
                return True

            async def predict_proba(self, query, prev_user_query=None):
                self.calls += 1
                self.prev = prev_user_query
                return {"knowledge": 0.8, "casual_chat": 0.1, "realtime": 0.1}

        rec = Recorder()
        agent = RouterAgent(intent_classifier=rec)
        result = asyncio.run(agent.classify("什么是Java线程池"))
        assert rec.calls == 1
        assert rec.prev is None
        assert result["intent"] == "knowledge"


class _CasualL4Classifier:
    """注入 router 的假 L4：把多轮省略句误判为 casual（真实分类器无历史上下文）"""

    async def load(self):
        return True

    async def predict_proba(self, query, prev_user_query=None):
        return {"knowledge": 0.28, "casual_chat": 0.65, "realtime": 0.07}


class TestL4L2Correction:
    """module-063：L4 路径同样走 L2 确定性信号确认（与 LLM 路径同款安全网）

    eval/golden_multi_turn 真实测量暴露：L4 单句分类无历史上下文，多轮省略句
    （"怎么解决呢"）可能被误判 casual → L2 fts_term 修正为 knowledge（零 LLM，
    红线不变；module-055 已证 L2 信号精确——golden 50 条非 knowledge 样本
    误确认 0）。
    """

    def test_l4_casual_corrected_by_l2_fts_term(self):
        """L4 判 casual 且 FTS 术语命中 → L2 修正为 knowledge"""
        agent = RouterAgent(intent_classifier=_CasualL4Classifier())
        agent._deterministic_confirm = AsyncConfirm(True, "fts_term")
        with mock.patch("llm.client.LLMFactory.get_client",
                        return_value=FakeLLMByQuery()):
            result = asyncio.run(agent.classify("怎么解决呢", history=[
                {"role": "user", "content": "Spring事务失效的场景有哪些？"}]))
        assert result["intent"] == "knowledge"
        assert result["confidence"] == 0.28  # L2 修正后取 knowledge 概率

    def test_l4_casual_kept_when_no_signal(self):
        """L4 判 casual 且无信号 → 保持 casual（真闲聊不误转）"""
        agent = RouterAgent(intent_classifier=_CasualL4Classifier())
        agent._deterministic_confirm = AsyncConfirm(False, "no_signal")
        with mock.patch("llm.client.LLMFactory.get_client",
                        return_value=FakeLLMByQuery()):
            result = asyncio.run(agent.classify("哈哈", history=[]))
        assert result["intent"] == "casual_chat"
        assert result["confidence"] == 0.65


# ─── WP-D：工具历史信号（规则层，轨迹不可得跳过） ───


class TestToolHistorySignal:
    """WP-D：上一轮 search_knowledge/generate_answer → 短 query 强制 knowledge"""

    @staticmethod
    def _classify(query, history, tool_history, confirm, llm=None):
        async def run():
            agent = RouterAgent()
            agent._deterministic_confirm = confirm
            with mock.patch("llm.client.LLMFactory.get_client",
                            return_value=llm or FakeLLMByQuery()):
                return await agent.classify(query, history=history,
                                            tool_history=tool_history)
        return asyncio.run(run())

    def test_kb_tool_history_forces_knowledge(self):
        result = self._classify(
            "为什么", KB_HISTORY, ["search_knowledge"],
            AsyncConfirm(False, "no_signal"))
        assert result["intent"] == "knowledge"
        assert "工具历史信号" in result["reason"]

    def test_generate_tool_history_forces_knowledge(self):
        result = self._classify(
            "为什么", KB_HISTORY, ["generate_answer"],
            AsyncConfirm(False, "no_signal"))
        assert result["intent"] == "knowledge"

    def test_non_kb_tool_history_normal_path(self):
        """上轮非知识工具（如 chat）→ 不强制，走短句继承（上一轮 knowledge）"""
        llm = FakeLLMByQuery({KB_HISTORY[0]["content"]: KNOWLEDGE_PAYLOAD})
        result = self._classify(
            "为什么", KB_HISTORY, ["chat"],
            AsyncConfirm(False, "no_signal"), llm=llm)
        # 工具信号不命中 → 短句继承（prev 路由 knowledge → 继承 knowledge）
        assert result["intent"] == "knowledge"
        assert "短句意图继承" in result["reason"]

    def test_no_tool_history_skips(self):
        """轨迹不可得（tool_history=None）→ 跳过工具信号（不阻塞）"""
        result = self._classify(
            "为什么", KB_HISTORY, None, AsyncConfirm(False, "no_signal"))
        assert "工具历史信号" not in result["reason"]
        assert "短句意图继承" in result["reason"]

    def test_normal_length_query_ignores_tool_signal(self):
        """工具信号只对短 query 生效（正常长度必须重新路由）"""
        result = self._classify(
            "线程池为什么这样设计", KB_HISTORY, ["search_knowledge"],
            AsyncConfirm(False, "no_signal"))
        assert "工具历史信号" not in result["reason"]


# ─── WP-C：改写喂路由（engine.chat 接入） ───


class TestEngineRewriteFeedsRouting:
    """WP-C：改写结果同时喂路由 + 检索；precise 短路；失败回退原 query"""

    FAKE_DOC = {"id": 1, "title": "t", "content": "c", "source": "s",
                "parent_id": None}
    REWRITTEN = "Java线程池 为什么核心线程数这样设置"

    @staticmethod
    def _chat(monkeypatch, query, history, prepare_ret=None):
        """跑一次 engine.chat，捕获 classify 调用"""
        from src.config import settings
        captured = {}

        async def run():
            with mock.patch("rag.engine.router_agent.classify",
                            new=mock.AsyncMock(
                                return_value={"intent": "knowledge",
                                              "confidence": 0.9})) as clf:
                with mock.patch("rag.engine.hybrid_retriever.retrieve",
                                new=mock.AsyncMock(return_value=[TestEngineRewriteFeedsRouting.FAKE_DOC])):
                    with mock.patch("rag.engine.reranker.rerank",
                                    new=mock.AsyncMock(side_effect=lambda q, d, top_k: d)):
                        with mock.patch("agent.reflector.reflector.check_sufficiency",
                                        new=mock.AsyncMock(
                                            return_value={"sufficient": True})):
                            with mock.patch("agent.reflector.reflector.generate_answer",
                                            new=mock.AsyncMock(return_value="答案")):
                                with mock.patch("agent.reflector.reflector.verify_answer",
                                                new=mock.AsyncMock(return_value=None)):
                                    with mock.patch("rag.engine.rag_engine._recall_memory",
                                                    new=mock.AsyncMock(return_value="")):
                                        with mock.patch(
                                            "rag.engine.rag_engine._resolve_session_history",
                                            new=mock.AsyncMock(side_effect=lambda i, h: h)):
                                            with mock.patch.object(
                                                rag_engine, "_persist_memory",
                                                new=mock.AsyncMock()):
                                                with mock.patch.object(
                                                    rag_engine, "_persist_session",
                                                    new=mock.AsyncMock()):
                                                    await rag_engine.chat(
                                                        ChatRequest(query=query,
                                                                    history=history),
                                                        identity="x")
            captured["clf"] = clf
        asyncio.run(run())
        return captured["clf"]

    def test_rewrite_success_feeds_routing(self, monkeypatch):
        """改写成功且保真通过 → 路由用改写后 query + history"""
        from src.config import settings
        monkeypatch.setattr(settings, "query_rewrite_enabled", True)
        history = [{"role": "user", "content": "什么是Java线程池"}]
        ret = (self.REWRITTEN, [self.FAKE_DOC],
               {"mode": "parallel", "used_rewrite": True, "rewritten": self.REWRITTEN})
        with mock.patch("rag.engine.query_rewrite.prepare",
                        new=mock.AsyncMock(return_value=ret)):
            clf = self._chat(monkeypatch, "为什么", history)
        assert clf.call_count == 1
        assert clf.call_args.args == (self.REWRITTEN,)  # 改写后 query 路由
        assert clf.call_args.kwargs["history"] == history

    def test_precise_triage_shortcircuits_knowledge(self, monkeypatch):
        """分诊命中 FTS 术语（precise）且非规则词 → 短路 knowledge，省一次路由"""
        from src.config import settings
        monkeypatch.setattr(settings, "query_rewrite_enabled", True)
        ret = ("G1垃圾收集器", None, {"mode": "precise"})
        with mock.patch("rag.engine.query_rewrite.prepare",
                        new=mock.AsyncMock(return_value=ret)):
            clf = self._chat(monkeypatch, "G1垃圾收集器", [])
        assert clf.call_count == 0  # 短路 → 不调 classify

    def test_precise_but_rule_word_not_shortcircuit(self, monkeypatch):
        """precise 但命中闲聊/实时规则词 → 不短路（防"你好"被强归 knowledge）"""
        from src.config import settings
        monkeypatch.setattr(settings, "query_rewrite_enabled", True)
        ret = ("你好", None, {"mode": "precise"})
        with mock.patch("rag.engine.query_rewrite.prepare",
                        new=mock.AsyncMock(return_value=ret)):
            clf = self._chat(monkeypatch, "你好", [])
        assert clf.call_count == 1  # 正常路由（不短路）
        assert clf.call_args.args == ("你好",)

    def test_rewrite_fallback_uses_original_query(self, monkeypatch):
        """改写失败/回退 → 原始 query 路由（零回归）"""
        from src.config import settings
        monkeypatch.setattr(settings, "query_rewrite_enabled", True)
        history = [{"role": "user", "content": "什么是Java线程池"}]
        ret = ("为什么", None, {"mode": "rewrite_fallback"})
        with mock.patch("rag.engine.query_rewrite.prepare",
                        new=mock.AsyncMock(return_value=ret)):
            clf = self._chat(monkeypatch, "为什么", history)
        assert clf.call_args.args == ("为什么",)  # 原始 query
        assert clf.call_args.kwargs["history"] == history

    def test_rewrite_disabled_uses_original_and_history(self, monkeypatch):
        """改写默认关（query_rewrite_enabled=False）→ 原始 query + history 路由"""
        from src.config import settings
        monkeypatch.setattr(settings, "query_rewrite_enabled", False)
        history = [{"role": "user", "content": "什么是Java线程池"}]
        clf = self._chat(monkeypatch, "为什么", history)
        assert clf.call_args.args == ("为什么",)
        assert clf.call_args.kwargs["history"] == history


class TestStreamingWiresHistory:
    """纪律 §八.2：流式路径（main.chat_stream）与 LangGraph 路径都接 history"""

    def test_chat_stream_passes_history(self):
        """chat_stream Step 1 路由调用带 history（空 history 零回归）"""
        import main as main_module
        from rag.schemas import ChatRequest

        called = {}
        history = [{"role": "user", "content": "什么是Java线程池"}]

        async def fake_classify(query, history=None, tool_history=None):
            called["query"] = query
            called["history"] = history
            return {"intent": "casual_chat", "confidence": 0.9, "reason": "x"}

        async def run():
            # fastapi_req=None 会令 resolve_identity 崩（None.state）→ mock 身份
            with mock.patch("main.resolve_identity", return_value="unknown"):
                # 流式 finally 的 persist_request_log(None) 会崩 → mock 掉
                with mock.patch("main.persist_request_log",
                                new=mock.AsyncMock()):
                    with mock.patch("agent.router.router_agent.classify",
                                    new=mock.AsyncMock(side_effect=fake_classify)):
                        resp = await main_module.chat_stream(
                            ChatRequest(query="哈哈", history=history), None)
                        # body_iterator = event_stream 生成器；拉第一个事件（intent step）
                        # 即触发 classify 调用，casual 分支/后续不执行
                        await anext(resp.body_iterator)
        asyncio.run(run())
        assert called.get("history") == history  # 流式路径接 history
        assert called.get("query") == "哈哈"

    def test_langgraph_classify_intent_passes_history(self):
        """rag/graph/graph.py classify_intent 带 history（空 history 零回归）"""
        import rag.graph as graph_module
        from rag.state import make_initial_state

        captured = {}

        async def fake_classify(query, history=None, tool_history=None):
            captured["history"] = history
            return {"intent": "knowledge", "confidence": 0.9, "reason": "x"}

        with mock.patch("agent.router.router_agent.classify",
                        new=mock.AsyncMock(side_effect=fake_classify)):
            state = make_initial_state("为什么", [
                {"role": "user", "content": "什么是Java线程池"}])
            asyncio.run(graph_module.classify_intent(state))
        assert captured["history"] == [{"role": "user", "content": "什么是Java线程池"}]

        # 空 history 零回归
        captured.clear()
        with mock.patch("agent.router.router_agent.classify",
                        new=mock.AsyncMock(side_effect=fake_classify)):
            state = make_initial_state("为什么", [])
            asyncio.run(graph_module.classify_intent(state))
        assert captured["history"] == []
