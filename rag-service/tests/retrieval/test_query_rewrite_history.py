"""module-072 WP-A：上下文改写接入生产链单元测试（query_rewrite.py history 分支 + engine 接线）

覆盖（验收 §1.1）：
- llm_rewrite prev 分支：prompt 走上下文改写模板（含"上一轮问题"/"当前省略句"段）；
  prev=None → 049 原模板（含"用户问题"段）逐字零回归
- llm_rewrite prev 失败/超时/空/无变化 → None（调用方回退原 query，链路不中断）
- extract_prev：取最近一条 user 消息 content；非 user/非字符串/空白 → None
- prepare/prepare_query：history 透传 prev；保真锚点 = f"{prev} {query}"（拼接双锚）；
  无 history → 锚点 = query（049 零回归）；precise 不受 history 影响
- contextual_rewrite 单一来源封装（golden_multi_turn 调用入口）：
  triage precise → None；保真未过 → None；成功 → 改写文本
- engine.chat：contextual 开 → prepare 收到 history；两开关全关 → prepare 不调；
  query_rewrite=false + contextual=true 独立生效（prepare 调用条件 OR）
- engine._retrieve：contextual 开 → prepare_query 收到 history；
  缓存 key 含 prev 哈希（防同 query 不同 prev 串话题）

实现说明：全部 mock（零真实 LLM/DB/嵌入）；同步用例内 asyncio.run（套件同款，
不依赖 pytest-asyncio）；打桩注入 retrieve_fn / 子函数。
"""
import asyncio
from unittest import mock

import rag.engine as engine_module
from rag import query_rewrite as qr
from rag.engine import rag_engine

ORIG = "为什么"
PREV = "什么是Java线程池？核心参数有哪些？"
REWRITTEN = "Java线程池为什么核心线程数这样设置"
HISTORY = [{"role": "user", "content": PREV}]


# ─── llm_rewrite prev 分支（WP-A 实现要点 1） ───

class TestLlmRewritePrev:
    """prev 非空 → 上下文改写 prompt；prev=None → 049 原 prompt（逐字零回归）"""

    @staticmethod
    def _patch_client(generate):
        fake = mock.MagicMock()
        fake.generate = generate
        return mock.patch.object(qr.LLMFactory, "get_client", return_value=fake)

    def test_prev_uses_contextual_prompt(self):
        captured = {}

        async def generate(prompt):
            captured["prompt"] = prompt
            return REWRITTEN

        async def run():
            with self._patch_client(mock.AsyncMock(side_effect=generate)):
                return await qr.llm_rewrite(ORIG, prev=PREV)

        assert asyncio.run(run()) == REWRITTEN
        assert "上一轮问题: " + PREV in captured["prompt"]
        assert "当前省略句: " + ORIG in captured["prompt"]

    def test_no_prev_uses_049_prompt(self):
        captured = {}

        async def generate(prompt):
            captured["prompt"] = prompt
            return REWRITTEN

        async def run():
            with self._patch_client(mock.AsyncMock(side_effect=generate)):
                return await qr.llm_rewrite(ORIG)

        assert asyncio.run(run()) == REWRITTEN
        assert "用户问题: " + ORIG in captured["prompt"]
        assert "上一轮问题" not in captured["prompt"]

    def test_prev_timeout_returns_none(self, monkeypatch):
        monkeypatch.setattr(qr, "_REWRITE_TIMEOUT", 0.01)

        async def slow_generate(prompt):
            await asyncio.sleep(0.1)
            return REWRITTEN

        async def run():
            with self._patch_client(mock.AsyncMock(side_effect=slow_generate)):
                return await qr.llm_rewrite(ORIG, prev=PREV)

        assert asyncio.run(run()) is None

    def test_prev_error_returns_none(self):
        async def run():
            with self._patch_client(
                    mock.AsyncMock(side_effect=RuntimeError("llm down"))):
                return await qr.llm_rewrite(ORIG, prev=PREV)

        assert asyncio.run(run()) is None

    def test_prev_empty_returns_none(self):
        async def run():
            with self._patch_client(mock.AsyncMock(return_value="  ")):
                return await qr.llm_rewrite(ORIG, prev=PREV)

        assert asyncio.run(run()) is None

    def test_prev_unchanged_returns_none(self):
        # 改写结果与原句相同 → 视为无效（避免无意义并行）
        async def run():
            with self._patch_client(mock.AsyncMock(return_value=ORIG)):
                return await qr.llm_rewrite(ORIG, prev=PREV)

        assert asyncio.run(run()) is None


# ─── extract_prev（history → prev，取最近一条 user 消息） ───

class TestExtractPrev:
    """history 提取：最近一条 user 消息 content；无 user/非字符串 → None"""

    def test_last_user_message(self):
        history = [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧回答"},
            {"role": "user", "content": PREV},
        ]
        assert qr.extract_prev(history) == PREV

    def test_empty_and_none(self):
        assert qr.extract_prev(None) is None
        assert qr.extract_prev([]) is None

    def test_no_user_message(self):
        assert qr.extract_prev([{"role": "assistant", "content": "x"}]) is None

    def test_skip_non_string_content(self):
        history = [{"role": "user", "content": ["多模态", "数组"]},
                   {"role": "user", "content": PREV}]
        assert qr.extract_prev(history) == PREV

    def test_skip_blank_content(self):
        assert qr.extract_prev([{"role": "user", "content": "  "}]) is None


# ─── prepare / prepare_query history 透传 + 拼接双锚（WP-A 实现要点 2） ───

class TestPrepareHistory:
    """history 非空 → prev 透传 llm_rewrite；保真锚点 = prev+query 拼接"""

    @staticmethod
    def _stub_retrieve():
        return mock.AsyncMock(return_value=[{
            "id": 1, "title": "t", "content": "c",
            "abs_cosine": 0.7, "parent_id": None}])

    def test_prepare_passes_prev_and_double_anchor(self):
        captured = {}

        async def fake_rewrite(query, prev=None):
            captured["query"] = query
            captured["prev"] = prev
            return REWRITTEN

        async def fake_fidelity(original, rewritten):
            captured["anchor"] = original
            return 0.8

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(side_effect=fake_rewrite)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(side_effect=fake_fidelity)):
                        return await qr.prepare(ORIG, self._stub_retrieve(),
                                                history=HISTORY)

        _, docs, info = asyncio.run(run())
        assert captured["query"] == ORIG
        assert captured["prev"] == PREV
        # 保真锚点 = 拼接双锚（主题+原句），非裸省略句
        assert captured["anchor"] == f"{PREV} {ORIG}"
        assert info["mode"] == "parallel"

    def test_prepare_without_history_anchor_is_query(self):
        captured = {}

        async def fake_rewrite(query, prev=None):
            captured["prev"] = prev
            return REWRITTEN

        async def fake_fidelity(original, rewritten):
            captured["anchor"] = original
            return 0.8

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(side_effect=fake_rewrite)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(side_effect=fake_fidelity)):
                        return await qr.prepare(ORIG, self._stub_retrieve())

        _, _, info = asyncio.run(run())
        assert captured["prev"] is None
        # 无 history → 锚点 = 原 query（049 语义逐字零回归）
        assert captured["anchor"] == ORIG
        assert info["mode"] == "parallel"

    def test_prepare_precise_ignores_history(self):
        """precise 分支不受 history 影响：不调 LLM、不并行（precise 零 LLM）"""
        llm_mock = mock.AsyncMock()

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="precise")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=llm_mock):
                    return await qr.prepare(ORIG, self._stub_retrieve(),
                                            history=HISTORY)

        search, docs, info = asyncio.run(run())
        assert search == ORIG
        assert docs is None
        assert info["mode"] == "precise"
        llm_mock.assert_not_called()

    def test_prepare_query_passes_prev_and_double_anchor(self):
        captured = {}

        async def fake_rewrite(query, prev=None):
            captured["prev"] = prev
            return REWRITTEN

        async def fake_fidelity(original, rewritten):
            captured["anchor"] = original
            return 0.8

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(side_effect=fake_rewrite)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(side_effect=fake_fidelity)):
                        return await qr.prepare_query(ORIG, history=HISTORY)

        base, info = asyncio.run(run())
        assert captured["prev"] == PREV
        assert captured["anchor"] == f"{PREV} {ORIG}"
        assert base == REWRITTEN
        assert info["mode"] == "rewrite_accepted"

    def test_prepare_query_no_history_anchor_is_query(self):
        captured = {}

        async def fake_rewrite(query, prev=None):
            captured["prev"] = prev
            return REWRITTEN

        async def fake_fidelity(original, rewritten):
            captured["anchor"] = original
            return 0.8

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(side_effect=fake_rewrite)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(side_effect=fake_fidelity)):
                        return await qr.prepare_query(ORIG)

        base, _info = asyncio.run(run())
        assert captured["prev"] is None
        assert captured["anchor"] == ORIG
        assert base == REWRITTEN


# ─── contextual_rewrite 单一来源封装（golden_multi_turn 调用入口） ───

class TestContextualRewrite:
    """生产封装：triage precise → None；保真未过 → None；成功 → 改写"""

    def test_triage_precise_returns_none(self):
        """句子已自包含（如"那CMS呢"术语命中）→ 不改写（precise 零 LLM）"""
        llm_mock = mock.AsyncMock()

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="precise")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=llm_mock):
                    return await qr.contextual_rewrite(PREV, ORIG)

        assert asyncio.run(run()) is None
        llm_mock.assert_not_called()

    def test_fidelity_reject_returns_none(self):
        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(return_value=REWRITTEN)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(return_value=0.3)):
                        return await qr.contextual_rewrite(PREV, ORIG)

        assert asyncio.run(run()) is None

    def test_success_returns_rewritten(self):
        captured = {}

        async def fake_fidelity(original, rewritten):
            captured["anchor"] = original
            return 0.8

        async def run():
            with mock.patch("rag.query_rewrite.triage",
                            new=mock.AsyncMock(return_value="vague")):
                with mock.patch("rag.query_rewrite.llm_rewrite",
                                new=mock.AsyncMock(return_value=REWRITTEN)):
                    with mock.patch("rag.query_rewrite.fidelity_check",
                                    new=mock.AsyncMock(side_effect=fake_fidelity)):
                        return await qr.contextual_rewrite(PREV, ORIG)

        assert asyncio.run(run()) == REWRITTEN
        assert captured["anchor"] == f"{PREV} {ORIG}"


# ─── engine 接线（WP-A 实现要点 3） ───

class TestEngineContextualWiring:
    """engine.chat / _retrieve：开关组合 + history 透传 + 缓存 key"""

    FAKE_DOC = {"id": 1, "title": "t", "content": "c", "source": "s",
                "hybrid_score": 0.9, "parent_id": None}

    @staticmethod
    def _chat_patches(prepare_ret):
        """engine.chat 最小 mock 栈（resolve_tool_history 钉住 None 保 hermetic）"""
        return [
            mock.patch("rag.engine.resolve_tool_history",
                       new=mock.AsyncMock(return_value=None)),
            mock.patch("rag.engine.router_agent.classify",
                       new=mock.AsyncMock(
                           return_value={"intent": "knowledge",
                                         "confidence": 0.9})),
            mock.patch("rag.engine.memory_service.recall",
                       new=mock.AsyncMock(return_value=[])),
            mock.patch("rag.engine.memory_service.recall_short",
                       new=mock.AsyncMock(return_value=[])),
            mock.patch("rag.engine.hybrid_retriever.retrieve",
                       new=mock.AsyncMock(return_value=[TestEngineContextualWiring.FAKE_DOC])),
            mock.patch("rag.engine.reranker.rerank",
                       new=mock.AsyncMock(
                           side_effect=lambda q, d, top_k=5: d)),
            mock.patch("agent.reflector.reflector.check_sufficiency",
                       new=mock.AsyncMock(return_value={"sufficient": True})),
            mock.patch("agent.reflector.reflector.generate_answer",
                       new=mock.AsyncMock(return_value="答案")),
            mock.patch("agent.reflector.reflector.verify_answer",
                       new=mock.AsyncMock(return_value=None)),
            mock.patch("rag.engine.query_rewrite.prepare",
                       new=mock.AsyncMock(return_value=prepare_ret)),
            mock.patch.object(rag_engine, "_persist_memory", new=mock.AsyncMock()),
            mock.patch.object(rag_engine, "_persist_session", new=mock.AsyncMock()),
        ]

    def test_chat_contextual_on_passes_history(self, monkeypatch):
        from rag.schemas import ChatRequest

        monkeypatch.setattr(engine_module.settings, "contextual_rewrite_enabled", True)
        history = [{"role": "user", "content": PREV}]
        ret = (ORIG, None, {"mode": "rewrite_fallback"})
        patches = self._chat_patches(ret)
        prepare_mock = patches[9].new

        async def run():
            for p in patches:
                p.start()
            try:
                await rag_engine.chat(ChatRequest(query=ORIG, history=history),
                                      identity="x")
            finally:
                for p in reversed(patches):
                    p.stop()

        asyncio.run(run())
        assert prepare_mock.call_count == 1
        assert prepare_mock.call_args.kwargs["history"] == history

    def test_chat_contextual_off_passes_none(self, monkeypatch):
        """contextual=false + query_rewrite=true → prepare 收到 history=None（049 零回归）"""
        from rag.schemas import ChatRequest

        monkeypatch.setattr(engine_module.settings, "query_rewrite_enabled", True)
        monkeypatch.setattr(engine_module.settings, "contextual_rewrite_enabled", False)
        history = [{"role": "user", "content": PREV}]
        ret = (ORIG, None, {"mode": "rewrite_fallback"})
        patches = self._chat_patches(ret)
        prepare_mock = patches[9].new

        async def run():
            for p in patches:
                p.start()
            try:
                await rag_engine.chat(ChatRequest(query=ORIG, history=history),
                                      identity="x")
            finally:
                for p in reversed(patches):
                    p.stop()

        asyncio.run(run())
        assert prepare_mock.call_count == 1
        assert prepare_mock.call_args.kwargs["history"] is None

    def test_chat_both_off_prepare_not_called(self, monkeypatch):
        """两开关全关 → 生产行为与改动前逐字一致（零回归）"""
        from rag.schemas import ChatRequest

        monkeypatch.setattr(engine_module.settings, "query_rewrite_enabled", False)
        monkeypatch.setattr(engine_module.settings, "contextual_rewrite_enabled", False)
        ret = (ORIG, None, {"mode": "rewrite_fallback"})
        patches = self._chat_patches(ret)
        prepare_mock = patches[9].new

        async def run():
            for p in patches:
                p.start()
            try:
                await rag_engine.chat(ChatRequest(query=ORIG, history=HISTORY),
                                      identity="x")
            finally:
                for p in reversed(patches):
                    p.stop()

        asyncio.run(run())
        prepare_mock.assert_not_called()

    def test_chat_contextual_independent_of_query_rewrite(self, monkeypatch):
        """query_rewrite=false + contextual=true → prepare 独立生效（调用条件 OR）"""
        from rag.schemas import ChatRequest

        monkeypatch.setattr(engine_module.settings, "query_rewrite_enabled", False)
        monkeypatch.setattr(engine_module.settings, "contextual_rewrite_enabled", True)
        ret = (ORIG, None, {"mode": "rewrite_fallback"})
        patches = self._chat_patches(ret)
        prepare_mock = patches[9].new

        async def run():
            for p in patches:
                p.start()
            try:
                await rag_engine.chat(ChatRequest(query=ORIG, history=HISTORY),
                                      identity="x")
            finally:
                for p in reversed(patches):
                    p.stop()

        asyncio.run(run())
        assert prepare_mock.call_count == 1

    def test_retrieve_passes_history_to_prepare_query(self, monkeypatch):
        """contextual 开 → _retrieve 透传 history 给 prepare_query"""
        monkeypatch.setattr(engine_module.settings, "contextual_rewrite_enabled", True)
        captured = {}

        async def fake_prepare_query(query, history=None):
            captured["history"] = history
            return ORIG, {"mode": "rewrite_fallback"}

        async def run():
            with mock.patch("rag.engine.cache.get", mock.AsyncMock(return_value=None)):
                with mock.patch("rag.engine.cache.set", mock.AsyncMock(return_value=True)):
                    with mock.patch.object(rag_engine, "_hyde_expand",
                                           mock.AsyncMock(return_value="HyDE")):
                        with mock.patch("rag.engine.hybrid_retriever.retrieve",
                                        mock.AsyncMock(return_value=[self.FAKE_DOC])):
                            with mock.patch("rag.engine.graph_extractor.extract_from_query",
                                            mock.AsyncMock(return_value=[])):
                                with mock.patch("rag.engine.graph_store.search_related",
                                                mock.AsyncMock(return_value=[])):
                                    with mock.patch("agent.reflector.reflector.check_sufficiency",
                                                    mock.AsyncMock(return_value={"sufficient": True})):
                                        with mock.patch("rag.engine.query_rewrite.prepare_query",
                                                        new=mock.AsyncMock(side_effect=fake_prepare_query)):
                                            return await rag_engine._retrieve(
                                                ORIG, history=HISTORY)

        docs = asyncio.run(run())
        assert captured["history"] == HISTORY
        assert len(docs) == 1

    def test_retrieve_cache_key_suffix_with_prev(self, monkeypatch):
        """contextual 开 + history → 缓存 key 含 prev 哈希（防同 query 串话题）"""
        monkeypatch.setattr(engine_module.settings, "contextual_rewrite_enabled", True)
        captured = {}

        async def fake_cache_get(key):
            captured["key"] = key
            return None

        async def run():
            with mock.patch("rag.engine.cache.get",
                            new=mock.AsyncMock(side_effect=fake_cache_get)):
                with mock.patch("rag.engine.cache.set", mock.AsyncMock(return_value=True)):
                    with mock.patch.object(rag_engine, "_hyde_expand",
                                           mock.AsyncMock(return_value="HyDE")):
                        with mock.patch("rag.engine.hybrid_retriever.retrieve",
                                        mock.AsyncMock(return_value=[self.FAKE_DOC])):
                            with mock.patch("rag.engine.graph_extractor.extract_from_query",
                                            mock.AsyncMock(return_value=[])):
                                with mock.patch("rag.engine.graph_store.search_related",
                                                mock.AsyncMock(return_value=[])):
                                    with mock.patch("agent.reflector.reflector.check_sufficiency",
                                                    mock.AsyncMock(return_value={"sufficient": True})):
                                        with mock.patch("rag.engine.query_rewrite.prepare_query",
                                                        mock.AsyncMock(return_value=(ORIG, {"mode": "precise"}))):
                                            return await rag_engine._retrieve(
                                                ORIG, history=HISTORY)

        asyncio.run(run())
        assert ":ctx:" in captured["key"]
        assert captured["key"] != engine_module._retrieve_cache_key(ORIG, 30, 0.6)

    def test_retrieve_cache_key_no_suffix_when_off(self, monkeypatch):
        """contextual 关（默认）→ 缓存 key 与 049 完全一致（零回归）"""
        monkeypatch.setattr(engine_module.settings, "contextual_rewrite_enabled", False)
        captured = {}

        async def fake_cache_get(key):
            captured["key"] = key
            return None

        async def run():
            with mock.patch("rag.engine.cache.get",
                            new=mock.AsyncMock(side_effect=fake_cache_get)):
                with mock.patch("rag.engine.cache.set", mock.AsyncMock(return_value=True)):
                    with mock.patch.object(rag_engine, "_hyde_expand",
                                           mock.AsyncMock(return_value="HyDE")):
                        with mock.patch("rag.engine.hybrid_retriever.retrieve",
                                        mock.AsyncMock(return_value=[self.FAKE_DOC])):
                            with mock.patch("rag.engine.graph_extractor.extract_from_query",
                                            mock.AsyncMock(return_value=[])):
                                with mock.patch("rag.engine.graph_store.search_related",
                                                mock.AsyncMock(return_value=[])):
                                    with mock.patch("agent.reflector.reflector.check_sufficiency",
                                                    mock.AsyncMock(return_value={"sufficient": True})):
                                        with mock.patch("rag.engine.query_rewrite.prepare_query",
                                                        mock.AsyncMock(return_value=(ORIG, {"mode": "precise"}))):
                                            return await rag_engine._retrieve(
                                                ORIG, history=HISTORY)

        asyncio.run(run())
        assert captured["key"] == engine_module._retrieve_cache_key(ORIG, 30, 0.6)
