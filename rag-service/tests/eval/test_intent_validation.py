"""module-043: Intent 校验体系测试 — L2 前置校验 / L3 后置反证 / L4 分类器

覆盖 acceptance-criteria §2/§3/§4/§5/§6：
- L2：低置信触发 / 信号命中修正 knowledge / 未命中保持原判 / 异常保守
      knowledge / 规则表否决 / 确认路径零 LLM 调用（红线）
- L3：精排 top-1 abs_cosine<0.3 → suspected_misclassify 标记写入 ChatSteps
      （不阻塞、不改回答路径）
- L4：fit/predict_proba 校准概率（mock 特征）/ 模型缺失回退 LLM / 注入优先

同步 def + 函数内 asyncio.run 执行，不依赖 pytest-asyncio
（与套件其余用例同款模式）。
"""
import asyncio
import os
from unittest import mock

import pytest

from agent.router import RouterAgent, _FUNCTION_STOPWORDS, _RULE_TABLE
from agent.intent_classifier import IntentClassifier
from rag.engine import RAGEngine
from rag.schemas import ChatRequest


class FakeLLM:
    """模拟 LLM 分类返回（JSON payload）"""

    def __init__(self, payload: str):
        self._payload = payload

    async def generate(self, prompt: str) -> str:
        return self._payload


class AsyncConfirm:
    """模块级 async 确认桩（module-055 新增用例：E2E query 类多用例复用）"""

    def __init__(self, confirmed: bool, signal: str):
        self._confirmed = confirmed
        self._signal = signal

    async def __call__(self, query: str) -> tuple[bool, str]:
        return self._confirmed, self._signal


# ─── L2 前置校验（AC §2 / §5 / §6） ───


class TestL2Trigger:
    """AC §2: intent≠knowledge 无条件触发确认（module-055 扩展）

    module-055 行为升级：原"且 confidence<0.5"限制在 module-054 E2E 暴露缺口
    ——LLM 高置信误判 casual_chat 直接漏检（"G1垃圾收集器的核心创新是什么？"
    被判闲聊返回来源 0）；确定性信号便宜且精确（golden 50 条非 knowledge
    样本误确认 0），规则表否决闲聊/实时特征词，扩展零风险。相应更新
    test_high_confidence_skips_l2 / test_missing_confidence_skips_l2 断言
    （行为升级非掩盖）。
    """

    def _classify(self, payload: str, confirm=None, query: str = "测试查询"):
        async def run():
            agent = RouterAgent()
            if confirm is not None:
                agent._deterministic_confirm = confirm
            with mock.patch("llm.client.LLMFactory.get_client",
                            return_value=FakeLLM(payload)):
                return await agent.classify(query)
        return asyncio.run(run())

    def test_low_confidence_casual_triggers_l2(self):
        called = []

        async def fake_confirm(query):
            called.append(query)
            return False, "no_signal"

        result = self._classify(
            '{"intent": "casual_chat", "confidence": 0.3, "reason": "闲聊"}',
            confirm=fake_confirm,
        )
        assert called  # 低置信触发确认
        assert result["intent"] == "casual_chat"  # 信号未命中 → 保持原判

    def test_low_confidence_realtime_triggers_l2(self):
        called = []

        async def fake_confirm(query):
            called.append(query)
            return False, "no_signal"

        self._classify(
            '{"intent": "realtime", "confidence": 0.4, "reason": "时间"}',
            confirm=fake_confirm,
        )
        assert called

    def test_high_confidence_casual_triggers_l2(self):
        """module-055 行为升级：高置信闲聊也触发 L2（module-054 E2E 缺口——
        LLM 高置信误判 casual_chat 直接漏检；信号命中 → 修正为 knowledge）"""
        called = []

        async def fake_confirm(query):
            called.append(query)
            return True, "fts_term"

        result = self._classify(
            '{"intent": "casual_chat", "confidence": 0.9, "reason": "闲聊"}',
            confirm=fake_confirm,
        )
        assert called  # 高置信同样触发确认
        assert result["intent"] == "knowledge"

    def test_high_confidence_no_signal_keeps_original(self):
        """高置信触发 L2 但信号未命中（真闲聊）→ 保持原判"""
        called = []

        async def fake_confirm(query):
            called.append(query)
            return False, "no_signal"

        result = self._classify(
            '{"intent": "casual_chat", "confidence": 0.9, "reason": "闲聊"}',
            confirm=fake_confirm,
        )
        assert called
        assert result["intent"] == "casual_chat"

    def test_knowledge_intent_skips_l2(self):
        called = []

        async def fake_confirm(query):
            called.append(query)
            return True, "fts_term"

        result = self._classify(
            '{"intent": "knowledge", "confidence": 0.2, "reason": "知识"}',
            confirm=fake_confirm,
        )
        assert not called  # 不对称投放：走 knowledge 是低风险路径，不校验
        assert result["intent"] == "knowledge"

    def test_missing_confidence_triggers_l2(self):
        """降级/外部 mock 结果无 confidence → 同样触发（module-055：触发条件
        与置信度解耦——高置信误判即 E2E 根因，无 confidence 不再作为豁免）"""
        called = []

        async def fake_confirm(query):
            called.append(query)
            return False, "no_signal"

        result = self._classify(
            '{"intent": "casual_chat", "reason": "闲聊"}',
            confirm=fake_confirm,
        )
        assert called
        assert result["intent"] == "casual_chat"  # 信号未命中 → 保持原判

    def test_e2e_query_high_confidence_casual_corrected(self):
        """E2E 场景 query（专有术语 G1 + 疑问句）：LLM 高置信误判 casual_chat
        → L2 确定性信号确认 → knowledge（module-054 E2E 回归场景）"""
        result = self._classify(
            '{"intent": "casual_chat", "confidence": 0.95, "reason": "闲聊"}',
            confirm=AsyncConfirm(True, "fts_term"),
            query="G1垃圾收集器的核心创新是什么？",
        )
        assert result["intent"] == "knowledge"
        assert "fts_term" in result["reason"]

    def test_boundary_term_question_queries_corrected(self):
        """专有术语 + 疑问句边界样本：JVM/Redis 类查询均被确定性信号拉回"""
        for q in ("JVM内存溢出怎么排查？", "Redis 的持久化机制有哪些？"):
            result = self._classify(
                '{"intent": "casual_chat", "confidence": 0.9, "reason": "误判闲聊"}',
                confirm=AsyncConfirm(True, "graph_entity"),
                query=q,
            )
            assert result["intent"] == "knowledge", q

    def test_confirm_hit_corrects_to_knowledge(self):
        async def fake_confirm(query):
            return True, "fts_term"

        result = self._classify(
            '{"intent": "casual_chat", "confidence": 0.2, "reason": "看似闲聊"}',
            confirm=fake_confirm,
        )
        assert result["intent"] == "knowledge"
        assert "fts_term" in result["reason"]  # 可观测：原因含确认信号

    def test_confirm_unexpected_error_conservative_knowledge(self):
        """AC 场景 4: 确认环节异常 → 保守 knowledge（宁多检不漏检）"""

        async def fake_confirm(query):
            raise RuntimeError("DB down")

        result = self._classify(
            '{"intent": "casual_chat", "confidence": 0.3, "reason": "闲聊"}',
            confirm=fake_confirm,
        )
        assert result["intent"] == "knowledge"

    def test_llm_parse_failure_still_conservative(self):
        """LLM 分类自身失败 → 原保守策略不变"""
        result = self._classify("not json")
        assert result["intent"] == "knowledge"


class TestL2DeterministicConfirm:
    """确定性信号：FTS 术语 / 图谱实体 / 规则表，任何异常保守 knowledge"""

    def test_fts_hit_confirms(self):
        agent = RouterAgent()
        with mock.patch.object(agent, "_fts_term_hit",
                               new=mock.AsyncMock(return_value=True)):
            confirmed, signal = asyncio.run(agent._deterministic_confirm("你知道 GC 是什么吗"))
        assert confirmed is True
        assert signal == "fts_term"

    def test_graph_hit_confirms(self):
        agent = RouterAgent()
        with mock.patch.object(agent, "_fts_term_hit",
                               new=mock.AsyncMock(return_value=False)):
            with mock.patch.object(agent, "_graph_entity_hit",
                                   new=mock.AsyncMock(return_value=True)):
                confirmed, signal = asyncio.run(agent._deterministic_confirm("GC 是什么"))
        assert confirmed is True
        assert signal == "graph_entity"

    def test_no_signal_keeps_original(self):
        agent = RouterAgent()
        with mock.patch.object(agent, "_fts_term_hit",
                               new=mock.AsyncMock(return_value=False)):
            with mock.patch.object(agent, "_graph_entity_hit",
                                   new=mock.AsyncMock(return_value=False)):
                # "周末去哪玩"：无规则词、无 FTS/图谱信号 → no_signal 保持原判
                confirmed, signal = asyncio.run(agent._deterministic_confirm("周末去哪玩"))
        assert confirmed is False
        assert signal == "no_signal"

    def test_rule_veto_overrides_fts_hit(self):
        """规则表命中（闲聊/实时特征词）→ 保持原判，否决 FTS 巧合命中

        场景："现在几点了" 即使 FTS 命中（如"现在"出现在文档），
        规则表"几点"命中 → 不修正为 knowledge。
        """
        agent = RouterAgent()
        with mock.patch.object(agent, "_fts_term_hit",
                               new=mock.AsyncMock(return_value=True)):
            confirmed, signal = asyncio.run(agent._deterministic_confirm("现在几点了"))
        assert confirmed is False
        assert signal == "rule_veto"

    def test_rule_check_short_circuits_before_db(self):
        """module-055：规则表提前短路——规则词命中时 FTS/图谱信号零调用

        L2 无条件触发后，闲聊/实时请求每轮都进确认；规则词命中直接返回
        rule_veto，不浪费 DB 查询（原实现 FTS 先行属无效开销）。
        """
        agent = RouterAgent()
        fts = mock.AsyncMock(return_value=True)
        graph = mock.AsyncMock(return_value=True)
        with mock.patch.object(agent, "_fts_term_hit", new=fts):
            with mock.patch.object(agent, "_graph_entity_hit", new=graph):
                confirmed, signal = asyncio.run(agent._deterministic_confirm("现在几点了"))
        assert confirmed is False
        assert signal == "rule_veto"
        fts.assert_not_awaited()   # 规则命中 → 零 DB 查询
        graph.assert_not_awaited()

    def test_kb_terms_filters_module055_noise_words(self):
        """module-055 数据驱动停用词：golden 扫描实测噪声词不再参与 FTS 确认

        "今天/问题/怎么样" 等词在知识库文档中广泛存在，命中无判别力
        （实测会导致闲聊/实时样本 20/50 误确认 → 补入后归零）。
        """
        for noisy in ("今天心情不太好", "最近在忙什么呀", "周末过得怎么样", "没问题"):
            terms = RouterAgent._kb_terms(noisy)
            assert "今天" not in terms and "问题" not in terms
            assert "怎么样" not in terms and "最近" not in terms

    def test_signal_exception_conservative_knowledge(self):
        """AC 场景 4: 信号查询异常 → 保守 knowledge

        module-055 规则表提前短路后，异常路径仅对无规则词的 query 可达
        （"你好呀"现走 rule_veto 短路，改用无规则词 query 保持原测试意图）。
        """
        agent = RouterAgent()
        with mock.patch.object(agent, "_fts_term_hit",
                               new=mock.AsyncMock(side_effect=RuntimeError("db"))):
            confirmed, signal = asyncio.run(agent._deterministic_confirm("周末去哪玩"))
        assert confirmed is True
        assert signal == "error_conservative"

    def test_confirm_path_never_calls_llm(self):
        """红线（AC §6）：确认动作与 LLM 完全无关——LLM 可用性不影响确认结果"""
        agent = RouterAgent()

        def boom(*args, **kwargs):
            raise AssertionError("确认路径禁止调用 LLM")

        with mock.patch("llm.client.LLMFactory.get_client", side_effect=boom):
            with mock.patch.object(agent, "_fts_term_hit",
                                   new=mock.AsyncMock(return_value=False)):
                with mock.patch.object(agent, "_graph_entity_hit",
                                       new=mock.AsyncMock(return_value=False)):
                    confirmed, signal = asyncio.run(agent._deterministic_confirm("周末去哪玩"))
        assert confirmed is False
        assert signal == "no_signal"

    def test_fts_hit_path_no_llm_dependency(self):
        """FTS 命中确认也不依赖 LLM"""
        agent = RouterAgent()

        def boom(*args, **kwargs):
            raise AssertionError("确认路径禁止调用 LLM")

        with mock.patch("llm.client.LLMFactory.get_client", side_effect=boom):
            with mock.patch.object(agent, "_fts_term_hit",
                                   new=mock.AsyncMock(return_value=True)):
                confirmed, signal = asyncio.run(agent._deterministic_confirm("GC 是什么"))
        assert confirmed is True
        assert signal == "fts_term"

    def test_kb_terms_filters_stopwords(self):
        """FTS 术语提取：过滤功能词，保留专有术语"""
        terms = RouterAgent._kb_terms("你知道 GC 和 JVM 的区别是什么吗")
        assert "GC" in terms
        assert "JVM" in terms
        assert all(t not in _FUNCTION_STOPWORDS for t in terms)
        assert "什么" not in terms and "区别" not in terms

    def test_kb_terms_empty_for_pure_chitchat(self):
        terms = RouterAgent._kb_terms("你好呀")
        assert all(t not in _FUNCTION_STOPWORDS for t in terms)  # 至少不输出功能词

    def test_rule_table_contains_documented_examples(self):
        """规则表覆盖 ADR-0003 文档示例词"""
        for word in ("几点", "天气", "你是谁"):
            assert any(word in rule for rule in _RULE_TABLE), word

    def test_rule_table_keeps_kb_boundary_samples(self):
        """规则表不误伤边界易混样本（module-045 WP2a）

        golden 边界样本"你能做什么？这个系统能帮我解决什么问题？"标注 knowledge
        （问系统能力）：移除"你能做什么/你会什么"后 _rule_hits 返回 False，
        不再被规则表否决 → L2 可走 FTS/图谱信号确认。
        """
        assert not RouterAgent._rule_hits("你能做什么？这个系统能帮我解决什么问题？")
        assert not RouterAgent._rule_hits("你们网站有哪些功能")
        assert not RouterAgent._rule_hits("你知道 GC 是什么吗")
        # 移除确认：规则表不再含"你能做什么/你会什么"
        assert not any("你能做什么" in rule for rule in _RULE_TABLE)
        assert not any("你会什么" in rule for rule in _RULE_TABLE)

    def test_boundary_sample_not_vetoed_when_fts_hit(self):
        """边界样本 FTS 命中 → 确认成功（规则词移除后不再被否决）"""
        agent = RouterAgent()
        with mock.patch.object(agent, "_fts_term_hit",
                               new=mock.AsyncMock(return_value=True)):
            confirmed, signal = asyncio.run(
                agent._deterministic_confirm("你能做什么？这个系统能帮我解决什么问题？"))
        assert confirmed is True
        assert signal == "fts_term"


# ─── L3 后置校验（AC §3 / §5） ───


class TestL3PostValidation:
    """精排 top-1 abs_cosine < 0.3 → suspected_misclassify（先度量后干预）

    module-045 WP2c: 返回 (flag, top1_abs) 二元组（判定与展示同源存档）。
    """

    def test_flag_when_top1_abs_below_threshold(self):
        flag, top1_abs = RAGEngine._check_suspected_misclassify([{"abs_cosine": 0.29}])
        assert flag is True
        assert top1_abs == 0.29

    def test_no_flag_when_abs_above_threshold(self):
        flag, top1_abs = RAGEngine._check_suspected_misclassify([{"abs_cosine": 0.5}])
        assert flag is False
        assert top1_abs == 0.5

    def test_no_flag_when_boundary_equal(self):
        # 阈值 0.3：等于阈值不算疑似误判（保守标记而非激进标记）
        flag, top1_abs = RAGEngine._check_suspected_misclassify([{"abs_cosine": 0.3}])
        assert flag is False
        assert top1_abs == 0.3

    def test_no_docs_no_flag(self):
        flag, top1_abs = RAGEngine._check_suspected_misclassify([])
        assert flag is False
        assert top1_abs == 0.0

    def test_all_missing_abs_cosine_no_flag(self):
        """module-055 行为升级：整组文档均无 abs_cosine（向量通道整体降级——
        module-054 方案 A 合法生产状态）→ 语义证据未度量，不标记

        原"缺字段视为 0.0 保守标记"在该状态恒误触发 suspected_misclassify
        （module-054 E2E 实测 top_abs_cosine=0.0 + 误标记；rrf 融合路径
        向量路降级即全组缺字段）。
        """
        flag, top1_abs = RAGEngine._check_suspected_misclassify([{"hybrid_score": 0.9}])
        assert flag is False
        assert top1_abs == 0.0

    def test_top1_missing_no_flag(self):
        """module-055 行为升级：top-1 无 abs_cosine（FTS/图谱独有命中排首——
        rrf 三通道下图谱通道返回父块文档可排 top-1，HyDE 查询实测复现）→
        缺向量分数 ≠ 低分，不标记（图谱实体命中本身就是相关证据）"""
        docs = [{"hybrid_score": 0.9}, {"abs_cosine": 0.8, "hybrid_score": 0.8}]
        flag, top1_abs = RAGEngine._check_suspected_misclassify(docs)
        assert flag is False
        assert top1_abs == 0.0

    def test_multiple_docs_uses_top1_only(self):
        docs = [{"abs_cosine": 0.2}, {"abs_cosine": 0.9}]
        flag, top1_abs = RAGEngine._check_suspected_misclassify(docs)
        assert flag is True
        assert top1_abs == 0.2
        docs = [{"abs_cosine": 0.5}, {"abs_cosine": 0.1}]
        flag, top1_abs = RAGEngine._check_suspected_misclassify(docs)
        assert flag is False
        assert top1_abs == 0.5


class TestL3ChatStepsObservable:
    """AC §3: 标记写入 ChatSteps（可观测），不阻塞、不改回答路径"""

    def test_suspected_misclassify_written_to_steps(self):
        from rag.engine import rag_engine

        fake_doc = {"id": 1, "title": "t", "content": "c", "source": "s",
                    "abs_cosine": 0.1, "parent_id": None}

        async def run():
            with mock.patch("rag.engine.router_agent.classify",
                            new=mock.AsyncMock(
                                return_value={"intent": "knowledge", "confidence": 0.8})):
                with mock.patch("rag.retriever.hybrid_retriever.retrieve",
                                new=mock.AsyncMock(return_value=[fake_doc])):
                    with mock.patch("rag.reranker.reranker.rerank",
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
                                                    result = await rag_engine.chat(
                                                        ChatRequest(query="你好呀"),
                                                        identity="x")
            return result

        result = asyncio.run(run())
        assert result.steps is not None
        assert result.steps.retrieval.get("suspected_misclassify") is True
        assert result.steps.retrieval.get("top_abs_cosine") == 0.1
        assert result.message == "ok"  # 不阻塞、不改回答路径
        assert result.answer == "答案"

    def test_top_abs_cosine_archived_before_parent_mapping(self):
        """module-045 WP2b: steps.top_abs_cosine 用 round 0 存档值（判定同源），
        父块映射重建 dict 丢 abs_cosine 后展示值不恒 0.0"""
        from rag.engine import rag_engine

        fake_doc = {"id": 1, "title": "t", "content": "c", "source": "s",
                    "parent_id": None}

        async def run():
            with mock.patch("rag.engine.router_agent.classify",
                            new=mock.AsyncMock(
                                return_value={"intent": "knowledge", "confidence": 0.9})):
                with mock.patch("rag.retriever.hybrid_retriever.retrieve",
                                new=mock.AsyncMock(return_value=[fake_doc])):
                    with mock.patch("rag.reranker.reranker.rerank",
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
                                                    # 模拟 round 0 判定返回存档值：
                                                    # 即使最终 docs 无 abs_cosine 字段
                                                    # （父块映射后），steps 仍展示真实值
                                                    with mock.patch(
                                                        "rag.engine.RAGEngine._check_suspected_misclassify",
                                                        return_value=(True, 0.42)):
                                                        result = await rag_engine.chat(
                                                            ChatRequest(query="什么是GC"),
                                                            identity="x")
            return result

        result = asyncio.run(run())
        assert result.steps is not None
        assert result.steps.retrieval.get("suspected_misclassify") is True
        assert result.steps.retrieval.get("top_abs_cosine") == 0.42

    def test_high_similarity_no_flag_in_steps(self):
        from rag.engine import rag_engine

        fake_doc = {"id": 1, "title": "t", "content": "c", "source": "s",
                    "abs_cosine": 0.7, "parent_id": None}

        async def run():
            with mock.patch("rag.engine.router_agent.classify",
                            new=mock.AsyncMock(
                                return_value={"intent": "knowledge", "confidence": 0.9})):
                with mock.patch("rag.retriever.hybrid_retriever.retrieve",
                                new=mock.AsyncMock(return_value=[fake_doc])):
                    with mock.patch("rag.reranker.reranker.rerank",
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
                                                    result = await rag_engine.chat(
                                                        ChatRequest(query="什么是GC"),
                                                        identity="x")
            return result

        result = asyncio.run(run())
        assert result.steps is not None
        assert result.steps.retrieval.get("suspected_misclassify") is False


class TestExpandToParentsAbsCosine:
    """module-045 WP2b: 父块映射保留 abs_cosine（子块最大值，与 hybrid_score 同策略）

    根因修复：_expand_to_parents 重建 dict 丢 abs_cosine → chat/_retrieve 的
    ChatSteps.top_abs_cosine 恒 0.0（失真）。存档方案（round 0 判定处同源存档）
    之外，父块映射层同步透传字段，流式路径（_retrieve）同样不丢。
    """

    @staticmethod
    def _run_expand(child_docs):
        from rag.engine import rag_engine

        class _Parent:
            def __init__(self, pid):
                self.id = pid
                self.title = "父块"
                self.content = "父块内容"
                self.source = "s"

        async def run():
            session = mock.AsyncMock()
            result_mock = mock.MagicMock()
            result_mock.scalars.return_value.all.return_value = [_Parent(10), _Parent(20)]
            session.execute = mock.AsyncMock(return_value=result_mock)
            cm = mock.MagicMock()
            cm.__aenter__ = mock.AsyncMock(return_value=session)
            cm.__aexit__ = mock.AsyncMock(return_value=False)
            factory = mock.MagicMock(return_value=cm)
            with mock.patch("rag.engine.async_session_factory", factory):
                return await rag_engine._expand_to_parents(child_docs)

        return asyncio.run(run())

    def test_parent_mapping_preserves_max_abs_cosine(self):
        docs = self._run_expand([
            {"id": 1, "title": "c1", "content": "c1", "source": "s",
             "parent_id": 10, "hybrid_score": 0.6, "abs_cosine": 0.35},
            {"id": 2, "title": "c2", "content": "c2", "source": "s",
             "parent_id": 10, "hybrid_score": 0.8, "abs_cosine": 0.55},
            {"id": 3, "title": "c3", "content": "c3", "source": "s",
             "parent_id": 20, "hybrid_score": 0.5, "abs_cosine": 0.2},
        ])
        by_id = {d["id"]: d for d in docs}
        assert by_id[10]["abs_cosine"] == 0.55  # 子块最大值透传
        assert by_id[10]["hybrid_score"] == 0.8
        assert by_id[20]["abs_cosine"] == 0.2

    def test_parent_without_abs_cosine_defaults_zero(self):
        """子块无 abs_cosine（fts-only 命中）→ 父块按 0.0 保守处理"""
        docs = self._run_expand([
            {"id": 1, "title": "c1", "content": "c1", "source": "s",
             "parent_id": 10, "hybrid_score": 0.7},
        ])
        assert docs[0]["id"] == 10
        assert docs[0]["abs_cosine"] == 0.0


# ─── L4 分类器（AC §4 / §5） ───


class _FakeEmbedding:
    """mock 特征：确定性向量，三类线性可分（knowledge/casual/realtime 坐标轴）"""

    def __init__(self):
        self._map = {
            "什么是GC": [1.0, 0.0, 0.0],
            "G1收集器": [0.9, 0.1, 0.0],
            "JVM原理": [1.0, 0.0, 0.0],
            "CMS和G1区别": [0.9, 0.1, 0.0],
            "你好": [0.0, 1.0, 0.0],
            "在吗": [0.1, 0.9, 0.0],
            "谢谢": [0.0, 1.0, 0.0],
            "哈哈": [0.1, 0.9, 0.0],
            "现在几点": [0.0, 0.0, 1.0],
            "今天天气": [0.0, 0.1, 0.9],
            "几点了": [0.0, 0.0, 1.0],
            "气温多少": [0.0, 0.1, 0.9],
        }

    async def embed_text(self, text: str) -> list:
        return self._map.get(text, [0.5, 0.5, 0.5])

    async def embed_documents(self, texts: list) -> list:
        return [await self.embed_text(t) for t in texts]


class TestL4Classifier:
    """AC §4: fit / predict_proba 输出校准概率（mock 特征）"""

    def test_fit_and_predict_proba(self, tmp_path):
        samples = [
            ("什么是GC", "knowledge"), ("G1收集器", "knowledge"),
            ("JVM原理", "knowledge"), ("CMS和G1区别", "knowledge"),
            ("你好", "casual_chat"), ("在吗", "casual_chat"),
            ("谢谢", "casual_chat"), ("哈哈", "casual_chat"),
            ("现在几点", "realtime"), ("今天天气", "realtime"),
            ("几点了", "realtime"), ("气温多少", "realtime"),
        ]
        model_path = os.path.join(str(tmp_path), "intent_clf.joblib")
        clf = IntentClassifier(model_path=model_path,
                               embedding_service=_FakeEmbedding())
        metrics = asyncio.run(clf.fit(samples))
        assert metrics["n_samples"] == 12
        assert os.path.isfile(model_path)  # 模型落盘

        # 重新加载后 predict_proba：三类键齐全、概率和≈1、knowledge 占优
        clf2 = IntentClassifier(model_path=model_path,
                                embedding_service=_FakeEmbedding())
        assert asyncio.run(clf2.load()) is True
        probs = asyncio.run(clf2.predict_proba("什么是GC"))
        assert set(probs) == {"knowledge", "casual_chat", "realtime"}
        assert abs(sum(probs.values()) - 1.0) < 0.01
        assert probs["knowledge"] > probs["casual_chat"]
        assert probs["knowledge"] > probs["realtime"]

    def test_load_missing_model_returns_false(self):
        clf = IntentClassifier(model_path="nonexistent_model.joblib",
                               embedding_service=_FakeEmbedding())
        assert asyncio.run(clf.load()) is False

    def test_predict_without_model_raises(self):
        clf = IntentClassifier(model_path="x.joblib",
                               embedding_service=_FakeEmbedding())
        with pytest.raises(RuntimeError):
            asyncio.run(clf.predict_proba("你好"))


class _FakeClassifier:
    """可注入 router 的假 L4 分类器"""

    def __init__(self, probs=None):
        self._probs = probs or {"knowledge": 0.8, "casual_chat": 0.1, "realtime": 0.1}
        self.calls = 0

    async def load(self):
        return True

    async def predict_proba(self, query):
        self.calls += 1
        return self._probs


class TestL4RouterInjection:
    """AC §4/§5: router 可注入分类器（配置开关），默认仍用 LLM；失败回退零影响"""

    def test_injected_classifier_used(self):
        clf = _FakeClassifier()
        agent = RouterAgent(intent_classifier=clf)

        def boom(*args, **kwargs):
            raise AssertionError("注入分类器时不应调用 LLM")

        with mock.patch("llm.client.LLMFactory.get_client", side_effect=boom):
            result = asyncio.run(agent.classify("什么是GC"))
        assert clf.calls == 1
        assert result["intent"] == "knowledge"
        assert result["confidence"] == 0.8

    def test_classifier_bogus_intent_whitelisted_to_knowledge(self):
        """module-045 WP2d: L4 返回非法 intent → 白名单归 knowledge（与 LLM 路径一致）"""
        clf = _FakeClassifier(
            probs={"knowledge": 0.1, "casual_chat": 0.2, "realtime": 0.1,
                   "bogus": 0.9},
        )
        agent = RouterAgent(intent_classifier=clf)
        result = asyncio.run(agent.classify("什么是GC"))
        assert result["intent"] == "knowledge"  # bogus 最高分被白名单拦截
        assert result["confidence"] == 0.1  # 置信度取白名单后 intent 的概率

    def test_classifier_missing_knowledge_key_no_keyerror(self):
        """module-048 WP5: probs 缺 knowledge 键 → 不抛 KeyError，回退默认置信度

        真实分类器可能缺键：bogus 最高分被白名单修正为 knowledge 后，
        probs["knowledge"] 会 KeyError（旧实现静默回退 LLM）。修复后
        probs.get(intent, 0.0) 回退默认置信度，且不调用 LLM。
        """
        clf = _FakeClassifier(
            probs={"casual_chat": 0.2, "realtime": 0.1, "bogus": 0.9},
        )
        agent = RouterAgent(intent_classifier=clf)

        def boom(*args, **kwargs):
            raise AssertionError("缺键防御不应触发 LLM 回退")

        with mock.patch("llm.client.LLMFactory.get_client", side_effect=boom):
            result = asyncio.run(agent.classify("什么是GC"))
        assert result["intent"] == "knowledge"  # bogus 白名单归 knowledge
        assert result["confidence"] == 0.0  # 缺 knowledge 键 → 回退默认置信度

    def test_default_llm_when_not_injected(self):
        agent = RouterAgent()
        payload = '{"intent": "casual_chat", "confidence": 0.9, "reason": "闲聊"}'
        with mock.patch("llm.client.LLMFactory.get_client",
                        return_value=FakeLLM(payload)):
            result = asyncio.run(agent.classify("你好"))
        assert result["intent"] == "casual_chat"
        assert result["confidence"] == 0.9

    def test_classifier_failure_falls_back_to_llm(self):
        """AC §5: L4 推理失败 → 回退 LLM 分类，零影响"""

        class Broken(_FakeClassifier):
            async def predict_proba(self, query):
                raise RuntimeError("模型推理失败")

        agent = RouterAgent(intent_classifier=Broken())
        payload = '{"intent": "casual_chat", "confidence": 0.9, "reason": "闲聊"}'
        with mock.patch("llm.client.LLMFactory.get_client",
                        return_value=FakeLLM(payload)):
            result = asyncio.run(agent.classify("你好"))
        assert result["intent"] == "casual_chat"

    def test_lazy_load_failure_falls_back_llm(self, monkeypatch):
        """配置开关开启但模型缺失 → 惰性加载失败 → 回退 LLM（零影响）"""
        from src.config import settings
        monkeypatch.setattr(settings, "intent_classifier_enabled", True)
        agent = RouterAgent()
        payload = '{"intent": "knowledge", "confidence": 0.9, "reason": "知识"}'
        with mock.patch("llm.client.LLMFactory.get_client",
                        return_value=FakeLLM(payload)):
            result = asyncio.run(agent.classify("什么是GC"))
        assert result["intent"] == "knowledge"

    def test_disabled_switch_skips_lazy_load(self, monkeypatch):
        """默认（开关关闭）：不尝试加载分类器，纯 LLM 路径"""
        from src.config import settings
        monkeypatch.setattr(settings, "intent_classifier_enabled", False)
        agent = RouterAgent()
        payload = '{"intent": "casual_chat", "confidence": 0.9, "reason": "闲聊"}'
        with mock.patch("llm.client.LLMFactory.get_client",
                        return_value=FakeLLM(payload)):
            result = asyncio.run(agent.classify("你好"))
        assert result["intent"] == "casual_chat"
