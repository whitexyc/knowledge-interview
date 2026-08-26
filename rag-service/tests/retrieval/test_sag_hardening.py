"""SAG 补强轮单测（module-082）

覆盖：search 端点三模式 / 兜底三态（LLM 正常/失败/空）/
boost 行为 / hybrid 零回归
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── 子任务 2：兜底实体提取纯函数 ──


class TestFallbackExtractEntities:
    """_fallback_extract_entities 三态覆盖"""

    def test_filters_stopwords(self):
        """停用词被过滤，保留有效候选"""
        from rag.retrieval.sag_retriever import _fallback_extract_entities
        result = _fallback_extract_entities("什么是G1 GC的原理")
        # 分隔符切词 + 停用词过滤后，应保留非停用词非单字符候选
        # 精确 token 取决于中文分隔符行为，但"G1"和"GC"应被保留
        assert len(result) > 0
        # 所有结果应非停用词、长度>1
        from rag.retrieval.sag_retriever import _STOPWORDS
        for token in result:
            assert len(token) > 1
            assert token.lower() not in _STOPWORDS

    def test_filters_single_char(self):
        """单字符词被过滤"""
        from rag.retrieval.sag_retriever import _fallback_extract_entities
        result = _fallback_extract_entities("a b 你好世界")
        # "a" 和 "b" 是单字符，应被过滤
        assert "a" not in result
        assert "b" not in result
        assert "你好世界" in result

    def test_empty_when_only_stopwords(self):
        """纯停用词返回空"""
        from rag.retrieval.sag_retriever import _fallback_extract_entities
        result = _fallback_extract_entities("的 了 是")
        assert result == []

    def test_max_entities_limit(self):
        """候选数量受 max_entities 限制"""
        from rag.retrieval.sag_retriever import _fallback_extract_entities
        result = _fallback_extract_entities(
            "alpha beta gamma delta epsilon zeta", max_entities=3,
        )
        assert len(result) == 3

    def test_english_tokens(self):
        """英文 token 正确过滤"""
        from rag.retrieval.sag_retriever import _fallback_extract_entities
        result = _fallback_extract_entities("What is the garbage collector")
        # "What"/"is"/"the" 是停用词或单字符
        assert "garbage" in result
        assert "collector" in result
        assert "the" not in result
        assert "is" not in result

    def test_mixed_delimiters(self):
        """中英文混合分隔符正确切词"""
        from rag.retrieval.sag_retriever import _fallback_extract_entities
        result = _fallback_extract_entities("Java,线程池；核心参数？")
        assert "Java" in result
        assert "线程池" in result
        assert "核心参数" in result


# ── 子任务 2：retrieve 三态（LLM 正常/失败/空）──


class TestSAGRetrieveFallback:
    """retrieve() LLM 失败/空时兜底"""

    @pytest.mark.asyncio
    async def test_llm_normal_uses_llm_result(self):
        """LLM 正常返回非空实体 → 使用 LLM 结果"""
        from rag.retrieval.sag_retriever import retrieve
        mock_row = (1, "Title", "Content", "source", {})
        with patch("rag.retrieval.sag_retriever.graph_extractor") as mock_ge, \
             patch("rag.retrieval.sag_retriever.async_session_factory") as mock_factory:
            mock_ge.extract_from_query = AsyncMock(return_value=["G1 GC"])
            mock_session = AsyncMock()
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = mock_ctx
            mock_session.execute = AsyncMock(side_effect=[
                MagicMock(fetchall=lambda: [([1],)]),
                MagicMock(fetchall=lambda: [mock_row]),
            ])
            docs = await retrieve("What is G1 GC?", top_k=5)
        assert len(docs) >= 1
        assert docs[0]["score"] == 1.0

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self):
        """LLM 抛异常 → 启用兜底实体提取"""
        from rag.retrieval.sag_retriever import retrieve
        mock_row = (1, "Title", "Content", "source", {})
        with patch("rag.retrieval.sag_retriever.graph_extractor") as mock_ge, \
             patch("rag.retrieval.sag_retriever.async_session_factory") as mock_factory:
            mock_ge.extract_from_query = AsyncMock(side_effect=RuntimeError("LLM down"))
            mock_session = AsyncMock()
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = mock_ctx
            mock_session.execute = AsyncMock(side_effect=[
                MagicMock(fetchall=lambda: [([1],)]),
                MagicMock(fetchall=lambda: [mock_row]),
            ])
            docs = await retrieve("G1 GC 原理", top_k=5)
        assert len(docs) >= 1  # 兜底应能提取出实体

    @pytest.mark.asyncio
    async def test_llm_empty_falls_back(self):
        """LLM 返回空列表 → 启用兜底实体提取"""
        from rag.retrieval.sag_retriever import retrieve
        mock_row = (1, "Title", "Content", "source", {})
        with patch("rag.retrieval.sag_retriever.graph_extractor") as mock_ge, \
             patch("rag.retrieval.sag_retriever.async_session_factory") as mock_factory:
            mock_ge.extract_from_query = AsyncMock(return_value=[])
            mock_session = AsyncMock()
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = mock_ctx
            mock_session.execute = AsyncMock(side_effect=[
                MagicMock(fetchall=lambda: [([1],)]),
                MagicMock(fetchall=lambda: [mock_row]),
            ])
            docs = await retrieve("Kafka topics", top_k=5)
        assert len(docs) >= 1  # 兜底应能提取出 "Kafka" 和 "topics"

    @pytest.mark.asyncio
    async def test_llm_timeout_falls_back(self):
        """LLM 超时 → 启用兜底实体提取"""
        from rag.retrieval.sag_retriever import retrieve

        async def slow_extract(query):
            await asyncio.sleep(20)  # 超过 10s 超时
            return ["entity"]

        with patch("rag.retrieval.sag_retriever.graph_extractor") as mock_ge, \
             patch("rag.retrieval.sag_retriever.async_session_factory") as mock_factory:
            mock_ge.extract_from_query = slow_extract
            mock_session = AsyncMock()
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = mock_ctx
            # 兜底提取到实体但 DB 无匹配
            mock_session.execute = AsyncMock(return_value=MagicMock(fetchall=lambda: []))
            docs = await retrieve("Redis cache", top_k=5)
        # 不抛异常，fail-open
        assert isinstance(docs, list)


# ── 子任务 3：boost 行为 ──


class TestSAGScoreBoost:
    """SAG 命中项 hybrid_score ×1.2 boost"""

    def test_boost_basic(self):
        """score 0.5 → boost 后 0.6"""
        from rag.engine import _SAG_SCORE_BOOST
        score = 0.5
        boosted = min(score * _SAG_SCORE_BOOST, 1.0)
        assert abs(boosted - 0.6) < 1e-9

    def test_boost_capped_at_1(self):
        """score 0.9 → boost 后 1.0（上限截断）"""
        from rag.engine import _SAG_SCORE_BOOST
        score = 0.9
        boosted = min(score * _SAG_SCORE_BOOST, 1.0)
        assert boosted == 1.0

    def test_boost_zero_stays_zero(self):
        """score 0.0 → boost 后仍 0.0"""
        from rag.engine import _SAG_SCORE_BOOST
        score = 0.0
        boosted = min(score * _SAG_SCORE_BOOST, 1.0)
        assert boosted == 0.0

    def test_boost_missing_score_defaults_zero(self):
        """hybrid_score 缺失时默认 0.0 → boost 后 0.0"""
        from rag.engine import _SAG_SCORE_BOOST
        doc = {"id": 1, "title": "test"}
        score = doc.get("hybrid_score", doc.get("score", 0.0))
        boosted = min(score * _SAG_SCORE_BOOST, 1.0)
        assert boosted == 0.0


# ── 子任务 1：search 端点三模式 ──


class TestSearchRetrievalMode:
    """search 端点对 retrieval_mode 的感知"""

    @pytest.mark.asyncio
    async def test_hybrid_mode_no_sag(self):
        """hybrid 模式不执行 SAG 检索（零行为变化）"""
        from rag.engine import rag_engine
        from rag.schemas import SearchRequest
        with patch("rag.engine.settings") as mock_settings, \
             patch("rag.engine.hybrid_retriever") as mock_hybrid, \
             patch("rag.engine.reranker") as mock_reranker:
            # 复制真实 settings 属性
            from src.config import settings as real_settings
            for attr in dir(real_settings):
                if not attr.startswith("_"):
                    try:
                        setattr(mock_settings, attr, getattr(real_settings, attr))
                    except Exception:
                        pass
            mock_settings.retrieval_mode = "hybrid"
            mock_hybrid.retrieve = AsyncMock(return_value=[
                {"id": 1, "title": "T", "content": "C", "source": "S", "hybrid_score": 0.8},
            ])
            mock_reranker.rerank = AsyncMock(side_effect=lambda q, docs, top_k: docs[:top_k])

            request = SearchRequest(query="test", top_k=5)
            response = await rag_engine.search(request)
        assert response.message == "ok"
        assert len(response.results) >= 1

    @pytest.mark.asyncio
    async def test_sag_mode_pure_sag(self):
        """sag 模式只返回 SAG 结果"""
        from rag.engine import rag_engine
        from rag.schemas import SearchRequest
        with patch("rag.engine.settings") as mock_settings, \
             patch("rag.engine.hybrid_retriever") as mock_hybrid, \
             patch("rag.engine.reranker") as mock_reranker, \
             patch("rag.retrieval.sag_retriever.graph_extractor") as mock_ge, \
             patch("rag.retrieval.sag_retriever.async_session_factory") as mock_factory:
            from src.config import settings as real_settings
            for attr in dir(real_settings):
                if not attr.startswith("_"):
                    try:
                        setattr(mock_settings, attr, getattr(real_settings, attr))
                    except Exception:
                        pass
            mock_settings.retrieval_mode = "sag"
            mock_ge.extract_from_query = AsyncMock(return_value=["G1 GC"])
            mock_session = AsyncMock()
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = mock_ctx
            mock_session.execute = AsyncMock(side_effect=[
                MagicMock(fetchall=lambda: [([10],)]),
                MagicMock(fetchall=lambda: [(10, "SAG Doc", "Content", "src", {})]),
            ])
            mock_reranker.rerank = AsyncMock(side_effect=lambda q, docs, top_k: docs[:top_k])

            request = SearchRequest(query="G1 GC", top_k=5)
            response = await rag_engine.search(request)
        # hybrid_retriever 不应被调用（sag 模式跳过）
        mock_hybrid.retrieve.assert_not_called()
        assert response.message == "ok"

    @pytest.mark.asyncio
    async def test_hybrid_sag_merges(self):
        """hybrid_sag 模式合并 SAG + 常规结果，SAG 在前"""
        from rag.engine import rag_engine
        from rag.schemas import SearchRequest
        with patch("rag.engine.settings") as mock_settings, \
             patch("rag.engine.hybrid_retriever") as mock_hybrid, \
             patch("rag.engine.reranker") as mock_reranker, \
             patch("rag.retrieval.sag_retriever.graph_extractor") as mock_ge, \
             patch("rag.retrieval.sag_retriever.async_session_factory") as mock_factory:
            from src.config import settings as real_settings
            for attr in dir(real_settings):
                if not attr.startswith("_"):
                    try:
                        setattr(mock_settings, attr, getattr(real_settings, attr))
                    except Exception:
                        pass
            mock_settings.retrieval_mode = "hybrid_sag"
            mock_ge.extract_from_query = AsyncMock(return_value=["G1 GC"])
            mock_session = AsyncMock()
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = mock_ctx
            mock_session.execute = AsyncMock(side_effect=[
                MagicMock(fetchall=lambda: [([10],)]),
                MagicMock(fetchall=lambda: [(10, "SAG Doc", "Content", "src", {})]),
            ])
            mock_hybrid.retrieve = AsyncMock(return_value=[
                {"id": 20, "title": "Regular", "content": "C", "source": "S", "hybrid_score": 0.7},
            ])
            mock_reranker.rerank = AsyncMock(side_effect=lambda q, docs, top_k: docs[:top_k])

            request = SearchRequest(query="G1 GC", top_k=5)
            response = await rag_engine.search(request)
        assert response.message == "ok"
        # SAG 文档在前（id=10），常规在后（id=20）
        ids = [r["id"] for r in response.results]
        assert 10 in ids
        assert 20 in ids
        assert ids.index(10) < ids.index(20)

    @pytest.mark.asyncio
    async def test_sag_failure_degrades(self):
        """SAG 检索失败 → 降级为仅常规结果（fail-open）"""
        from rag.engine import rag_engine
        from rag.schemas import SearchRequest
        with patch("rag.engine.settings") as mock_settings, \
             patch("rag.engine.hybrid_retriever") as mock_hybrid, \
             patch("rag.engine.reranker") as mock_reranker, \
             patch("rag.retrieval.sag_retriever.graph_extractor") as mock_ge:
            from src.config import settings as real_settings
            for attr in dir(real_settings):
                if not attr.startswith("_"):
                    try:
                        setattr(mock_settings, attr, getattr(real_settings, attr))
                    except Exception:
                        pass
            mock_settings.retrieval_mode = "hybrid_sag"
            mock_ge.extract_from_query = AsyncMock(side_effect=RuntimeError("LLM down"))
            mock_hybrid.retrieve = AsyncMock(return_value=[
                {"id": 20, "title": "Regular", "content": "C", "source": "S", "hybrid_score": 0.7},
            ])
            mock_reranker.rerank = AsyncMock(side_effect=lambda q, docs, top_k: docs[:top_k])

            request = SearchRequest(query="test", top_k=5)
            response = await rag_engine.search(request)
        # 不抛异常，常规结果正常返回
        assert response.message == "ok"
        assert len(response.results) >= 1
