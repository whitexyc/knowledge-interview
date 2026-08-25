"""Module-021 graph_store.search_related 真实分数单元测试

覆盖（验收 §4.1「分数归一化有单测（含保底分支）」+「排序正确性」）：
- _normalize_graph_scores：min-max 归一化 / 全同分保底 0.6 / 空列表
- search_related：按命中实体数降序排序、接口字段完整、分数 ∈ [0,1]
- 边界：空实体列表返回空、无 Cypher 命中返回空

实现说明：
- 用 mock.AsyncMock 打桩 async_session_factory（两个 `async with` 会话分别返回
  Cypher 行与 Document 查询结果），不依赖真实数据库
- 同步用例内 asyncio.run 执行，不依赖 pytest-asyncio（与 test_golden_retrieval.py
  同款模式，规避既有 pytest-asyncio 缺失问题）
"""
import asyncio
from unittest import mock

from rag.graph_store import GraphStore, _escape


class TestNormalizeGraphScores:
    """_normalize_graph_scores 归一化（含保底分支）"""

    def test_minmax_distinct_counts(self):
        # [2, 3, 5] → min=2, max=5 → [0.0, 0.333, 1.0]
        scores = GraphStore._normalize_graph_scores([2, 3, 5])
        assert scores[0] == 0.0
        assert round(scores[1], 3) == round(1 / 3, 3)
        assert scores[2] == 1.0

    def test_all_same_counts_fallback_06(self):
        # 全同分 → 保底 0.6（与历史硬编码一致，非 1.0）
        scores = GraphStore._normalize_graph_scores([3, 3, 3])
        assert scores == [0.6, 0.6, 0.6]

    def test_single_count_fallback_06(self):
        # 单结果 → 保底 0.6
        assert GraphStore._normalize_graph_scores([4]) == [0.6]

    def test_empty_counts(self):
        assert GraphStore._normalize_graph_scores([]) == []


class TestSearchRelated:
    """search_related 排序 + 接口（打桩 DB）"""

    @staticmethod
    def _mock_doc(did):
        d = mock.MagicMock()
        d.id = did
        d.title = f"title-{did}"
        d.content = f"content-{did}"
        d.source = "test"
        d.parent_id = None
        return d

    @staticmethod
    def _factory_returning(rows, docs):
        """async_session_factory 打桩：两次 `async with` 分别返回 Cypher 行与文档"""
        cypher_result = mock.MagicMock()
        cypher_result.fetchall.return_value = rows
        doc_result = mock.MagicMock()
        doc_result.scalars.return_value.all.return_value = docs

        s1 = mock.AsyncMock()
        s1.execute = mock.AsyncMock(return_value=cypher_result)
        s2 = mock.AsyncMock()
        s2.execute = mock.AsyncMock(return_value=doc_result)

        cm1 = mock.AsyncMock()
        cm1.__aenter__.return_value = s1
        cm2 = mock.AsyncMock()
        cm2.__aenter__.return_value = s2

        factory = mock.MagicMock()
        factory.side_effect = [cm1, cm2]
        return factory

    def test_sorted_by_hits_desc(self):
        # Cypher 返回 doc 75(hits=5), 91(hits=3), 95(hits=1) → 排序应 75,91,95
        rows = [('"75"', '5'), ('"91"', '3'), ('"95"', '1')]
        docs = [self._mock_doc(75), self._mock_doc(91), self._mock_doc(95)]

        async def run():
            store = GraphStore()
            with mock.patch("rag.graph_store.async_session_factory",
                            self._factory_returning(rows, docs)):
                return await store.search_related(["Java", "线程池"], top_k=3)

        result = asyncio.run(run())
        assert [d["id"] for d in result] == [75, 91, 95]

    def test_interface_fields_and_score_range(self):
        rows = [('"75"', '5'), ('"91"', '3')]
        docs = [self._mock_doc(75), self._mock_doc(91)]

        async def run():
            store = GraphStore()
            with mock.patch("rag.graph_store.async_session_factory",
                            self._factory_returning(rows, docs)):
                return await store.search_related(["Java"], top_k=5)

        result = asyncio.run(run())
        for d in result:
            assert set(d.keys()) == {"id", "title", "content", "source", "hybrid_score", "parent_id"}
            assert 0.0 <= d["hybrid_score"] <= 1.0
            assert isinstance(d["hybrid_score"], float)
        # 区分度：命中数不同 → 分数不同
        assert result[0]["hybrid_score"] != result[1]["hybrid_score"]

    def test_empty_entities(self):
        store = GraphStore()
        assert asyncio.run(store.search_related([], top_k=5)) == []

    def test_no_cypher_hits_returns_empty(self):
        # Cypher 无行 → 不查 Document，直接返回空
        async def run():
            store = GraphStore()
            factory = self._factory_returning([], [])
            with mock.patch("rag.graph_store.async_session_factory", factory):
                return await store.search_related(["不存在的实体"], top_k=5)

        assert asyncio.run(run()) == []


class TestEscape:
    """_escape Cypher 字符串字面量转义（module-031 修复）"""

    def test_brace_not_escaped(self):
        # } 不再转义：`\}` 在 openCypher（AGE 1.6）是非法转义序列，
        # 会导致含 `}` 的实体/关系写入失败（InvalidEscapeSequenceError）
        assert _escape("abc}def") == "abc}def"
        assert _escape("#{}") == "#{}"
        assert _escape("${}") == "${}"

    def test_single_quote_escaped(self):
        assert _escape("it's") == r"it\'s"

    def test_backslash_escaped(self):
        assert _escape("a\\b") == "a\\\\b"

    def test_mixed_special_chars(self):
        assert _escape("it's a\\b}") == r"it\'s a\\b}"
