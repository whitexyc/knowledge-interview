"""Module-020 _fts_search 查询逻辑单元测试

覆盖（验收 §4.1「_fts_search 查询逻辑」+ Reviewer 建议 #1）：
- SQL 用 search_tokens 列建 tsvector（而非 content）
- WHERE search_tokens IS NOT NULL 过滤未分词文档 + parent_id IS NOT NULL 只查子块
- query 侧 jieba 分词结果作为 :query 参数透传（与入库侧一致）
- 分词后为空（空串/纯标点）提前返回空列表，不执行 SQL
- 返回 list[dict] 且 score 转为 float（None → 0.0）

实现说明：
- 用 mock.AsyncMock 打桩 AsyncSession，不依赖真实数据库
- 同步用例内 asyncio.run 执行，不依赖 pytest-asyncio（规避既有环境问题，
  与 test_golden_retrieval.py 同款模式）
"""
import asyncio
from unittest import mock

from rag.text_tokenizer import tokenize
from rag.retriever import HybridRetriever


class TestFtsSearch:
    """_fts_search 查询逻辑"""

    @staticmethod
    def _make_retriever():
        return HybridRetriever(embedding_service=mock.MagicMock(), alpha=0.3)

    @staticmethod
    def _session_returning(rows):
        session = mock.AsyncMock()
        session.execute = mock.AsyncMock(
            return_value=mock.MagicMock(mappings=mock.MagicMock(return_value=rows))
        )
        return session

    def test_sql_builds_tsvector_on_search_tokens(self):
        # 核心：tsvector 建在 search_tokens（module-020 复活中文 FTS 的关键）
        async def run():
            retriever = self._make_retriever()
            session = self._session_returning([])
            await retriever._fts_search("线程池", 10, session)
            sql = session.execute.call_args.args[0].text
            assert "to_tsvector('simple', search_tokens)" in sql
            assert "to_tsvector('simple', content)" not in sql
            assert "search_tokens IS NOT NULL" in sql
            assert "parent_id IS NOT NULL" in sql
            assert "plainto_tsquery('simple', :query)" in sql
        asyncio.run(run())

    def test_tokenized_query_passed_as_param(self):
        # query 侧 jieba 分词结果（空格连接）作为 :query 透传，与入库侧一致
        async def run():
            retriever = self._make_retriever()
            session = self._session_returning([])
            await retriever._fts_search("Java线程池核心参数", 10, session)
            params = session.execute.call_args.args[1]
            assert params["query"] == tokenize("Java线程池核心参数")
            assert "线程" in params["query"] and " " in params["query"]
        asyncio.run(run())

    def test_empty_tokenized_query_returns_empty(self):
        # 纯标点/空 query 分词后为空 → 提前返回 []，不执行 SQL
        async def run():
            retriever = self._make_retriever()
            session = mock.AsyncMock()
            session.execute = mock.AsyncMock(side_effect=AssertionError("不应执行 SQL"))
            assert await retriever._fts_search("？？？---", 10, session) == []
            session.execute.assert_not_called()
        asyncio.run(run())

    def test_returns_list_of_dict_with_float_score(self):
        # 返回 list[dict]；score 转 float（None → 0.0）
        rows = [
            {"id": 1, "title": "T1", "content": "c1", "parent_id": 2, "score": "0.5"},
            {"id": 3, "title": "T2", "content": "c2", "parent_id": 4, "score": None},
        ]

        async def run():
            retriever = self._make_retriever()
            session = self._session_returning(rows)
            result = await retriever._fts_search("线程池", 10, session)
            assert isinstance(result, list)
            assert isinstance(result[0], dict)
            assert result[0]["score"] == 0.5
            assert result[1]["score"] == 0.0
        asyncio.run(run())
