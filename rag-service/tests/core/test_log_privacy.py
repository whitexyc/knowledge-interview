"""Module-073 WP-C 日志隐私单元测试（正常截断 / 异常完整）

覆盖（对齐验收 AC-12~AC-15 / AC-18 / AC-25）：
- engine.search 正常路径 INFO 日志 query 截断 [:50]，完整 query 不出现
- engine.chat 正常路径 INFO 日志 query 截断 [:50]
- engine.chat 异常路径 ERROR 日志含完整 query + 错误信息 + 堆栈
- 边界：query 恰好 50 字符不截断

实现说明：
- caplog + levelno==INFO 过滤断言——WP-C 落地后错误路径日志也含完整 query，
  不按级别过滤会假阴性（plan §6 日志断言脆弱性提示）
- mock 依赖：resolve_tool_history fail-open None + 路由走 realtime 快捷路径
  （chat 正常路径最小 mock 面）；search 用 hybrid_retriever.retrieve 返回 []
  （"未检索到相关内容"提前返回）
- 对齐 test_degradation_fix.py caplog 先例；同步用例内 asyncio.run 执行
"""
import asyncio
import logging
from unittest import mock

from rag.schemas import SearchRequest, ChatRequest
from rag.engine import rag_engine


_LONG_QUERY = "测" * 60        # > 50 字符，断言截断语义
_EXACT_50 = "测" * 50          # 恰好 50 字符，断言不截断


def _search(query: str, top_k: int = 5):
    async def run():
        return await rag_engine.search(SearchRequest(query=query, top_k=top_k))
    return asyncio.run(run())


def _chat(query: str):
    async def run():
        return await rag_engine.chat(ChatRequest(query=query), identity="test")
    return asyncio.run(run())


def _info_records(caplog) -> list[str]:
    return [r.message for r in caplog.records if r.levelno == logging.INFO]


def _error_records(caplog) -> list[str]:
    return [r.message for r in caplog.records if r.levelno == logging.ERROR]


class TestSearchLogTruncation:
    """AC-12：engine.search 正常路径 query 截断 [:50]"""

    def test_long_query_truncated(self, caplog):
        caplog.set_level(logging.INFO)
        with mock.patch("rag.engine.hybrid_retriever.retrieve",
                        new=mock.AsyncMock(return_value=[])):
            result = _search(_LONG_QUERY)
        assert result.message == "未检索到相关内容"
        msgs = _info_records(caplog)
        assert any("RAG search: query=" in m for m in msgs)
        logged = [m for m in msgs if "RAG search: query=" in m][0]
        assert _LONG_QUERY[:50] in logged
        assert _LONG_QUERY not in logged  # 完整 query 不出现

    def test_exact_50_chars_not_truncated(self, caplog):
        # AC-18：query 恰好 50 字符不截断（[:50] 恒等）
        caplog.set_level(logging.INFO)
        with mock.patch("rag.engine.hybrid_retriever.retrieve",
                        new=mock.AsyncMock(return_value=[])):
            _search(_EXACT_50)
        logged = [m for m in _info_records(caplog) if "RAG search: query=" in m][0]
        assert _EXACT_50 in logged


class TestChatLogPrivacy:
    """AC-13/AC-14：engine.chat 正常截断 / 异常完整"""

    def test_normal_path_truncated(self, caplog):
        # 正常路径：INFO 含截断 query、不含完整 query（levelno==INFO 过滤——
        # 本用例下方无错误日志，但过滤是断言契约防假阴性）
        caplog.set_level(logging.INFO)
        with mock.patch("rag.engine.resolve_tool_history",
                        new=mock.AsyncMock(return_value=None)), \
             mock.patch("rag.engine.router_agent.classify",
                        new=mock.AsyncMock(return_value={"intent": "realtime"})):
            result = _chat(_LONG_QUERY)
        assert result.message == "realtime_not_implemented"
        logged = [m for m in _info_records(caplog) if "RAG chat: query=" in m][0]
        assert _LONG_QUERY[:50] in logged
        assert _LONG_QUERY not in logged

    def test_error_path_full_query(self, caplog):
        # AC-14：异常路径 ERROR 含完整 query + 错误信息 + 堆栈（排查需要）
        caplog.set_level(logging.INFO)
        with mock.patch("rag.engine.resolve_tool_history",
                        new=mock.AsyncMock(side_effect=RuntimeError("模拟失败"))):
            result = _chat(_LONG_QUERY)
        assert result.message == "internal_error"
        msgs = _error_records(caplog)
        assert msgs, "应有 ERROR 日志"
        assert any("RAG chat 失败" in m for m in msgs)
        logged = [m for m in msgs if "RAG chat 失败" in m][0]
        assert _LONG_QUERY in logged         # 完整 query（异常完整原则）
        assert "模拟失败" in logged          # 错误信息
        rec = [r for r in caplog.records if r.levelno == logging.ERROR][0]
        assert rec.exc_info is not None      # 堆栈（exc_info=True）

    def test_error_path_empty_query_ok(self, caplog):
        # AC-18：异常路径 query 为空字符串也能完整记录（不崩）
        caplog.set_level(logging.INFO)
        with mock.patch("rag.engine.resolve_tool_history",
                        new=mock.AsyncMock(side_effect=RuntimeError("模拟失败"))):
            result = _chat("")
        assert result.message == "internal_error"
        logged = [m for m in _error_records(caplog) if "RAG chat 失败" in m][0]
        assert "query=" in logged
        assert "模拟失败" in logged
