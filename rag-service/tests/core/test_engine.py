"""RAG 引擎骨架单元测试

同步 def + 函数内 asyncio.run 执行，不依赖 pytest-asyncio
（与套件其余用例同款模式，规避既有 pytest-asyncio 缺失问题）。
"""
import asyncio

from rag.schemas import SearchRequest, ChatRequest
from rag.engine import rag_engine


def test_search_returns_response():
    async def run():
        r = SearchRequest(query="测试")
        result = await rag_engine.search(r)
        assert result.message is not None

    asyncio.run(run())


def test_chat_returns_response():
    async def run():
        r = ChatRequest(query="你好")
        result = await rag_engine.chat(r)
        assert result.answer is not None
        assert isinstance(result.sources, list)

    asyncio.run(run())
