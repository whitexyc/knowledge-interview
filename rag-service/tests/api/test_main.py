"""module-042: 答案截断测试 — AC 1.5"""
import asyncio
import json
from unittest import mock

import httpx

from rag.schemas import ChatResponse

TRUNC_MARKER = "\n\n[答案过长，已截断]"
MAX_LEN = 10000


def test_answer_truncation():
    """AC 1.5: 答案 >10000 字符 → 截断 + 标记追加；短答案不受影响"""
    import main as main_module

    async def run():
        # 场景 1: 超长答案 → 截断
        long_ans = "A" * 15000
        sources = [{"id": 1, "title": "文档1", "content": "内容", "source": "test", "ref_index": 1}]
        fake_response = ChatResponse(answer=long_ans, sources=sources, message="ok")

        with mock.patch("rag.engine.rag_engine.chat",
                        new=mock.AsyncMock(return_value=fake_response)):
            with mock.patch("main.save_messages_to_session"):
                transport = httpx.ASGITransport(
                    app=main_module.app, raise_app_exceptions=True)
                async with httpx.AsyncClient(
                        transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/ai/rag/chat",
                        json={"query": "测试截断", "history": []},
                    )
                assert resp.status_code == 200
                data = resp.json()
                # 截断后长度 ≤ MAX_LEN + 标记长度
                assert len(data["answer"]) <= MAX_LEN + len(TRUNC_MARKER)
                assert data["answer"].endswith(TRUNC_MARKER)
                assert data["answer"].startswith("A" * MAX_LEN)
                # sources 完整保留 (AC 2.3)
                assert len(data["sources"]) == 1
                assert data["sources"][0]["id"] == 1

        # 场景 2: 短答案 → 不截断
        short = "B" * 100
        fake_response2 = ChatResponse(answer=short, sources=[], message="ok")

        with mock.patch("rag.engine.rag_engine.chat",
                        new=mock.AsyncMock(return_value=fake_response2)):
            with mock.patch("main.save_messages_to_session"):
                transport = httpx.ASGITransport(
                    app=main_module.app, raise_app_exceptions=True)
                async with httpx.AsyncClient(
                        transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/ai/rag/chat",
                        json={"query": "测试", "history": []},
                    )
                assert resp.status_code == 200
                data = resp.json()
                assert data["answer"] == "B" * 100
                assert TRUNC_MARKER not in data["answer"]

    asyncio.run(run())
