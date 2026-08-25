"""Schema 模型单元测试"""
import pytest
from pydantic import ValidationError
from rag.schemas import SearchRequest, SearchResponse, ChatRequest, ChatResponse


def test_search_request_defaults():
    r = SearchRequest(query="test")
    assert r.query == "test"
    assert r.top_k == 5


def test_search_response():
    r = SearchResponse(results=[{"id": 1}], message="ok")
    assert len(r.results) == 1
    assert r.message == "ok"


def test_chat_request_with_history():
    r = ChatRequest(query="你好", history=[{"role": "user", "content": "hi"}])
    assert len(r.history) == 1


def test_chat_response():
    r = ChatResponse(answer="回答", sources=[{"id": 1}])
    assert r.answer == "回答"
    assert len(r.sources) == 1


class TestChatRequestValidation:
    """module-042: ChatRequest Pydantic Field 校验边界测试"""

    def test_query_exactly_2000_chars_passes(self):
        """AC 1.3: query=2000 字符 → 应该通过"""
        r = ChatRequest(query="A" * 2000)
        assert len(r.query) == 2000

    def test_query_over_2000_chars_raises_422(self):
        """AC 1.3: query>2000 字符 → 422 ValidationError"""
        with pytest.raises(ValidationError) as exc:
            ChatRequest(query="A" * 2001)
        assert "String should have at most 2000 characters" in str(exc.value)

    def test_history_exactly_20_items_passes(self):
        """AC 1.4: history=20 条 → 应该通过，不截断"""
        h = [{"role": "user", "content": f"msg{i}"} for i in range(20)]
        r = ChatRequest(query="test", history=h)
        assert len(r.history) == 20

    def test_history_over_20_items_silently_truncates(self):
        """AC 1.4: history>20 条 → 静默截断保留最近 20 条"""
        h = [{"role": "user", "content": f"msg{i}"} for i in range(25)]
        r = ChatRequest(query="test", history=h)
        assert len(r.history) == 20
        # 保留的是最近 20 条（后 20 条）
        assert r.history[0]["content"] == "msg5"
        assert r.history[-1]["content"] == "msg24"

    def test_short_history_unchanged(self):
        """history < 20 条 → 原样保留不截断"""
        h = [{"role": "user", "content": "hi"}]
        r = ChatRequest(query="test", history=h)
        assert r.history == h

    def test_empty_history_unchanged(self):
        """空 history → 保持空列表"""
        r = ChatRequest(query="test", history=[])
        assert r.history == []

    def test_default_history_empty(self):
        """不传 history → 默认空列表"""
        r = ChatRequest(query="test")
        assert r.history == []
