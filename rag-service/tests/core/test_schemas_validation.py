"""module-042/043: 请求模型 Pydantic 校验测试 — AC 1.3/1.4 + WP1 三端点加固"""
import pytest
from pydantic import ValidationError
from rag.schemas import ChatRequest, SearchRequest, MemorySaveRequest, MemoryRecallRequest


def test_query_too_long():
    """AC 1.3: query > 2000 字符 → 422 ValidationError"""
    with pytest.raises(ValidationError) as exc:
        ChatRequest(query="A" * 2001)
    assert "String should have at most 2000 characters" in str(exc.value)


def test_history_too_many():
    """AC 1.4: history > 20 条 → 静默截断保留最近 20 条"""
    h = [{"role": "user", "content": f"msg{i}"} for i in range(25)]
    r = ChatRequest(query="test", history=h)
    assert len(r.history) == 20
    # 保留的是最近 20 条（后 20 条，索引 5~24）
    assert r.history[0]["content"] == "msg5"
    assert r.history[-1]["content"] == "msg24"


def test_search_request_query_too_long():
    """module-043 WP1: SearchRequest.query > 2000 字符 → 422 ValidationError（与 ChatRequest 同值）"""
    with pytest.raises(ValidationError) as exc:
        SearchRequest(query="A" * 2001)
    assert "String should have at most 2000 characters" in str(exc.value)


def test_memory_save_content_too_long():
    """module-043 WP1: MemorySaveRequest.content > 2000 字符 → 422 ValidationError（落库防污染）"""
    with pytest.raises(ValidationError) as exc:
        MemorySaveRequest(content="A" * 2001)
    assert "String should have at most 2000 characters" in str(exc.value)


def test_memory_recall_query_too_long():
    """module-043 WP1: MemoryRecallRequest.query > 2000 字符 → 422 ValidationError"""
    with pytest.raises(ValidationError) as exc:
        MemoryRecallRequest(query="A" * 2001)
    assert "String should have at most 2000 characters" in str(exc.value)
