"""Module-032 JWT 身份解析单元测试

覆盖（验收 §1.3 + §4.1 Python 单测）：
- parse_jwt：合法 token → user_id；无 token / 非 Bearer / 非法签名 / 过期 / sub 缺失 → ""
- resolve_identity：user_id 非空优先，否则 client_ip（匿名降级）
- 中间件链路：带合法 token 的 /ai 请求注入 request.state.user_id（端点收到 identity=user_id）；
  无 token / 非法 token → 降级 client_ip
- memory source 前缀：identity 优先 user_id（memory:<user_id>:，否则 memory:<client_ip>:）
- engine._recall_memory：identity 透传 memory_service.recall

实现说明：
- 测试密钥显式写入 settings.jwt_secret（与 .env 解耦，确定性）
- 中间件/端点链路用 httpx ASGITransport 直连 app，mock rag_engine.chat，
  不依赖真实 DB / Redis / LLM（与 test_stream_memory.py 同款模式）
- 同步用例内 asyncio.run 执行，不依赖 pytest-asyncio（规避既有环境问题）
"""
import asyncio
import time
from unittest import mock

import httpx
import jwt

import main
from rag.engine import rag_engine
from rag.memory import memory_service, _normalize_identity
from rag.schemas import ChatResponse
from src.config import settings
from src.identity import parse_jwt, resolve_identity

# 显式指定测试密钥：与 .env 解耦，保证用例确定性（encode/decode 同密钥）。
# 32 字节以上，避免 pyjwt 的 InsecureKeyLengthWarning（HS256 建议 ≥32 字节）。
_TEST_SECRET = "test-module-032-jwt-secret-0123456789abcdef"
settings.jwt_secret = _TEST_SECRET


def _make_token(sub: str, exp=None) -> str:
    """用测试密钥签发一个 HS256 JWT"""
    payload = {"sub": sub, "username": "test"}
    if exp is not None:
        payload["exp"] = exp
    return jwt.encode(payload, _TEST_SECRET, algorithm="HS256")


class _State:
    def __init__(self):
        self.user_id = ""
        self.client_ip = ""


class _Req:
    """最小假 Request：只带 state（供 resolve_identity 读取）"""

    def __init__(self):
        self.state = _State()


class _FakeSession:
    """假 AsyncSession：记录 add 的对象（供 memory save 断言 source）"""

    def __init__(self, scalar=0):
        self.added: list = []
        self._scalar = scalar

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        for i, obj in enumerate(self.added):
            if getattr(obj, "parent_id", None) is None:
                obj.id = i + 1

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def execute(self, stmt):
        result = mock.MagicMock()
        result.scalar.return_value = self._scalar
        result.scalars.return_value = mock.MagicMock(all=mock.MagicMock(return_value=[]))
        return result


def _fake_factory(session):
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=session)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    return mock.MagicMock(return_value=cm)


class TestParseJwt:
    """parse_jwt：Authorization 头 → user_id"""

    def test_valid_token_returns_user_id(self):
        token = _make_token("42")
        assert parse_jwt(f"Bearer {token}") == "42"

    def test_no_token_returns_empty(self):
        assert parse_jwt(None) == ""
        assert parse_jwt("") == ""

    def test_non_bearer_header_returns_empty(self):
        token = _make_token("42")
        assert parse_jwt(f"Basic {token}") == ""
        assert parse_jwt("Bearer") == ""
        assert parse_jwt("Bearer a b") == ""

    def test_invalid_signature_returns_empty(self):
        token = jwt.encode(
            {"sub": "42"}, "a-different-secret-that-is-32-bytes-long!!", algorithm="HS256")
        assert parse_jwt(f"Bearer {token}") == ""

    def test_expired_token_returns_empty(self):
        token = _make_token("42", exp=int(time.time()) - 3600)
        assert parse_jwt(f"Bearer {token}") == ""

    def test_missing_sub_returns_empty(self):
        token = jwt.encode({"username": "x"}, _TEST_SECRET, algorithm="HS256")
        assert parse_jwt(f"Bearer {token}") == ""


class TestResolveIdentity:
    """resolve_identity：user_id 优先，否则 client_ip"""

    def test_user_id_priority(self):
        req = _Req()
        req.state.user_id = "42"
        req.state.client_ip = "1.2.3.4"
        assert resolve_identity(req) == "42"

    def test_empty_user_id_falls_back_to_client_ip(self):
        req = _Req()
        req.state.client_ip = "1.2.3.4"
        assert resolve_identity(req) == "1.2.3.4"

    def test_no_identity_falls_back_to_unknown(self):
        # state 未注入 client_ip（getattr 默认 'unknown'）→ 返回 'unknown'
        class _StateOnlyUser:
            user_id = ""

        req = _Req()
        req.state = _StateOnlyUser()
        assert resolve_identity(req) == "unknown"


class TestMiddlewareIdentity:
    """中间件 → /ai/rag/chat 端点链路：identity 注入正确"""

    def _hit_chat(self, headers: dict) -> tuple[int, dict]:
        """发一次 /ai/rag/chat 请求（mock rag_engine.chat），返回 (status, chat_kwargs)"""
        out = {}

        async def run():
            with mock.patch(
                "main.rag_engine.chat",
                new=mock.AsyncMock(return_value=ChatResponse(
                    answer="", sources=[], message="ok")),
            ) as chat:
                transport = httpx.ASGITransport(app=main.app, raise_app_exceptions=True)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/ai/rag/chat",
                        headers=headers,
                        json={"query": "hi", "history": []},
                    )
                out["status"] = resp.status_code
                out["kwargs"] = chat.call_args.kwargs

        asyncio.run(run())
        return out["status"], out["kwargs"]

    def test_valid_token_injects_user_id(self):
        token = _make_token("42")
        status, kwargs = self._hit_chat({
            "Authorization": f"Bearer {token}",
            "X-Forwarded-For": "9.9.9.9",
        })
        assert status == 200
        assert kwargs["identity"] == "42"

    def test_no_token_falls_back_to_client_ip(self):
        status, kwargs = self._hit_chat({"X-Forwarded-For": "10.0.0.8"})
        assert status == 200
        assert kwargs["identity"] == "10.0.0.8"

    def test_invalid_token_falls_back_to_client_ip(self):
        bad = jwt.encode(
            {"sub": "42"}, "a-different-secret-that-is-32-bytes-long!!", algorithm="HS256")
        status, kwargs = self._hit_chat({
            "Authorization": f"Bearer {bad}",
            "X-Forwarded-For": "10.0.0.9",
        })
        assert status == 200
        assert kwargs["identity"] == "10.0.0.9"


class TestNormalizeIdentity:
    """_normalize_identity：user_id / client_ip 规范化"""

    def test_user_id_passes_through(self):
        assert _normalize_identity("42") == "42"
        assert _normalize_identity("user-abc") == "user-abc"

    def test_client_ip_passes_through(self):
        assert _normalize_identity("192.168.1.1") == "192.168.1.1"

    def test_empty_defaults_unknown(self):
        assert _normalize_identity("") == "unknown"
        assert _normalize_identity("   ") == "unknown"

    def test_like_metacharacters_rejected(self):
        # LIKE 通配符降级 'unknown'，防止注入绕过身份隔离
        assert _normalize_identity("%") == "unknown"
        assert _normalize_identity("_") == "unknown"
        assert _normalize_identity("\\") == "unknown"


class TestMemorySourcePrefix:
    """memory source 前缀：identity 优先 user_id"""

    def test_save_uses_user_id_prefix(self):
        async def run():
            fs = _FakeSession(scalar=0)
            with mock.patch("rag.memory.async_session_factory", _fake_factory(fs)):
                with mock.patch("rag.memory.chunker") as chunker_mock:
                    with mock.patch("rag.memory.embedding_service") as emb_mock:
                        chunker_mock.chunk.return_value = {
                            "parents": [{"title": "记忆", "content": "x"}],
                            "children": [{"title": "记忆", "content": "x", "parent_index": 0}],
                        }
                        emb_mock.embed_documents = mock.AsyncMock(return_value=[[0.1]])
                        await memory_service.save("用户偏好简短回答", "42")

            # 父块 + 子块 source 均带 user_id：'memory:42:'（身份优先 user_id）
            assert {getattr(d, "source", None) for d in fs.added} == {"memory:42:"}

        asyncio.run(run())

    def test_save_client_ip_prefix(self):
        async def run():
            fs = _FakeSession(scalar=0)
            with mock.patch("rag.memory.async_session_factory", _fake_factory(fs)):
                with mock.patch("rag.memory.chunker") as chunker_mock:
                    with mock.patch("rag.memory.embedding_service") as emb_mock:
                        chunker_mock.chunk.return_value = {
                            "parents": [{"title": "记忆", "content": "x"}],
                            "children": [{"title": "记忆", "content": "x", "parent_index": 0}],
                        }
                        emb_mock.embed_documents = mock.AsyncMock(return_value=[[0.1]])
                        await memory_service.save("用户偏好简短回答", "192.168.1.1")

            assert {getattr(d, "source", None) for d in fs.added} == {"memory:192.168.1.1:"}

        asyncio.run(run())


class TestEngineRecallIdentity:
    """engine._recall_memory：identity 透传 memory_service.recall / recall_short"""

    def test_identity_passed_to_service(self):
        captured = {}

        async def run():
            # module-037: memory_service.recall / recall_short 的 top_k 默认值 3→5
            # 本测试验证"透传"语义，fake 默认值需与当前实现一致
            async def fake_recall(query, identity, top_k=5):
                captured["identity"] = identity
                captured["top_k"] = top_k
                return []

            async def fake_recall_short(query, identity, top_k=5):
                captured["short_identity"] = identity
                captured["short_top_k"] = top_k
                return []

            with mock.patch(
                "rag.engine.memory_service.recall",
                new=mock.AsyncMock(side_effect=fake_recall),
            ):
                with mock.patch(
                    "rag.engine.memory_service.recall_short",
                    new=mock.AsyncMock(side_effect=fake_recall_short),
                ):
                    await rag_engine._recall_memory("q", "42")

        asyncio.run(run())
        assert captured["identity"] == "42"
        assert captured["top_k"] == 5
        # module-034：短期召回同样按身份隔离透传
        assert captured["short_identity"] == "42"
        assert captured["short_top_k"] == 5

    def test_empty_identity_returns_without_calling_service(self):
        async def run():
            with mock.patch(
                "rag.engine.memory_service.recall", new=mock.AsyncMock()
            ) as recall:
                assert await rag_engine._recall_memory("q", "") == ""
                recall.assert_not_called()

        asyncio.run(run())
