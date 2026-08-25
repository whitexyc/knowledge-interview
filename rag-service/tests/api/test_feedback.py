"""module-048 反馈飞轮：POST /ai/feedback 端点测试

覆盖（验收 §1 功能 + §6 降级）：
- 落库：rating=±1（±comment）→ 200 {"status": "ok"} + feedback 记录字段完整
- 校验：rating 非法（0/2）→ 422；comment >500 字符 → 422
- identity：JWT user_id 优先，client_ip 兜底（对齐中间件注入口径）
- 降级：落库失败 → 500（前端 Toast 提示，不阻塞聊天）

实现说明：
- httpx.ASGITransport 打真实 app（与 test_main.py 同款），全量限流由
  conftest 取消；main.async_session_factory 打桩为假会话（记录 add 的对象），
  不依赖真实数据库（与 test_memory.py 同款模式）
- 同步用例内 asyncio.run 执行，不依赖 pytest-asyncio
"""
import asyncio
from unittest import mock

import httpx

import main as main_module


class _FakeSession:
    """假 AsyncSession：记录 add 的对象；可配置 commit 抛异常（降级用例）"""

    def __init__(self, commit_error: bool = False):
        self.added: list = []
        self._commit_error = commit_error

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        if self._commit_error:
            raise RuntimeError("数据库不可用")


def _fake_factory(session):
    """把 async_session_factory 打桩成返回 session 的异步上下文管理器

    factory 本身是同步可调用（async_session_factory() 立即返回 CM 对象），
    只有 __aenter__/__aexit__ 是异步的，故用 MagicMock 而非 AsyncMock。
    """
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=session)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    return mock.MagicMock(return_value=cm)


async def _post(payload: dict, fake_jwt: str = ""):
    """向 /ai/feedback 发一次真实请求；fake_jwt 非空时模拟 JWT 身份注入"""
    session = _FakeSession()
    with mock.patch("main.async_session_factory", _fake_factory(session)):
        with mock.patch("main.parse_jwt", return_value=fake_jwt):
            async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=main_module.app),
                    base_url="http://test") as client:
                resp = await client.post("/ai/feedback", json=payload)
            return resp, session


def test_feedback_persisted_success():
    """落库：rating=1 + comment → 200 {"status": "ok"}，记录字段完整（匿名 identity=client_ip）"""
    resp, session = asyncio.run(
        _post({"message_id": 1, "rating": 1, "comment": "很好"}))
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert len(session.added) == 1
    fb = session.added[0]
    assert fb.message_id == 1
    assert fb.rating == 1
    assert fb.comment == "很好"
    assert fb.identity == "127.0.0.1"  # 无 JWT → client_ip 兜底


def test_feedback_rating_minus_one():
    """落库：rating=-1（踩）同样成功；comment 可选（缺省 None）"""
    resp, session = asyncio.run(_post({"message_id": 2, "rating": -1}))
    assert resp.status_code == 200
    assert session.added[0].rating == -1
    assert session.added[0].comment is None


def test_feedback_rating_invalid_422():
    """校验：rating=0 / 2 → 422（仅允许 1/-1，防飞轮数据污染）"""
    for bad in (0, 2):
        resp, _ = asyncio.run(_post({"message_id": 1, "rating": bad}))
        assert resp.status_code == 422


def test_feedback_comment_too_long_422():
    """校验：comment >500 字符 → 422"""
    resp, _ = asyncio.run(
        _post({"message_id": 1, "rating": 1, "comment": "A" * 501}))
    assert resp.status_code == 422


def test_feedback_identity_user_id_priority():
    """identity：JWT user_id 优先 client_ip（对齐中间件注入与 /ai/rag/chat 口径）"""
    resp, session = asyncio.run(
        _post({"message_id": 1, "rating": 1}, fake_jwt="user-123"))
    assert resp.status_code == 200
    assert session.added[0].identity == "user-123"


def test_feedback_db_failure_500():
    """降级：落库失败 → 500 {"message": "反馈保存失败"}（前端 Toast 提示，聊天不受影响）"""
    session = _FakeSession(commit_error=True)

    async def run():
        with mock.patch("main.async_session_factory", _fake_factory(session)):
            async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=main_module.app),
                    base_url="http://test") as client:
                return await client.post(
                    "/ai/feedback", json={"message_id": 1, "rating": 1})

    resp = asyncio.run(run())
    assert resp.status_code == 500
    assert resp.json()["message"] == "反馈保存失败"
