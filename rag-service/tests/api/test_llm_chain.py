"""Module-029 降级链动态调序单元测试

覆盖（验收 §1.2 + §2.1）：
- validate_chain：合法链通过 / 空链拒绝 / 未知供应商拒绝 / 重复拒绝 / 非字符串拒绝
- LLMFactory 动态链：set_fallback_chain 后 get_client("fallback") 按新链重建 /
  get_fallback_chain 无覆盖时回退配置默认
- GET /ai/llm/chain：返回当前链
- PUT /ai/llm/chain：校验 → 存 Redis → 更新内存 + clear_cache；非法链拒绝不修改；
  Redis 失败拒绝不修改
- 启动加载：load_fallback_chain_from_redis 优先 Redis 链、Redis 不可用/不合法回退默认
- set_str/get_str：真实 Redis 持久化往返（跨重启保留）

实现说明：
- 用 mock 打桩 cache.set_str/get_str，不依赖真实 Redis（除真实 Redis 往返用例）
- 同步用例内 asyncio.run 执行，不依赖 pytest-asyncio（沿用既有模式）
- 每个用例前后重置 LLMFactory 运行时链与实例缓存，避免污染其他测试
"""
import asyncio
from unittest import mock

import httpx

import main
from llm.client import LLMFactory, FallbackClient
from src.cache import cache
from src.config import settings

# 配置默认链：从实际设置读取（不硬编码，避免与 .env 不一致）
CONFIG_DEFAULT = [p.strip() for p in settings.fallback_chain.split(",") if p.strip()]


def _reset_factory():
    """重置运行时链与实例缓存，避免用例间相互污染"""
    LLMFactory._fallback_chain = None
    LLMFactory.clear_cache()


# ─── validate_chain ───

class TestValidateChain:
    """降级链合法性校验"""

    def test_valid_chain_passes(self):
        assert LLMFactory.validate_chain(["zhipu", "deepseek", "qwen"]) == [
            "zhipu", "deepseek", "qwen",
        ]

    def test_chain_normalized_lowercase_strip(self):
        assert LLMFactory.validate_chain(["  Qwen ", "DeepSeek"]) == ["qwen", "deepseek"]

    def test_empty_chain_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="不能为空"):
            LLMFactory.validate_chain([])

    def test_unknown_provider_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="不支持的供应商"):
            LLMFactory.validate_chain(["qwen", "bogus"])

    def test_duplicate_provider_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="重复"):
            LLMFactory.validate_chain(["qwen", "deepseek", "qwen"])

    def test_non_string_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="非法供应商类型"):
            LLMFactory.validate_chain(["qwen", 123])

    def test_fallback_not_allowed_in_chain(self):
        """链内不允许嵌套 fallback（白名单不包含 fallback）"""
        import pytest
        with pytest.raises(ValueError, match="不支持的供应商"):
            LLMFactory.validate_chain(["fallback", "qwen"])


# ─── LLMFactory 动态链 ───

class TestDynamicChain:
    """set_fallback_chain 后 FallbackClient 按新链重建"""

    def _reset(self):
        _reset_factory()

    def test_fallback_client_uses_updated_chain(self):
        self._reset()
        try:
            LLMFactory.set_fallback_chain(["zhipu", "deepseek"])
            LLMFactory.clear_cache()
            client = LLMFactory.get_client("fallback")
            assert isinstance(client, FallbackClient)
            assert client._chain == ["zhipu", "deepseek"]
        finally:
            _reset_factory()

    def test_get_fallback_chain_defaults_to_config(self):
        self._reset()
        try:
            # 无运行时覆盖 → 配置默认
            assert LLMFactory.get_fallback_chain() == CONFIG_DEFAULT
        finally:
            _reset_factory()

    def test_get_fallback_chain_prefers_runtime(self):
        self._reset()
        try:
            LLMFactory.set_fallback_chain(["deepseek", "qwen"])
            assert LLMFactory.get_fallback_chain() == ["deepseek", "qwen"]
        finally:
            _reset_factory()

    def test_clear_cache_rebuilds_all_clients(self):
        self._reset()
        try:
            before = LLMFactory.get_client("fallback")
            LLMFactory.set_fallback_chain(["qwen"])
            LLMFactory.clear_cache()
            after = LLMFactory.get_client("fallback")
            assert before is not after  # clear_cache 后重建
            assert after._chain == ["qwen"]
        finally:
            _reset_factory()


# ─── GET / PUT /ai/llm/chain 端点 ───

class TestChainEndpoints:
    """降级链 API（httpx ASGITransport 直连 main.app）"""

    def _reset(self):
        _reset_factory()

    async def _get(self, path: str):
        transport = httpx.ASGITransport(app=main.app, raise_app_exceptions=True)
        async with httpx.AsyncClient(
                transport=transport, base_url="http://test") as client:
            resp = await client.get(path)
        return resp

    async def _put(self, path: str, body: dict):
        transport = httpx.ASGITransport(app=main.app, raise_app_exceptions=True)
        async with httpx.AsyncClient(
                transport=transport, base_url="http://test") as client:
            resp = await client.put(path, json=body)
        return resp

    def test_get_returns_config_default(self):
        self._reset()
        try:
            resp = asyncio.run(self._get("/ai/llm/chain"))
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 0
            assert body["data"]["chain"] == CONFIG_DEFAULT
        finally:
            _reset_factory()

    def test_put_updates_chain_and_persists(self):
        self._reset()
        try:
            with mock.patch.object(cache, "set_str", mock.AsyncMock(return_value=True)) as m_set:
                resp = asyncio.run(self._put(
                    "/ai/llm/chain", {"chain": ["zhipu", "deepseek", "qwen"]},
                ))
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 0
            assert body["data"]["chain"] == ["zhipu", "deepseek", "qwen"]
            # Redis 写入的是逗号拼接字符串
            m_set.assert_awaited_once_with("llm:fallback_chain", "zhipu,deepseek,qwen")
            # 内存链已更新
            assert LLMFactory.get_fallback_chain() == ["zhipu", "deepseek", "qwen"]
            # GET 应返回新链
            resp2 = asyncio.run(self._get("/ai/llm/chain"))
            assert resp2.json()["data"]["chain"] == ["zhipu", "deepseek", "qwen"]
        finally:
            _reset_factory()

    def test_put_invalid_chain_rejected_without_save(self):
        self._reset()
        try:
            with mock.patch.object(cache, "set_str", mock.AsyncMock(return_value=True)) as m_set:
                resp = asyncio.run(self._put(
                    "/ai/llm/chain", {"chain": ["zhipu", "zhipu"]},
                ))
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 1
            m_set.assert_not_awaited()
            # 内存链未被修改，仍是配置默认
            assert LLMFactory.get_fallback_chain() == CONFIG_DEFAULT
        finally:
            _reset_factory()

    def test_put_unknown_provider_rejected(self):
        self._reset()
        try:
            resp = asyncio.run(self._put(
                "/ai/llm/chain", {"chain": ["qwen", "huggingface"]},
            ))
            assert resp.json()["code"] == 1
            assert LLMFactory.get_fallback_chain() == CONFIG_DEFAULT
        finally:
            _reset_factory()

    def test_put_empty_chain_rejected(self):
        self._reset()
        try:
            resp = asyncio.run(self._put("/ai/llm/chain", {"chain": []}))
            assert resp.json()["code"] == 1
            assert LLMFactory.get_fallback_chain() == CONFIG_DEFAULT
        finally:
            _reset_factory()

    def test_put_redis_failure_keeps_chain(self):
        """Redis 不可用：调序不生效但服务正常（内存链不变）"""
        self._reset()
        try:
            LLMFactory.set_fallback_chain(["zhipu", "deepseek", "qwen"])
            with mock.patch.object(cache, "set_str", mock.AsyncMock(return_value=False)) as m_set:
                resp = asyncio.run(self._put(
                    "/ai/llm/chain", {"chain": ["deepseek", "qwen", "zhipu"]},
                ))
            assert resp.status_code == 200
            assert resp.json()["code"] == 2
            m_set.assert_awaited_once()
            # 内存链保持原样（未被新顺序覆盖）
            assert LLMFactory.get_fallback_chain() == ["zhipu", "deepseek", "qwen"]
        finally:
            _reset_factory()


# ─── 启动加载（Redis 优先） ───

class TestStartupLoad:
    """load_fallback_chain_from_redis：启动时读 Redis 链优先"""

    def _reset(self):
        _reset_factory()

    def test_loads_chain_from_redis(self):
        self._reset()
        try:
            with mock.patch.object(
                    cache, "get_str",
                    mock.AsyncMock(return_value="zhipu,deepseek,qwen")):
                asyncio.run(main.load_fallback_chain_from_redis())
            assert LLMFactory.get_fallback_chain() == ["zhipu", "deepseek", "qwen"]
        finally:
            _reset_factory()

    def test_no_redis_chain_uses_config_default(self):
        self._reset()
        try:
            with mock.patch.object(cache, "get_str", mock.AsyncMock(return_value=None)):
                asyncio.run(main.load_fallback_chain_from_redis())
            assert LLMFactory.get_fallback_chain() == CONFIG_DEFAULT
        finally:
            _reset_factory()

    def test_invalid_redis_chain_falls_back(self):
        """Redis 里存了非法链 → 忽略，用配置默认"""
        self._reset()
        try:
            with mock.patch.object(
                    cache, "get_str",
                    mock.AsyncMock(return_value="qwen,bogus,deepseek")):
                asyncio.run(main.load_fallback_chain_from_redis())
            assert LLMFactory.get_fallback_chain() == CONFIG_DEFAULT
        finally:
            _reset_factory()

    def test_redis_unavailable_falls_back(self):
        self._reset()
        try:
            with mock.patch.object(
                    cache, "get_str",
                    mock.AsyncMock(side_effect=Exception("redis down"))):
                asyncio.run(main.load_fallback_chain_from_redis())
            assert LLMFactory.get_fallback_chain() == CONFIG_DEFAULT
        finally:
            _reset_factory()


# ─── set_str / get_str 真实 Redis 往返（跨重启持久） ───

class TestChainPersistence:
    """降级链 Redis 持久化（真实 Redis，需可达 localhost:6379）"""

    def test_set_get_roundtrip_real_redis(self):
        from src.cache import RedisCache

        async def run():
            c = RedisCache()
            key = "llm:fallback_chain:ut"
            # 写入后删除再写，保证用例可重复运行
            await c._ensure_client()
            assert await c.set_str(key, "zhipu,deepseek,qwen") is True
            assert await c.get_str(key) == "zhipu,deepseek,qwen"
            # 覆写后读取新值
            assert await c.set_str(key, "qwen,zhipu") is True
            assert await c.get_str(key) == "qwen,zhipu"
            # 清理测试 key
            await c._ensure_client()
            await c._client.delete(key)
            assert await c.get_str(key) is None
            return True

        assert asyncio.run(run()) is True
