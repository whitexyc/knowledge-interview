"""Module-026 Reflector 低温度改造单元测试

覆盖（验收 §1.2 Reflector 改造 + §4.1「Reflector 温度 / 降级链 provider 单测」）：
- Reflector 默认 _provider="fallback"（走降级链，消除硬编码 deepseek）
- 反思用温度 0.1（结构化 JSON 稳定），生成保持 0.7
- LLMFactory.get_client 支持按温度创建，(provider, temperature) 缓存隔离
- FallbackClient 低温度贯穿降级链（各供应商都用 0.1）

实现说明：
- 用 mock 打桩 LLMFactory.get_client / ChatOpenAI / settings，不依赖真实 API key
- 同步用例内 asyncio.run 执行，不依赖 pytest-asyncio
- 用例内 clear_cache 隔离 LLMFactory 实例缓存
"""
import asyncio
from unittest import mock

from llm.client import LLMFactory, FallbackClient, LLMException
from agent.reflector import Reflector


class TestReflectorProviderAndTemperature:
    """Reflector provider + 温度设置"""

    def test_defaults_to_fallback_and_low_reflection_temperature(self):
        r = Reflector()
        assert r._provider == "fallback"          # 走降级链（消除单点）
        assert r._reflection_temperature == 0.1   # 反思低温度（JSON 稳定）
        assert r._generation_temperature == 0.7   # 生成保持默认

    def test_custom_provider_preserved(self):
        r = Reflector(provider="qwen")
        assert r._provider == "qwen"

    def test_check_sufficiency_uses_low_temperature_client(self):
        captured = {}

        def fake_get(provider=None, temperature=None):
            # LLMFactory.get_client 是同步方法，side_effect 须返回 client 而非 coroutine
            captured["provider"] = provider
            captured["temperature"] = temperature
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(
                return_value='{"sufficient": true, "reason": "ok"}',
            )
            return client

        async def run():
            r = Reflector()
            with mock.patch("llm.client.LLMFactory.get_client", side_effect=fake_get):
                return await r.check_sufficiency(
                    "什么是线程池",
                    # module-044：数量闸门 <2 篇直接不充分（零 LLM），
                    # 该用例验证反思温度，需 ≥2 篇且分数达标才走 LLM 路径
                    [{"title": "T", "content": "c", "abs_cosine": 0.7},
                     {"title": "T2", "content": "c2", "abs_cosine": 0.6}],
                )

        result = asyncio.run(run())
        assert captured["provider"] == "fallback"
        assert captured["temperature"] == 0.1
        assert result["sufficient"] is True

    def test_generate_answer_uses_default_temperature(self):
        captured = {}

        def fake_get(provider=None, temperature=None):
            captured["provider"] = provider
            captured["temperature"] = temperature
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(return_value="答案是线程池[1]")
            return client

        async def run():
            r = Reflector()
            with mock.patch("llm.client.LLMFactory.get_client", side_effect=fake_get):
                return await r.generate_answer(
                    "什么是线程池", [{"id": 1, "title": "T", "source": "s", "content": "c"}],
                )

        result = asyncio.run(run())
        assert captured["provider"] == "fallback"
        assert captured["temperature"] == 0.7   # 生成保持 0.7
        assert result == "答案是线程池[1]"

    def test_generate_answer_stream_uses_default_temperature(self):
        captured = {}

        async def fake_stream(prompt):
            yield "答案"
            yield "[1]"

        def fake_get(provider=None, temperature=None):
            captured["provider"] = provider
            captured["temperature"] = temperature
            client = mock.MagicMock()
            client.generate_stream = fake_stream
            return client

        async def run():
            r = Reflector()
            out = []
            with mock.patch("llm.client.LLMFactory.get_client", side_effect=fake_get):
                async for tok in r.generate_answer_stream(
                    "什么是线程池", [{"id": 1, "title": "T", "source": "s", "content": "c"}],
                ):
                    out.append(tok)
            return out

        out = asyncio.run(run())
        assert captured["temperature"] == 0.7
        assert out == ["答案", "[1]"]


class TestLLMFactoryTemperature:
    """LLMFactory 按温度创建 + (provider, temperature) 缓存隔离"""

    def test_get_client_accepts_temperature(self):
        constructed = []

        def fake_openai(*args, **kwargs):
            constructed.append(kwargs)
            return mock.MagicMock()

        try:
            with mock.patch("llm.client.settings.deepseek_api_key", "fake-key"), \
                 mock.patch("llm.client.ChatOpenAI", side_effect=fake_openai):
                LLMFactory.clear_cache()
                low = LLMFactory.get_client("deepseek", temperature=0.1)
                low_again = LLMFactory.get_client("deepseek", temperature=0.1)
                default = LLMFactory.get_client("deepseek")  # 默认 0.7
        finally:
            LLMFactory.clear_cache()

        # 构造参数里温度正确：先 0.1，后默认 0.7
        assert constructed[0]["temperature"] == 0.1
        assert constructed[-1]["temperature"] == 0.7
        # 同 (provider, temperature) 缓存复用；不同温度不同实例
        assert low is low_again
        assert low is not default
        assert low._llm is not None

    def test_fallback_passes_temperature_to_chain(self):
        # 低温度贯穿降级链：第一个失败 → 第二个（同为温度 0.1）
        calls = []

        def fake_get(provider=None, temperature=None):
            # LLMFactory.get_client 是同步方法，side_effect 须返回 client 而非 coroutine
            calls.append((provider, temperature))
            client = mock.MagicMock()
            if len(calls) == 1:
                client.generate = mock.AsyncMock(
                    side_effect=LLMException("deepseek", "服务不可用"),
                )
            else:
                client.generate = mock.AsyncMock(return_value="ok")
            return client

        async def run():
            client = FallbackClient(["deepseek", "qwen"], temperature=0.1)
            with mock.patch("llm.client.LLMFactory.get_client", side_effect=fake_get):
                return await client.generate("prompt")

        result = asyncio.run(run())
        assert result == "ok"
        assert calls == [("deepseek", 0.1), ("qwen", 0.1)]  # 温度 0.1 贯穿降级链
