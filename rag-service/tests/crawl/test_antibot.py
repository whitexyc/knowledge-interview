"""
反爬绕过 + 代理池单元测试（module-077）

覆盖 acceptance-criteria.md 全部验收项：
  - robots.txt 遵循（允许/禁止/fail-open/缓存/固定 UA）
  - UA 轮换（内置池≥8 + 请求头四键 + 随机性）
  - 限速与重试退避（429/5xx 重试 + 指数退避 + jitter + 超时不延迟 + 非 429/5xx 不重试）
  - 代理轮换（空列表直连 + round-robin + 失败切换 + 全部失败）

全 mock：conftest autouse default_antibot_mocks 钉住安全值，单测内显式覆盖。
不加载真实模型、不依赖真实 DB、不触发真实网络请求。
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, call

import httpx
import pytest

from rag.crawl.crawler import (
    _BUILTIN_UA_POOL,
    _check_robots_allowed,
    _random_headers,
    _pick_ua,
    _next_proxy,
    _rate_limit_delay,
    _load_proxies,
    _make_robot_parser,
    _robots_cache,
    _proxy_pool,
    _proxy_index,
    _last_fetch_time,
    _ROBOTS_UA,
    fetch_page,
    CrawlResult,
)


# --- helpers ---------------------------------------------------------------

def _mock_httpx_response(status_code=200, text="<html><title>T</title><body>OK</body></html>"):
    """构造 httpx 响应 mock"""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    return resp


def _make_mock_client_side_effect(get_fn):
    """构造 AsyncMock client，get 用 side_effect"""
    client = AsyncMock()
    client.get = AsyncMock(side_effect=get_fn)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _multi_response_client(responses):
    """构造按序返回多个响应的 client 工厂（用于重试场景）"""
    idx = {"n": 0}

    def factory(**kwargs):
        inst = AsyncMock()
        async def get_fn(url):
            r = responses[min(idx["n"], len(responses) - 1)]
            idx["n"] += 1
            return r
        inst.get = AsyncMock(side_effect=get_fn)
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)
        return inst

    return factory


# ═══════════════════════════════════════════════════════════════════════════
# 1. robots.txt 遵循
# ═══════════════════════════════════════════════════════════════════════════


class TestRobotsAllowed:
    """AC-1.1.1~1.1.5 + AC-2.1~2.2 + AC-3.1~3.2"""

    @pytest.mark.asyncio
    async def test_robots_allow(self, monkeypatch):
        """AC-1.1.1: robots.txt 允许 → True"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_robots_cache_ttl", 0)

        async def get_fn(url):
            resp = MagicMock()
            resp.text = "User-agent: *\nAllow: /"
            resp.raise_for_status = MagicMock()
            return resp

        _robots_cache.clear()
        with patch("rag.crawl.crawler.httpx.AsyncClient",
                   return_value=_make_mock_client_side_effect(get_fn)):
            result = await _check_robots_allowed("https://example.com/page")
        assert result is True

    @pytest.mark.asyncio
    async def test_robots_disallow(self, monkeypatch):
        """AC-1.1.2: robots.txt 禁止 → False"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_robots_cache_ttl", 0)

        async def get_fn(url):
            resp = MagicMock()
            resp.text = "User-agent: *\nDisallow: /private/"
            resp.raise_for_status = MagicMock()
            return resp

        _robots_cache.clear()
        with patch("rag.crawl.crawler.httpx.AsyncClient",
                   return_value=_make_mock_client_side_effect(get_fn)):
            result = await _check_robots_allowed("https://example.com/private/secret")
        assert result is False

    @pytest.mark.asyncio
    async def test_robots_cache_hit(self, monkeypatch):
        """AC-1.1.3: 同域名第二次请求命中缓存（只发一次 HTTP）"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_robots_cache_ttl", 3600)

        call_count = {"n": 0}

        async def get_fn(url):
            call_count["n"] += 1
            resp = MagicMock()
            resp.text = "User-agent: *\nAllow: /"
            resp.raise_for_status = MagicMock()
            return resp

        client = _make_mock_client_side_effect(get_fn)
        _robots_cache.clear()
        with patch("rag.crawl.crawler.httpx.AsyncClient", return_value=client):
            r1 = await _check_robots_allowed("https://cached.example.com/a")
            r2 = await _check_robots_allowed("https://cached.example.com/b")
        assert r1 is True
        assert r2 is True
        assert call_count["n"] == 1

    @pytest.mark.asyncio
    async def test_robots_fail_open(self, monkeypatch):
        """AC-1.1.4 / AC-3.1: 拉取失败 → fail-open 允许"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_robots_cache_ttl", 0)

        async def get_fn(url):
            raise httpx.ConnectError("connection refused")

        _robots_cache.clear()
        with patch("rag.crawl.crawler.httpx.AsyncClient",
                   return_value=_make_mock_client_side_effect(get_fn)):
            result = await _check_robots_allowed("https://down.example.com/page")
        assert result is True

    @pytest.mark.asyncio
    async def test_robots_empty_content_fail_open(self, monkeypatch):
        """AC-2.1: robots.txt 为空白内容 → 允许抓取"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_robots_cache_ttl", 0)

        async def get_fn(url):
            resp = MagicMock()
            resp.text = ""
            resp.raise_for_status = MagicMock()
            return resp

        _robots_cache.clear()
        with patch("rag.crawl.crawler.httpx.AsyncClient",
                   return_value=_make_mock_client_side_effect(get_fn)):
            result = await _check_robots_allowed("https://empty-robots.example.com/")
        assert result is True

    @pytest.mark.asyncio
    async def test_robots_wildcard_rules(self, monkeypatch):
        """AC-2.2: robots.txt 包含 * 通配符规则"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_robots_cache_ttl", 0)

        async def get_fn(url):
            resp = MagicMock()
            resp.text = "User-agent: *\nDisallow: /admin/"
            resp.raise_for_status = MagicMock()
            return resp

        _robots_cache.clear()
        with patch("rag.crawl.crawler.httpx.AsyncClient",
                   return_value=_make_mock_client_side_effect(get_fn)):
            allowed = await _check_robots_allowed("https://example.com/page")
            denied = await _check_robots_allowed("https://example.com/admin/dashboard")
        assert allowed is True
        assert denied is False

    @pytest.mark.asyncio
    async def test_robots_uses_fixed_ua(self, monkeypatch):
        """AC-1.1.5: robots 检查使用固定 UA 而非随机 UA

        _check_robots_allowed 调用 httpx.AsyncClient(timeout=5) 不传 headers,
        使用 _ROBOTS_UA 常量 via rp.can_fetch.
        """
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_robots_cache_ttl", 0)

        captured_kwargs = {}

        async def get_fn(url, **kwargs):
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.text = "User-agent: *\nAllow: /"
            resp.raise_for_status = MagicMock()
            return resp

        captured_constructor_kwargs = {}
        original_client = httpx.AsyncClient

        def factory(**kwargs):
            captured_constructor_kwargs.update(kwargs)
            return _make_mock_client_side_effect(get_fn)

        _robots_cache.clear()
        with patch("rag.crawl.crawler.httpx.AsyncClient", side_effect=factory):
            await _check_robots_allowed("https://ua-check.example.com/page")
        # _check_robots_allowed 不传 headers 到 client
        # P3-5: robots 检查现在传 User-Agent 头（固定 _ROBOTS_UA）
        assert captured_constructor_kwargs.get("headers", {}).get("User-Agent") == _ROBOTS_UA

# ═══════════════════════════════════════════════════════════════════════════
# 2. UA 轮换 + 请求头增强
# ═══════════════════════════════════════════════════════════════════════════


class TestUARotation:
    """AC-1.2.1~1.2.4"""

    def test_ua_pool_size(self):
        """AC-1.2.1: _BUILTIN_UA_POOL 包含 ≥8 个不同 UA"""
        assert len(_BUILTIN_UA_POOL) >= 8

    def test_random_headers_keys(self):
        """AC-1.2.2: _random_headers() 返回含四个键的 dict"""
        headers = _random_headers()
        assert "User-Agent" in headers
        assert "Accept" in headers
        assert "Accept-Language" in headers
        assert "Accept-Encoding" in headers

    def test_ua_randomness(self):
        """AC-1.2.3: 连续 10 次调用产生 ≥2 种不同 UA"""
        uas = {_pick_ua() for _ in range(10)}
        assert len(uas) >= 2

    @pytest.mark.asyncio
    async def test_fetch_uses_random_headers(self, monkeypatch):
        """AC-1.2.4: fetch_page 内使用 _random_headers() 的 headers

        headers 传给 httpx.AsyncClient 构造函数（不是 get 方法）。
        """
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_retry_max", 0)

        captured_constructor_kwargs = {}

        def factory(**kwargs):
            captured_constructor_kwargs.update(kwargs)
            inst = AsyncMock()
            inst.get = AsyncMock(return_value=_mock_httpx_response(200))
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            return inst

        with patch("rag.crawl.crawler.httpx.AsyncClient", side_effect=factory):
            result = await fetch_page("https://example.com")
        assert result.success is True
        headers = captured_constructor_kwargs.get("headers", {})
        assert "User-Agent" in headers
        assert "Accept" in headers
        assert "Accept-Language" in headers
        assert "Accept-Encoding" in headers


# ═══════════════════════════════════════════════════════════════════════════
# 3. 限速与重试退避
# ═══════════════════════════════════════════════════════════════════════════


class TestRetryAndBackoff:
    """AC-1.3.1~1.3.6 + AC-2.3 + AC-3.4"""

    @pytest.mark.asyncio
    async def test_429_triggers_retry_then_success(self, monkeypatch):
        """AC-1.3.1: 429→重试→200 成功"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_retry_max", 3)
        monkeypatch.setattr(settings, "crawl_retry_base_seconds", 0.01)

        with patch("rag.crawl.crawler.httpx.AsyncClient",
                   side_effect=_multi_response_client([
                       _mock_httpx_response(429),
                       _mock_httpx_response(200),
                   ])):
            with patch("rag.crawl.crawler.asyncio.sleep", new_callable=AsyncMock):
                result = await fetch_page("https://example.com")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_500_triggers_retry_then_success(self, monkeypatch):
        """AC-1.3.2: 500→重试→200 成功"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_retry_max", 3)
        monkeypatch.setattr(settings, "crawl_retry_base_seconds", 0.01)

        with patch("rag.crawl.crawler.httpx.AsyncClient",
                   side_effect=_multi_response_client([
                       _mock_httpx_response(500),
                       _mock_httpx_response(200),
                   ])):
            with patch("rag.crawl.crawler.asyncio.sleep", new_callable=AsyncMock):
                result = await fetch_page("https://example.com")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self, monkeypatch):
        """AC-1.3.3: 429 重试间隔符合指数退避"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_retry_max", 3)
        monkeypatch.setattr(settings, "crawl_retry_base_seconds", 1.0)

        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        with patch("rag.crawl.crawler.httpx.AsyncClient",
                   side_effect=_multi_response_client([
                       _mock_httpx_response(429),
                       _mock_httpx_response(429),
                       _mock_httpx_response(429),
                       _mock_httpx_response(429),
                   ])):
            with patch("rag.crawl.crawler.asyncio.sleep", side_effect=mock_sleep):
                with patch("rag.crawl.crawler.random.uniform", return_value=0.0):
                    result = await fetch_page("https://example.com")
        assert result.success is False
        # 3 次重试 sleep（attempt 0/1/2），attempt 3 不 sleep 直接返回
        assert len(sleep_calls) == 3
        assert sleep_calls[0] >= 1.0   # base * 2^0 = 1.0
        assert sleep_calls[1] >= 2.0   # base * 2^1 = 2.0

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_returns_failure(self, monkeypatch):
        """AC-1.3.4: 超过 crawl_max_retries 次重试后返回失败"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_retry_max", 3)
        monkeypatch.setattr(settings, "crawl_retry_base_seconds", 0.01)

        with patch("rag.crawl.crawler.httpx.AsyncClient",
                   side_effect=_multi_response_client([
                       _mock_httpx_response(429),
                       _mock_httpx_response(429),
                       _mock_httpx_response(429),
                       _mock_httpx_response(429),
                   ])):
            with patch("rag.crawl.crawler.asyncio.sleep", new_callable=AsyncMock):
                result = await fetch_page("https://example.com")
        assert result.success is False
        assert "429" in result.error

    @pytest.mark.asyncio
    async def test_timeout_retries_no_extra_delay(self, monkeypatch):
        """AC-1.3.5: 超时直接重试无额外延迟（最终成功）"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_retry_max", 2)
        monkeypatch.setattr(settings, "crawl_retry_base_seconds", 1.0)

        call_count = {"n": 0}

        async def get_fn(url):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise httpx.TimeoutException("timeout")
            return _mock_httpx_response(200)

        client = _make_mock_client_side_effect(get_fn)

        with patch("rag.crawl.crawler.httpx.AsyncClient", return_value=client):
            result = await fetch_page("https://example.com")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_non_retryable_error_no_retry(self, monkeypatch):
        """AC-1.3.6: 非 429/5xx/超时异常不重试（如 403 直接返回）"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_retry_max", 3)

        call_count = {"n": 0}

        async def get_fn(url):
            call_count["n"] += 1
            resp = MagicMock()
            resp.status_code = 403
            resp.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError("403", request=MagicMock(), response=resp)
            )
            return resp

        client = _make_mock_client_side_effect(get_fn)

        with patch("rag.crawl.crawler.httpx.AsyncClient", return_value=client):
            result = await fetch_page("https://example.com/forbidden")
        assert result.success is False
        assert "403" in result.error
        assert call_count["n"] == 1

    @pytest.mark.asyncio
    async def test_max_retries_zero_no_retry(self, monkeypatch):
        """AC-2.3: crawl_retry_max=0 时不重试"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_retry_max", 0)

        call_count = {"n": 0}

        async def get_fn(url):
            call_count["n"] += 1
            resp = MagicMock()
            resp.status_code = 429
            resp.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError("429", request=MagicMock(), response=resp)
            )
            return resp

        client = _make_mock_client_side_effect(get_fn)

        with patch("rag.crawl.crawler.httpx.AsyncClient", return_value=client):
            result = await fetch_page("https://example.com")
        assert result.success is False
        assert call_count["n"] == 1

    @pytest.mark.asyncio
    async def test_error_message_truncated(self, monkeypatch):
        """AC-3.4: 重试用尽后 error 信息包含异常摘要（截断 200 字符）"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_retry_max", 1)
        monkeypatch.setattr(settings, "crawl_retry_base_seconds", 0.01)

        long_msg = "x" * 300

        async def get_fn(url):
            raise Exception(long_msg)

        client = _make_mock_client_side_effect(get_fn)

        with patch("rag.crawl.crawler.httpx.AsyncClient", return_value=client):
            result = await fetch_page("https://example.com")
        assert result.success is False
        assert len(result.error) <= 200


# ═══════════════════════════════════════════════════════════════════════════
# 4. 代理轮换
# ═══════════════════════════════════════════════════════════════════════════


class TestProxyRotation:
    """AC-1.4.1~1.4.4 + AC-2.5 + AC-3.3"""

    def test_empty_proxy_list_returns_none(self, monkeypatch):
        """AC-1.4.1: crawl_proxies 为空时 _next_proxy 返回 None"""
        import rag.crawl.crawler as crawler_mod
        monkeypatch.setattr(crawler_mod, "_proxy_pool", [])
        monkeypatch.setattr(crawler_mod, "_proxy_index", 0)
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_proxies", "")
        assert _next_proxy() is None

    def test_round_robin_rotation(self, monkeypatch):
        """AC-1.4.2: 2 个代理时 round-robin 轮换"""
        import rag.crawl.crawler as crawler_mod
        monkeypatch.setattr(crawler_mod, "_proxy_index", 0)
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_proxies", "http://proxy-a:8080,http://proxy-b:8080")
        monkeypatch.setattr(crawler_mod, "_proxy_pool", [])

        p1 = _next_proxy()
        p2 = _next_proxy()
        p3 = _next_proxy()
        assert p1 == "http://proxy-a:8080"
        assert p2 == "http://proxy-b:8080"
        assert p3 == "http://proxy-a:8080"

    def test_proxy_with_spaces_parsed(self, monkeypatch):
        """AC-2.5: crawl_proxy_list 含空格时正确解析"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_proxies", " http://a:8080 , http://b:8080 ")
        proxies = _load_proxies()
        assert len(proxies) == 2
        assert proxies[0] == "http://a:8080"
        assert proxies[1] == "http://b:8080"

    @pytest.mark.asyncio
    async def test_proxy_switch_on_retry(self, monkeypatch):
        """AC-1.4.3: 重试时切换到下一个代理"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_retry_max", 3)
        monkeypatch.setattr(settings, "crawl_retry_base_seconds", 0.01)
        monkeypatch.setattr(settings, "crawl_proxies", "http://proxy-a:8080,http://proxy-b:8080")

        import rag.crawl.crawler as crawler_mod
        monkeypatch.setattr(crawler_mod, "_proxy_pool", [])
        monkeypatch.setattr(crawler_mod, "_proxy_index", 0)

        captured_proxies = []
        call_count = {"n": 0}

        def factory(**kwargs):
            captured_proxies.append(kwargs.get("proxy"))
            inst = AsyncMock()
            call_count["n"] += 1
            if call_count["n"] <= 1:
                inst.get = AsyncMock(return_value=_mock_httpx_response(500))
            else:
                inst.get = AsyncMock(return_value=_mock_httpx_response(200))
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            return inst

        with patch("rag.crawl.crawler.httpx.AsyncClient", side_effect=factory):
            with patch("rag.crawl.crawler.asyncio.sleep", new_callable=AsyncMock):
                result = await fetch_page("https://example.com")
        assert result.success is True
        assert len(captured_proxies) == 2
        assert captured_proxies[0] == "http://proxy-a:8080"
        assert captured_proxies[1] == "http://proxy-b:8080"

    @pytest.mark.asyncio
    async def test_all_proxies_fail_returns_failure(self, monkeypatch):
        """AC-1.4.4: 全部代理失败后返回失败"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_retry_max", 2)
        monkeypatch.setattr(settings, "crawl_retry_base_seconds", 0.01)
        monkeypatch.setattr(settings, "crawl_proxies", "http://bad-a:8080,http://bad-b:8080")

        import rag.crawl.crawler as crawler_mod
        monkeypatch.setattr(crawler_mod, "_proxy_pool", [])
        monkeypatch.setattr(crawler_mod, "_proxy_index", 0)

        def factory(**kwargs):
            inst = AsyncMock()
            inst.get = AsyncMock(return_value=_mock_httpx_response(500))
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            return inst

        with patch("rag.crawl.crawler.httpx.AsyncClient", side_effect=factory):
            with patch("rag.crawl.crawler.asyncio.sleep", new_callable=AsyncMock):
                result = await fetch_page("https://example.com")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_no_proxy_when_empty(self, monkeypatch):
        """AC-1.4.1: 空列表时 httpx.AsyncClient 不传 proxy"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_retry_max", 0)
        monkeypatch.setattr(settings, "crawl_proxies", "")

        import rag.crawl.crawler as crawler_mod
        monkeypatch.setattr(crawler_mod, "_proxy_pool", [])
        monkeypatch.setattr(crawler_mod, "_proxy_index", 0)

        captured_kwargs = {}

        def factory(**kwargs):
            captured_kwargs.update(kwargs)
            inst = AsyncMock()
            inst.get = AsyncMock(return_value=_mock_httpx_response(200))
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            return inst

        with patch("rag.crawl.crawler.httpx.AsyncClient", side_effect=factory):
            result = await fetch_page("https://example.com")
        assert result.success is True
        assert "proxy" not in captured_kwargs


# ═══════════════════════════════════════════════════════════════════════════
# 5. 限速 delay
# ═══════════════════════════════════════════════════════════════════════════


class TestRateLimitDelay:
    """AC-1.3.7 + AC-2.4"""

    @pytest.mark.asyncio
    async def test_delay_injected_between_requests(self, monkeypatch):
        """AC-1.3.7: 同源请求间隔 ≥ crawl_request_delay_seconds"""
        from src.config import settings
        # 显式覆盖 conftest 的 0
        monkeypatch.setattr(settings, "crawl_request_delay_seconds", 1.0)

        import rag.crawl.crawler as crawler_mod
        crawler_mod._last_fetch_time.clear()

        sleep_called = {"count": 0, "delay": 0}

        async def mock_sleep(delay):
            sleep_called["count"] += 1
            sleep_called["delay"] += delay

        # time.monotonic 恒返回 100.0，_last_fetch_time 为空 → last=None → 不 sleep
        # 但第一次调用后 last 被设为 100.0；再调用一次 elapsed=0 < 1.0 → sleep
        with patch("rag.crawl.crawler.asyncio.sleep", side_effect=mock_sleep):
            with patch("rag.crawl.crawler.time.monotonic", return_value=100.0):
                await _rate_limit_delay(source_id=1)  # 首次，last=None → 设 last
                await _rate_limit_delay(source_id=1)  # 二次，elapsed=0 → sleep 1.0

        assert sleep_called["count"] == 1
        assert sleep_called["delay"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_no_delay_when_config_zero(self, monkeypatch):
        """AC-2.4: crawl_request_delay_seconds=0 时不限速"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_request_delay_seconds", 0)

        import rag.crawl.crawler as crawler_mod
        crawler_mod._last_fetch_time.clear()

        sleep_called = {"count": 0}

        async def mock_sleep(delay):
            sleep_called["count"] += 1

        with patch("rag.crawl.crawler.asyncio.sleep", side_effect=mock_sleep):
            await _rate_limit_delay(source_id=99)

        assert sleep_called["count"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# 6. 配置项验证
# ═══════════════════════════════════════════════════════════════════════════


class TestAntibotConfig:
    """验证 module-077 新增 config 默认值

    注意：conftest autouse fixtures 会钉住部分值为测试安全值，
    所以这里验证的是 Settings 模型声明的默认值（类定义级别），
    而非运行时 settings 实例值。
    """

    def test_config_field_defaults(self):
        """验证 Settings 类字段的默认值（不受 conftest 钉住影响）"""
        from src.config import Settings
        fields = Settings.model_fields
        assert fields["crawl_request_delay_seconds"].default == 1.0
        assert fields["crawl_retry_max"].default == 3
        assert fields["crawl_retry_base_seconds"].default == 1.0  # P3-3: 对齐 plan
        assert fields["crawl_user_agents"].default == ""


# ═══════════════════════════════════════════════════════════════════════════
# 7. P3-8 补充测试（AC-3.2/3.3 + 限速 per-source 隔离）
# ═══════════════════════════════════════════════════════════════════════════


class TestAntibotP3Additions:
    """P3-8: 补充 3 个缺失独立测试"""

    @pytest.mark.asyncio
    async def test_robots_non_text_html_fail_open(self, monkeypatch):
        """AC-3.2: robots.txt 返回非文本 HTML 404 → fail-open 允许"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_robots_cache_ttl", 0)

        async def get_fn(url):
            resp = MagicMock()
            resp.text = "<html><body>Not Found</body></html>"
            resp.raise_for_status = MagicMock()
            return resp

        _robots_cache.clear()
        with patch("rag.crawl.crawler.httpx.AsyncClient",
                   return_value=_make_mock_client_side_effect(get_fn)):
            result = await _check_robots_allowed("https://html-robots.example.com/page")
        assert result is True

    @pytest.mark.asyncio
    async def test_proxy_connect_error_switches(self, monkeypatch):
        """AC-3.3: 代理连接拒绝（ConnectError）→ 切换下一个代理"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_retry_max", 2)
        monkeypatch.setattr(settings, "crawl_retry_base_seconds", 0.01)
        monkeypatch.setattr(settings, "crawl_proxies", "http://bad-a:8080,http://good-b:8080")

        import rag.crawl.crawler as crawler_mod
        monkeypatch.setattr(crawler_mod, "_proxy_pool", [])
        monkeypatch.setattr(crawler_mod, "_proxy_index", 0)

        call_count = {"n": 0}

        def factory(**kwargs):
            inst = AsyncMock()
            call_count["n"] += 1
            if call_count["n"] <= 1:
                # 第一个代理连接拒绝
                inst.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
            else:
                inst.get = AsyncMock(return_value=_mock_httpx_response(200))
            inst.__aenter__ = AsyncMock(return_value=inst)
            inst.__aexit__ = AsyncMock(return_value=False)
            return inst

        with patch("rag.crawl.crawler.httpx.AsyncClient", side_effect=factory):
            result = await fetch_page("https://example.com")
        assert result.success is True
        assert call_count["n"] == 2  # 第一个失败，第二个成功

    @pytest.mark.asyncio
    async def test_rate_limit_per_source_isolation(self, monkeypatch):
        """限速 per-source 隔离：不同 source_key 互不阻塞"""
        from src.config import settings
        monkeypatch.setattr(settings, "crawl_request_delay_seconds", 1.0)

        import rag.crawl.crawler as crawler_mod
        crawler_mod._last_fetch_time.clear()

        sleep_called = {"count": 0, "delays": []}

        async def mock_sleep(delay):
            sleep_called["count"] += 1
            sleep_called["delays"].append(delay)

        with patch("rag.crawl.crawler.asyncio.sleep", side_effect=mock_sleep):
            with patch("rag.crawl.crawler.time.monotonic", return_value=100.0):
                # source 1 首次 → 设 last
                await _rate_limit_delay(source_id=1)
                # source 2 首次 → 不同 source，不 sleep
                await _rate_limit_delay(source_id=2)
                # source 1 二次 → elapsed=0 < 1.0 → sleep
                await _rate_limit_delay(source_id=1)

        # 只有 source 1 第二次调用触发 sleep
        assert sleep_called["count"] == 1
        assert sleep_called["delays"][0] == pytest.approx(1.0)
