"""MCP Server 集成单测（module-067 / ADR-0018）"""
import asyncio

import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError

import mcp_server as mcp_module
from agent.tool_registry import ToolRegistry, _ENTITY_SCHEMA, _MEMORY_SCHEMA, _SEARCH_SCHEMA
from mcp_server import READ_ONLY_TOOLS, _make_ctx, _truncate_result, _TRUNCATE_SUFFIX, build_server


async def _noop(ctx, args):
    return "noop"


def _list_tools(server):
    """await server.list_tools() → {name: MCPTool}"""
    async def run():
        return {t.name: t for t in await server.list_tools()}
    return asyncio.run(run())


def _call(server, name, args):
    """await server.call_tool(name, args) → TextContent 文本"""
    async def run():
        content, _ = await server.call_tool(name, args)
        return content[0].text
    return asyncio.run(run())


def _reg_with(func, schema=None, name="search_knowledge"):
    """构造单工具注册表（自定义 func/schema；工具名须在白名单内才会被注册）"""
    reg = ToolRegistry()
    reg.register(
        name, "测试工具描述",
        schema or {"type": "object", "properties": {"query": {"type": "string"}},
                   "required": ["query"]},
        func,
    )
    return reg


class TestBuildServer:
    def test_default_registers_exactly_read_only_tools(self):
        """默认白名单：恰好 6 个只读工具，与 READ_ONLY_TOOLS 集合一致"""
        names = set(_list_tools(build_server(mcp_module.registry)))
        assert names == set(READ_ONLY_TOOLS)
        assert len(names) == 6

    def test_non_read_only_tools_excluded(self):
        """只读过滤：generate/verify/re_search/note_to_self 不在注册列表（显式白名单意义）"""
        names = set(_list_tools(build_server(mcp_module.registry)))
        for excluded in ("generate_answer", "verify_answer", "re_search", "note_to_self"):
            assert excluded not in names

    def test_explicit_groups_filters_by_group(self):
        """显式 groups 按 group 过滤（仅测试/扩展用，含双组工具）"""
        names = set(_list_tools(build_server(mcp_module.registry, groups=["generation"])))
        assert names == {"generate_answer", "verify_answer", "re_search", "note_to_self"}

    def test_description_passthrough(self):
        """description 透传：改 registry 工具描述后 build_server 自动同步（单一事实源）"""
        reg = _reg_with(_noop)
        tools = _list_tools(build_server(reg))
        assert tools["search_knowledge"].description == "测试工具描述"
        reg.get("search_knowledge").description = "新描述"
        tools = _list_tools(build_server(reg))
        assert tools["search_knowledge"].description == "新描述"

    def test_schema_search_variant(self):
        """_SEARCH_SCHEMA：query 必填 string，top_k 可选 integer"""
        reg = _reg_with(_noop, schema=_SEARCH_SCHEMA)
        schema = _list_tools(build_server(reg))["search_knowledge"].inputSchema
        assert schema["required"] == ["query"]
        assert schema["properties"]["query"]["type"] == "string"
        # 可选参数生成 anyOf [integer, null]（FastMCP 从 Optional[int] 推导）
        assert {"type": "integer"} in schema["properties"]["top_k"]["anyOf"]
        assert "top_k" not in schema["required"]

    def test_schema_entity_variant(self):
        """_ENTITY_SCHEMA：仅 query 必填"""
        reg = _reg_with(_noop, schema=_ENTITY_SCHEMA)
        schema = _list_tools(build_server(reg))["search_knowledge"].inputSchema
        assert schema["required"] == ["query"]
        assert set(schema["properties"]) == {"query"}

    def test_schema_memory_variant(self):
        """_MEMORY_SCHEMA：query 必填 + top_k 可选"""
        reg = _reg_with(_noop, schema=_MEMORY_SCHEMA)
        schema = _list_tools(build_server(reg))["search_knowledge"].inputSchema
        assert schema["required"] == ["query"]
        assert {"type": "integer"} in schema["properties"]["top_k"]["anyOf"]
        assert "top_k" not in schema["required"]

    def test_schema_unknown_type_falls_back_to_str(self):
        """未知 type → str 兜底"""
        reg = _reg_with(_noop, schema={"type": "object", "properties": {"x": {"type": "weird"}}})
        schema = _list_tools(build_server(reg))["search_knowledge"].inputSchema
        assert {"type": "string"} in schema["properties"]["x"]["anyOf"]

    def test_schema_schema_default_applied(self):
        """properties 带 default → 参数带默认值（非必填）"""
        schema = {"type": "object", "properties": {"n": {"type": "integer", "default": 5}}}
        reg = _reg_with(_noop, schema=schema)
        out = _list_tools(build_server(reg))["search_knowledge"].inputSchema
        assert out["properties"]["n"]["default"] == 5
        assert "n" not in out.get("required", [])


class TestTruncate:
    def test_under_limit_unchanged(self):
        text = "字" * 2000
        assert _truncate_result(text) == text

    def test_empty_passthrough(self):
        assert _truncate_result("") == ""

    def test_over_limit_truncated_with_suffix(self):
        text = "字" * 2500
        out = _truncate_result(text)
        assert out.startswith("字" * 2000)
        assert out.endswith(_TRUNCATE_SUFFIX)
        assert len(out) == 2000 + len(_TRUNCATE_SUFFIX)


class TestExecution:
    def test_call_returns_tool_result(self):
        async def func(ctx, args):
            return "检索结果: " + args["query"]
        server = build_server(_reg_with(func))
        assert _call(server, "search_knowledge", {"query": "hello"}) == "检索结果: hello"

    def test_ctx_query_and_identity(self):
        """ctx 合成：query 透传 + identity=mcp（search 系语义）"""
        async def func(ctx, args):
            return f"{ctx.identity}|{ctx.query}"
        server = build_server(_reg_with(func))
        assert _call(server, "search_knowledge", {"query": "q1"}) == "mcp|q1"

    def test_ctx_memory_writable(self):
        """recall_memory 写 ctx.memory 不抛错（SimpleNamespace 可赋值）"""
        async def func(ctx, args):
            ctx.memory = "m1"
            return str(ctx.memory)
        server = build_server(_reg_with(func))
        assert _call(server, "search_knowledge", {"query": "q"}) == "m1"

    def test_failure_returns_readable_message(self):
        """工具执行失败（run 捕获返回空串）→ 包装可读提示，不抛裸 Exception"""
        async def func(ctx, args):
            raise RuntimeError("boom")
        server = build_server(_reg_with(func))
        assert _call(server, "search_knowledge", {"query": "q"}) == "（工具执行失败）"

    def test_optional_param_none_dropped(self):
        """可选参数缺省：args 剔除 None（工具 int(args.get('top_k', 5)) 语义不被破坏）"""
        seen = {}
        async def func(ctx, args):
            seen["args"] = dict(args)
            return "ok"
        reg = _reg_with(func, schema=_SEARCH_SCHEMA)
        _call(build_server(reg), "search_knowledge", {"query": "q"})
        assert seen["args"] == {"query": "q"}
        _call(build_server(reg), "search_knowledge", {"query": "q", "top_k": 3})
        assert seen["args"] == {"query": "q", "top_k": 3}

    def test_missing_required_param_rejected(self):
        """缺必填参数：MCP 参数校验拒绝（可读错误，非 500/裸异常）"""
        server = build_server(_reg_with(_noop))
        async def run():
            with pytest.raises(ToolError) as exc:
                await server.call_tool("search_knowledge", {})
            assert "query" in str(exc.value)
        asyncio.run(run())

    def test_wrong_type_rejected(self):
        """top_k 类型错误：参数校验拒绝，不崩溃"""
        server = build_server(_reg_with(_noop, schema=_SEARCH_SCHEMA))
        async def run():
            with pytest.raises(ToolError) as exc:
                await server.call_tool("search_knowledge", {"query": "q", "top_k": "abc"})
            assert "top_k" in str(exc.value)
        asyncio.run(run())

    def test_make_ctx_shape(self):
        """_make_ctx 字段齐全：query/identity/docs/add_docs/memory"""
        ctx = _make_ctx({"query": "q1"})
        assert ctx.query == "q1"
        assert ctx.identity == "mcp"
        assert ctx.docs == []
        assert ctx.memory == ""
        ctx.add_docs([{"id": 1}])
        assert ctx.docs == []
        assert _make_ctx({}).query == ""


# MCP initialize 握手（Streamable HTTP 有界响应；GET 是长连接 SSE 通道不适用 200 断言）
_MCP_INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
               "clientInfo": {"name": "test", "version": "1.0.0"}},
}
_MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


class TestHttpAuth:
    """/ai/mcp 认证（ASGITransport 直连 main.app）"""

    @staticmethod
    def _request(method, path="/ai/mcp/", headers=None, json=None):
        import main as main_module

        async def run():
            # 生产等价：宿主 lifespan 启动 MCP session 任务组（Mount 不转发
            # lifespan scope，mcp_http_lifespan 手动进入；401 路径用不到但无害）
            async with main_module.mcp_http_lifespan():
                transport = httpx.ASGITransport(app=main_module.app, raise_app_exceptions=True)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    # 规范 URL 带尾斜杠（Starlette Mount 对 /ai/mcp 307 → /ai/mcp/）
                    return await client.request(method, path, headers=headers, json=json)
        return asyncio.run(run())

    @classmethod
    def _init(cls, token):
        """带 token 的 initialize 握手请求"""
        return cls._request("POST", headers={"Authorization": f"Bearer {token}", **_MCP_HEADERS},
                            json=_MCP_INIT)

    def test_no_token_401(self, monkeypatch):
        """无 Authorization 头 → 401"""
        from src.config import settings
        monkeypatch.setattr(settings, "mcp_token", "s3cret")
        assert self._request("GET").status_code == 401

    def test_wrong_token_401(self, monkeypatch):
        """错误 token → 401"""
        from src.config import settings
        monkeypatch.setattr(settings, "mcp_token", "s3cret")
        resp = self._request("GET", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401
        assert "未授权" in resp.json()["message"]

    def test_correct_token_200(self, monkeypatch):
        """正确 token → initialize 握手 200（含 session 任务组初始化——Mount 不转发
        lifespan 的修复验证；MCP GET 是长连接 SSE 通道，200 断言用 POST 握手）"""
        from src.config import settings
        monkeypatch.setattr(settings, "mcp_token", "s3cret")
        resp = self._init("s3cret")
        assert resp.status_code == 200
        assert resp.json()["result"]["serverInfo"]["name"] == "personal-knowledge-kb"

    def test_no_slash_redirects_to_slash(self, monkeypatch):
        """/ai/mcp（无尾斜杠）→ 307 /ai/mcp/（Starlette Mount 规范 URL）"""
        from src.config import settings
        monkeypatch.setattr(settings, "mcp_token", "s3cret")
        resp = self._request("GET", path="/ai/mcp")
        assert resp.status_code == 307
        assert resp.headers["location"].endswith("/ai/mcp/")

    def test_empty_token_always_401(self, monkeypatch):
        """token 为空：即使带正确格式头也恒 401（fail-closed 双保险）"""
        from src.config import settings
        monkeypatch.setattr(settings, "mcp_token", "")
        resp = self._request("GET", headers={"Authorization": "Bearer anything"})
        assert resp.status_code == 401

    def test_token_change_takes_effect_immediately(self, monkeypatch):
        """运行时改 token：中间件每次请求实时读 settings（不缓存）"""
        from src.config import settings
        monkeypatch.setattr(settings, "mcp_token", "a")
        assert self._init("a").status_code == 200
        assert self._init("b").status_code == 401
        monkeypatch.setattr(settings, "mcp_token", "b")
        assert self._init("b").status_code == 200


class TestFailClosed:
    def test_lifespan_raises_without_token(self, monkeypatch):
        """PW_MCP_TOKEN 未设置 → lifespan 启动即抛 RuntimeError（fail-closed 拒绝启动）"""
        import main as main_module
        from src.config import settings
        monkeypatch.setattr(settings, "jwt_secret", "test")
        monkeypatch.setattr(settings, "mcp_token", "")

        async def enter():
            async with main_module.lifespan(main_module.app):
                pass
        with pytest.raises(RuntimeError, match="PW_MCP_TOKEN"):
            asyncio.run(enter())
