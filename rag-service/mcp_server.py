"""
MCP Server 适配层 — ToolRegistry → 标准 MCP Server（module-067 / ADR-0018）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

把 ToolRegistry 的 6 个只读工具动态注册为标准 MCP server（官方 FastMCP）：
  - stdio（默认传输，本地 Cursor / Claude Code / Claude Desktop 即插即用）
  - Streamable HTTP（挂载进 main.py /ai/mcp，token 认证见 main._mcp_auth_middleware）

设计要点：
  1. ToolRegistry 保持单一事实源——遍历 registry.list_tools() 动态注册，
     改工具定义（description/args_schema）MCP 自动同步，零双份维护。
  2. 默认只暴露 6 个只读检索工具（READ_ONLY_TOOLS 显式白名单）——不按
     group 过滤（检索组含 re_search 双组状态类工具，按 group 会多暴露）。
  3. 参数模型从 args_schema 动态生成（exec 构造带 type hints 的闭包），
     FastMCP 用 type hints 生成 MCP schema。
  4. 执行统一走 AgentTool.run(args, ctx)（复用 15s 超时 + 异常降级语义），
     ctx 用轻量 SimpleNamespace 合成，不构造完整 ReactContext。
  5. 日志走 logging（stderr）——stdio 模式 stdout 是协议通道，禁 print。

已知适配（mcp 1.26.0 实测，勿照抄 ADR 旧写法）：
  - stateless_http / json_response 是 FastMCP 构造参数（streamable_http_app 零参）
  - 1.26.0 FastMCP 构造无 version 参数（server version 由 SDK 内部管理），省略
"""
import logging
import types
from contextlib import asynccontextmanager
from typing import Callable, Optional

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from agent.tool_registry import AgentTool, ToolRegistry, registry

logger = logging.getLogger(__name__)

# 默认只暴露的只读工具白名单（显式 6 名——检索组共 7 个含 re_search 双组
# 状态类工具，按 group 过滤会多暴露，故用显式白名单）
READ_ONLY_TOOLS = frozenset({
    "search_knowledge",
    "search_fts",
    "search_vector",
    "search_graph",
    "extract_entities",
    "recall_memory",
})

_SERVER_NAME = "personal-knowledge-kb"

# JSON Schema type → Python type hint 映射（未知/缺失 → str 兜底）
_TYPE_MAP = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}

# 工具返回截断（防大文档撑爆 client 上下文）
_TRUNCATE_LIMIT = 2000
_TRUNCATE_SUFFIX = "…（结果已截断，完整内容需更多上下文）"


def _truncate_result(text: str, limit: int = _TRUNCATE_LIMIT) -> str:
    """截断超长工具结果；未超限原样返回"""
    if not text or len(text) <= limit:
        return text
    return text[:limit] + _TRUNCATE_SUFFIX


def _make_ctx(args: dict) -> types.SimpleNamespace:
    """合成 MCP 调用的一次性轻量 ctx（不构造完整 ReactContext）

    search 系用 ctx.query 与 ctx.add_docs（MCP 单次调用无后续消费者，no-op）；
    recall_memory 用 ctx.identity 并写 ctx.memory；extract_entities 只用 args。
    """
    return types.SimpleNamespace(
        query=args.get("query") or "",
        identity="mcp",
        history=[],
        docs=[],
        memory="",
        scratchpad=[],
        add_docs=lambda docs: None,
        add_note=lambda note: None,
    )


async def _invoke_tool(tool: AgentTool, args: dict) -> str:
    """执行工具（复用 AgentTool.run 的 15s 超时 + 异常降级语义）+ 截断

    工具返回空串 = 执行失败（工具内部防御文案均非空，如"（无检索结果）"），
    包装为可读提示，不抛裸 Exception 给 MCP client。
    """
    ctx = _make_ctx(args)
    result = await tool.run(args, ctx)
    if not result:
        return "（工具执行失败）"
    return _truncate_result(result)


def _make_tool_fn(tool: AgentTool) -> Callable:
    """从 args_schema 动态构造带 type hints 的闭包（FastMCP 用 hints 生成 schema）

    properties 键 → 参数名（与 args_schema 一致）；type 映射 string→str /
    integer→int / number→float / boolean→bool，未知 → str 兜底；required 字段
    必填无默认值，properties 带 default 的给默认值，其余可选（None）。None
    值参数在构造 args 时剔除——工具函数 `args.get("top_k", 5)` 的缺省语义
    不被 `int(None)` 破坏。
    """
    schema = tool.args_schema or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    # 必填参数在前、可选参数在后（Python 函数签名要求非默认参数不能跟在
    # 默认参数之后——如 _VERIFY_SCHEMA 只 required answer，query 可选）
    required_params, optional_params = [], []
    for name, spec in props.items():
        ann = _TYPE_MAP.get(str(spec.get("type", "")), "str")
        if name in required:
            required_params.append(f"    {name}: {ann},")
        elif "default" in spec:
            optional_params.append(f"    {name}: {ann} = {spec['default']!r},")
        else:
            optional_params.append(f"    {name}: Optional[{ann}] = None,")
    params = required_params + optional_params
    body = "    args = {k: v for k, v in locals().items() if v is not None}\n"
    body += "    return await _invoke_tool(_tool, args)\n"
    source = "async def _mcp_tool(\n" + "\n".join(params) + ") -> str:\n" + body
    namespace = {"_tool": tool, "_invoke_tool": _invoke_tool, "Optional": Optional}
    exec(source, namespace)
    return namespace["_mcp_tool"]


def build_server(registry: ToolRegistry,
                 groups: Optional[list[str]] = None) -> FastMCP:
    """遍历 registry 动态注册工具为标准 MCP server

    Args:
        registry: ToolRegistry（工具定义单一事实源）
        groups: None → 只注册 READ_ONLY_TOOLS 白名单（默认）；显式传值
            （如 ["retrieval"]）→ 按 group 过滤（仅测试/扩展用，含双组工具）

    Returns:
        已注册工具的 FastMCP 实例
    """
    # streamable_http_path="/"：MCP 端点落在挂载根（/ai/mcp/），而非默认
    # /mcp（挂载后会是 /ai/mcp/mcp，与 plan 声明不一致——mcp 1.26.0 实测）。
    # transport_security：关闭 FastMCP 默认 DNS rebinding 保护（只放行
    # localhost Host，0.0.0.0 部署下任何远程 Host 请求被 421 拒绝）——本项目
    # MCP 端点安全边界是 token 认证（fail-closed），host 校验属冗余且有害
    server = FastMCP(
        _SERVER_NAME,
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    group_set = set(groups) if groups is not None else None
    for tool in registry.list_tools():
        if group_set is not None:
            if not (tool.group & group_set):
                continue
        elif tool.name not in READ_ONLY_TOOLS:
            continue
        server.tool(name=tool.name, description=tool.description)(_make_tool_fn(tool))
    return server


# 模块级实例：main.py 挂载复用同一实例（stdio 入口共用；stateless_http/
# json_response 构造参数不影响 stdio 模式）
mcp = build_server(registry)


@asynccontextmanager
async def mcp_http_lifespan():
    """MCP Streamable HTTP 会话任务组生命周期（挂载进 FastAPI 时由宿主 lifespan 进入）

    Starlette Mount 不把 lifespan scope 转发给挂载子应用，而 FastMCP 的
    session_manager 任务组只在自身 Starlette lifespan 里初始化（等价独立
    uvicorn 运行 `mcp.run(transport="streamable-http")` 的行为）——挂载场景
    不手动进入，每个 /ai/mcp 请求都会抛 "Task group is not initialized"
    （mcp 1.26.0 实测）。本函数复刻 StreamableHTTPSessionManager.run() 的
    核心语义（create_task_group + _task_group 注入 + cancel 关闭），但去掉
    其"单次调用 guard"（生产启动进入一次；测试每用例可多轮进入，等价
    FastMCP 实例复用的真实行为）。main.py lifespan 的 yield 前后进入/退出。
    """
    http_app = mcp.streamable_http_app()  # 确保 session manager 已创建
    sm = mcp._session_manager
    async with anyio.create_task_group() as tg:
        sm._task_group = tg
        try:
            yield
        finally:
            tg.cancel_scope.cancel()
            sm._task_group = None


if __name__ == "__main__":
    # stdio 默认传输（本地 Cursor / Claude Code / Claude Desktop 配置
    # {"command": "python", "args": ["mcp_server.py"]} 即可连接）
    mcp.run()
