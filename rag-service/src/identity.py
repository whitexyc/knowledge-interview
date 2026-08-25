"""
JWT 身份解析（module-032）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

登录用户身份作为记忆隔离键：从 Authorization: Bearer <token> 解析 JWT
（HS256，共享密钥 settings.jwt_secret，与 Java 后端一致），得到 user_id。
无 / 非法 / 过期 token → 返回 ""（降级 client_ip，匿名零回归）。

- parse_jwt(authorization): 解析 Authorization 头 → user_id 或 ""
- resolve_identity(request): user_id 非空优先，否则 client_ip

中间件（main.py rate_limit_middleware）负责把 parse_jwt 结果注入
request.state.user_id；各端点用 resolve_identity 取最终身份。
"""
import logging

import jwt
from fastapi import Request

from src.config import settings

logger = logging.getLogger(__name__)

_ALGORITHM = "HS256"


def parse_jwt(authorization: str) -> str:
    """从 Authorization 头解析 JWT，返回 user_id；无/非法/过期返回 ''

    只接受 "Bearer <token>" 格式（大小写不敏感）；token 用 HS256 + 共享
    JWT_SECRET 校验。任何失败（无头、非 Bearer、签名非法、过期、sub 缺失）
    一律返回空串，由调用方降级 client_ip（匿名零回归）。

    Args:
        authorization: Authorization 请求头原始值（可 None）

    Returns:
        JWT payload.sub（str），失败时 ""
    """
    if not settings.jwt_secret:
        logger.warning("JWT_SECRET 未配置，跳过 token 解析（全部按匿名处理）")
        return ""
    if not authorization:
        return ""
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return ""
    token = parts[1]
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[_ALGORITHM])
    except jwt.PyJWTError:
        return ""
    sub = payload.get("sub")
    if not sub:
        return ""
    return str(sub)


def resolve_identity(request: Request) -> str:
    """解析请求身份：user_id（JWT.sub）非空优先，否则 client_ip（匿名降级）

    Args:
        request: FastAPI Request（限流中间件已注入
            request.state.client_ip / request.state.user_id）

    Returns:
        identity 字符串：user_id 或 client_ip（均取不到时 "unknown"）
    """
    user_id = getattr(request.state, "user_id", "")
    if user_id:
        return user_id
    return getattr(request.state, "client_ip", "unknown")
