"""
IP 限流器 — 滑动窗口算法
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

实现基于 IP 的请求频率限制，防止滥用。

算法：滑动窗口计数
  用 dict[IP] = list[timestamp] 记录每个 IP 的请求时间戳。
  每次请求时，剔除窗口外的旧时间戳，检查剩余数量是否超限。

为什么用简单内存 dict 而不是 Redis？
  单机部署时，内存 dict 足够简单可靠，无额外依赖。
  如果后续扩展到多实例，可以替换为 Redis sorted set。
"""
import logging
import time
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# 默认限流配置
_DEFAULT_MAX_REQUESTS = 20  # 窗口内最大请求数
_DEFAULT_WINDOW_SECONDS = 60  # 窗口大小（秒）

# 存储结构：{client_ip: [timestamp1, timestamp2, ...]}
_request_records: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(
    client_ip: str,
    max_requests: int = _DEFAULT_MAX_REQUESTS,
    window_seconds: int = _DEFAULT_WINDOW_SECONDS,
) -> tuple[bool, int]:
    """检查是否超过限流阈值

    Args:
        client_ip: 客户端 IP
        max_requests: 窗口内允许的最大请求数
        window_seconds: 时间窗口（秒）

    Returns:
        (allowed: bool, retry_after: int)
        allowed=False 时 retry_after 表示需要等待的秒数
    """
    now = time.time()
    window_start = now - window_seconds

    # 获取该 IP 的请求记录
    records = _request_records[client_ip]

    # 剔除窗口外的旧记录
    _request_records[client_ip] = [t for t in records if t > window_start]

    # 检查是否超限
    if len(_request_records[client_ip]) >= max_requests:
        # 计算最快什么时候可以重试
        oldest = min(_request_records[client_ip])
        retry_after = int(window_seconds - (now - oldest))
        logger.warning("IP %s 触发限流: %d 次/%ds", client_ip, max_requests, window_seconds)
        return False, max(retry_after, 1)

    # 记录本次请求
    _request_records[client_ip].append(now)
    return True, 0


def get_client_ip(
    forwarded: Optional[str],
    remote_addr: Optional[str],
) -> str:
    """从请求头中提取客户端真实 IP

    优先取 X-Forwarded-For 的第一个 IP（最接近客户端的），
    没有则取 remote_addr。
    """
    if forwarded:
        # X-Forwarded-For: client_ip, proxy1, proxy2, ...
        return forwarded.split(",")[0].strip()
    return remote_addr or "unknown"
