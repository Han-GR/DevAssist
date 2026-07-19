from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import uuid4

import structlog

from app.core.config import Settings


_RATE_LIMIT_LUA = """
local key = KEYS[1]
local now_ms = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local ttl_seconds = tonumber(ARGV[4])
local member = ARGV[5]

redis.call("ZREMRANGEBYSCORE", key, 0, now_ms - window_ms)
redis.call("ZADD", key, now_ms, member)
local count = redis.call("ZCARD", key)
redis.call("EXPIRE", key, ttl_seconds)

if count > limit then
  redis.call("ZREM", key, member)
  return {0, count - 1}
end

return {1, count}
"""


@dataclass(frozen=True)
class RateLimitDecision:
    """
    一次限流判断的结果。

    Args:
        allowed (bool): 是否允许本次请求通过。
        count (int): 当前滑动窗口内的请求数（包含本次请求，若被拒绝则不计入）。
        limit (int): 窗口内允许的最大请求数。
        window_seconds (int): 窗口大小（秒）。

    Returns:
        RateLimitDecision: 判断结果。

    Raises:
        None
    """

    allowed: bool
    count: int
    limit: int
    window_seconds: int


class RateLimiter:
    """
    基于 Redis ZSET 的滑动窗口限流器。

    Args:
        redis_url (str): Redis 连接串。
        key_prefix (str): Redis key 前缀，避免冲突。
        requests_per_minute (int): 每分钟最大请求数。
        window_seconds (int): 滑动窗口大小（秒），默认 60。
        fail_open (bool): Redis 不可用时是否放行（True=放行，False=拒绝）。

    Returns:
        RateLimiter: 限流器实例。

    Raises:
        ValueError: 配置非法时抛出。

    Notes:
        该实现用于“每用户/每 IP”的请求级限流，覆盖所有 API。
    """

    def __init__(
        self,
        *,
        redis_url: str,
        key_prefix: str,
        requests_per_minute: int,
        window_seconds: int,
        fail_open: bool,
    ) -> None:
        if not redis_url.strip():
            raise ValueError("redis_url is required")
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be a positive integer")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be a positive integer")

        from redis.asyncio import Redis

        self._redis = Redis.from_url(redis_url, decode_responses=True)
        self._key_prefix = key_prefix.strip(":")
        self._requests_per_minute = requests_per_minute
        self._window_seconds = window_seconds
        self._fail_open = fail_open
        self._logger = structlog.get_logger()
        self._script = self._redis.register_script(_RATE_LIMIT_LUA)

    @classmethod
    def from_settings(cls, *, settings: Settings) -> RateLimiter | None:
        """
        从 Settings 构建限流器。

        Args:
            settings (Settings): 后端配置对象。

        Returns:
            RateLimiter | None: 未启用限流时返回 None。

        Raises:
            ValueError: 配置非法时抛出。

        Notes:
            测试环境默认不启用，避免单测依赖外部 Redis。
        """
        if settings.env == "test":
            return None
        if not settings.rate_limit_enabled:
            return None
        if not settings.rate_limit_redis_url.strip():
            return None

        return cls(
            redis_url=settings.rate_limit_redis_url,
            key_prefix=settings.rate_limit_key_prefix,
            requests_per_minute=settings.rate_limit_requests_per_minute,
            window_seconds=settings.rate_limit_window_seconds,
            fail_open=settings.rate_limit_fail_open,
        )

    async def check(self, *, identity: str) -> RateLimitDecision:
        """
        对单个 identity 做一次滑动窗口限流判断。

        Args:
            identity (str): 用户标识（优先 user_id，其次 IP）。

        Returns:
            RateLimitDecision: 判断结果。

        Raises:
            Exception: 当 Redis 异常且 fail_open=False 时，可能抛出异常交由上层处理。
        """
        key = f"{self._key_prefix}:{identity}"
        now_ms = int(time.time() * 1000)
        window_ms = self._window_seconds * 1000
        ttl_seconds = self._window_seconds + 5
        member = f"{now_ms}-{uuid4().hex}"

        try:
            raw = await self._script(
                keys=[key],
                args=[now_ms, window_ms, self._requests_per_minute, ttl_seconds, member],
            )
            allowed = bool(int(raw[0]))
            count = int(raw[1])
            return RateLimitDecision(
                allowed=allowed,
                count=count,
                limit=self._requests_per_minute,
                window_seconds=self._window_seconds,
            )
        except Exception as exc:
            self._logger.warning(
                "rate_limit_check_failed",
                identity=identity,
                error=str(exc),
                fail_open=self._fail_open,
            )
            if self._fail_open:
                return RateLimitDecision(
                    allowed=True,
                    count=0,
                    limit=self._requests_per_minute,
                    window_seconds=self._window_seconds,
                )
            raise
