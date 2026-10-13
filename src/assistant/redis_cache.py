
import hashlib
import json
import os
from typing import Any

import redis
from dotenv import load_dotenv


load_dotenv()

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0",
)

DEFAULT_TTL_SECONDS = 300

_client = None


def get_redis_client():
    """Create and reuse a Redis client."""
    global _client

    if _client is None:
        _client = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )

    return _client


def cache_key(namespace: str, payload: dict[str, Any]) -> str:
    """Build a stable cache key without storing raw questions in keys."""
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    return f"matchpulse:{namespace}:{digest}"


def get_cached_json(key: str) -> Any | None:
    """Return cached JSON, or None if missing/unavailable."""
    try:
        value = get_redis_client().get(key)

        if value is None:
            return None

        return json.loads(value)

    except (redis.RedisError, json.JSONDecodeError, OSError):
        return None


def set_cached_json(
    key: str,
    value: Any,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> bool:
    """Store JSON with a time-to-live."""
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

        return bool(
            get_redis_client().set(
                key,
                serialized,
                ex=ttl_seconds,
            )
        )

    except (redis.RedisError, TypeError, ValueError, OSError):
        return False


def health_check() -> bool:
    """Check Redis availability."""
    try:
        return bool(get_redis_client().ping())
    except (redis.RedisError, OSError):
        return False
