import redis.asyncio as redis
import json
import hashlib
from typing import Optional

REDIS_URL = "redis://localhost:6379"

redis_client = redis.from_url(
    REDIS_URL,
    encoding="utf-8",
    decode_responses=True
)

CACHE_TTL = 3600  # 1 hour


def make_cache_key(data: dict) -> str:
    """
    Deterministic hash of request input
    """
    sorted_data = json.dumps(data, sort_keys=True)
    hash_key = hashlib.sha256(sorted_data.encode()).hexdigest()
    return f"prediction:{hash_key}"


async def cache_get(key: str) -> Optional[dict]:
    cached = await redis_client.get(key)
    if cached:
        return json.loads(cached)
    return None


async def cache_set(key: str, value: dict):
    await redis_client.set(
        key,
        json.dumps(value),
        ex=CACHE_TTL
    )