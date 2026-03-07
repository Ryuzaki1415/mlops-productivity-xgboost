import redis 
import json
import hashlib
from typing import Optional

from api.api_config import REDIS_HOST,REDIS_PORT

import os
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

redis_client_sync = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
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

def cache_get_sync(key: str):
    cached = redis_client_sync.get(key)
    if cached:
        return json.loads(cached)
    return None


def cache_set_sync(key: str, value: dict):
    redis_client_sync.set(
        key,
        json.dumps(value),
        ex=CACHE_TTL
    )