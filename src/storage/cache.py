import os

import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://cache:6379/0")
client = redis.from_url(REDIS_URL, decode_responses=True)


def set_cache(key: str, value: str, expire: int = 300):
    client.setex(key, expire, value)


def get_cache(key: str):
    return client.get(key)


def get_redis_client():
    """Creates and returns a connection to the Redis cache."""
    # Pull the URL from the environment, defaulting to our Docker Compose service name
    redis_url = os.getenv("REDIS_URL", "redis://cache:6379/0")

    # decode_responses=True automatically converts byte strings to normal strings
    return redis.from_url(redis_url, decode_responses=True)
