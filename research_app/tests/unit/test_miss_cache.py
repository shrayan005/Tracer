import pytest
import redis.asyncio as aioredis

from app.integrations.miss_cache import MissCache, is_permanent


@pytest.mark.parametrize("failures, permanent", [
    (["no free full text found"], True),
    (["metadata: HTTP 403", "unpaywall: expected pdf, got an HTML page"], True),
    (["europepmc: timed out after 60s", "metadata: HTTP 403"], False),     # one temporary failure: retry later
    (["europepmc: network error: ReadTimeout"], False),
    (["europepmc: HTTP 503"], False),
    (["unpaywall: HTTP 429"], False),
    ([], False),
])
def test_is_permanent(failures, permanent):
    assert is_permanent(failures) is permanent


class DictRedis:
    """Just enough of a Redis client for the cache."""
    def __init__(self):
        self.data, self.ttl = {}, {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.data[key], self.ttl[key] = value.encode(), ex

    async def delete(self, key):
        self.data.pop(key, None)


class BrokenRedis:
    async def get(self, key):
        raise aioredis.ConnectionError("redis is down")

    async def set(self, key, value, ex=None):
        raise aioredis.ConnectionError("redis is down")

    async def delete(self, key):
        raise aioredis.ConnectionError("redis is down")


async def test_remember_get_forget_with_expiry():
    redis = DictRedis()
    cache = MissCache(redis, ttl_seconds=3600)
    assert await cache.get(7) is None
    await cache.remember(7, "no free full text found")
    assert await cache.get(7) == "no free full text found"
    assert redis.ttl["fulltext:miss:7"] == 3600
    await cache.forget(7)
    assert await cache.get(7) is None


async def test_redis_down_never_breaks_the_app():
    cache = MissCache(BrokenRedis())
    assert await cache.get(7) is None              # behaves like "nothing remembered"
    await cache.remember(7, "x")                    # no exception
    await cache.forget(7)
