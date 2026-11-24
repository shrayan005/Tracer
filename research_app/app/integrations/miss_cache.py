"""
"No full text for this paper" memory (integration layer, step 4.7), kept in Redis with an expiry.

Without it, every run retries papers we already know are paywalled or blocked (10-30 s each run).
Only PERMANENT failures are remembered (nothing found, HTTP 403/404, an HTML page instead of a PDF);
temporary ones (timeouts, network errors, HTTP 5xx/429) are not: the service may be back in a minute.

A cache must never break the app: if Redis is down, it simply remembers nothing.
"""

import redis.asyncio as aioredis

from app.common.logging import get_logger

log = get_logger(__name__)

KEY = "fulltext:miss:{paper_id}"
TEMPORARY = ("timed out", "network error", "HTTP 5", "HTTP 429")     # reasons worth retrying soon


def is_permanent(failures: list[str]) -> bool:
    """True if retrying soon is pointless: no failure looks temporary."""
    return bool(failures) and not any(word in failure for failure in failures for word in TEMPORARY)


class MissCache:
    def __init__(self, redis: aioredis.Redis, ttl_seconds: int = 24 * 3600):
        self.redis = redis
        self.ttl_seconds = ttl_seconds

    async def get(self, paper_id: int) -> str | None:
        """Why this paper had no full text recently, or None if it should be tried."""
        try:
            value = await self.redis.get(KEY.format(paper_id=paper_id))
        except aioredis.RedisError as e:
            log.warning("miss cache unavailable (get): %s", e)
            return None
        return value.decode() if isinstance(value, bytes) else value

    async def remember(self, paper_id: int, reason: str) -> None:
        try:
            await self.redis.set(KEY.format(paper_id=paper_id), reason[:500], ex=self.ttl_seconds)
        except aioredis.RedisError as e:
            log.warning("miss cache unavailable (set): %s", e)

    async def forget(self, paper_id: int) -> None:
        try:
            await self.redis.delete(KEY.format(paper_id=paper_id))
        except aioredis.RedisError as e:
            log.warning("miss cache unavailable (delete): %s", e)
