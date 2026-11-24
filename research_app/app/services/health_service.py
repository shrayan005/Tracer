"""
Health checks: is every dependency the app needs working?

Async: the three checks (database, Redis, LLM) run AT THE SAME TIME with asyncio.gather,
so the whole check takes as long as the slowest one, not the sum of all three.
A service only uses what it is given (session factory, redis client, gateway),
so tests can give it fakes.
"""

import asyncio
from collections.abc import Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.logging import get_logger
from app.domain.errors import LlmError
from app.domain.schemas import ComponentHealth, HealthReport
from app.integrations.llm.gateway import LlmGateway

log = get_logger(__name__)


class HealthService:
    def __init__(self, session_factory: Callable[[], AsyncSession], redis_client, gateway: LlmGateway):
        self.session_factory = session_factory
        self.redis = redis_client          # a redis.asyncio client
        self.gateway = gateway

    async def check(self, deep: bool = False) -> HealthReport:
        """deep=True also sends a tiny real request to the LLM (costs a few tokens)."""
        components = await asyncio.gather(self._database(), self._redis(), self._llm(deep))
        status = "ok" if all(c.ok for c in components) else "degraded"
        return HealthReport(status=status, components=list(components))

    async def _database(self) -> ComponentHealth:
        try:
            async with self.session_factory() as session:
                await session.execute(text("SELECT 1"))
                has_vector = await session.scalar(
                    text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'"))
            detail = "connected, pgvector enabled" if has_vector else "connected, pgvector MISSING"
            return ComponentHealth(name="database", ok=bool(has_vector), detail=detail)
        except Exception as e:                   # any failure = "down"; logged so it is not only on /health
            log.warning("health check: database is down: %s", _short(e))
            return ComponentHealth(name="database", ok=False, detail=_short(e))

    async def _redis(self) -> ComponentHealth:
        try:
            return ComponentHealth(name="redis", ok=bool(await self.redis.ping()), detail="PONG")
        except Exception as e:                   # any failure = "down"; logged so it is not only on /health
            log.warning("health check: redis is down: %s", _short(e))
            return ComponentHealth(name="redis", ok=False, detail=_short(e))

    async def _llm(self, deep: bool) -> ComponentHealth:
        names = self.gateway.provider_names
        if not names:
            return ComponentHealth(name="llm", ok=False, detail="no providers have API keys")
        if not deep:
            return ComponentHealth(name="llm", ok=True, detail=f"configured: {', '.join(names)}")
        try:
            result = await self.gateway.complete([{"role": "user", "content": "Reply with the word: ok"}],
                                                 purpose="health", max_tokens=200)
            return ComponentHealth(name="llm", ok=True,
                                   detail=f"{result.provider} answered in {result.latency_s}s")
        except LlmError as e:
            return ComponentHealth(name="llm", ok=False, detail=_short(e))


def _short(error: Exception) -> str:
    return f"{type(error).__name__}: {str(error)[:200]}"
