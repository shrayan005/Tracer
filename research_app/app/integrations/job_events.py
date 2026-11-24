"""
"This job changed" signals between the worker and the web server (SSE step 6), over Redis pub/sub.

The worker is a separate process, so the API can't see its progress directly. Redis is already shared
(it carries the job queue), so:
    worker   after every change of a job (running, a progress message, done, failed): publish "job-events:42"
    API      a browser following job 42 subscribes to that channel, and re-reads the job on each signal
The signal carries NO data: the jobs table stays the one truth, and a lost signal only delays an update
(the follower also re-reads now and then, see JobService.follow). Pub/sub keeps nothing: whoever isn't
listening at that moment doesn't get it, which is fine for a nudge.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.common.logging import get_logger

log = get_logger(__name__)
Changed = Callable[[float], Awaitable[bool]]     # wait up to N seconds: True if a signal came, False if not


def channel(job_id: int) -> str:
    return f"job-events:{job_id}"


class JobEvents:
    def __init__(self, redis: aioredis.Redis):
        self.redis = redis

    async def publish(self, job_id: int) -> None:
        """Tell any follower that job `job_id` changed. Never fails the job: a lost signal only delays."""
        try:
            await self.redis.publish(channel(job_id), "changed")
        except (RedisError, OSError) as e:
            log.warning("job %s: could not publish its change (%s)", job_id, e)

    @asynccontextmanager
    async def listen(self, job_id: int) -> AsyncIterator[Changed]:
        """Subscribe to one job's signals for as long as the `async with` lasts."""
        pubsub = self.redis.pubsub()
        try:
            await pubsub.subscribe(channel(job_id))
        except (RedisError, OSError) as e:                 # Redis down: followers fall back to re-reading
            log.warning("job %s: can't subscribe to its changes (%s): re-reading on a timer", job_id, e)
            await pubsub.aclose()
            yield _just_wait
            return

        async def changed(timeout: float) -> bool:
            try:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=timeout)
            except (RedisError, OSError, asyncio.TimeoutError):
                await asyncio.sleep(min(timeout, 1.0))     # a hiccup: don't spin, the timer still re-reads
                return False
            return message is not None

        try:
            yield changed
        finally:
            try:
                await pubsub.unsubscribe(channel(job_id))
            except (RedisError, OSError):
                pass
            await pubsub.aclose()


class NoJobEvents:
    """No Redis (tests, the CLI): nothing is published, and followers re-read on a timer."""

    async def publish(self, job_id: int) -> None:
        pass

    @asynccontextmanager
    async def listen(self, job_id: int) -> AsyncIterator[Changed]:
        yield _just_wait


async def _just_wait(timeout: float) -> bool:
    await asyncio.sleep(timeout)
    return False
