"""SSE step 6: "job changed" signals through the REAL Redis (pub/sub).
Run with: python -m pytest -m integration tests/integration/test_job_events.py"""

import asyncio

import pytest

from app.dependencies import close_resources, get_events_redis
from app.integrations.job_events import JobEvents

pytestmark = pytest.mark.integration


async def test_a_listener_gets_a_published_signal_and_only_for_its_job():
    events = JobEvents(get_events_redis())
    try:
        async with events.listen(987654) as changed:
            assert await changed(0.3) is False                   # nothing yet (the timeout, not an error)
            await events.publish(987655)                         # another job: not heard
            assert await changed(0.3) is False
            await events.publish(987654)
            assert await changed(2.0) is True
            # a long wait is not cut off by a socket timeout (the 3 s one of the other Redis client)
            started = asyncio.get_running_loop().time()
            assert await changed(4.0) is False
            assert asyncio.get_running_loop().time() - started >= 3.5
    finally:
        await close_resources()
