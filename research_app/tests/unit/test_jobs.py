"""Background jobs (step 7.1): submit, run, fail, retry, ownership. In-memory fakes: no Redis, no database."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from sqlalchemy.exc import IntegrityError

from app.domain.errors import ConflictError, NotFoundError, QuotaExceededError
from app.integrations.llm.run_context import current_run_id, current_user_id
from app.services import job_service as js
from app.services.job_service import JobService

ROWS: dict[int, SimpleNamespace] = {}
EVENTS: list[str] = []


class FakeJobRepository:
    def __init__(self, session):
        pass

    async def create(self, *, user_id, kind, params, retry_of=None):
        job = SimpleNamespace(id=len(ROWS) + 1, user_id=user_id, kind=kind, params=params, status="queued",
                              progress=[], topic_id=None, error=None, retry_of=retry_of,
                              created_at=datetime.now(timezone.utc), started_at=None, finished_at=None)
        ROWS[job.id] = job
        EVENTS.append(f"row {job.id}")
        return job

    async def get(self, job_id):
        return ROWS.get(job_id)

    async def list_for_user(self, user_id, limit=20):
        return [j for j in reversed(ROWS.values()) if j.user_id == user_id][:limit]

    async def active(self, user_id, kind):
        return [j for j in ROWS.values() if j.user_id == user_id and j.kind == kind
                and j.status in ("queued", "running")]

    async def mark_running(self, job_id):
        job = ROWS.get(job_id)
        if job:
            job.status, job.started_at = "running", datetime.now(timezone.utc)
        return job

    async def add_progress(self, job_id, message):
        ROWS[job_id].progress = [*ROWS[job_id].progress, {"at": "2026-11-23T10:00:00", "message": message}]

    async def finish(self, job_id, topic_id):
        ROWS[job_id].status, ROWS[job_id].topic_id = "done", topic_id

    async def fail(self, job_id, error, topic_id=None):
        ROWS[job_id].status, ROWS[job_id].error = "failed", error


class FakeUsage:
    TODAY = 0                                    # tokens the user already used today
    PROVIDER_QUERIES = 0

    def __init__(self, session):
        pass

    async def usage(self, run_id):
        return 1234, 0.0

    async def user_usage_since(self, user_id, since):
        assert since.hour == 0 and since.minute == 0           # counted from midnight UTC
        return FakeUsage.TODAY, 0.01

    async def user_providers_since(self, user_id, since):
        FakeUsage.PROVIDER_QUERIES += 1
        return [{"provider": "gemini", "ok_calls": 12, "failed_calls": 0, "tokens": FakeUsage.TODAY,
                 "last_error": None, "last_error_at": None},
                {"provider": "groq", "ok_calls": 0, "failed_calls": 3, "tokens": 0,
                 "last_error": "429 rate limited", "last_error_at": since}]


@asynccontextmanager
async def fake_unit_of_work():
    yield object()


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    ROWS.clear()
    EVENTS.clear()
    FakeUsage.TODAY = 0
    FakeUsage.PROVIDER_QUERIES = 0
    monkeypatch.setattr(js, "JobRepository", FakeJobRepository)
    monkeypatch.setattr(js, "LlmCallRepository", FakeUsage)


def service(fail_enqueue=False, budget=0) -> JobService:
    async def enqueue(job_id):
        if fail_enqueue:
            raise ConnectionError("redis is down")
        EVENTS.append(f"queued {job_id}")
    return JobService(fake_unit_of_work, enqueue, daily_token_budget=budget)


class FakeResearch:
    def __init__(self, error=None, cancel=False):
        self.error, self.cancel, self.seen_run_id = error, cancel, None

    async def research(self, topic, user_id, progress=None):
        self.seen_run_id, self.seen_user_id = current_run_id(), current_user_id()
        await progress("planning and searching...")
        await progress("summarising 10 selected papers...")
        if self.cancel:
            raise asyncio.CancelledError()
        if self.error:
            raise self.error
        return SimpleNamespace(topic_id=259)

    async def analyze_topic(self, topic_id, user_id, *, refresh=False, progress=None):
        await progress(f"analysing topic {topic_id} (refresh={refresh})")
        return SimpleNamespace(topic_id=topic_id)


# ---------- submit ----------

async def test_the_row_is_saved_before_the_job_is_queued():
    job = await service().submit("research", {"topic": "creatine"}, user_id=1)
    assert EVENTS == ["row 1", "queued 1"] and job.status == "queued" and job.tokens == 1234


async def test_a_queue_failure_marks_the_job_failed_instead_of_queued_forever():
    with pytest.raises(ConnectionError):
        await service(fail_enqueue=True).submit("research", {"topic": "x"}, user_id=1)
    assert ROWS[1].status == "failed" and "could not queue" in ROWS[1].error


async def test_unknown_kinds_are_refused():
    with pytest.raises(ValueError, match="unknown job kind"):
        await service().submit("delete everything", {}, user_id=1)


# ---------- run (the worker) ----------

async def test_run_saves_progress_in_order_and_the_result():
    s, research = service(), FakeResearch()
    job = await s.submit("research", {"topic": "creatine"}, user_id=1)
    await s.run(job.id, research)
    done = await s.get(job.id, user_id=1)
    assert done.status == "done" and done.topic_id == 259
    assert [p.message for p in done.progress] == ["planning and searching...", "summarising 10 selected papers..."]
    assert research.seen_run_id == job.id                 # LLM calls are tagged with the job id ...
    assert research.seen_user_id == 1                     # ... and its owner (the daily budget, 7.4)


async def test_analyze_jobs_pass_the_topic_and_refresh():
    s = service()
    job = await s.submit("analyze", {"topic_id": 119, "refresh": True}, user_id=1)
    await s.run(job.id, FakeResearch())
    done = await s.get(job.id, user_id=1)
    assert done.topic_id == 119 and done.progress[0].message == "analysing topic 119 (refresh=True)"


async def test_a_crash_is_saved_on_the_job_and_does_not_stop_the_worker():
    s = service()
    job = await s.submit("research", {"topic": "x"}, user_id=1)
    await s.run(job.id, FakeResearch(error=RuntimeError("all LLM providers failed")))   # no exception escapes
    failed = await s.get(job.id, user_id=1)
    assert failed.status == "failed" and failed.error == "RuntimeError: all LLM providers failed"
    assert len(failed.progress) == 2                      # what happened before the crash is kept


async def test_the_time_limit_marks_the_job_failed_and_still_stops_it():
    s = service()
    job = await s.submit("research", {"topic": "x"}, user_id=1)
    with pytest.raises(asyncio.CancelledError):           # the worker must still see the cancellation
        await s.run(job.id, FakeResearch(cancel=True))
    assert ROWS[job.id].status == "failed" and "took too long" in ROWS[job.id].error


# ---------- retry and ownership ----------

async def test_only_a_failed_job_can_be_retried_as_a_new_job():
    s = service()
    job = await s.submit("research", {"topic": "x"}, user_id=1)
    with pytest.raises(ConflictError, match="only a failed job"):
        await s.retry(job.id, user_id=1)
    await s.run(job.id, FakeResearch(error=RuntimeError("boom")))
    again = await s.retry(job.id, user_id=1)
    assert again.id != job.id and again.retry_of == job.id and again.params == {"topic": "x"}
    assert again.status == "queued"


async def test_another_users_job_looks_like_a_missing_one():
    s = service()
    job = await s.submit("research", {"topic": "private"}, user_id=1)
    with pytest.raises(NotFoundError, match=f"job {job.id} does not exist"):
        await s.get(job.id, user_id=2)
    assert await s.list(user_id=2) == [] and len(await s.list(user_id=1)) == 1


async def test_one_research_at_a_time_per_user():
    s = service()
    first = await s.submit("research", {"topic": "creatine"}, user_id=1)
    with pytest.raises(ConflictError, match="already have a research running"):
        await s.submit("research", {"topic": "creatine again"}, user_id=1)      # a double click
    assert (await s.submit("research", {"topic": "mine"}, user_id=2)).status == "queued"   # other users: fine
    assert (await s.submit("analyze", {"topic_id": 119}, user_id=1)).status == "queued"   # analyze: fine
    await s.run(first.id, FakeResearch())
    assert (await s.submit("research", {"topic": "next"}, user_id=1)).status == "queued"  # after it ends: fine


# ---------- per-user limits (step 7.4) ----------

async def test_a_job_over_the_daily_budget_is_refused_with_the_reset_time():
    FakeUsage.TODAY = 250_000
    with pytest.raises(QuotaExceededError, match="daily LLM budget") as error:
        await service(budget=250_000).submit("analyze", {"topic_id": 119}, user_id=1)
    assert 0 < error.value.retry_after <= 24 * 3600 and ROWS == {}          # nothing was created


async def test_usage_and_an_unlimited_budget():
    FakeUsage.TODAY = 40_000
    usage = await service(budget=250_000).usage(user_id=1)
    assert (usage.tokens_today, usage.remaining) == (40_000, 210_000) and usage.resets_at.hour == 0
    FakeUsage.TODAY = 10_000_000
    assert (await service(budget=0).usage(user_id=1)).remaining is None       # 0 = no limit
    assert (await service(budget=0).submit("analyze", {"topic_id": 1}, user_id=1)).status == "queued"


async def test_usage_shows_which_provider_works_and_which_fails():
    FakeUsage.TODAY = 40_000
    usage = await service(budget=250_000).usage(user_id=1)
    gemini, groq = usage.providers
    assert (gemini.provider, gemini.tokens, gemini.failed_calls) == ("gemini", 40_000, 0)
    assert (groq.failed_calls, groq.last_error) == (3, "429 rate limited")


async def test_the_budget_check_skips_the_provider_breakdown():
    await service(budget=250_000).submit("analyze", {"topic_id": 1}, user_id=1)   # runs check_budget
    assert FakeUsage.PROVIDER_QUERIES == 0                                      # one query per submit, not two


async def test_the_database_guarantee_becomes_a_friendly_409(monkeypatch):
    async def racing_create(self, **kw):           # another request created the active research first
        raise IntegrityError("INSERT", {}, Exception(
            'duplicate key value violates unique constraint "uq_jobs_one_active_research"'))
    monkeypatch.setattr(FakeJobRepository, "create", racing_create)
    with pytest.raises(ConflictError, match="already have a research running"):
        await service().submit("research", {"topic": "x"}, user_id=1)


# ---------- following a job live (SSE step 6) ----------

class MemoryJobEvents:
    """In-memory stand-in for the Redis signals: publish() wakes whoever listens to that job."""

    def __init__(self):
        self.published: list[int] = []
        self.waiting: dict[int, asyncio.Event] = {}

    async def publish(self, job_id):
        self.published.append(job_id)
        if job_id in self.waiting:
            self.waiting[job_id].set()

    @asynccontextmanager
    async def listen(self, job_id):
        signal = self.waiting.setdefault(job_id, asyncio.Event())

        async def changed(timeout):
            try:
                await asyncio.wait_for(signal.wait(), timeout)
            except asyncio.TimeoutError:
                return False
            signal.clear()
            return True
        yield changed


class SlowResearch(FakeResearch):
    """Like FakeResearch, but takes a moment between steps (as a real run does)."""

    async def research(self, topic, user_id, progress=None):
        for step in ("planning and searching...", "screening 40 papers...", "summarising 10 papers..."):
            await asyncio.sleep(0.02)
            await progress(step)
        return SimpleNamespace(topic_id=259)


async def test_a_follower_sees_each_change_as_the_worker_publishes_it():
    events = MemoryJobEvents()
    jobs = JobService(fake_unit_of_work, lambda _id: asyncio.sleep(0), events=events,
                      follow_check_s=30, follow_max_s=30)          # long timer: only signals can explain updates
    job = await jobs.submit("research", {"topic": "creatine"}, user_id=1)
    updates = await jobs.follow(job.id, user_id=1)
    seen = [await anext(updates)]                                  # subscribed and has the first state
    worker = asyncio.create_task(jobs.run(job.id, SlowResearch()))
    seen += [info async for info in updates if info is not None]   # ends by itself when the job is done
    await worker
    assert [i.status for i in seen][0] == "queued" and seen[-1].status == "done" and seen[-1].topic_id == 259
    counts = [len(i.progress) for i in seen]
    assert counts == sorted(counts) and len(set(counts)) >= 3      # the steps arrived one after another ...
    assert counts[-1] == 3                                         # ... and changes close together merge:
    assert events.published == [job.id] * 5                       # 5 signals (running, 3 messages, done)


async def test_without_signals_a_follower_still_catches_up_on_its_timer():
    jobs = JobService(fake_unit_of_work, lambda _id: asyncio.sleep(0), follow_check_s=0.02, follow_max_s=5)
    job = await jobs.submit("research", {"topic": "creatine"}, user_id=1)
    updates = await jobs.follow(job.id, user_id=1)
    assert (await anext(updates)).status == "queued"
    ROWS[job.id].status = "failed"                                 # changed with nobody publishing
    assert (await anext(updates)).status == "failed"
    assert [i async for i in updates] == []                       # a finished job ends the stream


async def test_a_quiet_job_gets_keep_alives_and_the_stream_ends_after_its_time():
    # (a follower waits at least 0.1 s per round, so it can never spin)
    jobs = JobService(fake_unit_of_work, lambda _id: asyncio.sleep(0), follow_check_s=0.1, follow_max_s=0.35)
    job = await jobs.submit("research", {"topic": "creatine"}, user_id=1)
    rest = [i async for i in await jobs.follow(job.id, user_id=1)]
    assert rest[0].status == "queued" and len(rest) > 2 and all(i is None for i in rest[1:])


async def test_following_someone_elses_job_fails_before_any_stream():
    jobs = service()
    job = await jobs.submit("research", {"topic": "creatine"}, user_id=1)
    with pytest.raises(NotFoundError):
        await jobs.follow(job.id, user_id=2)


async def test_a_redis_failure_never_fails_the_job():
    from redis.exceptions import ConnectionError as RedisDown

    from app.integrations.job_events import JobEvents

    class DeadRedis:
        async def publish(self, *_):
            raise RedisDown("redis is down")

    jobs = JobService(fake_unit_of_work, lambda _id: asyncio.sleep(0), events=JobEvents(DeadRedis()))
    job = await jobs.submit("research", {"topic": "creatine"}, user_id=1)
    await jobs.run(job.id, FakeResearch())
    assert ROWS[job.id].status == "done"


async def test_the_events_endpoint_streams_the_job_until_it_ends():
    import httpx

    from app.dependencies import get_job_service, require_user_id
    from app.main import app

    jobs = JobService(fake_unit_of_work, lambda _id: asyncio.sleep(0), follow_check_s=0.01)
    job = await jobs.submit("research", {"topic": "creatine"}, user_id=1)
    ROWS[job.id].status = "done"
    app.dependency_overrides.update({get_job_service: lambda: jobs, require_user_id: lambda: 1})
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get(f"/api/jobs/{job.id}/events")
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            assert r.text.startswith("retry: 1000") and r.text.count("event: job") == 1 and '"status": "done"' in r.text
            assert (await client.get("/api/jobs/999/events")).status_code == 404
    finally:
        app.dependency_overrides.clear()
