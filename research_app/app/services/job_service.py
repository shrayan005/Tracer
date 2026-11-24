"""
Background jobs (step 7.1): long runs leave the request and go to a worker.

    submit   save a 'queued' row FIRST, then put its id on the queue (a queued id always has a row)
    run      (in the worker) queued -> running -> done | failed; every progress message is saved as it
             happens, and every LLM call is tagged with the job id (cost per job)
    retry    a failed job is never retried automatically (research creates a topic: a blind retry would
             make a duplicate and spend tokens unseen); the user starts a NEW job that points at the old one

Limits per user (step 7.4):
    budget   a daily LLM token budget (UTC), checked at submit: a job that would START over budget is refused
             (429). A running job is never cut off halfway, so one job can go past the budget once.
    one      one active research per user: checked here for a friendly message, and GUARANTEED by a partial
             unique index in the database (two requests in the same instant can't both get through)
    tagged   every LLM call of a job carries its user (user_scope) -> llm_calls.user_id -> the budget

The queue itself (arq on Redis) is injected as `enqueue`, so tests need no Redis.

Following a job live (SSE step 6): the worker publishes "job changed" after each change (job_events.py);
follow() re-reads the job on each signal and hands every new state to the browser's open stream.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timedelta, timezone

from sqlalchemy.exc import IntegrityError

from app.common.logging import get_logger
from app.common.progress import Progress
from app.domain.errors import ConflictError, NotFoundError, QuotaExceededError
from app.domain.schemas import JobDetails, JobInfo, JobStepUsage, LlmTrace, ProviderUsage, UsageInfo
from app.integrations.job_events import JobEvents, NoJobEvents
from app.integrations.llm.run_context import run_scope, user_scope
from app.repositories.job_repository import JobRepository
from app.repositories.llm_call_repository import LlmCallRepository
from app.repositories.models import Job
from app.services.discovery_service import UnitOfWork

log = get_logger(__name__)
Enqueue = Callable[..., Awaitable[None]]            # put a job id on the queue (optionally: defer_s=seconds)
MAX_PREPARE_TRIES = 6        # preparing for questions continues by itself at most this many times
PREPARE_MARGIN_S = 60        # stop preparing this long before the job's time limit (save + queue the rest)
KINDS = ("research", "analyze", "index")
ACTIVE = ("queued", "running")


class JobService:
    def __init__(self, unit_of_work: UnitOfWork, enqueue: Enqueue, daily_token_budget: int = 0,
                 events: JobEvents | NoJobEvents | None = None, follow_check_s: float = 15.0,
                 follow_max_s: float = 60.0, job_timeout_s: float = 15 * 60):
        self.unit_of_work = unit_of_work
        self.job_timeout_s = job_timeout_s                  # the worker's limit per job (preparing stops before it)
        self.enqueue = enqueue
        self.daily_token_budget = daily_token_budget        # 0 = no limit
        self.events = events or NoJobEvents()               # "job changed" signals (SSE step 6)
        self.follow_check_s = follow_check_s                # re-read this often even without a signal
        self.follow_max_s = follow_max_s                    # then end the stream: the browser reconnects

    # ---------- what users do ----------

    async def submit(self, kind: str, params: dict, user_id: int, retry_of: int | None = None,
                     defer_s: float = 0) -> JobInfo:
        if kind not in KINDS:
            raise ValueError(f"unknown job kind {kind!r} (known: {', '.join(KINDS)})")
        await self.check_budget(user_id)
        busy = ConflictError("you already have a research running: wait for it to finish")
        try:
            async with self.unit_of_work() as session:      # committed BEFORE it is queued
                if kind == "research" and await JobRepository(session).active(user_id, "research"):
                    raise busy                               # the friendly check (a double click)
                job = await JobRepository(session).create(user_id=user_id, kind=kind, params=params,
                                                          retry_of=retry_of)
                job_id = job.id
        except IntegrityError as e:                          # the database's guarantee (two requests at once)
            if "uq_jobs_one_active_research" in str(e):
                raise busy from None
            raise
        try:
            await (self.enqueue(job_id, defer_s=defer_s) if defer_s > 0 else self.enqueue(job_id))
        except Exception as e:                               # Redis down: say so instead of 'queued' forever
            async with self.unit_of_work() as session:
                await JobRepository(session).fail(job_id, f"could not queue the job: {e}")
            raise
        return await self.get(job_id, user_id)

    async def get(self, job_id: int, user_id: int) -> JobInfo:
        """One of the user's jobs. Another user's job looks exactly like a missing one."""
        async with self.unit_of_work() as session:
            job = await JobRepository(session).get(job_id)
            if job is None or job.user_id != user_id:
                raise NotFoundError(f"job {job_id} does not exist")
            tokens, cost = await LlmCallRepository(session).usage(job.id)
            return to_info(job, tokens, cost)

    async def details(self, job_id: int, user_id: int) -> JobDetails:
        """What happened inside one of the user's jobs: LLM use per step + the traced prompts/answers."""
        await self.get(job_id, user_id)                      # ownership: another user's job is 'missing'
        async with self.unit_of_work() as session:
            calls = LlmCallRepository(session)
            steps = [JobStepUsage(**s) for s in await calls.steps(job_id)]
            traces = [LlmTrace(id=c.id, created_at=c.created_at, purpose=c.purpose, provider=c.provider,
                               model=c.model, tokens=c.input_tokens + c.output_tokens + c.thinking_tokens,
                               latency_ms=c.latency_ms, prompt=c.prompt, response=c.response)
                      for c in await calls.traces(job_id)]
        return JobDetails(job_id=job_id, steps=steps, traces=traces)

    async def follow(self, job_id: int, user_id: int) -> AsyncIterator[JobInfo | None]:
        """SSE step 6: the job now, then again each time it changes, until it is done or failed.
        Checks the job is yours FIRST (404 before any stream starts), then returns the updates.
        None = nothing changed for a while (the caller sends a keep-alive)."""
        await self.get(job_id, user_id)
        return self._updates(job_id, user_id)

    async def _updates(self, job_id: int, user_id: int) -> AsyncIterator[JobInfo | None]:
        deadline = time.monotonic() + self.follow_max_s
        async with self.events.listen(job_id) as changed:
            info = await self.get(job_id, user_id)           # read AFTER subscribing: no change falls in between
            yield info
            while info.status in ACTIVE and time.monotonic() < deadline:
                signalled = await changed(max(0.1, min(self.follow_check_s, deadline - time.monotonic())))
                latest = await self.get(job_id, user_id)     # the table is the truth; the signal only says "look"
                if latest != info:
                    info = latest
                    yield info
                elif not signalled:
                    yield None

    async def list(self, user_id: int, limit: int = 20) -> list[JobInfo]:
        async with self.unit_of_work() as session:
            return [to_info(j) for j in await JobRepository(session).list_for_user(user_id, limit)]

    async def check_budget(self, user_id: int) -> None:
        """Raise QuotaExceededError (429) if the user's daily LLM budget is used up (jobs and questions)."""
        usage = await self.usage(user_id, with_providers=False)     # runs before every job and question: keep it light
        if usage.remaining is not None and usage.remaining <= 0:
            wait = int((usage.resets_at - datetime.now(timezone.utc)).total_seconds()) + 1
            raise QuotaExceededError(f"your daily LLM budget ({usage.daily_budget:,} tokens) is used up: it "
                                     f"resets at {usage.resets_at:%H:%M} UTC", retry_after=wait)

    async def usage(self, user_id: int, with_providers: bool = True) -> UsageInfo:
        """The user's LLM use today (UTC) against the daily budget, and (with_providers) which provider did the
        work and which one is failing: free-tier trouble shows up here before it breaks a job."""
        now = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        async with self.unit_of_work() as session:
            calls = LlmCallRepository(session)
            tokens, cost = await calls.user_usage_since(user_id, midnight)
            providers = await calls.user_providers_since(user_id, midnight) if with_providers else []
        budget = self.daily_token_budget
        return UsageInfo(tokens_today=tokens, cost_today_usd=cost, daily_budget=budget,
                         remaining=max(0, budget - tokens) if budget else None,
                         resets_at=midnight + timedelta(days=1),
                         providers=[ProviderUsage(**p) for p in providers])

    async def retry(self, job_id: int, user_id: int) -> JobInfo:
        """A NEW job with the same request, pointing at the failed one (option A of step 7.1)."""
        old = await self.get(job_id, user_id)
        if old.status != "failed":
            raise ConflictError(f"job {job_id} is {old.status}: only a failed job can be retried")
        return await self.submit(old.kind, old.params, user_id, retry_of=old.id)

    # ---------- what the worker does ----------

    async def run(self, job_id: int, research) -> None:
        """Run one queued job to the end. Never raises for a failed run: the failure is saved on the job."""
        async with self.unit_of_work() as session:
            job = await JobRepository(session).mark_running(job_id)
            if job is None:
                log.warning("job %s: no such row, skipped", job_id)
                return
            kind, params, user_id = job.kind, dict(job.params), job.user_id
        await self.events.publish(job_id)                    # queued -> running
        say = self._progress(job_id)
        started = time.monotonic()
        try:
            with run_scope(job_id), user_scope(user_id):     # every LLM call: its job (cost) and user (budget)
                if kind == "research":
                    topic_id = (await research.research(params["topic"], user_id, progress=say, index=False,
                                                        papers=params.get("papers"))).topic_id
                elif kind == "index":
                    if getattr(research, "indexer", None) is None:
                        raise RuntimeError("questions are switched off (INDEX_ENABLED=false)")
                    topic_id = params["topic_id"]
                else:
                    topic_id = (await research.analyze_topic(params["topic_id"], user_id, index=False,
                                                             refresh=bool(params.get("refresh")),
                                                             progress=say)).topic_id
                if getattr(research, "indexer", None) is not None:   # the results are saved: now prepare
                    left = self.job_timeout_s - (time.monotonic() - started) - PREPARE_MARGIN_S
                    await self._prepare(research, topic_id, user_id, say, int(params.get("attempt", 0)), left)
            async with self.unit_of_work() as session:
                await JobRepository(session).finish(job_id, topic_id)
            await self.events.publish(job_id)                # done
        except asyncio.CancelledError:                       # the worker's time limit, or it is shutting down
            await self._fail(job_id, "stopped: the job took too long or the worker was shut down")
            raise
        except Exception as e:                               # noqa: BLE001 (any failure ends up on the job)
            log.exception("job %s failed", job_id)
            await self._fail(job_id, f"{type(e).__name__}: {e}")

    async def _prepare(self, research, topic_id: int, user_id: int, say: Progress, attempt: int,
                       time_left_s: float) -> None:
        """Prepare the papers for questions INSIDE the run, after its results are saved (measured 24 Nov
        2026: embedding ~900 passages takes ~10 min on the free tier; when that ran out the job's time, the
        unsaved report was lost). If it can't finish (Gemini's limit, or the run's time is up), the run still
        ends 'done' and preparing continues BY ITSELF in a follow-up job, after the wait Google asked for.
        Only after MAX_PREPARE_TRIES does it give up (an index job then fails; the Ask tab offers a retry)."""
        outcome = await research.prepare(topic_id, user_id, progress=say, time_budget_s=max(30.0, time_left_s))
        if outcome.done:
            await say("ready for questions")
            return
        if attempt + 1 >= MAX_PREPARE_TRIES:
            message = f"not ready for questions after {MAX_PREPARE_TRIES} tries: {outcome.reason}"
            if attempt > 0:                                  # a follow-up index job: this is its failure
                raise RuntimeError(message)
            await say(message)
            return
        wait = (outcome.retry_after + 5) if outcome.retry_after else 0
        at = datetime.now(timezone.utc) + timedelta(seconds=wait)
        try:
            job = await self.submit("index", {"topic_id": topic_id, "attempt": attempt + 1,
                                              "not_before": at.isoformat()}, user_id, defer_s=wait)
        except Exception as e:                               # noqa: BLE001 (never fail a saved research)
            log.warning("could not queue the follow-up index job for topic %s: %s", topic_id, e)
            await say(f"not ready for questions yet ({outcome.reason}); open the Ask tab to try again")
            return
        why = ("Gemini's embedding limit is reached" if outcome.retry_after else outcome.reason)
        when = f"at {at:%H:%M} UTC" if wait else "now"
        await say(f"{why}: preparing for questions continues by itself {when} (job #{job.id})")

    def _progress(self, job_id: int) -> Progress:
        async def say(message: str) -> None:                 # own short transaction: visible at once
            async with self.unit_of_work() as session:
                await JobRepository(session).add_progress(job_id, message)
            await self.events.publish(job_id)                # after the commit: a follower reads the new row
        return say

    async def _fail(self, job_id: int, error: str) -> None:
        async with self.unit_of_work() as session:
            await JobRepository(session).fail(job_id, error)
        await self.events.publish(job_id)


def to_info(job: Job, tokens: int = 0, cost: float = 0.0) -> JobInfo:
    return JobInfo(id=job.id, kind=job.kind, params=job.params, status=job.status, progress=job.progress,
                   topic_id=job.topic_id, error=job.error, retry_of=job.retry_of, created_at=job.created_at,
                   started_at=job.started_at, finished_at=job.finished_at, tokens=tokens, cost_usd=cost)
