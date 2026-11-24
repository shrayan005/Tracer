"""Data access for the jobs table (step 7.1)."""

from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.models import Job


def now() -> datetime:
    return datetime.now(timezone.utc)


class JobRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, *, user_id: int, kind: str, params: dict, retry_of: int | None = None) -> Job:
        job = Job(user_id=user_id, kind=kind, params=params, status="queued", progress=[], retry_of=retry_of)
        self.session.add(job)
        await self.session.flush()
        return job

    async def get(self, job_id: int) -> Job | None:
        return await self.session.get(Job, job_id)

    async def list_for_user(self, user_id: int, limit: int = 20) -> list[Job]:
        query = select(Job).where(Job.user_id == user_id).order_by(Job.id.desc()).limit(limit)
        return list(await self.session.scalars(query))

    async def active(self, user_id: int, kind: str) -> list[Job]:
        """The user's jobs of this kind that are still queued or running."""
        query = select(Job).where(Job.user_id == user_id, Job.kind == kind, Job.status.in_(("queued", "running")))
        return list(await self.session.scalars(query))

    async def mark_running(self, job_id: int) -> Job | None:
        job = await self.get(job_id)
        if job is not None:
            job.status, job.started_at = "running", now()
        return job

    async def add_progress(self, job_id: int, message: str) -> None:
        job = await self.get(job_id)
        if job is not None:            # JSONB: assign a NEW list, or the change isn't saved
            job.progress = [*job.progress, {"at": now().isoformat(timespec="seconds"), "message": message}]

    async def finish(self, job_id: int, topic_id: int | None) -> None:
        job = await self.get(job_id)
        if job is not None:
            job.status, job.topic_id, job.finished_at = "done", topic_id, now()

    async def fail(self, job_id: int, error: str, topic_id: int | None = None) -> None:
        job = await self.get(job_id)
        if job is not None:
            job.status, job.error, job.finished_at = "failed", error[:2000], now()
            job.topic_id = topic_id or job.topic_id

    async def fail_stuck(self, older_than: timedelta) -> int:
        """Jobs still 'running' long after they should have ended (the worker was stopped): failed."""
        result = await self.session.execute(
            update(Job).where(Job.status == "running", Job.started_at < now() - older_than)
            .values(status="failed", error="the worker stopped during the run", finished_at=now()))
        return result.rowcount or 0
