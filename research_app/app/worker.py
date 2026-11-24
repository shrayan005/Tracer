"""
The background worker (step 7.1): runs research / analyze jobs from the Redis queue (arq).

Run it (in WSL, from research_app/), next to the web server:
    arq app.worker.WorkerSettings

Each job's state lives in the jobs table (see job_service.py); Redis only carries "please run job 42".
"""

from datetime import timedelta

from arq.connections import RedisSettings

from app.common.logging import get_logger, setup_logging
from app.config import get_settings
from app.dependencies import close_resources, get_job_service, get_research_service
from app.repositories.job_repository import JobRepository
from app.repositories.session import session_scope

log = get_logger(__name__)
settings = get_settings()


async def run_job(ctx: dict, job_id: int) -> None:
    await get_job_service().run(job_id, get_research_service())


async def startup(ctx: dict) -> None:
    setup_logging(settings.log_level)
    # a job still 'running' well past the time limit belongs to a worker that was stopped mid-run
    async with session_scope() as session:
        stuck = await JobRepository(session).fail_stuck(timedelta(minutes=settings.job_timeout_minutes + 1))
    if stuck:
        log.warning("marked %d stuck job(s) as failed", stuck)
    log.info("worker ready: up to %d jobs at a time", settings.worker_max_jobs)


async def shutdown(ctx: dict) -> None:
    await close_resources()


class WorkerSettings:
    functions = [run_job]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = settings.worker_max_jobs          # free-tier LLM limits: a few topics at a time is plenty
    job_timeout = settings.job_timeout_minutes * 60
    max_tries = 1                                # never retried automatically (see job_service.py)
    keep_result = 3600                           # arq's own result record; OUR state is in the jobs table
