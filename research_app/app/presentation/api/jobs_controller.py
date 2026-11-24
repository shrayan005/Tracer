"""Jobs (step 7.2): start a research, follow any job, retry a failed one. Long work runs in the worker (7.1)."""

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.dependencies import require_user_id, get_job_service
from app.domain.schemas import JobInfo, UsageInfo
from app.presentation.api.sse import keep_alive, retry_after, sse, stream
from app.services.job_service import JobService

router = APIRouter(prefix="/api", tags=["jobs"])


class ResearchRequest(BaseModel):
    """Length limits stop empty topics and pasted essays (prompt-injection hygiene)."""
    topic: str = Field(min_length=3, max_length=300, examples=["creatine supplementation in older adults"])


@router.post("/research", response_model=JobInfo, status_code=status.HTTP_202_ACCEPTED)
async def start_research(request: ResearchRequest, jobs: JobService = Depends(get_job_service),
                         user_id: int = Depends(require_user_id)) -> JobInfo:
    """Queue a research job and return it at once (202). Follow it with GET /api/jobs/{id}.
    409 if you already have a research running; 429 if your daily LLM budget is used up."""
    return await jobs.submit("research", {"topic": request.topic.strip()}, user_id)


@router.get("/jobs", response_model=list[JobInfo])
async def list_jobs(limit: int = Query(20, ge=1, le=100), jobs: JobService = Depends(get_job_service),
                    user_id: int = Depends(require_user_id)) -> list[JobInfo]:
    return await jobs.list(user_id, limit)


@router.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job(job_id: int, jobs: JobService = Depends(get_job_service),
                  user_id: int = Depends(require_user_id)) -> JobInfo:
    """Status, progress messages, the topic when done, the error when failed, and its LLM use."""
    return await jobs.get(job_id, user_id)


@router.get("/jobs/{job_id}/events", response_class=StreamingResponse,
            responses={200: {"content": {"text/event-stream": {}},
                             "description": "event `job` (a JobInfo) now and after every change, until done or "
                                            "failed; keep-alive comments in between"}})
async def job_events(job_id: int, jobs: JobService = Depends(get_job_service),
                     user_id: int = Depends(require_user_id)) -> StreamingResponse:
    """Follow a job live (Server-Sent Events), instead of asking GET /api/jobs/{id} again and again.
    A GET with the login cookie, so the browser's own EventSource can read it. The stream ends when the job
    does, or after a minute (the browser then reconnects by itself): a short stream never blocks a server
    reload or sits forever in a proxy."""
    updates = await jobs.follow(job_id, user_id)                # 404 before the stream starts

    async def events() -> AsyncIterator[str]:
        yield retry_after(1000)
        async for info in updates:
            yield keep_alive() if info is None else sse("job", info)

    return stream(events())


@router.post("/jobs/{job_id}/retry", response_model=JobInfo, status_code=status.HTTP_202_ACCEPTED)
async def retry_job(job_id: int, jobs: JobService = Depends(get_job_service),
                    user_id: int = Depends(require_user_id)) -> JobInfo:
    """Start a failed job again, as a NEW job (409 if it did not fail)."""
    return await jobs.retry(job_id, user_id)


@router.get("/usage", response_model=UsageInfo)
async def usage(jobs: JobService = Depends(get_job_service), user_id: int = Depends(require_user_id)) -> UsageInfo:
    """Your LLM use today (UTC) against your daily budget."""
    return await jobs.usage(user_id)
