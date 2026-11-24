"""Topics (step 7.2): the user's topics, their reports, one paper, and starting an analysis.
Reading never calls an LLM: reports are rebuilt from the database."""

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.dependencies import require_user_id, get_job_service, get_research_service
from app.domain.schemas import JobInfo, PaperDetail, ResearchReport, TopicSummary
from app.presentation.report_markdown import render, slugify
from app.services.job_service import JobService
from app.services.research_service import ResearchService

router = APIRouter(prefix="/api/topics", tags=["topics"])


class AnalyzeRequest(BaseModel):
    refresh: bool = False           # re-analyse papers even if their stored analysis is still valid


@router.get("", response_model=list[TopicSummary])
async def list_topics(limit: int = Query(50, ge=1, le=200), research: ResearchService = Depends(get_research_service),
                      user_id: int = Depends(require_user_id)) -> list[TopicSummary]:
    return await research.topics(user_id, limit)


@router.get("/{topic_id}", response_model=ResearchReport)
async def get_report(topic_id: int, research: ResearchService = Depends(get_research_service),
                     user_id: int = Depends(require_user_id)) -> ResearchReport:
    """The full report as JSON. 404 if it isn't yours or doesn't exist; 409 if it has no results yet."""
    return await research.load(topic_id, user_id)


@router.get("/{topic_id}/report.md", response_class=PlainTextResponse)
async def get_report_markdown(topic_id: int, research: ResearchService = Depends(get_research_service),
                              user_id: int = Depends(require_user_id)) -> PlainTextResponse:
    report = await research.load(topic_id, user_id)
    name = f"{report.topic_id}-{slugify(report.topic)}.md"
    return PlainTextResponse(render(report), media_type="text/markdown; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/{topic_id}/papers/{rank}", response_model=PaperDetail)
async def get_paper(topic_id: int, rank: int, research: ResearchService = Depends(get_research_service),
                    user_id: int = Depends(require_user_id)) -> PaperDetail:
    return await research.paper_detail(topic_id, rank, user_id)


@router.post("/{topic_id}/analyze", response_model=JobInfo, status_code=status.HTTP_202_ACCEPTED)
async def analyze_topic(topic_id: int, request: AnalyzeRequest | None = None,
                        research: ResearchService = Depends(get_research_service),
                        jobs: JobService = Depends(get_job_service),
                        user_id: int = Depends(require_user_id)) -> JobInfo:
    """Queue claims + quality + connections + gaps for this topic. Checked BEFORE queueing: a topic that
    isn't yours (404) or has no results yet (409) never becomes a job that fails later."""
    await research.load(topic_id, user_id)
    refresh = request.refresh if request else False
    return await jobs.submit("analyze", {"topic_id": topic_id, "refresh": refresh}, user_id)
