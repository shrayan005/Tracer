"""Saving research results and rebuilding the report, against the real Postgres (rolled back)."""

from contextlib import asynccontextmanager

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.domain.errors import NotFoundError
from app.domain.schemas import PaperSummary, ResearchPlan, SummarizedPaper, TrustAssessment
from app.repositories.models import Paper
from app.repositories.paper_repository import PaperRepository
from app.repositories.topic_repository import TopicRepository
from app.repositories.user_repository import UserRepository
from app.services.research_service import ResearchService

pytestmark = pytest.mark.integration

PLAN = ResearchPlan(field="sports science", is_biomedical=True, is_cs_math_physics=False,
                    research_brief="How the triceps extends the elbow.", queries=["triceps EMG"])


@pytest.fixture
async def session():
    engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        db = AsyncSession(bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False)
        yield db
        await db.close()
        await transaction.rollback()          # undo everything the test did
    await engine.dispose()


def service_on(session) -> ResearchService:
    """A ResearchService whose unit of work is the test's session (so the rollback still applies)."""
    @asynccontextmanager
    async def unit_of_work():
        yield session
        await session.flush()
    return ResearchService(None, None, None, unit_of_work)


async def make_topic_with_papers(session):
    user = await UserRepository(session).get_or_create("test@example.com")
    topics, papers = TopicRepository(session), PaperRepository(session)
    topic = await topics.create("biomechanics of triceps", user_id=user.id, field="sports science")
    a = await papers.add(Paper(title="Triceps torque", authors=["John Smith"], year=2020, doi="10.991/A",
                               source_urls={"openalex": "https://openalex.org/W1", "pdf": "https://x.org/a.pdf"}))
    b = await papers.add(Paper(title="Triceps EMG", authors=["Kim Lee", "Ann Park"], year=2022))
    c = await papers.add(Paper(title="Shoulder swimming", year=2019))
    for p in (a, b, c):
        await topics.link_paper(topic.id, p.id, [{"source": "openalex", "query": "q"}])
    return topic, a, b, c


async def test_save_and_rebuild_report(session):
    topic, a, b, c = await make_topic_with_papers(session)
    topics = TopicRepository(session)
    for paper, relevance in ((a, 9), (b, 8), (c, 1)):
        await topics.save_screening(topic.id, paper.id, relevance=relevance, reason="r", study_type="cohort")
    trust = TrustAssessment(level=3, label="moderate", reasons=["cohort"]).model_dump()
    summary = SummarizedPaper(based_on="abstract", summary=PaperSummary(
        tldr="t", objective="o", methods="m", key_findings=["f"], why_relevant="w")).model_dump(mode="json")
    # saved in rank order b, a (rank = MMR pick order, not relevance order)
    await topics.save_selection(topic.id, b.id, rank=1, score=0.8, score_parts={"relevance": 0.4},
                                trust=trust, summary=summary)
    await topics.save_selection(topic.id, a.id, rank=2, score=0.7, score_parts={"relevance": 0.45},
                                trust=trust, summary=None)
    await topics.save_run(topic.id, plan=PLAN.model_dump(), run_stats={
        "field_family": "biomedical", "sources_used": ["openalex"],
        "counts": {"found": 3, "prefiltered": 3, "screened": 3, "selected": 2},
        "timings_s": {"discovery": 1.0}, "warnings": ["w1"]})

    report = await service_on(session).load(topic.id)

    assert [p.paper_id for p in report.papers] == [b.id, a.id]        # ordered by rank; c not selected
    first, second = report.papers
    assert first.in_text == "(Lee & Park, 2022)" and first.summary.summary.tldr == "t"
    assert second.summary is None
    assert second.paper.pdf_url == "https://x.org/a.pdf" and second.paper.landing_url == "https://openalex.org/W1"
    assert second.apa.endswith("https://doi.org/10.991/a")
    assert report.plan.research_brief == PLAN.research_brief and report.warnings == ["w1"]
    assert first.trust.label == "moderate" and first.relevance == 8


async def test_load_unknown_or_unfinished_topic(session):
    topic, *_ = await make_topic_with_papers(session)
    service = service_on(session)
    with pytest.raises(NotFoundError, match="no research results"):
        await service.load(topic.id)                 # searched, but research never ran
    with pytest.raises(NotFoundError, match="does not exist"):
        await service.load(999_999_999)


async def test_another_users_topic_looks_like_it_does_not_exist(session):
    topic, *_ = await make_topic_with_papers(session)
    await TopicRepository(session).save_run(topic.id, plan=PLAN.model_dump(), run_stats={
        "field_family": "biomedical", "sources_used": [], "counts": {}, "timings_s": {}, "warnings": []})
    service = service_on(session)
    owner_report = await service.load(topic.id, topic.user_id)          # the owner can load it
    assert owner_report.topic_id == topic.id
    stranger = await UserRepository(session).get_or_create("stranger@example.com")
    with pytest.raises(NotFoundError, match="does not exist"):          # same message as a missing topic
        await service.load(topic.id, stranger.id)
