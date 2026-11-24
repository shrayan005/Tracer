"""
Milestone 5 storage + pipeline (step 5.5) against the real Postgres (rolled back), with fake LLM services.
Checks: analyses stored per paper and REUSED, re-analysis rules, quality applied to the BASE trust only
(idempotent), connections/gaps stored, and the report loading all of it (old reports still load).
"""

from contextlib import asynccontextmanager

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.domain.schemas import (AnalyzedPaper, Connection, ConnectionMember, EvidenceGaps, PaperRecord,
                                ResearchPlan, TrustAssessment)
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.models import Paper
from app.repositories.paper_repository import PaperRepository
from app.repositories.topic_repository import TopicRepository
from app.repositories.user_repository import UserRepository
from app.services import synthesis_service
from app.services.fulltext_context import PaperText
from app.services.research_service import ResearchService
from app.services.synthesis_service import SynthesisInput, SynthesisService, Stored, needs_analysis

pytestmark = pytest.mark.integration

PLAN = ResearchPlan(field="sports science", is_biomedical=True, is_cs_math_physics=False,
                    research_brief="How the triceps extends the elbow.", queries=["triceps"])
FACTS = ("control_group", "randomized", "blinded", "statistics_reported", "effect_sizes",
         "conflicts_declared", "data_available")


def analyzed(n: int, based_on: str = "full text") -> AnalyzedPaper:
    appraisal = {"sample_size": n, "sex": "mixed", "duration": "single session",
                 **{f: {"value": "not reported"} for f in FACTS}}
    return AnalyzedPaper(based_on=based_on, analysis={"appraisal": appraisal, "claims": [
        {"factor": "training (before vs after)", "outcome": "strength", "direction": "increase",
         "population": "adults", "quote": "q"}]})


class FakeAnalyzer:
    def __init__(self, sizes: dict[int, int]):
        self.sizes, self.asked = sizes, []

    async def analyze(self, papers, texts=None):
        self.asked.append(sorted(papers))
        return {pid: analyzed(self.sizes[pid], "full text" if pid in (texts or {}) else "abstract")
                for pid in papers}, []


class FakeConnector:
    async def connect(self, brief, papers):
        keys = [key for key, *_ in papers]
        members = [ConnectionMember(paper=k, claim=0, stance="supports",
                                    claim_snapshot=papers[i][3].analysis.claims[0]) for i, k in enumerate(keys)]
        return [Connection(question="Does training increase strength?", relation="agree", finding="yes",
                           members=members)], ["a connection flag"]


class FakeGaps:
    async def find(self, brief, papers):
        return EvidenceGaps(n_papers=len(papers), gaps=["No study reports blinding."])


@pytest.fixture
async def session():
    engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        db = AsyncSession(bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False)
        yield db
        await db.close()
        await transaction.rollback()
    await engine.dispose()


def uow_for(session):
    @asynccontextmanager
    async def unit_of_work():
        yield session
        await session.flush()
    return unit_of_work


BASE = TrustAssessment(level=3, label="moderate", reasons=["lab experiment"])


async def make_topic(session, sizes=(25, 7)):
    """A researched topic with selected papers whose base trust is moderate (3)."""
    user = await UserRepository(session).get_or_create("synth@example.com")
    topics = TopicRepository(session)
    topic = await topics.create("triceps", user_id=user.id, field="sports science")
    papers = []
    for rank, _ in enumerate(sizes, start=1):
        paper = await PaperRepository(session).add(Paper(title=f"Paper {rank}", year=2020,
                                                         authors=[f"Author{rank} A"], abstract="An abstract."))
        await topics.link_paper(topic.id, paper.id, [])
        await topics.save_screening(topic.id, paper.id, relevance=8, reason="r", study_type="lab_experiment")
        await topics.save_selection(topic.id, paper.id, rank=rank, score=0.7, score_parts={}, trust=BASE.model_dump(),
                                    summary=None)
        papers.append(paper)
    await topics.save_run(topic.id, plan=PLAN.model_dump(), run_stats={
        "field_family": "sport_biomechanics", "sources_used": [], "counts": {}, "timings_s": {}, "warnings": []})
    return topic, papers


def inputs(papers, texts=()):
    return [SynthesisInput(paper_id=p.id, label=f"(Author{i}, 2020)", study_type="lab_experiment",
                           paper=PaperRecord(source="db", title=p.title, abstract="An abstract."), base_trust=BASE,
                           text=PaperText(context="c", source="s") if p.id in texts else None)
            for i, p in enumerate(papers, start=1)]


def make_service(session, analyzer):
    return SynthesisService(analyzer, FakeConnector(), FakeGaps(), uow_for(session))


# ---------- the re-analysis rule ----------

def test_needs_analysis_rules():
    paper = SynthesisInput(paper_id=1, label="x", study_type="cohort", paper=PaperRecord(source="x", title="t"),
                           base_trust=BASE)
    with_text = SynthesisInput(**{**paper.__dict__, "text": PaperText(context="c", source="s")})
    current = synthesis_service.ANALYSIS_VERSION
    assert needs_analysis(None, paper)                                              # never analysed
    assert not needs_analysis(Stored(current, "full text", analyzed(10)), paper)    # still valid
    assert needs_analysis(Stored("old", "full text", analyzed(10)), paper)          # prompt changed
    assert needs_analysis(Stored(current, "abstract", analyzed(10)), with_text)     # full text appeared
    assert not needs_analysis(Stored(current, "full text", analyzed(10)), paper)    # full text vanished: keep


# ---------- run: store, reuse, quality, synthesis ----------

async def test_run_stores_everything_and_the_report_loads_it(session):
    topic, papers = await make_topic(session)
    analyzer = FakeAnalyzer({papers[0].id: 25, papers[1].id: 7})
    flags = await make_service(session, analyzer).run(topic.id, "brief", "sport_biomechanics",
                                                       inputs(papers, texts={papers[1].id}))
    assert flags == ["a connection flag"]
    stored = await AnalysisRepository(session).get_many([p.id for p in papers])
    assert stored[papers[0].id].based_on == "abstract" and stored[papers[1].id].based_on == "full text"

    report = await ResearchService(None, None, None, uow_for(session)).load(topic.id)
    first, second = report.papers
    assert first.analysis.analysis.appraisal.sample_size == 25
    assert first.trust.level == 3 and first.trust.reasons[-1] == "checklist not judged (abstract only)"
    assert second.trust.level == 2 and second.base_trust.level == 3        # full text, weak + n=7: 3 -> 2
    assert "very small sample (n=7): trust -1" in second.trust.reasons[-1]  # (both reasons, still one step)
    assert report.connections[0].members[0].claim_snapshot.outcome == "strength"
    assert report.gaps.gaps == ["No study reports blinding."]
    assert report.synthesis_flags == ["a connection flag"]


async def test_second_run_reuses_analyses_and_does_not_lower_trust_twice(session):
    topic, papers = await make_topic(session)
    analyzer = FakeAnalyzer({papers[0].id: 25, papers[1].id: 7})
    service = make_service(session, analyzer)
    await service.run(topic.id, "brief", "sport_biomechanics", inputs(papers))
    await service.run(topic.id, "brief", "sport_biomechanics", inputs(papers))
    assert analyzer.asked[1] == []                                          # 2nd run: everything reused
    report = await ResearchService(None, None, None, uow_for(session)).load(topic.id)
    assert report.papers[1].trust.level == 2                                # still 2, not 1

    await service.run(topic.id, "brief", "sport_biomechanics", inputs(papers), refresh=True)
    assert analyzer.asked[2] == sorted(p.id for p in papers)               # refresh: all again


async def test_full_text_appearing_triggers_reanalysis_of_that_paper_only(session):
    topic, papers = await make_topic(session)
    analyzer = FakeAnalyzer({papers[0].id: 25, papers[1].id: 7})
    service = make_service(session, analyzer)
    await service.run(topic.id, "brief", "sport_biomechanics", inputs(papers))                     # abstracts
    await service.run(topic.id, "brief", "sport_biomechanics", inputs(papers, texts={papers[1].id}))
    assert analyzer.asked[1] == [papers[1].id]


async def test_report_without_milestone_5_still_loads(session):
    topic, _ = await make_topic(session)
    report = await ResearchService(None, None, None, uow_for(session)).load(topic.id)
    assert report.connections == [] and report.gaps is None and report.synthesis_flags == []
    assert all(p.analysis is None and p.trust == p.base_trust for p in report.papers)


async def test_analyze_topic_runs_synthesis_for_an_existing_topic(session):
    topic, papers = await make_topic(session)
    analyzer = FakeAnalyzer({papers[0].id: 25, papers[1].id: 7})
    research = ResearchService(None, None, None, uow_for(session), synthesis=make_service(session, analyzer))
    report = await research.analyze_topic(topic.id, topic.user_id)
    assert report.gaps is not None and len(report.connections) == 1
    assert analyzer.asked == [sorted(p.id for p in papers)]
