"""
Frozen eval cases: a topic's selected papers saved as a file, so eval runs never depend on today's
search results, a source outage or a changed database. Same input -> any change in scores comes from
the code, prompt or model being tested.

Cases contain the papers' extracted full text, so they live in evals/cases/ (git-ignored, like
data/files): fine on your machine, not for publishing. Gold labels (evals/gold/) are small and in git.
"""

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import ROOT_DIR
from app.domain.schemas import FieldFamily, PaperRecord, ResearchPlan, TrustAssessment
from app.repositories.document_repository import DocumentRepository
from app.repositories.paper_repository import to_record
from app.repositories.topic_repository import TopicRepository
from app.services.fulltext_context import PaperText, paper_text
from app.services.research_service import ResearchService

CASES_DIR = ROOT_DIR / "evals" / "cases"


class CasePaper(BaseModel):
    key: int                               # the paper's database id when frozen (stable label for gold files)
    label: str                             # in-text citation
    paper: PaperRecord
    study_type: str
    base_trust: TrustAssessment
    sections: list[tuple[str, str | None, str]] = Field(default_factory=list)   # empty: abstract only

    def text(self) -> PaperText | None:
        return paper_text(self.sections) if self.sections else None


class ScreenedCandidate(BaseModel):
    """A paper the LLM screened when the topic was researched (step 6.5: the screening eval re-screens these)."""
    key: int                               # database id when frozen
    paper: PaperRecord
    relevance: int                         # the AI's score at the time
    reason: str = ""
    study_type: str = "other"


class EvalCase(BaseModel):
    name: str                              # e.g. "triceps"
    topic_id: int
    topic: str
    brief: str
    field_family: FieldFamily
    frozen_at: datetime
    papers: list[CasePaper]
    candidates: list[ScreenedCandidate] = Field(default_factory=list)   # empty in cases frozen before 6.5
    plan: ResearchPlan | None = None       # the topic's search plan: search evals reuse it (no planner noise)


def case_path(name: str) -> Path:
    return CASES_DIR / f"{name}.json"


def save_case(case: EvalCase) -> Path:
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    path = case_path(case.name)
    path.write_text(case.model_dump_json(indent=1), encoding="utf-8")
    return path


def load_case(name: str) -> EvalCase:
    path = case_path(name)
    if not path.exists():
        raise FileNotFoundError(f"no eval case '{name}' (freeze one with: python -m app.evals freeze <topic> {name})")
    return EvalCase.model_validate_json(path.read_text(encoding="utf-8"))


def list_cases() -> list[str]:
    return sorted(p.stem for p in CASES_DIR.glob("*.json")) if CASES_DIR.exists() else []


async def freeze(research: ResearchService, topic_id: int, name: str, user_id: int | None = None,
                 force: bool = False) -> EvalCase:
    """Copy a researched topic's selected papers (with their extracted sections) and every screened
    candidate into a case file. Gold files refer to cases, so replacing one needs force=True."""
    if case_path(name).exists() and not force:
        raise FileExistsError(f"eval case '{name}' already exists (gold labels may refer to it; use --force)")
    report = await research.load(topic_id, user_id)
    papers = []
    async with research.unit_of_work() as session:
        screened = [ScreenedCandidate(key=paper.id, paper=to_record(paper), relevance=link.relevance,
                                      reason=link.relevance_reason or "", study_type=link.study_type or "other")
                    for link, paper in await TopicRepository(session).screened(topic_id)]
        documents = DocumentRepository(session)
        for p in report.papers:
            usable = [d for d in await documents.for_paper(p.paper_id) if d.status == "extracted"]
            best = min(usable, key=lambda d: d.content_type != "application/xml") if usable else None
            sections = [(s.kind, s.title, s.text) for s in await documents.sections(best.id)] if best else []
            papers.append(CasePaper(key=p.paper_id, label=p.in_text, paper=p.paper, study_type=p.study_type,
                                    base_trust=p.base_trust or p.trust, sections=sections))
    case = EvalCase(name=name, topic_id=report.topic_id, topic=report.topic, brief=report.plan.research_brief,
                    field_family=report.field_family, frozen_at=datetime.now(timezone.utc), papers=papers,
                    candidates=screened, plan=report.plan)
    save_case(case)
    return case
