"""
Milestone 5 for one topic (step 5.5): analyse -> quality -> connect -> gaps -> save.

    analyse   per paper: REUSE the stored analysis (paper_analyses) unless it was made with an older
              prompt version, or it came from the abstract and a full text exists now
    quality   adjusted trust = apply_quality(BASE trust, ...): always from the design-based trust,
              so running this twice can never lower a paper twice
    connect   claims of all papers -> checked connections (with claim snapshots)
    gaps      counted gaps + checked write-up
    save      paper_analyses, topic_papers.adjusted_trust, topic_syntheses
"""

from app.common.progress import Progress, quiet
from dataclasses import dataclass, field

from app.domain.schemas import AnalyzedPaper, FieldFamily, PaperRecord, TrustAssessment
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.topic_repository import TopicRepository
from app.services.analysis_service import ANALYSIS_VERSION, AnalysisService
from app.services.connection_service import ConnectionService
from app.services.fulltext_context import PaperText
from app.services.fulltext_fetcher import UnitOfWork
from app.services.gap_service import GapService, paper_facts
from app.services.quality import apply_quality



@dataclass
class SynthesisInput:
    """One selected paper, with everything milestone 5 needs."""
    paper_id: int
    label: str                       # in-text citation
    study_type: str
    paper: PaperRecord
    base_trust: TrustAssessment      # design-based trust from selection (never the adjusted one)
    text: PaperText | None = None    # full text, if we have it
    limitations: list[str] = field(default_factory=list)   # stated by the authors (summary)


@dataclass
class Stored:
    version: str
    based_on: str
    analysis: AnalyzedPaper


def needs_analysis(stored: Stored | None, paper: SynthesisInput) -> bool:
    """Re-analyse if: never analysed, made with another prompt version, or a full text appeared since."""
    return (stored is None or stored.version != ANALYSIS_VERSION
            or (stored.based_on != "full text" and paper.text is not None))


class SynthesisService:
    def __init__(self, analyzer: AnalysisService, connector: ConnectionService, gap_finder: GapService,
                 unit_of_work: UnitOfWork):
        self.analyzer = analyzer
        self.connector = connector
        self.gap_finder = gap_finder
        self.unit_of_work = unit_of_work

    async def run(self, topic_id: int, brief: str, field_name: FieldFamily, papers: list[SynthesisInput], *,
                  refresh: bool = False, progress: Progress | None = None) -> list[str]:
        """Analyse, connect and find gaps for a topic's selected papers; save everything.
        Returns warnings (also stored with the synthesis)."""
        say = progress or quiet
        ids = [p.paper_id for p in papers]
        async with self.unit_of_work() as session:
            rows = await AnalysisRepository(session).get_many(ids)
            stored = {pid: Stored(row.version, row.based_on, AnalyzedPaper.model_validate(row.analysis))
                      for pid, row in rows.items()}

        todo = [p for p in papers if refresh or needs_analysis(stored.get(p.paper_id), p)]
        await say(f"analysing {len(papers)} papers ({len(papers) - len(todo)} reused)...")
        fresh, warnings = await self.analyzer.analyze({p.paper_id: p.paper for p in todo},
                                                      {p.paper_id: p.text for p in todo if p.text is not None})
        analyses = {pid: s.analysis for pid, s in stored.items()} | fresh     # a failed re-analysis keeps the old

        async with self.unit_of_work() as session:
            for pid, analyzed in fresh.items():
                await AnalysisRepository(session).save(pid, analyzed, ANALYSIS_VERSION)
            topics = TopicRepository(session)
            for p in papers:
                if p.paper_id in analyses:
                    a = analyses[p.paper_id]
                    adjusted = apply_quality(p.base_trust, a.analysis.appraisal, field_name, a.based_on)
                    await topics.save_adjusted_trust(topic_id, p.paper_id, adjusted.model_dump())

        analysed = [p for p in papers if p.paper_id in analyses]
        await say("connecting the papers...")
        connections, connection_flags = await self.connector.connect(
            brief, [(p.paper_id, p.label, p.study_type, analyses[p.paper_id]) for p in analysed])
        await say("finding gaps in the evidence...")
        gaps = await self.gap_finder.find(
            brief, [paper_facts(p.label, p.study_type, analyses[p.paper_id], p.limitations) for p in analysed])

        flags = warnings + connection_flags
        async with self.unit_of_work() as session:
            await TopicRepository(session).save_synthesis(
                topic_id, connections=[c.model_dump(mode="json") for c in connections],
                gaps=gaps.model_dump(mode="json"), flags=flags)
        return flags
