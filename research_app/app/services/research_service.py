"""
Research pipeline (step 3.6): one call runs every step of milestone 3.

    discover (plan + search + save candidates)
      -> prefilter (cheap, code)          -> screen (LLM, batched)       -> shortlist (code: trust, score, MMR)
      -> full text for the SHORTLIST (find, download, extract; milestone 4, optional)
      -> final pick (code): papers with a readable full text first, abstract-only ones only fill gaps
      -> summarize (LLM, one per paper: full text or abstract)
      -> synthesis (milestone 5, optional: analyse, quality, connect, gaps)
      -> save results -> load the report from the DB

The report is always built by load() FROM THE DATABASE, also right after a run. So a fresh run and a
later `report <topic_id>` give exactly the same report, and rebuilding one costs no LLM calls.

Failures are collected as warnings instead of crashing: a failed source, a failed screening batch or
a failed summary still gives you a report, with the gap clearly stated.
"""

import asyncio
import time
from dataclasses import dataclass
from app.common.progress import Progress, quiet

from app.domain.errors import NotFoundError, NotReadyError
from app.domain.schemas import (AnalyzedPaper, Connection, DocumentInfo, EvidenceGaps, PaperDetail, PaperRecord,
                                PaperScreening, ReportPaper, ResearchPlan, ResearchReport, SearchAgentRound,
                                SectionInfo,
                                SummarizedPaper, TopicSummary, TrustAssessment)
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.paper_repository import to_record
from app.repositories.topic_repository import TopicRepository
from app.services.citations import apa, bibtex, in_text, unique_bibtex_keys
from app.services.discovery_service import DiscoveryService, UnitOfWork
from app.services.fulltext_context import PaperText
from app.services.fulltext_service import FullTextService
from app.services.index_service import IndexService
from app.services.prefilter import prefilter
from app.services.screening_service import ScreeningService
from app.services.selection import MIN_RELEVANCE, prefer_full_text, select_best
from app.services.study_type import reconcile
from app.services.summary_service import SummaryService
from app.services.synthesis_service import SynthesisInput, SynthesisService
from app.services.trust import field_family



class ResearchService:
    def __init__(self, discovery: DiscoveryService, screener: ScreeningService, summarizer: SummaryService,
                 unit_of_work: UnitOfWork, *, prefilter_keep: int = 40, screen_first: int = 20,
                 screen_step: int = 20, select_max: int = 10, select_min: int = 5, fulltext_shortlist: int = 12,
                 fulltext: FullTextService | None = None, synthesis: SynthesisService | None = None,
                 indexer: IndexService | None = None):
        self.discovery = discovery
        self.screener = screener
        self.summarizer = summarizer
        self.unit_of_work = unit_of_work
        self.fulltext = fulltext                 # None: summaries from abstracts only (as in milestone 3)
        self.synthesis = synthesis               # None: no claims, quality, connections or gaps (milestone 5)
        self.indexer = indexer                   # None: no passages for questions (step 7.6)
        self.prefilter_keep = prefilter_keep     # most papers that MAY be screened
        self.screen_first = screen_first         # papers screened in the first round
        self.screen_step = screen_step           # papers per extra round, if not enough were relevant
        self.select_max = select_max
        self.select_min = select_min
        self.fulltext_shortlist = fulltext_shortlist   # papers whose full text is tried before the final pick

    @property
    def shortlist_size(self) -> int:
        """With full text on: more papers than we keep, so one without a readable full text can be swapped for
        one with. Without full text there is nothing to swap: the shortlist IS the selection."""
        return max(self.fulltext_shortlist, self.select_max) if self.fulltext is not None else self.select_max

    def sizes(self, papers: int | None = None) -> "RunSizes":
        """How many papers a run keeps at each stage, for `papers` final papers (the user's choice, e.g. 5-20).
        The defaults (10 papers) stay exactly as before; other counts scale the same margins: a shortlist
        `fulltext_shortlist - select_max` bigger (swap papers without full text), and at least 3 candidates
        per final paper kept for screening."""
        if not papers or papers == self.select_max:
            return RunSizes(self.select_max, self.shortlist_size, self.prefilter_keep, self.select_min)
        extra = max(0, self.fulltext_shortlist - self.select_max) if self.fulltext is not None else 0
        return RunSizes(select_max=papers, shortlist=papers + extra,
                        prefilter_keep=max(self.prefilter_keep, papers * 3), select_min=min(self.select_min, papers))

    async def research(self, topic: str, user_id: int, progress: Progress | None = None, *,
                       index: bool = True, papers: int | None = None) -> ResearchReport:
        """index=False: don't prepare for questions here (background jobs queue a separate 'index' job, so
        slow embeddings can never time out a research and lose its saved report; the CLI keeps index=True)."""
        say = progress or quiet
        size = self.sizes(papers)                      # how many papers to pick, read and prepare (user's choice)
        timings: dict[str, float] = {}
        clock = time.monotonic()

        def lap(stage: str) -> None:
            nonlocal clock
            now = time.monotonic()
            timings[stage] = round(now - clock, 1)
            clock = now

        await say("planning and searching the academic sources...")
        found = await self.discovery.discover(topic, user_id, say=say)
        lap("discovery")
        plan, brief = found.plan, found.plan.research_brief
        warnings = list(found.errors)
        await say(f"found {found.unique_papers} unique papers in {timings['discovery']}s")

        # prefilter returns Candidate objects; remember each one's database id by object identity
        id_of = {id(c): paper_id for c, paper_id in zip(found.candidates, found.paper_ids)}
        kept = prefilter(found.candidates, plan.queries + found.agent_queries, keep=size.prefilter_keep)
        papers = [c.paper for c in kept]
        paper_ids = [id_of[id(c)] for c in kept]

        screenings, attempted = await self._screen_adaptive(brief, papers, say, enough=size.shortlist)
        lap("screening")
        if len(screenings) < attempted:
            warnings.append(f"{attempted - len(screenings)} of {attempted} papers could not be screened")

        field = field_family(plan)
        shortlist = select_best(papers, screenings, field, k_max=size.shortlist, k_min=size.select_min)
        warnings += shortlist.warnings
        picked = shortlist.picked[:size.select_max]
        lap("selection")

        texts: dict[int, PaperText] = {}
        if self.fulltext is not None and shortlist.picked:
            await say(f"looking for free full texts of the best {len(shortlist.picked)} papers...")
            by_paper = await self.fulltext.texts_for({paper_ids[sp.index]: sp.paper for sp in shortlist.picked})
            final = prefer_full_text(shortlist.picked, {sp.index for sp in shortlist.picked
                                                        if paper_ids[sp.index] in by_paper}, size.select_max)
            picked, warnings = final.picked, warnings + final.warnings
            texts = {sp.index: by_paper[paper_ids[sp.index]] for sp in picked if paper_ids[sp.index] in by_paper}
            lap("full text")
            await say(f"full text for {len(texts)} of {len(picked)} selected papers")

        await say(f"summarising {len(picked)} selected papers...")
        summaries, summary_warnings = await self.summarizer.summarize(
            brief, {sp.index: sp.paper for sp in picked}, texts)
        warnings += summary_warnings
        lap("summaries")

        if self.synthesis is not None and picked:
            inputs = [SynthesisInput(
                paper_id=paper_ids[sp.index], label=in_text(sp.paper), study_type=sp.screening.study_type,
                paper=sp.paper, base_trust=sp.trust, text=texts.get(sp.index),
                limitations=_limitations(summaries.get(sp.index))) for sp in picked]
            await self.synthesis.run(found.topic_id, brief, field, inputs, progress=say)
            lap("analysis")

        if index:
            warnings += await self._index([paper_ids[sp.index] for sp in picked], say)

        async with self.unit_of_work() as session:        # all results saved together, or nothing
            topics = TopicRepository(session)
            for i, s in screenings.items():
                await topics.save_screening(found.topic_id, paper_ids[i], relevance=s.relevance,
                                            reason=s.reason, study_type=s.study_type)
            for rank, sp in enumerate(picked, start=1):
                summary = summaries.get(sp.index)
                await topics.save_selection(
                    found.topic_id, paper_ids[sp.index], rank=rank, score=sp.score, score_parts=sp.parts,
                    trust=sp.trust.model_dump(), summary=summary.model_dump(mode="json") if summary else None)
            await topics.save_run(found.topic_id, plan=plan.model_dump(), run_stats={
                "field_family": field,
                "sources_used": found.sources_used,
                "search_agent": found.agent_rounds,
                "counts": {"found": found.unique_papers, "prefiltered": len(papers),
                           "screened": len(screenings), "skipped": len(papers) - attempted,
                           "shortlisted": len(shortlist.picked), "selected": len(picked),
                           "full_text": len(texts)},
                "timings_s": timings,
                "warnings": warnings,
            })
        await say("saved")
        return await self.load(found.topic_id, user_id)

    async def analyze_topic(self, topic_id: int, user_id: int, *, refresh: bool = False,
                            progress: Progress | None = None, index: bool = True) -> ResearchReport:
        """Run milestone 5 for an EXISTING topic (e.g. researched before milestone 5, or after a prompt
        change with refresh=True). Uses stored full texts; analyses are reused where still valid."""
        if self.synthesis is None:
            raise RuntimeError("analysis is switched off (ANALYSIS_ENABLED=false)")
        report = await self.load(topic_id, user_id)
        texts = {}
        if self.fulltext is not None:
            texts = await self.fulltext.texts_for({p.paper_id: p.paper for p in report.papers})
        inputs = [SynthesisInput(
            paper_id=p.paper_id, label=p.in_text, study_type=p.study_type, paper=p.paper,
            base_trust=p.base_trust or p.trust, text=texts.get(p.paper_id), limitations=_limitations(p.summary))
            for p in report.papers]
        await self.synthesis.run(topic_id, report.plan.research_brief, report.field_family, inputs,
                                 refresh=refresh, progress=progress)
        if index:
            await self._index([p.paper_id for p in report.papers], progress or quiet)
        return await self.load(topic_id, user_id)

    async def prepare(self, topic_id: int, user_id: int, progress: Progress | None = None,
                      time_budget_s: float | None = None) -> "PrepareOutcome":
        """Prepare a topic's papers for questions as part of a run. Never raises: says whether it finished,
        and if not, why and how long the embedding API asked to wait (the job then continues it later).
        Resumable: papers already embedded are skipped, so a later try only does what is missing."""
        if self.indexer is None:
            return PrepareOutcome(done=True)
        report = await self.load(topic_id, user_id)
        work = self.indexer.index([p.paper_id for p in report.papers], progress=progress)
        try:
            result = await (asyncio.wait_for(work, time_budget_s) if time_budget_s else work)
        except TimeoutError:
            return PrepareOutcome(done=False, reason="this run's time is up")
        if result.failed:
            return PrepareOutcome(done=False, reason=result.failed[0], retry_after=result.retry_after)
        return PrepareOutcome(done=True)

    async def index_topic(self, topic_id: int, user_id: int, progress: Progress | None = None) -> ResearchReport:
        """Prepare an EXISTING topic's papers for questions (step 7.6: the 'index' job)."""
        if self.indexer is None:
            raise RuntimeError("questions are switched off (INDEX_ENABLED=false)")
        report = await self.load(topic_id, user_id)
        result = await self.indexer.index([p.paper_id for p in report.papers], progress=progress)
        if result.failed:
            raise RuntimeError(f"{len(result.failed)} papers could not be prepared: {result.failed[0]}")
        return report

    async def _index(self, paper_ids: list[int], say: Progress) -> list[str]:
        """Prepare the selected papers for questions (7.6). Never fails a research: problems become warnings."""
        if self.indexer is None or not paper_ids:
            return []
        result = await self.indexer.index(paper_ids, progress=say)
        return [f"not ready for questions: {f}" for f in result.failed]

    async def topics(self, user_id: int, limit: int = 50) -> list[TopicSummary]:
        """The user's topics for a list page (no LLM calls)."""
        async with self.unit_of_work() as session:
            return [TopicSummary(**row) for row in await TopicRepository(session).summaries(user_id, limit)]

    async def paper_detail(self, topic_id: int, rank: int, user_id: int | None = None) -> PaperDetail:
        """Everything known about the paper ranked `rank` in a topic (step 5.7). No LLM calls."""
        report = await self.load(topic_id, user_id)              # includes the ownership check
        paper = next((p for p in report.papers if p.rank == rank), None)
        if paper is None:
            raise NotFoundError(f"topic {topic_id} has no paper ranked {rank}")
        async with self.unit_of_work() as session:
            documents = DocumentRepository(session)
            usable = [d for d in await documents.for_paper(paper.paper_id) if d.status != "failed"] or \
                await documents.for_paper(paper.paper_id)        # a failed one still explains why
            document = None
            if usable:
                best = min(usable, key=lambda d: (d.status != "extracted", d.content_type != "application/xml"))
                sections = await documents.sections(best.id)
                document = DocumentInfo(
                    content_type=best.content_type, source_url=best.source_url, size_bytes=best.size_bytes,
                    page_count=best.page_count, status=best.status, error=best.error,
                    sections=[SectionInfo(kind=s.kind, title=s.title, characters=len(s.text)) for s in sections])
        return PaperDetail(
            topic_id=report.topic_id, topic=report.topic, field_family=report.field_family,
            n_papers=len(report.papers), paper=paper,
            connections=[c for c in report.connections if any(m.paper == paper.paper_id for m in c.members)],
            labels={p.paper_id: p.in_text for p in report.papers},
            ranks={p.paper_id: p.rank for p in report.papers}, document=document)

    async def _screen_adaptive(self, brief: str, papers: list[PaperRecord], say: Progress,
                               enough: int | None = None) -> tuple[dict[int, PaperScreening], int]:
        """
        Adaptive screening (stop early): screen the prefilter's best `screen_first` papers; if at least
        `shortlist_size` of them are relevant (>= MIN_RELEVANCE), stop. Otherwise screen the next
        `screen_step`, and so on. The prefilter already put the most promising papers first, so on a
        clear topic the rest are rarely needed. Saves ~50% of screening tokens there.
        Returns ({index in papers: screening}, how many papers were sent to the LLM).
        """
        return await screen_adaptive(self.screener, brief, papers, first=self.screen_first, step=self.screen_step,
                                     enough=enough or self.shortlist_size, say=say)

    async def load(self, topic_id: int, user_id: int | None = None) -> ResearchReport:
        """
        Rebuild a topic's report from the database. No LLM calls.
        With user_id: another user's topic looks exactly like a missing one ("does not exist"),
        so nobody can even find out which topic ids other users have.
        """
        async with self.unit_of_work() as session:
            topics = TopicRepository(session)
            topic = await topics.get(topic_id)
            if topic is None or (user_id is not None and topic.user_id != user_id):
                raise NotFoundError(f"topic {topic_id} does not exist")
            if topic.plan is None or topic.run_stats is None:
                raise NotReadyError(f"topic {topic_id} has no research results yet (still running, or it failed)")
            rows = await topics.selected(topic_id)
            analyses = await AnalysisRepository(session).get_many([paper.id for _, paper in rows])
            synthesis = await topics.get_synthesis(topic_id)

            records = [to_record(paper) for _, paper in rows]
            keys = unique_bibtex_keys(records)
            papers = []
            for (link, paper), record, key in zip(rows, records, keys):
                base = TrustAssessment.model_validate(link.trust)
                stored = analyses.get(paper.id)
                papers.append(ReportPaper(
                    rank=link.rank, paper_id=paper.id, paper=record,
                    relevance=link.relevance, relevance_reason=link.relevance_reason or "",
                    study_type=link.study_type or "other",
                    trust=TrustAssessment.model_validate(link.adjusted_trust) if link.adjusted_trust else base,
                    base_trust=base, score=link.score, score_parts=link.score_parts or {},
                    summary=SummarizedPaper.model_validate(link.summary) if link.summary else None,
                    apa=apa(record), in_text=in_text(record), bibtex_key=key, bibtex=bibtex(record, key),
                    analysis=AnalyzedPaper.model_validate(stored.analysis) if stored else None,
                ))
            stats = topic.run_stats
            return ResearchReport(
                topic_id=topic.id, topic=topic.query, created_at=topic.created_at,
                plan=ResearchPlan.model_validate(topic.plan), field_family=stats["field_family"],
                sources_used=stats["sources_used"], counts=stats["counts"], timings_s=stats["timings_s"],
                warnings=stats["warnings"], papers=papers,
                search_agent=[SearchAgentRound.model_validate(r) for r in stats.get("search_agent", [])],
                connections=[Connection.model_validate(c) for c in synthesis.connections] if synthesis else [],
                gaps=EvidenceGaps.model_validate(synthesis.gaps) if synthesis and synthesis.gaps else None,
                synthesis_flags=list(synthesis.flags) if synthesis else [],
            )


@dataclass
class RunSizes:
    select_max: int          # papers in the final selection (read in full text where free, prepared for chat)
    shortlist: int           # best papers whose full text is tried before the final pick
    prefilter_keep: int      # candidates that may be screened by the LLM
    select_min: int          # fewer relevant than this -> relax the gate + warn


@dataclass
class PrepareOutcome:
    done: bool
    reason: str = ""                     # why it stopped (when not done)
    retry_after: float | None = None     # seconds the embedding API asked to wait


async def screen_adaptive(screener: ScreeningService, brief: str, papers: list[PaperRecord], *, first: int,
                          step: int, enough: int, say: Progress | None = None
                          ) -> tuple[dict[int, PaperScreening], int]:
    """The adaptive screening loop (shared by the pipeline and the search-recall eval)."""
    say = say or quiet
    results: dict[int, PaperScreening] = {}
    start = 0
    while start < len(papers):
        end = min(start + (first if start == 0 else step), len(papers))
        await say(f"screening papers {start + 1}-{end} of {len(papers)} with the LLM...")
        found = await screener.screen(brief, papers[start:end])
        for i, s in found.items():                                  # batch index -> index in papers
            results[start + i] = with_official_type(s, papers[start + i])
        start = end
        relevant = sum(s.relevance >= MIN_RELEVANCE for s in results.values())
        if relevant >= enough:
            if start < len(papers):
                await say(f"{relevant} relevant papers already: skipped screening the other {len(papers) - start}")
            break
    return results, start


def with_official_type(screening: PaperScreening, paper: PaperRecord) -> PaperScreening:
    """The screening with its study type checked against the paper's official index tags."""
    study_type, note = reconcile(screening.study_type, paper.pub_types)
    if note is None:
        return screening
    return screening.model_copy(update={"study_type": study_type, "reason": f"{screening.reason} [{note}]"})


def _limitations(summary: SummarizedPaper | None) -> list[str]:
    """Limitations the authors stated, from a paper's summary (step 4.5), if any."""
    return list(summary.summary.limitations) if summary and summary.summary else []
