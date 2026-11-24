"""
Discovery: topic -> plan -> search every chosen source with every query (in parallel)
-> [search agent: look at the results, search new queries, at most N extra rounds]
-> merge duplicates -> save papers -> link them to the topic.

Business rules that live here:
  - which sources to use for which field
  - each source has its OWN limit of searches in flight (a slow source only delays itself)
  - each search has a time budget; slower ones are cancelled and reported
  - the whole search phase has a deadline; whatever finished by then is used
  - a failing source is reported, never fatal (other sources still count)
  - candidates are sorted so the most-found, most-cited papers come first
  - the search agent may add rounds, but CODE owns the loop: at most `agent_rounds` extra rounds,
    stop when it says done or suggests nothing usable, and an agent failure only ends the loop

Measured before these rules (topic "thyroid", Europe PMC having a slow day): 82 s in total.
One shared semaphore let slow Europe PMC calls (up to 56 s) and Semantic Scholar calls waiting
for their own lock hold the slots, so fast OpenAlex queries queued behind them.
"""

import asyncio
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.logging import get_logger
from app.common.progress import Progress, quiet
from app.domain.errors import LlmError, QuotaExceededError, SourceError
from app.domain.schemas import Candidate, DiscoveryResult, PaperRecord, ResearchPlan
from app.integrations.sources.base import SourceClient
from app.repositories.paper_repository import PaperRepository
from app.repositories.topic_repository import TopicRepository
from app.services.dedupe import merge_records
from app.services.planning_service import PlanningService
from app.services.search_agent import SearchAgent, round_summary

log = get_logger(__name__)
UnitOfWork = Callable[[], AbstractAsyncContextManager[AsyncSession]]   # e.g. session_scope

# How many searches each source may run at the same time.
# arXiv and Semantic Scholar allow ~1 request at a time (their clients also space calls).
PARALLEL_PER_SOURCE = {"openalex": 4, "europepmc": 3, "semantic_scholar": 1, "arxiv": 1}
DEFAULT_PARALLEL = 2       # for any source not listed above


@dataclass
class SearchOutcome:
    """Plan + search results before anything is saved."""
    plan: ResearchPlan
    sources: list[str]
    records: list[PaperRecord]          # raw results (one per source x query hit)
    candidates: list[Candidate]         # merged and sorted
    errors: list[str]
    agent_queries: list[str] = field(default_factory=list)   # queries the search agent added
    agent_rounds: list[dict] = field(default_factory=list)   # what the agent decided each round

    @property
    def queries(self) -> list[str]:
        """Every query that was run: the plan's, then the agent's (the prefilter ranks by all of them)."""
        return self.plan.queries + self.agent_queries


class DiscoveryService:
    def __init__(self, sources: dict[str, SourceClient], planner: PlanningService,
                 unit_of_work: UnitOfWork, results_per_query: int = 15,
                 search_timeout_s: float = 20.0, total_budget_s: float = 30.0,
                 agent: SearchAgent | None = None, agent_rounds: int = 2):
        self.sources = sources            # {"openalex": OpenAlexClient(...), ...}
        self.planner = planner
        self.unit_of_work = unit_of_work  # opens ONE database transaction
        self.results_per_query = results_per_query
        self.search_timeout_s = search_timeout_s   # time budget for ONE search (one source, one query)
        self.total_budget_s = total_budget_s       # time budget for the WHOLE search phase (per round)
        self.agent = agent                         # None = one round only (the old behaviour)
        self.agent_rounds = agent_rounds           # extra rounds the agent may add at most

    async def search_only(self, topic: str, plan: ResearchPlan | None = None,
                          say: Progress | None = None) -> SearchOutcome:
        """Plan and search, WITHOUT saving anything (the search-recall eval runs this live).
        A given `plan` is reused instead of asking the planner (evals: same queries every run)."""
        plan = plan or await self.planner.plan(topic)
        source_names = self.choose_sources(plan)
        log.info("plan: field=%s sources=%s queries=%s", plan.field, source_names, plan.queries)
        records, errors = await self._search_all(source_names, plan.queries)
        outcome = SearchOutcome(plan, source_names, records, sort_candidates(merge_records(records)), errors)
        if self.agent is not None:
            say = say or quiet
            await say(f"round 1: {len(outcome.candidates)} papers from {len(plan.queries)} queries")
            await self._agent_rounds(outcome, say)
        return outcome

    async def _agent_rounds(self, outcome: SearchOutcome, say: Progress) -> None:
        """The agent loop: review the results -> search its new queries -> repeat (updates `outcome`).
        Every decision is shown as progress, with its reason (gaps): an agent's work should be visible."""
        for round_no in range(2, self.agent_rounds + 2):          # round 1 was the plan's own search
            summary = round_summary(outcome.candidates, outcome.queries)
            titles = [c.paper.title for c in outcome.candidates]
            try:
                review = await self.agent.review(outcome.plan.research_brief, summary, titles,
                                                 limit=self.results_per_query)
            except (LlmError, QuotaExceededError) as e:          # the agent is an extra: never fatal
                log.warning("search agent failed, keeping the results so far: %s", e)
                outcome.errors.append(f"search agent: {e}")
                await say("search agent unavailable: continuing with the papers found so far")
                return
            outcome.agent_rounds.append({"round": round_no, "gaps": review.gaps, "done": review.done,
                                         "new_queries": review.new_queries})
            log.info("search agent round %d: done=%s new=%s gaps=%s", round_no, review.done,
                     review.new_queries, review.gaps)
            if review.done:
                await say(f"search agent: done, the results cover the topic ({review.gaps})")
                return
            await say(f"search agent: {review.gaps}")
            await say(f"round {round_no}: searching " + ", ".join(f'"{q}"' for q in review.new_queries))
            before = len(outcome.candidates)
            records, errors = await self._search_all(outcome.sources, review.new_queries)
            outcome.records += records
            outcome.errors += errors
            outcome.agent_queries += review.new_queries
            outcome.candidates = sort_candidates(merge_records(outcome.records))
            outcome.agent_rounds[-1]["new_papers"] = len(outcome.candidates) - before
            await say(f"round {round_no}: +{len(outcome.candidates) - before} new papers")

    async def discover(self, topic: str, user_id: int, say: Progress | None = None) -> DiscoveryResult:
        """Plan, search and save a new topic owned by `user_id`."""
        found = await self.search_only(topic, say=say)
        plan, source_names, records, candidates, errors = (found.plan, found.sources, found.records,
                                                           found.candidates, found.errors)

        async with self.unit_of_work() as session:        # everything saved together, or nothing
            topics, papers = TopicRepository(session), PaperRepository(session)
            topic_row = await topics.create(topic, user_id=user_id, field=plan.field,
                                            clarified_query=plan.research_brief)
            paper_ids = []
            for candidate in candidates:
                paper = await papers.upsert_from_record(candidate.paper)
                await topics.link_paper(topic_row.id, paper.id,
                                        [f.model_dump() for f in candidate.found_by])
                paper_ids.append(paper.id)        # later steps save results per paper id
            topic_id = topic_row.id

        return DiscoveryResult(
            topic_id=topic_id, plan=plan, sources_used=source_names, raw_results=len(records),
            unique_papers=len(candidates), per_source=dict(Counter(r.source for r in records)),
            errors=errors, candidates=candidates, paper_ids=paper_ids,
            agent_queries=found.agent_queries, agent_rounds=found.agent_rounds,
        )

    def choose_sources(self, plan: ResearchPlan) -> list[str]:
        """OpenAlex and Semantic Scholar for everything; Europe PMC / arXiv only when the field fits."""
        wanted = ["openalex", "semantic_scholar"]
        if plan.is_biomedical:
            wanted.append("europepmc")
        if plan.is_cs_math_physics:
            wanted.append("arxiv")
        return [name for name in wanted if name in self.sources]   # only sources that are configured

    async def _search_all(self, source_names: list[str],
                          queries: list[str]) -> tuple[list[PaperRecord], list[str]]:
        # Fix A: one semaphore PER SOURCE (not one shared by all), so a slow or rate-limited
        # source only queues its own searches and never blocks the fast ones.
        semaphores = {name: asyncio.Semaphore(PARALLEL_PER_SOURCE.get(name, DEFAULT_PARALLEL))
                      for name in source_names}

        async def run(name: str, query: str) -> tuple[list[PaperRecord], str | None]:
            async with semaphores[name]:                   # wait for a free slot of THIS source
                try:
                    # Fix B: a time budget for this one search. The clock starts only after we got
                    # a slot, so waiting in line doesn't count. Too slow -> cancelled and reported.
                    found = await asyncio.wait_for(self.sources[name].search(query, self.results_per_query),
                                                   timeout=self.search_timeout_s)
                    return found, None
                except TimeoutError:
                    log.warning("search timed out: %s / %r after %.0fs", name, query, self.search_timeout_s)
                    return [], f"{name} / '{query}': timed out after {self.search_timeout_s:.0f}s"
                except SourceError as e:                   # one failure must not stop the others
                    log.warning("search failed: %s", e)
                    return [], f"{name} / '{query}': {e}"

        # Start every search as a Task (they all run concurrently) and remember which is which.
        tasks = {asyncio.create_task(run(name, query)): (name, query)
                 for name in source_names for query in queries}

        # Fix C: an overall deadline for the whole search phase. A rate-limited source runs its
        # searches one after another, so per-search budgets alone can still add up (measured:
        # Semantic Scholar 6 x ~13 s = 80 s). After the deadline we keep what finished and cancel the rest.
        done, pending = await asyncio.wait(tasks, timeout=self.total_budget_s)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)   # let cancelled tasks clean up (release locks)

        records, errors = [], []
        for task in done:
            found, error = task.result()
            records.extend(found)
            if error:
                errors.append(error)
        for task in pending:
            name, query = tasks[task]
            errors.append(f"{name} / '{query}': skipped, overall search budget of {self.total_budget_s:.0f}s reached")
        if pending:
            log.warning("search budget reached: %d searches skipped", len(pending))
        return records, errors


def sort_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Temporary ordering until real ranking (milestone 3): found by more queries/sources, then citations."""
    return sorted(candidates, key=lambda c: (len(c.found_by), c.paper.citation_count or 0), reverse=True)
