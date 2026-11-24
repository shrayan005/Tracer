"""
Planning + discovery with a fake LLM and fake sources (no network, no database).
These tests lock in the timing rules: per-source limits, per-search timeout, overall deadline.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager

from app.domain.errors import SourceError
from app.domain.schemas import PaperRecord, ResearchPlan
from app.integrations.llm.gateway import LlmGateway
from app.services.discovery_service import DiscoveryService
from app.services.planning_service import PlanningService
from tests.fakes import FakeProvider

PLAN_JSON = json.dumps({
    "field": "sports science", "is_biomedical": True, "is_cs_math_physics": False,
    "research_brief": "How the triceps produces elbow extension.",
    "queries": ["triceps brachii EMG", "elbow extension torque", "triceps brachii EMG"],
})


async def no_sleep(_seconds):
    pass


def planner(plan_json=PLAN_JSON):
    return PlanningService(LlmGateway([FakeProvider("fake", [plan_json])], sleep=no_sleep))


class FakeSource:
    """Returns one paper per query after `delay` seconds, or raises SourceError if fail=True."""

    def __init__(self, name, delay=0.0, fail=False):
        self.name, self.delay, self.fail = name, delay, fail
        self.calls = 0

    async def search(self, query, limit):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.fail:
            raise SourceError(self.name, "HTTP 429", retryable=True)
        return [PaperRecord(source=self.name, title=f"Triceps brachii activation study ({query})",
                            found_by_query=query)]


def plan(biomedical=True, cs=False):
    return ResearchPlan(field="x", is_biomedical=biomedical, is_cs_math_physics=cs,
                        research_brief="x", queries=["q"])


# ---------- planning ----------

async def test_plan_adds_topic_and_removes_duplicate_queries():
    result = await planner().plan("biomechanics of triceps")
    assert result.queries == ["biomechanics of triceps", "triceps brachii EMG", "elbow extension torque"]


# ---------- choosing sources ----------

def test_choose_sources_by_field_and_only_configured_ones():
    service = DiscoveryService({"openalex": None, "europepmc": None, "arxiv": None}, None, None)
    assert service.choose_sources(plan(biomedical=True)) == ["openalex", "europepmc"]
    assert service.choose_sources(plan(biomedical=False, cs=True)) == ["openalex", "arxiv"]
    # a switched-off source simply isn't in the dict, so it's never chosen
    assert DiscoveryService({"openalex": None}, None, None).choose_sources(plan()) == ["openalex"]


# ---------- searching: failures and timing rules ----------

async def test_failing_source_is_reported_not_fatal():
    service = DiscoveryService({"openalex": FakeSource("openalex"),
                                "europepmc": FakeSource("europepmc", fail=True)}, None, None)
    records, errors = await service._search_all(["openalex", "europepmc"], ["q1", "q2"])
    assert len(records) == 2                          # openalex still delivered
    assert len(errors) == 2 and all("europepmc" in e for e in errors)


async def test_slow_search_times_out():
    service = DiscoveryService({"europepmc": FakeSource("europepmc", delay=5)}, None, None,
                               search_timeout_s=0.1)
    records, errors = await service._search_all(["europepmc"], ["q1"])
    assert records == [] and "timed out" in errors[0]


async def test_slow_source_does_not_block_fast_source():
    # Fix A: per-source limits. OpenAlex must finish quickly even while Europe PMC is stuck.
    fast, slow = FakeSource("openalex", delay=0.05), FakeSource("europepmc", delay=5)
    service = DiscoveryService({"openalex": fast, "europepmc": slow}, None, None,
                               search_timeout_s=10, total_budget_s=0.5)
    start = time.monotonic()
    records, errors = await service._search_all(["openalex", "europepmc"], [f"q{i}" for i in range(6)])
    assert time.monotonic() - start < 1.0             # the overall deadline, not 5 s+
    assert sum(r.source == "openalex" for r in records) == 6    # every fast search finished
    assert all("europepmc" in e and "budget" in e for e in errors)


async def test_overall_budget_keeps_finished_results():
    # Fix C: a one-at-a-time source with 6 queries x 0.2 s would need 1.2 s; the budget is 0.5 s.
    slow_line = FakeSource("semantic_scholar", delay=0.2)
    service = DiscoveryService({"semantic_scholar": slow_line}, None, None, total_budget_s=0.5)
    records, errors = await service._search_all(["semantic_scholar"], [f"q{i}" for i in range(6)])
    assert 1 <= len(records) < 6                      # some finished in time and were KEPT
    assert len(records) + len(errors) == 6            # the rest are reported, nothing lost silently
    assert all("budget" in e for e in errors)


# ---------- a fixed plan (step 6.7: search evals reuse the stored plan) ----------

class RecordingSource(FakeSource):
    def __init__(self, name):
        super().__init__(name)
        self.queries = []

    async def search(self, query, limit):
        self.queries.append(query)
        return []


async def test_search_only_reuses_a_given_plan():
    source = RecordingSource("openalex")
    service = DiscoveryService({"openalex": source}, planner=None, unit_of_work=None)
    given = ResearchPlan(field="x", is_biomedical=True, is_cs_math_physics=False, research_brief="b",
                         queries=["creatine older adults", "creatine older men"])
    outcome = await service.search_only("creatine in older adults", plan=given)   # planner=None: never called
    assert outcome.plan is given
    assert sorted(source.queries) == ["creatine older adults", "creatine older men"]


# ---------- the whole discover() flow (repositories faked) ----------

async def test_discover_merges_and_saves(monkeypatch):
    saved = []

    class FakeTopicRepo:
        def __init__(self, session): pass
        async def create(self, query, **kw):
            assert kw["user_id"] == 1                 # the owner is passed on to the repository
            return type("Topic", (), {"id": 1})()
        async def link_paper(self, topic_id, paper_id, found_by):
            saved.append((paper_id, found_by))

    class FakePaperRepo:
        def __init__(self, session): pass
        async def upsert_from_record(self, record):
            return type("Paper", (), {"id": 100 + len(saved)})()

    monkeypatch.setattr("app.services.discovery_service.TopicRepository", FakeTopicRepo)
    monkeypatch.setattr("app.services.discovery_service.PaperRepository", FakePaperRepo)

    @asynccontextmanager
    async def fake_unit_of_work():
        yield object()                                # stands in for a database session

    class SamePaperSource(FakeSource):                # returns the SAME paper for every query
        async def search(self, query, limit):
            return [PaperRecord(source=self.name, title="Triceps brachii activation during push-ups",
                                doi="10.1/t", found_by_query=query)]

    service = DiscoveryService({"openalex": SamePaperSource("openalex"),
                                "europepmc": FakeSource("europepmc", fail=True)},
                               planner(), fake_unit_of_work)
    result = await service.discover("biomechanics of triceps", user_id=1)

    assert result.raw_results == 3                    # 1 paper x 3 queries from openalex
    assert result.unique_papers == 1                  # merged into one candidate
    assert len(result.candidates[0].found_by) == 3
    assert len(result.errors) == 3                    # europepmc failed for each query, no crash
    assert len(saved) == 1                            # saved and linked once


# ---------- the search agent loop (agent faked: no LLM) ----------

class ScriptedAgent:
    """Returns the given reviews in order; an Exception in the script is raised instead."""

    def __init__(self, *reviews):
        self.reviews, self.seen = list(reviews), []

    async def review(self, brief, summary, titles, limit):
        self.seen.append(dict(summary))
        step = self.reviews.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def review(*queries, done=False):
    from app.domain.schemas import SearchReview
    return SearchReview(gaps="g", done=done, new_queries=list(queries))


async def test_agent_rounds_add_queries_and_papers():
    from app.domain.schemas import ResearchPlan as RP
    source = FakeSource("openalex")
    agent = ScriptedAgent(review("q2"), review("q3"))
    service = DiscoveryService({"openalex": source}, None, None, agent=agent, agent_rounds=2)
    outcome = await service.search_only("t", plan=RP(field="x", is_biomedical=False, is_cs_math_physics=False,
                                                     research_brief="b", queries=["q1"]))
    assert source.calls == 3 and outcome.queries == ["q1", "q2", "q3"]
    assert len(outcome.candidates) == 3                           # one new paper per query
    assert [r["new_papers"] for r in outcome.agent_rounds] == [1, 1]
    assert list(agent.seen[1]) == ["q1", "q2"]                    # round 3 sees round 2's query too


async def test_agent_done_stops_and_max_rounds_is_enforced_by_code():
    source = FakeSource("openalex")
    service = DiscoveryService({"openalex": source}, None, None,
                               agent=ScriptedAgent(review("q2", done=True)), agent_rounds=2)
    outcome = await service.search_only("t", plan=plan(biomedical=False))
    assert source.calls == 1 and outcome.agent_queries == []      # done: nothing more searched

    agent = ScriptedAgent(review("a"), review("b"), review("c"))  # the agent would go on forever
    service = DiscoveryService({"openalex": FakeSource("openalex")}, None, None, agent=agent, agent_rounds=2)
    outcome = await service.search_only("t", plan=plan(biomedical=False))
    assert outcome.agent_queries == ["a", "b"] and len(agent.reviews) == 1   # 3rd review never asked


async def test_agent_failure_keeps_results_so_far():
    from app.domain.errors import AllProvidersFailed
    service = DiscoveryService({"openalex": FakeSource("openalex")}, None, None,
                               agent=ScriptedAgent(review("q2"), AllProvidersFailed([])), agent_rounds=2)
    outcome = await service.search_only("t", plan=plan(biomedical=False))
    assert outcome.queries == ["q", "q2"] and len(outcome.candidates) == 2
    assert any(e.startswith("search agent:") for e in outcome.errors)
