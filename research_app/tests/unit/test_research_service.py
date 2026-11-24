"""
The research pipeline with fakes: no network, no LLM, no database.
Checks that results are saved for the RIGHT paper ids and that failures become warnings.
"""

from contextlib import asynccontextmanager

import pytest

from app.common.progress import quiet
from app.domain.schemas import (Candidate, DiscoveryResult, FoundBy, PaperRecord, PaperScreening,
                                PaperSummary, ResearchPlan, SummarizedPaper)
from app.services import research_service
from app.services.fulltext_context import PaperText
from app.services.research_service import ResearchService

PLAN = ResearchPlan(field="sports science", is_biomedical=True, is_cs_math_physics=False,
                    research_brief="How the triceps extends the elbow.", queries=["triceps elbow extension"])

# title -> relevance the fake screener gives it
RELEVANCE = {
    "Triceps elbow extension torque": 9,
    "Triceps elbow extension EMG in athletes": 8,
    "Triceps muscle architecture ultrasound": 7,
    "Elbow extension after nerve transfer": 6,
    "Triceps tendon anatomy and extension": 6,
    "Swimming shoulder injuries": 1,
}


def discovery_result() -> DiscoveryResult:
    candidates = [Candidate(paper=PaperRecord(source="x", title=t, abstract=f"{t}. " * 10, year=2020,
                                              citation_count=10),
                            found_by=[FoundBy(source="openalex", query="q")]) for t in RELEVANCE]
    return DiscoveryResult(topic_id=7, plan=PLAN, sources_used=["openalex"], raw_results=6, unique_papers=6,
                           per_source={"openalex": 6}, errors=["europepmc / 'q': timed out after 20s"],
                           candidates=candidates, paper_ids=[100 + i for i in range(len(candidates))])


class FakeDiscovery:
    async def discover(self, topic, user_id, say=None):
        assert user_id == 3
        return discovery_result()


class FakeScreener:
    async def screen(self, brief, papers):
        return {i: PaperScreening(id=i, relevance=RELEVANCE[p.title], reason="r", study_type="lab_experiment")
                for i, p in enumerate(papers)}


class FakeSummarizer:
    """Summarises every paper except the 'nerve transfer' one, which 'fails'. Remembers the texts it got."""

    def __init__(self):
        self.texts = None

    async def summarize(self, brief, papers, texts=None):
        self.texts = texts
        out = {}
        for key, p in papers.items():
            if "nerve" in p.title:
                continue
            out[key] = SummarizedPaper(based_on="abstract", summary=PaperSummary(
                tldr=p.title, objective="o", methods="m", key_findings=["f"], why_relevant="w"))
        return out, ["could not summarise: Elbow extension after nerve transfer"]


class FakeTopicRepository:
    saved: dict = {}

    def __init__(self, session):
        pass

    async def save_screening(self, topic_id, paper_id, **kwargs):
        self.saved.setdefault("screening", {})[paper_id] = kwargs

    async def save_selection(self, topic_id, paper_id, **kwargs):
        self.saved.setdefault("selection", {})[paper_id] = kwargs

    async def save_run(self, topic_id, **kwargs):
        self.saved["run"] = (topic_id, kwargs)


@asynccontextmanager
async def fake_unit_of_work():
    yield None


@pytest.fixture
def saved(monkeypatch):
    FakeTopicRepository.saved = {}
    monkeypatch.setattr(research_service, "TopicRepository", FakeTopicRepository)
    return FakeTopicRepository.saved


async def run_pipeline(fulltext=None, select_max=10, shortlist=12) -> tuple[ResearchService, list[str]]:
    service = ResearchService(FakeDiscovery(), FakeScreener(), FakeSummarizer(), fake_unit_of_work,
                              prefilter_keep=40, select_max=select_max, select_min=3, fulltext_shortlist=shortlist,
                              fulltext=fulltext)
    loaded = []

    async def fake_load(topic_id, user_id=None):
        loaded.append((topic_id, user_id))
        return "report"

    service.load = fake_load
    messages = []

    async def collect(message):                 # progress is async (7.1: jobs save it to the database)
        messages.append(message)
    assert await service.research("triceps", 3, progress=collect) == "report"
    assert loaded == [(7, 3)]                # always rebuilt from the database, for the same user
    return service, messages


async def test_results_are_saved_for_the_right_paper_ids(saved):
    await run_pipeline()
    ids = {title: 100 + i for i, title in enumerate(RELEVANCE)}
    screening = saved["screening"]
    assert set(screening) == set(ids.values())                    # every screened paper is saved
    assert screening[ids["Swimming shoulder injuries"]]["relevance"] == 1
    assert screening[ids["Triceps elbow extension torque"]]["relevance"] == 9

    selection = saved["selection"]
    assert ids["Swimming shoulder injuries"] not in selection    # relevance 1: not selected
    assert sorted(s["rank"] for s in selection.values()) == list(range(1, len(selection) + 1))
    assert selection[ids["Triceps elbow extension torque"]]["rank"] == 1


async def test_failures_become_warnings_not_crashes(saved):
    _, messages = await run_pipeline()
    ids = {title: 100 + i for i, title in enumerate(RELEVANCE)}
    assert saved["selection"][ids["Elbow extension after nerve transfer"]]["summary"] is None

    topic_id, saved_run = saved["run"]
    assert topic_id == 7
    assert saved_run["plan"]["research_brief"] == PLAN.research_brief
    run = saved_run["run_stats"]
    assert "europepmc / 'q': timed out after 20s" in run["warnings"]
    assert "could not summarise: Elbow extension after nerve transfer" in run["warnings"]
    assert run["counts"] == {"found": 6, "prefiltered": 6, "screened": 6, "skipped": 0,
                             "shortlisted": len(saved["selection"]), "selected": len(saved["selection"]),
                             "full_text": 0}
    assert set(run["timings_s"]) == {"discovery", "screening", "selection", "summaries"}
    assert run["field_family"] == "sport_biomechanics"
    assert messages and messages[-1] == "saved"


# ---------- adaptive screening (stop early) ----------

class CountingScreener:
    """Gives relevance 8 to papers whose title starts with 'good', 2 otherwise; records each call's size."""

    def __init__(self):
        self.calls = []

    async def screen(self, brief, papers):
        self.calls.append(len(papers))
        return {i: PaperScreening(id=i, relevance=8 if p.title.startswith("good") else 2, reason="r",
                                  study_type="cohort") for i, p in enumerate(papers)}


def adaptive_service(screener) -> ResearchService:
    return ResearchService(None, screener, None, fake_unit_of_work, screen_first=20, screen_step=20,
                           select_max=10)


def titles(good: int, bad: int) -> list[PaperRecord]:
    return [PaperRecord(source="x", title=f"good {i}") for i in range(good)] + \
           [PaperRecord(source="x", title=f"bad {i}") for i in range(bad)]


async def test_stops_after_first_round_when_enough_are_relevant():
    screener = CountingScreener()
    papers = titles(good=12, bad=28)                      # 40 papers, the first 20 contain 12 relevant ones
    results, attempted = await adaptive_service(screener)._screen_adaptive("b", papers, quiet)
    assert screener.calls == [20]                         # the other 20 were never sent to the LLM
    assert attempted == 20 and len(results) == 20


async def test_screens_more_when_first_round_is_not_enough():
    screener = CountingScreener()
    papers = titles(good=0, bad=15) + titles(good=11, bad=14)   # relevant papers sit in the second half
    results, attempted = await adaptive_service(screener)._screen_adaptive("b", papers, quiet)
    assert screener.calls == [20, 20]
    assert attempted == 40
    assert {papers[i].title for i, s in results.items() if s.relevance >= 6} == {f"good {i}" for i in range(11)}


async def test_indexes_map_back_to_the_full_list():
    screener = CountingScreener()
    papers = titles(good=0, bad=20) + titles(good=3, bad=0)     # never enough: screens everything
    results, attempted = await adaptive_service(screener)._screen_adaptive("b", papers, quiet)
    assert screener.calls == [20, 3] and attempted == 23
    assert results[21].relevance == 8 and results[5].relevance == 2      # index 21 = 'good 1', not batch index 1


# ---------- full text (step 4.5) ----------

class FakeFullText:
    """Has a full text for two of the papers (by database id)."""

    def __init__(self, paper_ids):
        self.paper_ids, self.asked = paper_ids, None

    async def texts_for(self, papers):
        self.asked = set(papers)
        return {pid: PaperText(context=f"## Methods\npaper {pid}", source=f"paper {pid}")
                for pid in papers if pid in self.paper_ids}


async def test_full_texts_reach_the_summarizer_for_the_right_papers(saved):
    ids = {title: 100 + i for i, title in enumerate(RELEVANCE)}
    with_text = {ids["Triceps elbow extension torque"], ids["Triceps muscle architecture ultrasound"]}
    fulltext = FakeFullText(with_text)
    service, messages = await run_pipeline(fulltext)

    assert fulltext.asked == set(saved["selection"])                 # 5 relevant papers: all shortlisted + kept
    received = service.summarizer.texts                              # keyed by index, holding the right text
    assert sorted(t.source for t in received.values()) == sorted(f"paper {pid}" for pid in with_text)
    counts = saved["run"][1]["run_stats"]["counts"]
    assert counts["full_text"] == 2 and counts["shortlisted"] == 5 and counts["selected"] == 5
    assert "full text" in saved["run"][1]["run_stats"]["timings_s"]
    assert "full text for 2 of 5 selected papers" in messages


async def test_final_pick_prefers_papers_with_a_readable_full_text(saved):
    """Keep 2 of the 5 relevant papers: the two with full text win over better abstract-only ones."""
    ids = {title: 100 + i for i, title in enumerate(RELEVANCE)}
    with_text = {ids["Elbow extension after nerve transfer"], ids["Triceps tendon anatomy and extension"]}
    fulltext = FakeFullText(with_text)
    await run_pipeline(fulltext, select_max=2)

    assert len(fulltext.asked) == 5                                   # full text tried for the whole shortlist
    assert set(saved["selection"]) == with_text                       # relevance 6 + full text beat 9, 8, 7
    run = saved["run"][1]["run_stats"]
    assert run["counts"]["full_text"] == 2 and run["counts"]["selected"] == 2
    assert not any("abstract only" in w for w in run["warnings"])


async def test_abstract_only_papers_fill_gaps_with_a_warning(saved):
    ids = {title: 100 + i for i, title in enumerate(RELEVANCE)}
    await run_pipeline(FakeFullText({ids["Triceps tendon anatomy and extension"]}), select_max=3)

    selection = saved["selection"]
    assert ids["Triceps tendon anatomy and extension"] in selection   # the one with full text is kept
    assert ids["Triceps elbow extension torque"] in selection         # best abstract-only ones fill the rest
    run = saved["run"][1]["run_stats"]
    assert "only 1 of 3 selected papers have a readable free full text; 2 are abstract only" in run["warnings"]


# ---------- milestone 5 synthesis inside the pipeline (step 5.5) ----------

class FakeSynthesis:
    def __init__(self):
        self.calls = []

    async def run(self, topic_id, brief, field, papers, *, refresh=False, progress=None):
        self.calls.append((topic_id, field, papers))
        return []


async def test_synthesis_runs_once_with_the_selected_papers_and_their_texts(saved):
    ids = {title: 100 + i for i, title in enumerate(RELEVANCE)}
    synthesis = FakeSynthesis()
    service = ResearchService(FakeDiscovery(), FakeScreener(), FakeSummarizer(), fake_unit_of_work,
                              prefilter_keep=40, select_max=10, select_min=3,
                              fulltext=FakeFullText({ids["Triceps elbow extension torque"]}), synthesis=synthesis)

    async def fake_load(topic_id, user_id=None):
        return "report"
    service.load = fake_load
    await service.research("triceps", 3)

    (topic_id, field, papers), = synthesis.calls
    assert topic_id == 7 and field == "sport_biomechanics"
    assert {p.paper_id for p in papers} == set(saved["selection"])          # exactly the selected papers
    with_text = [p.paper_id for p in papers if p.text is not None]
    assert with_text == [ids["Triceps elbow extension torque"]]             # texts are passed along
    assert all(p.base_trust.level >= 0 for p in papers)
    assert "analysis" in saved["run"][1]["run_stats"]["timings_s"]


# ---------- how many papers a run keeps (the user's choice: 5 / 10 / 15 / 20) ----------

def test_sizes_default_is_unchanged_and_other_counts_scale_the_margins():
    service = ResearchService(None, None, None, fake_unit_of_work, prefilter_keep=40, select_max=10,
                              select_min=5, fulltext_shortlist=12, fulltext=object())
    assert service.sizes(None) == service.sizes(10)
    ten = service.sizes(10)
    assert (ten.select_max, ten.shortlist, ten.prefilter_keep, ten.select_min) == (10, 12, 40, 5)
    twenty = service.sizes(20)
    assert (twenty.select_max, twenty.shortlist, twenty.prefilter_keep) == (20, 22, 60)   # 2 spare, 3 per paper
    five = service.sizes(5)
    assert (five.select_max, five.shortlist, five.prefilter_keep, five.select_min) == (5, 7, 40, 5)
