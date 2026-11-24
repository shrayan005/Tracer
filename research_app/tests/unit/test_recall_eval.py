"""Step 6.5: screening eval + search recall ("borrowed gold"). Fakes only: no tokens, no network, no database."""

from datetime import datetime, timezone

import httpx
import pytest
import yaml

from app.domain.schemas import Candidate, FoundBy, PaperRecord, PaperScreening, ResearchPlan, TrustAssessment
from app.evals import cases as cases_module
from app.evals import labels as labels_module
from app.evals import recall as r
from app.evals import screening_eval as se
from app.evals.cases import CasePaper, EvalCase, ScreenedCandidate, freeze
from app.evals.runner import Services
from app.integrations.sources.openalex import OpenAlexClient
from app.services.discovery_service import SearchOutcome

TRUST = TrustAssessment(level=5, label="high", reasons=["meta-analysis"])


def paper(title, **kw) -> PaperRecord:
    return PaperRecord(source=kw.pop("source", "openalex"), title=title, **kw)


def make_case(candidates=(), papers=None) -> EvalCase:
    papers = papers or [CasePaper(key=1, label="(Chilibeck et al., 2017)", study_type="meta_analysis",
                                  base_trust=TRUST, paper=paper("Creatine meta-analysis", openalex_id="W1"))]
    return EvalCase(name="creatine", topic_id=1, topic="creatine supplementation older adults",
                    brief="Does creatine help older adults?", field_family="biomedical",
                    frozen_at=datetime(2026, 11, 23, tzinfo=timezone.utc), papers=papers,
                    candidates=list(candidates))


# ---------- cases ----------

def test_old_case_files_without_candidates_still_load():
    data = make_case().model_dump(mode="json")
    data.pop("candidates")
    assert EvalCase.model_validate(data).candidates == []


async def test_refreezing_needs_force(tmp_path, monkeypatch):
    monkeypatch.setattr(cases_module, "CASES_DIR", tmp_path)
    (tmp_path / "creatine.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--force"):
        await freeze(research=None, topic_id=1, name="creatine")


# ---------- OpenAlex references ----------

async def test_openalex_references_fetches_titles_in_batches_of_50():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/W1"):
            return httpx.Response(200, json={"referenced_works": [f"https://openalex.org/W{i}" for i in range(100, 160)]})
        ids = request.url.params["filter"].removeprefix("openalex_id:").split("|")
        return httpx.Response(200, json={"results": [{"id": f"https://openalex.org/{i}", "display_name": f"T {i}"}
                                                     for i in ids]})

    client = OpenAlexClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    refs = await client.references("W1")
    assert len(refs) == 60 and refs[0].openalex_id == "W100"
    assert len(seen) == 3                                    # the paper + 50 + 10


# ---------- matching and pre-marking ----------

def test_same_paper_by_doi_id_or_title():
    p = paper("Creatine and Resistance Training in Older Women: an RCT", doi="10.1/ABC", openalex_id="W9")
    assert r.same_paper({"doi": "10.1/abc"}, p)
    assert r.same_paper({"openalex_id": "W9"}, p)
    assert r.same_paper({"title": "creatine and resistance training in older women - an RCT"}, p)
    assert not r.same_paper({"title": "Something else", "doi": "10.1/zzz"}, p)


def test_premark_by_citing_reviews_or_topic_words():
    topic = {"creatine", "supplementation", "older", "adults"}
    assert r.premark("Muscle physiology basics", topic, cited_by=2)                   # key study: 2 reviews
    assert r.premark("Creatine supplementation in older adults: a trial", topic, cited_by=1)
    assert not r.premark("Creatine kinetics in rats", topic, cited_by=1)


async def test_prefill_recall_merges_references_and_counts_reviews(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "recall_path", lambda name: tmp_path / f"recall_{name}.yaml")
    two_reviews = make_case(papers=[
        CasePaper(key=1, label="(A, 2017)", study_type="meta_analysis", base_trust=TRUST, paper=paper("A", openalex_id="W1")),
        CasePaper(key=2, label="(B, 2019)", study_type="systematic_review", base_trust=TRUST, paper=paper("B", openalex_id="W2")),
        CasePaper(key=3, label="(C, 2003)", study_type="randomized_trial", base_trust=TRUST, paper=paper("C", openalex_id="W3"))])

    class FakeOpenAlex:
        async def references(self, oid):
            shared = paper("Trial in vulnerable women", openalex_id="W50")
            return [shared, paper("Mitochondria review", openalex_id=f"W6{oid}")]

    path, n, marked = await r.prefill_recall(two_reviews, FakeOpenAlex())
    data = yaml.safe_load(open(path, encoding="utf-8"))
    assert n == 3 and marked == 1 and data["reviewed"] is False
    assert data["papers"][0]["openalex_id"] == "W50" and data["papers"][0]["cited_by_reviews"] == ["(A, 2017)", "(B, 2019)"]
    with pytest.raises(FileExistsError):
        await r.prefill_recall(two_reviews, FakeOpenAlex())


# ---------- the search-recall suite ----------

class FakeScreener:
    def __init__(self, scores):           # {title: relevance}
        self.scores = scores

    async def screen(self, brief, papers):
        return {i: PaperScreening(id=i, relevance=self.scores.get(p.title, 1), reason="r", study_type="other")
                for i, p in enumerate(papers)}


async def test_search_recall_follows_gold_through_every_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(labels_module, "GOLD_DIR", tmp_path)
    gold = [{"title": t, "include": True} for t in ("Found and relevant", "Found but screened out",
                                                     "Lost in prefilter", "Never found")]
    (tmp_path / "recall_creatine.yaml").write_text(yaml.safe_dump({"reviewed": True, "papers": gold}), encoding="utf-8")
    monkeypatch.setattr(r.get_settings(), "prefilter_keep", 2)   # only 2 candidates survive the prefilter

    def cand(title, *sources):
        return Candidate(paper=paper(title), found_by=[FoundBy(source=s, query="q") for s in sources])
    candidates = [cand("Found and relevant", "openalex", "europepmc"), cand("Found but screened out", "europepmc"),
                  cand("Lost in prefilter", "openalex")]
    monkeypatch.setattr(r, "prefilter", lambda cands, queries, keep: cands[:keep])

    class FakeDiscovery:
        async def search_only(self, topic, plan=None):
            plan = ResearchPlan(field="x", is_biomedical=True, is_cs_math_physics=False, research_brief="b",
                                queries=["q"])
            return SearchOutcome(plan, ["openalex", "europepmc"], [c.paper for c in candidates], candidates,
                                 ["europepmc / 'q': HTTP 500"])

    services = Services(analysis=None, screener=FakeScreener({"Found and relevant": 8}), discovery=FakeDiscovery())
    metrics, details = await r.search_recall([make_case()], services)
    assert metrics["gold_size"] == 4
    assert metrics["recall_found"] == 0.75 and metrics["recall_prefiltered"] == 0.5
    assert metrics["recall_relevant"] == 0.25 and metrics["source_errors"] == 1
    assert metrics["gold_found_by_europepmc"] == 2 and metrics["gold_only_europepmc"] == 1
    assert metrics["gold_only_openalex"] == 1
    assert any(d.startswith("LOST AT SEARCH") and "Never found" in d for d in details)
    assert any(d.startswith("LOST AT PREFILTER") and "Lost in prefilter" in d for d in details)
    assert any(d.startswith("SCREENED OUT") and "screened out" in d for d in details)


async def test_search_recall_needs_a_reviewed_file(tmp_path, monkeypatch):
    monkeypatch.setattr(labels_module, "GOLD_DIR", tmp_path)
    services = Services(analysis=None, screener=FakeScreener({}), discovery=object())
    with pytest.raises(ValueError, match="prefill recall"):
        await r.search_recall([make_case()], services)
    (tmp_path / "recall_creatine.yaml").write_text("reviewed: false\npapers: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed: true"):
        await r.search_recall([make_case()], services)


# ---------- the screening suite ----------

def screened(key, title, relevance):
    return ScreenedCandidate(key=key, paper=paper(title), relevance=relevance, study_type="randomized_trial")


async def test_screening_prefill_and_suite(tmp_path, monkeypatch):
    path = tmp_path / "screening.yaml"
    monkeypatch.setattr(se, "SCREENING", path)
    monkeypatch.setattr(se, "load_screening_gold", lambda: labels_module.load_screening_gold(path))
    case = make_case(candidates=[screened(1, "Key RCT", 8), screened(2, "Side topic", 4), screened(3, "Rats", 1)])
    se.prefill_screening([case])
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert [p["gold"] for p in data["cases"][0]["papers"]] == ["relevant", "partly", "not relevant"]
    with pytest.raises(ValueError, match="reviewed: true"):
        labels_module.load_screening_gold(path)
    # you: "Side topic" is in fact relevant, and paper 1 is a randomized trial
    data["cases"][0]["reviewed"] = True
    data["cases"][0]["papers"][1]["gold"] = "relevant"
    data["cases"][0]["papers"][0]["gold_study_type"] = "randomized_trial"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    # the current screener: misses "Side topic", wrongly calls "Rats" relevant
    services = Services(analysis=None, screener=FakeScreener({"Key RCT": 9, "Side topic": 5, "Rats": 7}))
    metrics, details = await se.screening([case], services)
    assert metrics["relevant_recall"] == 0.5 and metrics["relevant_precision"] == 0.5
    assert metrics["bucket_accuracy"] == pytest.approx(1 / 3) and metrics["study_type_accuracy"] == 0.0
    assert any(d.startswith("MISSED") and "Side topic" in d for d in details)
    assert any(d.startswith("FALSE RELEVANT") and "Rats" in d for d in details)
