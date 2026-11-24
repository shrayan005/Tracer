"""Eval framework (step 6.1): run scope tagging, cases, scorecards and the runner. No tokens, no database."""

import asyncio
from datetime import datetime, timezone

import pytest

from app.domain.schemas import PaperRecord, TrustAssessment
from app.evals import cases as cases_module
from app.evals import runner as runner_module
from app.evals import scorecard as scorecard_module
from app.evals.cases import CasePaper, EvalCase, load_case, save_case
from app.evals.runner import SUITES, Services, run_suite, suite
from app.evals.scorecard import Scorecard, StageCost, compare, load_scorecard, render_scorecard, save_scorecard
from app.integrations.llm.gateway import LlmGateway
from app.integrations.llm.run_context import current_run_id, run_scope
from tests.fakes import FakeProvider

MSG = [{"role": "user", "content": "hi"}]


# ---------- run scope ----------

async def test_run_scope_reaches_tasks_and_nests():
    assert current_run_id() is None
    with run_scope(7):
        seen = await asyncio.gather(*(asyncio.sleep(0, result=None) for _ in range(2)),
                                    asyncio.create_task(_read_run_id()))
        assert seen[-1] == 7                                        # a task started inside sees it
        with run_scope(8):
            assert current_run_id() == 8
        assert current_run_id() == 7                                # the outer id comes back
    assert current_run_id() is None


async def _read_run_id():
    await asyncio.sleep(0)
    return current_run_id()


async def test_gateway_tags_calls_with_the_run_id():
    records = []

    async def sink(record):
        records.append(record)

    async def no_sleep(_s):
        pass

    gateway = LlmGateway([FakeProvider("a", ["one", "two", "three"])], usage_sink=sink, sleep=no_sleep)
    with run_scope(42):
        await gateway.complete(MSG, purpose="test")
        await gateway.complete(MSG, purpose="test", run_id=5)             # an explicit id wins
    await gateway.complete(MSG, purpose="test")                          # outside any scope
    assert [r.run_id for r in records] == [42, 5, None]


# ---------- cases ----------

def case(name="demo") -> EvalCase:
    trust = TrustAssessment(level=3, label="moderate", reasons=["lab experiment"])
    return EvalCase(name=name, topic_id=1, topic="t", brief="b", field_family="general",
                    frozen_at=datetime(2026, 11, 20, tzinfo=timezone.utc), papers=[
                        CasePaper(key=10, label="(A, 2020)", paper=PaperRecord(source="db", title="A"),
                                  study_type="cohort", base_trust=trust, sections=[("methods", "M", "Ten people.")]),
                        CasePaper(key=11, label="(B, 2021)", paper=PaperRecord(source="db", title="B"),
                                  study_type="cohort", base_trust=trust)])


def test_case_round_trip_and_text(tmp_path, monkeypatch):
    monkeypatch.setattr(cases_module, "CASES_DIR", tmp_path)
    save_case(case())
    loaded = load_case("demo")
    assert loaded == case()
    assert loaded.papers[0].text().source == "Ten people."
    assert loaded.papers[1].text() is None                               # abstract only
    assert cases_module.list_cases() == ["demo"]


def test_missing_case_explains_how_to_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(cases_module, "CASES_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="python -m app.evals freeze <topic> nope"):
        load_case("nope")


# ---------- scorecards ----------

def card(run_id=1, metrics=None, settings=None) -> Scorecard:
    return Scorecard(suite="demo", run_id=run_id, started_at=datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc),
                     seconds=12.0, cases=["triceps"], settings=settings or {"analysis_version": "5.3"},
                     metrics=metrics or {"accuracy": 0.8, "flags": 3.0}, details=["one detail"],
                     cost={"analyze": StageCost(calls=2, tokens=9000, seconds=6.0)})


def test_scorecard_save_load_and_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(scorecard_module, "RESULTS_DIR", tmp_path)
    json_path, md_path = save_scorecard(card())
    assert load_scorecard(json_path) == card()
    md = md_path.read_text(encoding="utf-8")
    assert "| accuracy | 0.800 |" in md and "| flags | 3 |" in md
    assert "| analyze | 2 | 9,000 | 6 |" in md and "- one detail" in md
    assert render_scorecard(card()) == md


def test_compare_shows_changes_and_changed_settings():
    old = card(1, {"accuracy": 0.8, "flags": 3.0})
    new = card(2, {"accuracy": 0.9, "recall": 0.7}, {"analysis_version": "5.4"})
    text = compare(old, new)
    assert "| accuracy | 0.800 | 0.900 | +0.1 |" in text
    assert "| flags | 3 | – |  |" in text and "| recall | – | 0.700 |  |" in text
    assert "- analysis_version: `5.3` → `5.4`" in text


# ---------- runner ----------

async def test_run_suite_runs_inside_the_scope_and_builds_a_card(monkeypatch):
    seen = {}

    @suite("test-only")
    async def test_suite(cases, services):
        seen["run_id"] = current_run_id()
        return {"cases": float(len(cases))}, ["ok"]

    async def no_cost(run_id):
        return {}

    monkeypatch.setattr(runner_module, "cost_of", no_cost)
    try:
        result = await run_suite("test-only", [case()], Services(analysis=None))
    finally:
        SUITES.pop("test-only")
    assert seen["run_id"] == result.run_id and result.metrics == {"cases": 1.0}
    assert result.cases == ["demo"] and "analysis_version" in result.settings


async def test_unknown_suite_lists_the_known_ones():
    with pytest.raises(KeyError, match="smoke"):
        await run_suite("nope", [case()], Services(analysis=None))
