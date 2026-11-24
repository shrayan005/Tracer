"""Per-paper analysis (step 5.1): schema, quote verification and the service with a fake LLM."""

import json

import pytest
from pydantic import ValidationError

from app.domain.errors import ProviderRejected
from app.domain.schemas import PaperAnalysis, PaperRecord
from app.integrations.llm.gateway import LlmGateway
from app.services.analysis_service import ANALYSIS_BUDGET, AnalysisService, verify
from app.services.fulltext_context import build_context, paper_text
from tests.fakes import FakeProvider

SOURCE = ("Ten untrained women trained for four weeks. Peak torque increased after training (p = 0.040). "
          "Participants were randomly assigned to the two groups. The authors declare no conflict of interest.")
CLAIM_QUOTE = "Peak torque increased after training (p = 0.040)"


def fact(value="not reported", quote=None):
    return {"value": value, "quote": quote}


def analysis_json(claims=None, **facts) -> str:
    appraisal = {"sample_size": 10, "sample_size_quote": "Ten untrained women trained for four weeks",
                 "sex": "female", "duration": "under 3 months",
                 **{name: fact() for name in ("control_group", "randomized", "blinded", "statistics_reported",
                                              "effect_sizes", "conflicts_declared", "data_available")}}
    appraisal.update(facts)
    claims = claims if claims is not None else [
        {"outcome": "peak torque", "factor": "training (before vs after)", "population": "10 untrained women",
         "direction": "increase", "magnitude": "p = 0.040", "quote": CLAIM_QUOTE}]
    return json.dumps({"claims": claims, "appraisal": appraisal})


# ---------- schema ----------

def test_schema_limits():
    PaperAnalysis.model_validate_json(analysis_json())
    with pytest.raises(ValidationError):
        PaperAnalysis.model_validate_json(analysis_json(claims=[]))                      # at least one claim
    bad = json.loads(analysis_json())
    bad["claims"][0]["direction"] = "much bigger"
    with pytest.raises(ValidationError):
        PaperAnalysis.model_validate(bad)                                                # only known directions


# ---------- verify ----------

def test_verified_analysis_has_no_flags():
    analysis = PaperAnalysis.model_validate_json(analysis_json(
        randomized=fact("yes", "Participants were randomly assigned to the two groups")))
    assert verify(analysis, SOURCE) == []
    assert analysis.appraisal.randomized.value == "yes"


def test_invented_claim_quote_is_flagged_but_kept():
    analysis = PaperAnalysis.model_validate_json(analysis_json(claims=[
        {"outcome": "x", "factor": "y", "population": "z", "direction": "increase",
         "quote": "Strength doubled in every single participant of the study"}]))
    assert verify(analysis, SOURCE) == ["claim 1: quote not found in the paper"]
    assert len(analysis.claims) == 1


def test_yes_without_quote_becomes_not_reported_with_a_flag():
    analysis = PaperAnalysis.model_validate_json(analysis_json(blinded=fact("yes", "Assessors were blinded")))
    assert verify(analysis, SOURCE) == ["blinded: 'yes' not supported by a quote -> not reported"]
    assert analysis.appraisal.blinded.value == "not reported"


def test_no_without_quote_quietly_becomes_not_reported_but_quoted_no_stays():
    analysis = PaperAnalysis.model_validate_json(analysis_json(
        control_group=fact("no"),
        conflicts_declared=fact("no", "The authors declare no conflict of interest")))
    assert verify(analysis, SOURCE) == []
    assert analysis.appraisal.control_group.value == "not reported"
    assert analysis.appraisal.conflicts_declared.value == "no"


def test_sample_size_without_quote_is_flagged():
    analysis = PaperAnalysis.model_validate_json(analysis_json(sample_size=25, sample_size_quote=None))
    assert verify(analysis, SOURCE) == ["sample size 25: no supporting quote"]


# ---------- the service ----------

def make_service(script):
    provider = FakeProvider("a", script)

    async def no_sleep(_s):
        pass

    return AnalysisService(LlmGateway([provider], max_retries=0, sleep=no_sleep)), provider


def paper(abstract=SOURCE, title="Training study"):
    return PaperRecord(source="x", title=title, abstract=abstract)


async def test_full_text_uses_the_analysis_budget_with_backmatter():
    sections = [("methods", "Methods", "Ten untrained women trained for four weeks."),
                ("results", "Results", "Peak torque increased after training (p = 0.040)."),
                ("backmatter", "Conflicts", "The authors declare no conflict of interest."),
                ("references", "References", "1. Smith")]
    service, provider = make_service([analysis_json()])
    results, warnings = await service.analyze({5: paper(abstract=None)}, {5: paper_text(sections)})
    assert warnings == [] and results[5].based_on == "full text" and results[5].flags == []
    prompt = provider.last_kwargs["messages"][1]["content"]
    assert "## Backmatter\nThe authors declare" in prompt          # in the analysis, not in summaries
    assert "1. Smith" not in prompt


async def test_abstract_only_and_no_text():
    service, provider = make_service([analysis_json()])
    results, _ = await service.analyze({1: paper(), 2: paper(abstract="too short")})
    assert results[1].based_on == "abstract"
    assert 2 not in results and provider.calls == 1                # no text: no LLM call


async def test_one_failure_does_not_stop_the_others():
    service, _ = make_service([ProviderRejected("a", "bad request"), analysis_json()])
    results, warnings = await service.analyze({1: paper(title="First"), 2: paper(title="Second")})
    assert len(results) == 1 and len(warnings) == 1 and "could not analyse" in warnings[0]


def test_build_context_with_a_custom_budget():
    sections = [("methods", "M", "m" * 50), ("backmatter", "B", "b" * 50)]
    assert "## Backmatter" not in build_context(sections)               # summary budget: no backmatter
    assert "## Backmatter" in build_context(sections, ANALYSIS_BUDGET)
