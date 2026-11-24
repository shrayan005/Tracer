"""Appraisal suite (step 6.4): hints, the label file, matching rules and metrics. No tokens, no database."""

from datetime import datetime, timezone

import pytest
import yaml

from app.domain.schemas import AnalyzedPaper, PaperRecord, TrustAssessment
from app.evals import appraisal as a
from app.evals import labels as labels_module
from app.evals.cases import CasePaper, EvalCase
from app.evals.labels import load_appraisal_gold
from app.evals.runner import Services

METHODS = ("Nine young, novice, female participants trained for 6 weeks. Participants were randomly assigned "
           "to two groups. Partial eta squared was used as an indicator of effect size. Data were analysed with "
           "ANOVA (p < 0.05).")
SECTIONS = [("methods", "Methods", METHODS), ("backmatter", "Conflicts", "The authors declare no conflict of interest."),
            ("references", "References", "1. Smith J. Randomly assigned trials. 2019.")]


def analysed(**values) -> AnalyzedPaper:
    facts = {f: {"value": values.get(f, "not reported"), "quote": "q" if values.get(f) == "yes" else None}
             for f in a.CHECKLIST}
    appraisal = {"sample_size": values.get("n", 9), "sample_size_quote": "Nine participants", "sex": "female",
                 "duration": "under 3 months", **facts}
    return AnalyzedPaper(based_on="full text", analysis={"appraisal": appraisal, "claims": [
        {"factor": "f", "outcome": "o", "direction": "increase", "population": "p", "quote": "q"}]})


def case(sections=SECTIONS) -> EvalCase:
    trust = TrustAssessment(level=3, label="moderate", reasons=["lab experiment"])
    papers = [CasePaper(key=1, label="(Stasinaki et al., 2018)", study_type="lab_experiment", base_trust=trust,
                        paper=PaperRecord(source="db", title="Triceps training", doi="10.3390/jfmk"), sections=sections),
              CasePaper(key=2, label="(Abstract only, 2020)", study_type="lab_experiment", base_trust=trust,
                        paper=PaperRecord(source="db", title="Other"))]
    return EvalCase(name="demo", topic_id=1, topic="t", brief="b", field_family="sport_biomechanics",
                    frozen_at=datetime(2026, 11, 20, tzinfo=timezone.utc), papers=papers)


# ---------- hints and matching ----------

def test_hints_find_the_deciding_sentences():
    assert a.hints(SECTIONS, "sample_size")[0].startswith("[methods] Nine young, novice, female participants")
    assert "randomly assigned to two groups" in a.hints(SECTIONS, "randomized")[0]
    assert all("[references]" not in h for h in a.hints(SECTIONS, "randomized"))     # references skipped
    assert "Partial eta squared" in a.hints(SECTIONS, "effect_sizes")[0]
    assert a.hints(SECTIONS, "conflicts_declared") == ["[backmatter] The authors declare no conflict of interest."]
    assert a.hints(SECTIONS, "blinded") == []


def test_matching_rules():
    assert a.same("randomized", "yes", "yes")
    assert a.same("blinded", "no", "not reported")                  # score the same in the quality check
    assert not a.same("effect_sizes", "not reported", "yes")
    assert a.same("sample_size", 17, 18) and not a.same("sample_size", 17, 20)
    assert a.same("sample_size", None, None) and not a.same("sample_size", 9, None)


# ---------- the label file ----------

@pytest.fixture
def gold_file(tmp_path, monkeypatch):
    path = tmp_path / "appraisal.yaml"
    monkeypatch.setattr(labels_module, "APPRAISAL", path)
    monkeypatch.setattr(a, "APPRAISAL", path)
    monkeypatch.setattr(a, "load_appraisal_gold", lambda: load_appraisal_gold(path))
    return path


def stored(analyses):
    async def claims_for(case, services):
        return analyses, []
    return claims_for


async def test_prefill_writes_full_text_papers_and_protects_your_file(gold_file, monkeypatch):
    monkeypatch.setattr(a, "claims_for", stored({1: analysed(randomized="yes"), 2: analysed()}))
    path, n = await a.prefill([case()], Services(analysis=None))
    data = yaml.safe_load(gold_file.read_text(encoding="utf-8"))
    paper = data["papers"][0]
    assert n == 1 and paper["id"] == "demo-1" and paper["reviewed"] is False     # abstract-only paper left out
    assert paper["link"] == "https://doi.org/10.3390/jfmk"
    assert paper["facts"]["randomized"]["gold"] == "yes" and paper["facts"]["randomized"]["hints"]
    with pytest.raises(FileExistsError, match="--force"):
        await a.prefill([case()], Services(analysis=None))


def write_gold(path, reviewed=True, **gold):
    facts = {f: {"gold": gold.get(f, "not reported")} for f in a.CHECKLIST}
    facts |= {"sample_size": {"gold": gold.get("n", 9)}, "sex": {"gold": "female"},
              "duration": {"gold": "under 3 months"}}
    path.write_text(yaml.safe_dump({"papers": [{"id": "demo-1", "reviewed": reviewed, "facts": facts}]}),
                    encoding="utf-8")


def test_gold_loading_errors_and_yaml_booleans(gold_file):
    with pytest.raises(FileNotFoundError, match="prefill appraisal"):
        load_appraisal_gold(gold_file)
    write_gold(gold_file, reviewed=False)
    with pytest.raises(ValueError, match="reviewed: true"):
        load_appraisal_gold(gold_file)
    gold_file.write_text("papers:\n- id: demo-1\n  reviewed: true\n  facts:\n    randomized: {gold: yes}\n"
                         "    blinded: {gold: no}\n", encoding="utf-8")        # unquoted yes/no -> booleans
    assert load_appraisal_gold(gold_file)[0]["facts"] == {"randomized": "yes", "blinded": "no"}
    write_gold(gold_file, randomized="yse")
    with pytest.raises(ValueError, match="unknown value 'yse'"):
        load_appraisal_gold(gold_file)


# ---------- the suite ----------

async def test_suite_counts_false_and_missed_yes_and_trust_impact(gold_file, monkeypatch):
    # truth: no control group, effect sizes reported, conflicts stated
    write_gold(gold_file, randomized="yes", statistics_reported="yes", effect_sizes="yes",
               conflicts_declared="yes", n=9)
    # AI: claims a control group (false ✓), misses effect sizes and conflicts (missed ✓)
    ai = analysed(control_group="yes", randomized="yes", statistics_reported="yes", n=9)
    monkeypatch.setattr(a, "claims_for", stored({1: ai}))
    metrics, details = await a.appraisal([case()], Services(analysis=None))
    assert metrics["papers"] == 1 and metrics["facts"] == 10
    assert metrics["false_yes"] == 1 and metrics["false_yes_rate"] == pytest.approx(1 / 3)
    assert metrics["missed_yes"] == 2 and metrics["missed_yes_rate"] == pytest.approx(2 / 4)
    assert metrics["fact_accuracy"] == pytest.approx(7 / 10)
    assert metrics["accuracy_control_group"] == 0.0 and metrics["accuracy_randomized"] == 1.0
    assert "demo · (Stasinaki et al., 2018) · control_group: AI yes, you not reported" in details
    assert metrics["trust_agreement"] == 1.0                # n=9 red flag dominates both: trust 2 either way


async def test_a_false_yes_that_lifts_trust_is_detected(gold_file, monkeypatch):
    write_gold(gold_file, n=40)                                         # truth: a weakly reported study
    ai = analysed(n=40, **{f: "yes" for f in a.CHECKLIST})               # AI: everything ✓
    monkeypatch.setattr(a, "claims_for", stored({1: ai}))
    metrics, details = await a.appraisal([case()], Services(analysis=None))
    assert metrics["false_yes"] == 7 and metrics["trust_agreement"] == 0.0
    assert any("TRUST: AI's facts give 4, yours give 2" in d for d in details)
