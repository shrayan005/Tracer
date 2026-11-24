"""Grounding suites (step 6.3) with a scripted judge: no tokens, no database."""

import json
from datetime import datetime, timezone

import pytest
import yaml

from app.domain.schemas import AnalyzedPaper, Claim, PaperRecord, TrustAssessment
from app.evals import grounding as g
from app.evals import labels as labels_module
from app.evals import scorecard as scorecard_module
from app.evals.cases import CasePaper, EvalCase
from app.evals.labels import load_labels, make_label_file
from app.evals.runner import Services
from app.integrations.llm.gateway import LlmGateway
from tests.fakes import FakeProvider

RESULT = "Peak torque increased after training (p = 0.040)."
INTRO = ("Resistance training increases strength. Previous studies have shown that triceps strength rises "
         "after eight weeks of training (Smith et al., 2019).")
SECTIONS = [("introduction", "Introduction", INTRO), ("results", "Results", f"Ten women trained.\n\n{RESULT}")]
FACTS = ("control_group", "randomized", "blinded", "statistics_reported", "effect_sizes",
         "conflicts_declared", "data_available")


def claim(quote=RESULT, direction="increase") -> Claim:
    return Claim(factor="training (before vs after)", outcome="peak torque", direction=direction,
                 population="10 untrained women", magnitude="p = 0.040", quote=quote)


def analysed(*claims) -> AnalyzedPaper:
    appraisal = {"sample_size": 10, "sex": "female", "duration": "under 3 months",
                 **{f: {"value": "not reported"} for f in FACTS}}
    return AnalyzedPaper(based_on="full text", analysis={"appraisal": appraisal,
                                                          "claims": [c.model_dump() for c in claims]})


def case_paper(key=1, sections=SECTIONS) -> CasePaper:
    return CasePaper(key=key, label="(Roberts et al., 2026)", paper=PaperRecord(source="db", title="T"),
                     study_type="lab_experiment", sections=sections,
                     base_trust=TrustAssessment(level=3, label="moderate", reasons=[]))


def item(c=None) -> g.Item:
    return g.Item("demo-1-1", "demo", case_paper(), c or claim(), "results", RESULT)


# ---------- finding paragraphs and planting errors ----------

def test_locate_finds_the_paragraph_or_nothing():
    assert g.locate(RESULT, SECTIONS, None) == ("results", RESULT)
    assert g.locate("Ten women trained.", SECTIONS, "Ten women trained. Abstract.")[0] == "abstract"
    assert g.locate("This sentence is nowhere in the paper at all.", SECTIONS, None) is None


def test_planted_errors_have_known_answers():
    fakes = {f.planted: f for f in g.plant_errors(item())}
    assert fakes["opposite"].claim.direction == "decrease"
    assert fakes["wrong population"].claim.population == g.WRONG_POPULATION
    assert "Previous studies" in fakes["other study"].claim.quote and fakes["other study"].section == "introduction"
    no_intro = g.Item("x", "demo", case_paper(sections=SECTIONS[1:]), claim(), "results", RESULT)
    assert {f.planted for f in g.plant_errors(no_intro)} == {"opposite", "wrong population"}


def test_item_text_puts_the_direction_first_and_cuts_long_paragraphs():
    block = g.item_block(3, item(claim(direction="decrease")))
    assert 'CLAIM SAYS: peak torque DECREASED (p = 0.040)' in block and '<item id="3">' in block
    long = g.Item("x", "demo", case_paper(), claim(), "results", "A" * 3000 + RESULT + "B" * 3000)
    cut = g.item_block(1, long)
    assert "Peak torque increased" in cut and len(cut) < 2000


def test_the_judge_is_told_when_a_paper_is_a_review():
    assert "PAPER TYPE: original study" in g.item_block(1, item())
    review = g.Item("x", "demo", case_paper().model_copy(update={"study_type": "meta_analysis"}), claim(),
                    "abstract", RESULT)
    assert "PAPER TYPE: meta-analysis (a review: its pooled findings and conclusions are its own result)" \
        in g.item_block(1, review)
    assert g.paper_type("other") == "unknown"


# ---------- the grounding suite with a scripted judge ----------

class ScriptedJudge(LlmGateway):
    """Reads the item ids from the prompt and answers by what each item looks like."""
    def __init__(self):
        super().__init__([FakeProvider("judge", [])])

    async def complete_structured(self, messages, schema, **kwargs):
        prompt = messages[-1]["content"]
        answers = []
        for block in prompt.split("<item id=")[1:]:
            i = int(block.split('"')[1])
            if "DECREASED" in block:
                verdict = "supports"                              # the judge MISSES the flipped claim
            elif g.WRONG_POPULATION in block:
                verdict = "partly"
            elif "Previous studies" in block:
                verdict = "other study"
            elif "muscle thickness" in block:
                verdict = "does not support"
            else:
                verdict = "supports"
            answers.append({"id": i, "verdict": verdict, "reason": "scripted"})
        return schema.model_validate({"items": answers}), None


def demo_case() -> EvalCase:
    return EvalCase(name="demo", topic_id=1, topic="t", brief="b", field_family="general",
                    frozen_at=datetime(2026, 11, 20, tzinfo=timezone.utc), papers=[case_paper()])


async def test_grounding_suite_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(scorecard_module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(g, "RESULTS_DIR", tmp_path)
    wrong = claim(quote="Ten women trained.").model_copy(update={"outcome": "muscle thickness"})
    missing = claim(quote="A sentence that the paper never contains anywhere at all.")

    async def stored(keys):
        return {1: analysed(claim(), wrong, missing)}
    monkeypatch.setattr(g, "stored_analyses", stored)

    metrics, details = await g.grounding([demo_case()], Services(analysis=None, judge=ScriptedJudge()))
    assert metrics["claims"] == 3
    assert metrics["supported_share"] == pytest.approx(1 / 3)
    assert metrics["unsupported_share"] == pytest.approx(1 / 3)
    assert metrics["quote_not_found_share"] == pytest.approx(1 / 3)
    assert metrics["planted_opposite_caught"] == 0.0              # the scripted judge misses flips
    assert metrics["planted_population_caught"] == 1.0 and metrics["planted_other_study_caught"] == 1.0
    assert metrics["planted_caught_share"] == pytest.approx(2 / 3)
    assert any("MISSED planted opposite" in d for d in details)
    assert any("does not support: muscle thickness" in d for d in details)
    items = yaml.safe_load(next(tmp_path.glob("*-grounding-items.yaml")).read_text(encoding="utf-8"))
    assert len(items) == 2 and {i["judge"] for i in items} == {"supports", "does not support"}


async def test_no_planted_errors_in_narrative_reviews(monkeypatch, tmp_path):
    monkeypatch.setattr(g, "RESULTS_DIR", tmp_path)

    async def stored(keys):
        return {1: analysed(claim())}
    monkeypatch.setattr(g, "stored_analyses", stored)
    review = demo_case()
    review.papers[0].study_type = "narrative_review"
    metrics, _ = await g.grounding([review], Services(analysis=None, judge=ScriptedJudge()))
    assert metrics["claims"] == 1 and metrics["planted_caught_share"] == 0.0
    assert "planted_opposite_caught" not in metrics                 # nothing planted, nothing measured


async def test_grounding_needs_a_judge():
    with pytest.raises(ValueError, match="judge"):
        await g.grounding([demo_case()], Services(analysis=None))


# ---------- labels + judge-check ----------

def items_file(tmp_path, n_supports=5, n_doubtful=2):
    rows = [{"id": f"i{k}", "case": "demo", "paper": "(A, 2020)", "claim": "c", "quote": RESULT,
             "section": "results", "paragraph": RESULT, "judge": "supports" if k < n_supports else "partly",
             "judge_reason": "r", "claim_data": claim().model_dump()} for k in range(n_supports + n_doubtful)]
    path = tmp_path / "items.yaml"
    path.write_text(yaml.safe_dump(rows), encoding="utf-8")
    return path


def test_label_file_keeps_every_doubtful_item_and_protects_your_work(tmp_path):
    out = tmp_path / "gold" / "judge_check.yaml"
    path, n = make_label_file(items_file(tmp_path), out=out, sample=4)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    judges = [i["judge"] for i in data["items"]]
    assert n == 4 and judges.count("partly") == 2               # both doubtful items are always included
    assert all(i["human"] is None for i in data["items"]) and "how_to" in data
    with pytest.raises(FileExistsError, match="--force"):
        make_label_file(items_file(tmp_path), out=out)


def test_load_labels_errors(tmp_path):
    out = tmp_path / "judge_check.yaml"
    with pytest.raises(FileNotFoundError, match="python -m app.evals label"):
        load_labels(out)
    make_label_file(items_file(tmp_path), out=out, sample=3)
    with pytest.raises(ValueError, match="3 of 3 items still have no `human:` value"):
        load_labels(out)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    for row in data["items"]:
        row["human"] = "supprts"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown verdict 'supprts'"):
        load_labels(out)


async def test_judge_check_agreement(tmp_path, monkeypatch):
    out = tmp_path / "judge_check.yaml"
    make_label_file(items_file(tmp_path, n_supports=3, n_doubtful=1), out=out, sample=4)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    for k, row in enumerate(data["items"]):
        row["human"] = "supports" if k < 3 else "opposite"       # judge will say supports to all 4
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(labels_module, "JUDGE_CHECK", out)
    monkeypatch.setattr(g, "load_labels", lambda: load_labels(out))
    metrics, details = await g.judge_check([], Services(analysis=None, judge=ScriptedJudge()))
    assert metrics["agreement"] == 0.75 and metrics["agreement_supported_or_not"] == 0.75
    assert len(details) == 1 and "you said opposite, judge said supports" in details[0]


def test_judge_gateway_has_only_the_judge_provider(monkeypatch):
    from app import dependencies
    from app.config import get_settings

    class Fake:
        def __init__(self, name):
            self.name = name
    monkeypatch.setattr(dependencies, "build_providers", lambda s: [Fake("groq"), Fake("gemini")])
    dependencies.get_judge_gateway.cache_clear()
    try:
        gateway = dependencies.get_judge_gateway()
        assert [p.name for p in gateway.providers] == [get_settings().judge_provider]
    finally:
        dependencies.get_judge_gateway.cache_clear()
