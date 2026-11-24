"""Stability suite (step 6.2) with scripted fake services: no tokens, no database."""

from datetime import datetime, timezone

from app.domain.schemas import (AnalyzedPaper, Claim, Connection, ConnectionMember, PaperRecord,
                                TrustAssessment)
from app.evals.cases import CasePaper, EvalCase
from app.evals.runner import Services, cost_key
from app.evals.stability import claim_overlap, claims_match, connection_pairs, jaccard, stability

FACTS = ("control_group", "randomized", "blinded", "statistics_reported", "effect_sizes",
         "conflicts_declared", "data_available")


def claim(factor="training (before vs after)", outcome="peak torque", direction="increase") -> Claim:
    return Claim(factor=factor, outcome=outcome, direction=direction, population="adults", quote="q")


def analysed(claims=None, n=20, **facts) -> AnalyzedPaper:
    appraisal = {"sample_size": n, "sex": "mixed", "duration": "single session",
                 **{f: {"value": facts.get(f, "not reported")} for f in FACTS}}
    return AnalyzedPaper(based_on="full text", analysis={
        "appraisal": appraisal, "claims": [c.model_dump() for c in (claims or [claim()])]})


# ---------- the comparison helpers ----------

def test_claims_match():
    assert claims_match(claim(), claim(outcome="peak torque of elbow extension"))
    assert not claims_match(claim(), claim(direction="decrease"))
    assert not claims_match(claim(), claim(factor="grip width", outcome="EMG amplitude"))


def test_claim_overlap():
    one = analysed([claim(), claim(outcome="muscle thickness", direction="no difference")])
    assert claim_overlap(one, one) == 1.0
    extra = analysed([claim(), claim(outcome="muscle thickness", direction="no difference"),
                      claim(factor="grip width", outcome="EMG amplitude")])
    assert claim_overlap(one, extra) == (1.0 + 2 / 3) / 2


def test_connection_pairs_and_jaccard():
    def group(*members):
        return Connection(question="q", relation="agree", finding="f",
                          members=[ConnectionMember(paper=p, claim=c, stance="supports") for p, c in members])
    a = [group((1, 0), (2, 0), (3, 0))]
    same_shuffled = [group((3, 0), (1, 0), (2, 0))]
    one_changed = [group((1, 0), (2, 0), (4, 0))]
    assert jaccard(connection_pairs(a), connection_pairs(same_shuffled)) == 1.0
    assert jaccard(connection_pairs(a), connection_pairs(one_changed)) == 1 / 5     # 1 shared pair of 5
    assert jaccard(set(), set()) == 1.0


# ---------- the suite with scripted fakes ----------

class ScriptedAnalyzer:
    """Returns run 1, run 2, ... in turn: lets a test decide exactly what changes between runs."""
    def __init__(self, runs):
        self.runs, self.calls = list(runs), 0

    async def analyze(self, papers, texts=None):
        result = self.runs[self.calls]
        self.calls += 1
        return result, []


class SameConnector:
    async def connect(self, brief, papers):
        return [Connection(question="q", relation="agree", finding="f", members=[
            ConnectionMember(paper=key, claim=0, stance="supports") for key, *_ in papers])], []


def case() -> EvalCase:
    trust = TrustAssessment(level=3, label="moderate", reasons=["lab experiment"])
    papers = [CasePaper(key=k, label=f"(P{k}, 2020)", paper=PaperRecord(source="db", title=f"P{k}"),
                        study_type="lab_experiment", base_trust=trust) for k in (1, 2)]
    return EvalCase(name="demo", topic_id=1, topic="t", brief="b", field_family="sport_biomechanics",
                    frozen_at=datetime(2026, 11, 20, tzinfo=timezone.utc), papers=papers)


async def test_suite_measures_what_changed():
    decent = {"control_group": "yes", "randomized": "yes", "statistics_reported": "yes", "effect_sizes": "yes"}
    run_1 = {1: analysed(blinded="not reported"), 2: analysed(n=25, **decent)}   # 4.5/6.5: as expected -> 3
    run_2 = {1: analysed(blinded="yes"), 2: analysed(n=8, **decent)}             # n=8 red flag -> 2
    services = Services(analysis=ScriptedAnalyzer([run_1, run_2]), connection=SameConnector(), runs=2)
    metrics, details = await stability([case()], services)
    assert metrics["facts_identical_share"] == 18 / 20                # blinded (paper 1) + sample size (paper 2)
    assert metrics["trust_identical_share"] == 0.5                    # paper 2: 3 -> 2
    assert metrics["claim_overlap_mean"] == 1.0
    assert metrics["connection_overlap_mean"] == 1.0 and metrics["connection_groups_mean"] == 1.0
    assert "demo · (P1, 2020) · blinded: not reported / yes" in details
    assert "demo · (P2, 2020) · trust level: 3 / 2" in details
    assert details[0].startswith("most unstable facts: ")


async def test_single_run_compares_nothing_safely():
    services = Services(analysis=ScriptedAnalyzer([{1: analysed(), 2: analysed()}]), runs=1)
    metrics, _ = await stability([case()], services)
    assert metrics["papers"] == 0 and metrics["facts_identical_share"] == 0.0
    assert "connection_overlap_mean" not in metrics                   # no connector given


def test_cost_key_names_purpose_and_provider():
    assert cost_key("analyze", "gemini") == "analyze · gemini"
