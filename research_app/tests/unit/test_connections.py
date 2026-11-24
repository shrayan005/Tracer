"""Connections between papers (step 5.3): code rules, validation of the LLM's groups, and the service."""

import json

import pytest

from app.domain.errors import ProviderRejected
from app.domain.schemas import AnalyzedPaper, ClaimGroup, Connections, GroupMember, PaperAnalysis
from app.integrations.llm.gateway import LlmGateway
from app.services.connection_service import (ConnectionService, candidate_groups, candidate_pairs, factor_kind, format_data,
                                             relation_of, validate)
from tests.fakes import FakeProvider

NA = {"value": "not reported"}


def analyzed(*claims) -> AnalyzedPaper:
    """claims: (factor, outcome, direction)"""
    appraisal = {"sample_size": 10, "sex": "mixed", "duration": "single session",
                 **{f: NA for f in ("control_group", "randomized", "blinded", "statistics_reported",
                                    "effect_sizes", "conflicts_declared", "data_available")}}
    return AnalyzedPaper(based_on="full text", analysis=PaperAnalysis(appraisal=appraisal, claims=[
        {"factor": f, "outcome": o, "direction": d, "population": "adults", "quote": "q"} for f, o, d in claims]))


# keys 101..105 stand for database ids; in the prompt they become P1..P5
PAPERS = [
    (101, "(Roberts et al., 2026)", "randomized_trial", analyzed(
        ("training (before vs after)", "peak torque", "increase"),
        ("training (before vs after)", "muscle thickness", "no difference"))),
    (102, "(Stasinaki et al., 2018)", "lab_experiment", analyzed(
        ("short vs long fascicle training (before vs after)", "1-RM elbow extension", "increase"),
        ("short vs long fascicle training (before vs after)", "muscle thickness at 50% length", "increase"))),
    (103, "(Popadić-Gaćeša et al., 2013)", "lab_experiment", analyzed(
        ("resistance training (before vs after)", "endurance strength", "increase"),
        ("resistance training (before vs after)", "decrease in strength", "decrease"))),
    (104, "(Hussain et al., 2020)", "lab_experiment", analyzed(
        ("exercise intensity (30% vs 60% 1RM)", "rate of fatigue", "increase"))),
    (105, "(Akima et al., 2021)", "lab_experiment", analyzed(
        ("TB-Long vs TB-Lat", "RMS EMG", "decrease"))),
]


def group(question, *members, why="different protocols"):
    return ClaimGroup(question=question, finding="f", why_differ=why,
                      members=[GroupMember(claim_id=c, stance=s) for c, s in members])


# ---------- small rules ----------

@pytest.mark.parametrize("stances, relation", [
    (["supports", "supports"], "agree"), (["no effect", "no effect"], "agree"),
    (["supports", "opposes"], "contradict"), (["supports", "no effect"], "mixed"),
    (["opposes", "no effect", "supports"], "contradict"),
])
def test_relation_is_computed_from_stances(stances, relation):
    assert relation_of(stances) == relation


@pytest.mark.parametrize("factor, kind", [
    ("training (before vs after)", "a change over time"), ("narrow vs wide grip", "a comparison"),
    ("body mass", "a relationship")])
def test_factor_kind(factor, kind):
    assert factor_kind(factor) == kind


def test_format_data_uses_positions_not_database_ids():
    text, claims = format_data(PAPERS)
    assert "P1 (Roberts et al., 2026): randomized trial, n=10" in text
    assert "  P1.C2 = training (before vs after) → muscle thickness | adults" in text
    assert claims["P3.C1"]["paper"] == 103 and claims["P3.C1"]["claim"] == 0


def test_candidate_pairs_match_outcome_words_across_papers_only():
    _, claims = format_data(PAPERS)
    pairs = candidate_pairs(claims)
    assert any(p.startswith("P1.C2 ↔ P2.C2") and "muscle, thickness" in p for p in pairs)
    assert not any(p.startswith("P3.C1 ↔ P3.C2") for p in pairs)            # never within one paper


def test_candidate_groups_join_different_words_for_the_same_kind_of_outcome():
    # measured (6.6): "peak torque" and "1-RM", "muscle mass" and "lean tissue mass" share no word
    claims = {"P1.C1": {"paper": 1, "outcome": "peak torque"}, "P2.C1": {"paper": 2, "outcome": "elbow extension 1-RM"},
              "P2.C2": {"paper": 2, "outcome": "lean tissue mass"}, "P3.C1": {"paper": 3, "outcome": "muscle mass"},
              "P3.C2": {"paper": 3, "outcome": "quality of life"}}
    groups = candidate_groups(claims)
    assert any(g.startswith("strength") and g.endswith("P1.C1, P2.C1") for g in groups)
    assert any(g.startswith("muscle size") and g.endswith("P2.C2, P3.C1") for g in groups)
    assert not any("P3.C2" in g for g in groups)
    assert candidate_groups({"P1.C1": {"paper": 1, "outcome": "peak torque"},
                             "P1.C2": {"paper": 1, "outcome": "1-RM"}}) == ["(none)"]   # one paper: no group


# ---------- validation ----------

def check(*groups):
    _, claims = format_data(PAPERS)
    return validate(Connections(groups=list(groups)), claims)


def test_agree_group_points_at_real_papers_and_claims():
    result, flags = check(group("Does training increase strength?",
                                ("P1.C1", "supports"), ("P2.C1", "supports"), ("P3.C1", "supports")))
    assert flags == [] and result[0].relation == "agree" and result[0].why_differ is None
    assert [(m.paper, m.claim) for m in result[0].members] == [(101, 0), (102, 0), (103, 0)]


def test_no_difference_claim_is_forced_to_no_effect_giving_mixed():
    result, flags = check(group("Does training increase muscle thickness?",
                                ("P1.C2", "opposes"), ("P2.C2", "supports")))
    assert result[0].relation == "mixed"
    assert flags == ["P1.C2: found no difference, so stance set to 'no effect' (was opposes)"]


def test_paper_on_both_sides_loses_its_outlier_claim():
    result, flags = check(group("Does training increase strength?",
                                ("P1.C1", "supports"), ("P3.C1", "supports"), ("P3.C2", "opposes")))
    assert result[0].relation == "agree" and len(result[0].members) == 2
    assert flags == ["P3.C2 dropped: its paper is on both sides of this question (probably a badly worded claim)"]


def test_disagreement_between_different_kinds_of_factor_is_not_a_contradiction():
    result, flags = check(group("Does more load increase fatigue?", ("P4.C1", "supports"), ("P3.C2", "opposes")))
    assert result == []
    assert "a different kind of question" in flags[0] and "fewer than 2 papers left" in flags[1]


def test_only_the_odd_kind_of_question_is_dropped_and_the_disagreement_kept():
    # measured (6.6): dropping the WHOLE group also threw away a real disagreement between the others
    result, flags = check(group("Does the first condition raise the outcome?", ("P4.C1", "supports"),
                                ("P5.C1", "opposes"), ("P1.C1", "supports")))
    assert len(result) == 1 and result[0].relation == "contradict"
    assert [m.paper for m in result[0].members] == [104, 105]
    assert flags[0].startswith("P1.C1 dropped from") and "(a comparison)" in flags[0]


def test_housekeeping_checks():
    result, flags = check(
        group("A?", ("P1.C1", "supports"), ("P9.C9", "supports")),                      # unknown id -> 1 paper
        group("B?", ("P1.C1", "supports"), ("P2.C1", "supports")),
        group("C?", ("P1.C1", "supports"), ("P3.C1", "supports")),                      # P1.C1 used already
        group("D?", ("P1.C2", "no effect"), ("P2.C2", "supports"), why=None))           # mixed without reason
    assert [c.question for c in result] == ["B?", "D?"]
    assert "unknown claim id P9.C9 dropped" in flags
    assert "P1.C1 already used in another group: dropped here" in flags
    assert "group 'D?': papers disagree but no reason was given" in flags


# ---------- the service ----------

async def no_sleep(_s):
    pass


async def test_service_maps_prompt_ids_back_to_paper_keys():
    reply = json.dumps({"groups": [{"question": "Does training increase strength?", "finding": "yes",
                                    "members": [{"claim_id": "P1.C1", "stance": "supports"},
                                                {"claim_id": "P2.C1", "stance": "supports"}]}]})
    provider = FakeProvider("a", [reply])
    connections, flags = await ConnectionService(LlmGateway([provider], sleep=no_sleep)).connect("brief", PAPERS)
    assert flags == [] and [m.paper for m in connections[0].members] == [101, 102]
    prompt = provider.last_kwargs["messages"][-1]["content"]      # the gateway puts its schema note first
    assert "Candidate pairs" in prompt and "P1.C2 ↔ P2.C2" in prompt


async def test_service_failure_and_too_few_papers():
    failing = ConnectionService(LlmGateway([FakeProvider("a", [ProviderRejected("a", "bad")])], sleep=no_sleep))
    assert await failing.connect("brief", PAPERS) == ([], ["could not connect the papers (LLM call failed)"])
    assert await failing.connect("brief", PAPERS[:1]) == ([], [])


# ---------- readers see citations, not prompt ids (step 5.6) ----------

def test_prompt_ids_are_replaced_with_citations():
    from app.services.connection_service import name_papers
    labels = {"P2": "(Roberts et al., 2026)", "P7": "(Stasinaki et al., 2018)"}
    assert name_papers("P2 used 10 women, whereas P7.C1 used 9.", labels) == \
        "(Roberts et al., 2026) used 10 women, whereas (Stasinaki et al., 2018) used 9."
    assert name_papers("P9 and P12 are unknown; PDF stays.", labels) == "P9 and P12 are unknown; PDF stays."


async def test_service_names_papers_in_finding_and_explanation():
    reply = json.dumps({"groups": [{"question": "Does training increase strength?",
                                    "finding": "P1 and P2 both found gains", "why_differ": "P1 was shorter",
                                    "members": [{"claim_id": "P1.C1", "stance": "supports"},
                                                {"claim_id": "P2.C1", "stance": "supports"}]}]})
    connections, _ = await ConnectionService(
        LlmGateway([FakeProvider("a", [reply])], sleep=no_sleep)).connect("brief", PAPERS)
    assert connections[0].finding == "(Roberts et al., 2026) and (Stasinaki et al., 2018) both found gains"
    assert connections[0].why_differ is None                      # agree: no "why they differ"
    assert connections[0].members[0].claim_snapshot.outcome == "peak torque"
