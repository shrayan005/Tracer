"""Research gaps (step 5.4): counting, gap rules, shared limitations and the checked write-up."""

import json

import pytest

from app.domain.errors import ProviderRejected
from app.integrations.llm.gateway import LlmGateway
from app.services.gap_service import GapService, PaperFacts, find_gaps, profile, shared_limitations
from tests.fakes import FakeProvider

FACT_NAMES = ("control_group", "randomized", "blinded", "statistics_reported", "effect_sizes",
              "conflicts_declared", "data_available")


def paper(label="(A, 2020)", study_type="lab_experiment", based_on="full text", n=12, sex="mixed",
          duration="single session", limitations=(), **facts) -> PaperFacts:
    values = {name: facts.get(name, "not reported") for name in FACT_NAMES}
    return PaperFacts(label=label, study_type=study_type, based_on=based_on, sample_size=n, sex=sex,
                      duration=duration, facts=values, limitations=list(limitations))


def gaps_of(papers):
    return find_gaps(profile(papers))


def test_profile_counts_and_skips_not_applicable():
    pr = profile([paper(sex="male", blinded="yes"), paper(sex="female", blinded="not applicable"),
                  paper(based_on="abstract")])
    assert pr.sexes["male"] == 1 and pr.abstract_only == 1
    assert pr.fact_applicable["blinded"] == 2 and pr.fact_yes["blinded"] == 1


def test_typical_small_lab_literature():
    papers = [paper(f"(P{i}, 2020)", n=n, sex=s) for i, (n, s) in
              enumerate([(7, "male"), (9, "female"), (12, "female"), (15, "mixed"), (20, "female")])]
    gaps = gaps_of(papers)
    assert "No randomised controlled trial among the selected papers." in gaps
    assert "No systematic review or meta-analysis pools these results." in gaps
    assert "Small samples: median n = 12 (range 7–20); 2 of 5 studies had fewer than 10 participants." in gaps
    assert "4 of 5 studies that report sex included only one sex (3 women only, 1 men only)." in gaps
    assert "No study reports blinding." in gaps                         # zero: "No study ...", not "0 of 5"


def test_rules_stay_silent_when_the_evidence_is_strong():
    papers = [paper(f"(P{i}, 2020)", study_type=t, n=60, duration="3 months or more", control_group="yes",
                    randomized="yes", blinded="yes", effect_sizes="yes", data_available="yes")
              for i, t in enumerate(["randomized_trial", "randomized_trial", "meta_analysis", "cohort"])]
    assert gaps_of(papers) == []


def test_some_but_rare_wording_and_abstract_share_is_a_note_not_a_gap():
    from app.services.gap_service import find_notes
    papers = [paper(f"(P{i}, 2020)", effect_sizes="yes" if i == 0 else "not reported",
                    based_on="abstract" if i < 2 else "full text") for i in range(6)]
    gaps = gaps_of(papers)
    assert "Effect sizes reported in only 1 of 6 studies." in gaps
    abstract_note = ("2 of 6 papers could only be judged from their abstracts (no free full text), so their "
                     "methods are less certain.")
    assert abstract_note not in gaps                                  # about this app, not the research
    assert find_notes(profile(papers)) == [abstract_note]


async def test_notes_are_kept_but_not_given_to_the_write_up():
    provider = FakeProvider("a", [json.dumps({"summary": "Samples are small.", "next_studies": []})])
    papers = [paper(f"(P{i}, 2020)", n=8, based_on="abstract") for i in range(3)]
    gaps = await GapService(LlmGateway([provider], sleep=no_sleep)).find("brief", papers)
    assert gaps.notes and "judged from their abstracts" in gaps.notes[0]
    assert "abstracts" not in provider.last_kwargs["messages"][-1]["content"]


def test_participant_rules_are_silent_for_non_human_studies():
    papers = [paper(f"(P{i}, 2020)", study_type="computational", n=None, sex="not applicable",
                    duration="not applicable") for i in range(4)]
    gaps = gaps_of(papers)
    assert not any("randomised" in g or "sex" in g or "Small samples" in g for g in gaps)


def test_shared_limitations_need_two_papers():
    papers = [paper("(A, 2020)", limitations=["The small sample size limits power."]),
              paper("(B, 2021)", limitations=["Only ten participants; a small number."]),
              paper("(C, 2022)", limitations=["Surface EMG crosstalk."])]
    shared = shared_limitations(papers)
    assert [(s.theme, s.papers) for s in shared] == [("small sample", ["(A, 2020)", "(B, 2021)"])]


# ---------- the service ----------

async def no_sleep(_s):
    pass


def service(reply):
    return GapService(LlmGateway([FakeProvider("a", [reply])], sleep=no_sleep))


PAPERS = [paper(f"(P{i}, 2020)", n=n) for i, n in enumerate([8, 12, 14])]


async def test_summary_with_an_invented_number_is_dropped():
    reply = json.dumps({"summary": "Samples are small: 37% of studies had under 10 people.", "next_studies": []})
    gaps = await service(reply).find("brief", PAPERS)
    assert gaps.write_up is None                        # 37 was never counted -> the summary is not shown
    assert gaps.flags == ["written summary dropped: it used numbers not in the counted facts (37)"]


async def test_only_the_suggested_study_with_an_invented_number_is_dropped():
    reply = json.dumps({"summary": "Samples are small (median n = 12) and no study reports blinding.",
                        "next_studies": ["A blinded study with 40 participants.", "A larger, blinded study."]})
    gaps = await service(reply).find("brief", PAPERS)
    assert gaps.write_up.summary.startswith("Samples are small")
    assert gaps.write_up.next_studies == ["A larger, blinded study."]
    assert gaps.flags == ["suggested study dropped (uses numbers not in the counted facts: 40): "
                          "A blinded study with 40 participants."]
    assert gaps.n_papers == 3 and gaps.papers_with_limitations == 0


async def test_failed_call_still_returns_the_counted_gaps():
    gaps = await service(ProviderRejected("a", "bad")).find("brief", PAPERS)
    assert gaps.gaps and gaps.write_up is None
    assert gaps.flags == ["no written summary (LLM call failed): the counted gaps are shown as they are"]
