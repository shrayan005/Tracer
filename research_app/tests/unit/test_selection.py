from datetime import date

from app.domain.schemas import PaperRecord, PaperScreening
from app.services.selection import WEIGHTS, prefer_full_text, select_best, select_diverse, score_paper
from app.services.trust import assess_trust

TODAY = date(2026, 11, 20)


def paper(title: str, abstract: str = "", **kwargs) -> PaperRecord:
    defaults = {"year": 2022, "citation_count": 20}
    return PaperRecord(source="x", title=title, abstract=abstract or title, **{**defaults, **kwargs})


def screening(i: int, relevance: int, study_type: str = "cohort") -> PaperScreening:
    return PaperScreening(id=i, relevance=relevance, reason="r", study_type=study_type)


def test_weights_add_up_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_parts_add_up_to_score():
    p = paper("Muscle fatigue", year=2020, citation_count=60, is_open_access=True)
    s = screening(0, 8, "randomized_trial")
    score, parts = score_paper(p, s, assess_trust(p, s.study_type, "biomedical", TODAY), TODAY)
    assert set(parts) == set(WEIGHTS)
    assert abs(sum(parts.values()) - score) < 1e-3
    assert 0 <= score <= 1


def test_retracted_and_irrelevant_are_excluded():
    papers = [paper(f"Relevant paper number {i} about topic{i}") for i in range(5)]
    papers.append(paper("Retracted but relevant", is_retracted=True, citation_count=999))
    papers.append(paper("Off topic paper"))
    screens = {i: screening(i, 8) for i in range(5)}
    screens[5] = screening(5, 10, "meta_analysis")
    screens[6] = screening(6, 3)

    sel = select_best(papers, screens, "biomedical", today=TODAY)

    titles = {sp.paper.title for sp in sel.picked}
    assert len(sel.picked) == 5
    assert "Retracted but relevant" not in titles
    assert "Off topic paper" not in titles
    assert sel.warnings == []


def test_higher_relevance_ranks_first_when_all_else_equal():
    papers = [paper("alpha beta gamma"), paper("delta epsilon zeta")]
    screens = {0: screening(0, 7), 1: screening(1, 10)}
    sel = select_best(papers, screens, k_min=1, today=TODAY)
    assert sel.picked[0].index == 1


def test_mmr_prefers_a_different_paper_over_a_near_duplicate():
    papers = [
        paper("Triceps activation bench press", "emg triceps brachii bench press activation load"),
        paper("Triceps activation bench press exercise", "emg triceps brachii bench press activation grip"),
        paper("Triceps activation bench press variation", "emg triceps brachii bench press activation incline"),
        paper("Elbow torque and muscle architecture", "ultrasound fascicle length pennation elbow torque"),
    ]
    screens = {0: screening(0, 9), 1: screening(1, 9), 2: screening(2, 9), 3: screening(3, 8)}
    sel = select_best(papers, screens, k_max=2, k_min=1, today=TODAY)
    assert sel.picked[0].index == 0
    assert sel.picked[1].index == 3          # lower score, but different -> picked before duplicates


def test_too_few_relevant_papers_relaxes_gate_and_warns():
    papers = [paper(f"paper {i} word{i}") for i in range(4)]
    screens = {0: screening(0, 8), 1: screening(1, 5), 2: screening(2, 5), 3: screening(3, 2)}
    sel = select_best(papers, screens, k_min=5, today=TODAY)
    assert {sp.index for sp in sel.picked} == {0, 1, 2}
    assert len(sel.warnings) == 2            # "relaxed" + "only 3 relevant papers"


def test_select_diverse_handles_empty_input():
    assert select_diverse([], k=10) == []


# ---------- final pick: readable full text first ----------

def shortlist(n: int):
    """n relevant papers (relevance 10, 10, 9, 9, ... all above the gate), best first, clearly different."""
    papers = [paper(f"topic{i} word{i} thing{i}") for i in range(n)]
    screens = {i: screening(i, 10 - i // 2) for i in range(n)}
    return select_best(papers, screens, k_max=n, k_min=1, today=TODAY).picked


def test_full_text_papers_replace_better_abstract_only_ones():
    short = shortlist(8)
    order = [s.index for s in short]
    with_text = set(order[2:7])                          # the 2 best have no full text; 5 lower ones do
    sel = prefer_full_text(short, with_text, k=5)
    assert [s.index for s in sel.picked] == order[2:7]   # all 5 with full text, still in "best first" order
    assert sel.warnings == []


def test_abstract_only_papers_fill_the_gaps_with_a_warning():
    short = shortlist(8)
    order = [s.index for s in short]
    with_text = {order[1], order[6]}                     # only 2 of 8 have a readable full text
    sel = prefer_full_text(short, with_text, k=5)
    picked = [s.index for s in sel.picked]
    assert picked == [order[0], order[1], order[2], order[3], order[6]]   # 2 with text + 3 best without
    assert sel.warnings == ["only 2 of 5 selected papers have a readable free full text; 3 are abstract only"]


def test_no_full_text_at_all_still_gives_the_best_papers():
    short = shortlist(8)                                 # e.g. Europe PMC down (topic 912)
    sel = prefer_full_text(short, set(), k=5)
    assert [s.index for s in sel.picked] == [s.index for s in short[:5]]
    assert "only 0 of 5" in sel.warnings[0]


def test_short_shortlist_gives_what_there_is():
    short = shortlist(3)
    sel = prefer_full_text(short, {s.index for s in short}, k=5)
    assert len(sel.picked) == 3 and sel.warnings == []
