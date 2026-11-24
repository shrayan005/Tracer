"""Quality score (step 5.2): checklist points per field, red flags, and how trust moves."""

import pytest

from app.domain.schemas import Appraisal, TrustAssessment
from app.services.quality import QUALITY_POINTS, apply_quality, sample_points, score

ITEMS = ("control_group", "randomized", "blinded", "statistics_reported", "effect_sizes",
         "conflicts_declared", "data_available")


def appraisal(n=20, sex="mixed", **values) -> Appraisal:
    facts = {item: {"value": values.get(item, "not reported")} for item in ITEMS}
    return Appraisal(sample_size=n, sex=sex, duration="single session", **facts)


def trust(level=3) -> TrustAssessment:
    labels = {0: "retracted", 1: "very low", 2: "low", 3: "moderate", 4: "high", 5: "high"}
    return TrustAssessment(level=level, label=labels[level], reasons=["lab experiment"])


ALL_YES = {item: "yes" for item in ITEMS}


def test_points_table_uses_only_known_items():
    for field, points in QUALITY_POINTS.items():
        assert set(points) <= set(ITEMS) | {"sample"}, field
        assert all(weight > 0 for weight in points.values()), field


@pytest.mark.parametrize("n, points", [(30, 1.0), (120, 1.0), (15, 0.5), (10, 0.5), (5, 0.0), (None, 0.0)])
def test_sample_points(n, points):
    assert sample_points(n) == points


def test_not_applicable_items_count_neither_way():
    earned, possible, _ = score(appraisal(n=40, control_group="not applicable", statistics_reported="yes"),
                                "sport_biomechanics")
    assert possible == 6.5 - 1                     # control group left out of the total
    assert earned == 1 + 1                         # statistics + sample size (n >= 30)


def test_strong_full_text_study_goes_up_with_the_reason():
    out = apply_quality(trust(3), appraisal(n=60, **ALL_YES), "biomedical", "full text")
    assert out.level == 4 and out.label == "high"
    assert out.reasons[0] == "lab experiment"                     # the design reason is kept
    assert out.reasons[-1] == "well-conducted study: trust +1"


def test_weak_full_text_study_goes_down():
    out = apply_quality(trust(3), appraisal(n=None), "biomedical", "full text")
    assert out.level == 2 and out.reasons[-1] == "weak methods reporting: trust -1"


def test_abstract_only_checklist_is_not_judged():
    out = apply_quality(trust(3), appraisal(n=None), "biomedical", "abstract")
    assert out.level == 3 and out.reasons[-1] == "checklist not judged (abstract only)"


def test_very_small_sample_is_a_red_flag_even_from_an_abstract():
    out = apply_quality(trust(4), appraisal(n=7), "sport_biomechanics", "abstract")
    assert out.level == 3
    assert out.reasons[-1] == "checklist not judged (abstract only); very small sample (n=7): trust -1"


def test_change_is_at_most_one_step_and_stays_in_range():
    weak_and_tiny = apply_quality(trust(3), appraisal(n=5), "biomedical", "full text")
    assert weak_and_tiny.level == 2                                   # -1 and -1 -> still only -1
    good_but_tiny = apply_quality(trust(3), appraisal(n=8, **ALL_YES), "biomedical", "full text")
    assert good_but_tiny.level == 2                                   # the red flag wins over +1
    assert apply_quality(trust(5), appraisal(n=90, **ALL_YES), "biomedical", "full text").level == 5
    assert apply_quality(trust(1), appraisal(n=None), "biomedical", "full text").level == 1


def test_retracted_and_missing_appraisal_are_unchanged():
    retracted = trust(0)
    assert apply_quality(retracted, appraisal(n=90, **ALL_YES), "biomedical", "full text") is retracted
    assert apply_quality(trust(3), None, "biomedical", "full text").level == 3


def test_cs_values_released_code_and_has_no_sample_item():
    _, possible, details = score(appraisal(n=None, sex="not applicable"), "cs")
    assert possible == 4.5 and not any(d.startswith("n=") for d in details)
    out = apply_quality(trust(3), appraisal(n=None, sex="not applicable", data_available="yes",
                                            statistics_reported="yes", control_group="yes"), "cs", "full text")
    assert out.level == 4                                             # 4 of 4.5 points
