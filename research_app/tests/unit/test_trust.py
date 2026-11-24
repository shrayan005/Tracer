from datetime import date

import pytest

from app.domain.schemas import PaperRecord, ResearchPlan
from app.services.trust import LEVELS_BY_FIELD, PREPRINT_PENALTY, assess_trust, field_family

TODAY = date(2026, 11, 20)


def plan(field: str, bio: bool = False, cs: bool = False) -> ResearchPlan:
    return ResearchPlan(field=field, is_biomedical=bio, is_cs_math_physics=cs, research_brief="x", queries=["q"])


@pytest.mark.parametrize("field, bio, cs, expected", [
    ("sports science", True, False, "sport_biomechanics"),
    ("Biomechanics", True, False, "sport_biomechanics"),
    ("exercise physiology", True, False, "sport_biomechanics"),
    ("medicine", True, False, "biomedical"),
    ("computer science", False, True, "cs"),
    ("mathematics", False, True, "physics_math"),
    ("economics", False, False, "general"),
])
def test_field_family(field, bio, cs, expected):
    assert field_family(plan(field, bio, cs)) == expected


def test_every_table_covers_every_study_type_and_has_a_preprint_rule():
    study_types = set(LEVELS_BY_FIELD["general"])
    for name, table in LEVELS_BY_FIELD.items():
        assert set(table) == study_types, name
        assert name in PREPRINT_PENALTY


def test_lab_experiment_is_moderate_in_sport_biomechanics_but_low_in_medicine():
    paper = PaperRecord(source="x", title="t", year=2018, citation_count=60, abstract="a")   # ~7 citations/yr
    assert assess_trust(paper, "lab_experiment", "sport_biomechanics", TODAY).label == "moderate"
    assert assess_trust(paper, "lab_experiment", "biomedical", TODAY).label == "low"


def test_modifiers():
    preprint = PaperRecord(source="x", title="t", year=2025, citation_count=40, abstract="a",
                           publication_type="preprint")                                   # 20 citations/yr
    t = assess_trust(preprint, "computational", "cs", TODAY)
    assert t.level == 5 and "preprint (normal in this field)" in t.reasons
    t = assess_trust(preprint, "computational", "biomedical", TODAY)
    assert t.level == 2 and "preprint (not peer-reviewed yet)" in t.reasons


def test_retracted_is_always_zero():
    paper = PaperRecord(source="x", title="t", is_retracted=True, citation_count=999)
    for family in LEVELS_BY_FIELD:
        assert assess_trust(paper, "meta_analysis", family).level == 0
