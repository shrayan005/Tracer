"""
Trust assessment: how strong is a paper as EVIDENCE? Transparent rules in code (not the LLM),
so every result is consistent and explained.

Field-aware: what counts as strong evidence differs by field. In medicine a randomized trial beats
a lab experiment; in maths a proof (theoretical) IS the evidence; in CS, arXiv preprints and
benchmark (computational) studies are normal. So each field family has its own table.

Base level from the study design, then modifiers: retracted -> 0, preprint -> minus the field's
penalty, well cited (>= 10 citations/year) -> +1. The study type comes from LLM screening (3.1).
A fuller quality appraisal (checklists: sample size, blinding...) comes in milestone 5.
"""

from datetime import date

from app.domain.schemas import FieldFamily, PaperRecord, ResearchPlan, StudyType, TrustAssessment

LEVELS_BY_FIELD: dict[str, dict[str, int]] = {
    "biomedical": {        # medicine, clinical biology: the classic evidence hierarchy
        "meta_analysis": 5, "systematic_review": 5, "randomized_trial": 4, "cohort": 3, "case_control": 3,
        "cross_sectional": 2, "lab_experiment": 2, "computational": 2, "narrative_review": 2,
        "case_report": 1, "theoretical": 1, "other": 1,
    },
    # sports science, biomechanics, exercise science: questions about HOW the body moves are answered
    # in the lab (EMG, force plates, motion capture) and with musculoskeletal models, so controlled
    # lab experiments and models are normal, solid evidence there (unlike for "does treatment X work?").
    "sport_biomechanics": {
        "meta_analysis": 5, "systematic_review": 5, "randomized_trial": 4, "lab_experiment": 3,
        "computational": 3, "cohort": 3, "case_control": 3, "cross_sectional": 3, "narrative_review": 2,
        "theoretical": 2, "case_report": 1, "other": 1,
    },
    "cs": {                # computer science / AI: benchmarks and experiments are the main evidence
        "meta_analysis": 4, "systematic_review": 4, "computational": 4, "randomized_trial": 4,
        "lab_experiment": 3, "theoretical": 3, "narrative_review": 3, "cohort": 3, "case_control": 3,
        "cross_sectional": 2, "case_report": 1, "other": 2,
    },
    "physics_math": {      # physics, maths, statistics: proofs and experiments are the evidence
        "theoretical": 4, "lab_experiment": 4, "meta_analysis": 4, "systematic_review": 4,
        "computational": 3, "narrative_review": 3, "randomized_trial": 3, "cohort": 2, "case_control": 2,
        "cross_sectional": 2, "case_report": 1, "other": 2,
    },
    "general": {           # social sciences, economics, everything else
        "meta_analysis": 5, "systematic_review": 5, "randomized_trial": 4, "cohort": 3, "case_control": 3,
        "cross_sectional": 3, "lab_experiment": 3, "computational": 3, "narrative_review": 2,
        "theoretical": 2, "case_report": 1, "other": 1,
    },
}
# arXiv preprints are the normal way to publish in CS, physics and maths: no penalty there
PREPRINT_PENALTY: dict[str, int] = {"biomedical": 1, "sport_biomechanics": 1, "general": 1,
                                    "cs": 0, "physics_math": 0}
# words in the plan's field name that mean "lab-based movement science"
SPORT_BIOMECHANICS_WORDS = ("sport", "exercise", "biomechanic", "kinesiolog", "movement science",
                            "motor control", "ergonomic")
LABELS = {5: "high", 4: "high", 3: "moderate", 2: "low", 1: "very low"}
WELL_CITED_PER_YEAR = 10


def field_family(plan: ResearchPlan) -> FieldFamily:
    """Which table to use, from the research plan (step 2.9)."""
    name = plan.field.lower()
    if any(w in name for w in SPORT_BIOMECHANICS_WORDS):   # checked first: the planner marks it biomedical
        return "sport_biomechanics"
    if plan.is_biomedical:
        return "biomedical"
    if plan.is_cs_math_physics:
        return "physics_math" if any(w in name for w in ("math", "physic", "statistic")) else "cs"
    return "general"


def citations_per_year(paper: PaperRecord, today: date | None = None) -> float | None:
    """Citations divided by age, so old and new papers can be compared fairly."""
    if paper.citation_count is None or not paper.year:
        return None
    years = max((today or date.today()).year - paper.year, 0) + 1   # a paper from this year counts as 1 year
    return paper.citation_count / years


def assess_trust(paper: PaperRecord, study_type: StudyType | None, field: FieldFamily = "general",
                 today: date | None = None) -> TrustAssessment:
    if paper.is_retracted:
        return TrustAssessment(level=0, label="retracted", reasons=["RETRACTED: do not rely on it"])

    design = study_type or "other"
    level = LEVELS_BY_FIELD[field].get(design, 1)
    reasons = [design.replace("_", " ")]

    is_preprint = paper.publication_type == "preprint" or paper.venue_type == "preprint"
    if is_preprint and PREPRINT_PENALTY[field]:
        level -= PREPRINT_PENALTY[field]
        reasons.append("preprint (not peer-reviewed yet)")
    elif is_preprint:
        reasons.append("preprint (normal in this field)")

    per_year = citations_per_year(paper, today)
    if per_year is not None and per_year >= WELL_CITED_PER_YEAR:
        level += 1
        reasons.append(f"well cited ({per_year:.0f}/yr)")
    elif per_year is not None:
        reasons.append(f"{paper.citation_count} citation{'' if paper.citation_count == 1 else 's'}")

    if not paper.abstract:
        reasons.append("no abstract (judged from title)")

    level = max(1, min(level, 5))          # keep it within 1..5 (0 is reserved for retracted)
    return TrustAssessment(level=level, label=LABELS[level], reasons=reasons)
