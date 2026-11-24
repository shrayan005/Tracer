"""
Quality score (step 5.2): appraisal facts (5.1) -> quality points -> trust moves up or down one step.
Code only (no LLM): the same facts always give the same score, with the reasons shown.

Two parts:
  1. the CHECKLIST (control group, randomised, blinded, statistics, effect sizes, sample size, ...),
     scored per field family. Only with the full text: an abstract hides most items, and
     "not reported in the abstract" is not the same as "not done".
     >= 70% of the applicable points -> trust +1;  <= 30% -> trust -1
  2. RED FLAGS that are visible whatever we read (now: fewer than 10 participants) -> trust -1
Quality moves trust by at most one step; a retracted paper stays 0.
"""

from app.domain.schemas import Appraisal, FieldFamily, TrustAssessment
from app.services.trust import LABELS

# Points per checklist item, per field family (policy as data). 0 = not part of this field's checklist.
#   sample = sample size (n >= 30 full point, 10-29 half, < 10 none)
QUALITY_POINTS: dict[str, dict[str, float]] = {
    "biomedical":         {"control_group": 1, "randomized": 1, "blinded": 1, "statistics_reported": 1,
                           "effect_sizes": 1, "sample": 1, "conflicts_declared": 0.5, "data_available": 0.5},
    # exercise studies can rarely blind participants (they know how hard they lift): blinding counts half
    "sport_biomechanics": {"control_group": 1, "randomized": 1, "blinded": 0.5, "statistics_reported": 1,
                           "effect_sizes": 1, "sample": 1, "conflicts_declared": 0.5, "data_available": 0.5},
    "general":            {"control_group": 1, "randomized": 1, "blinded": 0.5, "statistics_reported": 1,
                           "effect_sizes": 1, "sample": 1, "conflicts_declared": 0.5, "data_available": 0.5},
    # CS: baselines, statistics and above all released code/data (reproducibility)
    "cs":                 {"control_group": 1, "statistics_reported": 1, "data_available": 2,
                           "conflicts_declared": 0.5},
    "physics_math":       {"statistics_reported": 1, "data_available": 1, "conflicts_declared": 0.5},
}
NAMES = {"control_group": "control group", "randomized": "randomised", "blinded": "blinded",
         "statistics_reported": "statistics", "effect_sizes": "effect sizes", "sample": "sample size",
         "conflicts_declared": "conflicts stated", "data_available": "data available"}
RAISE_AT, LOWER_AT = 0.70, 0.30     # share of applicable points -> trust +1 / -1
VERY_SMALL_SAMPLE = 10              # fewer participants than this is a red flag (visible even in an abstract)
MIN_APPLICABLE = 3.0                # fewer applicable points than this: too little to judge


def sample_points(n: int | None) -> float:
    if n is None:
        return 0.0
    return 1.0 if n >= 30 else 0.5 if n >= 10 else 0.0


def score(appraisal: Appraisal, field: FieldFamily) -> tuple[float, float, list[str]]:
    """(points earned, points possible, details) -- 'not applicable' items don't count either way."""
    earned = possible = 0.0
    details = []
    for item, weight in QUALITY_POINTS[field].items():
        if item == "sample":
            if appraisal.sample_size is None and appraisal.sex == "not applicable":
                continue                                      # no participants (e.g. a simulation)
            got = sample_points(appraisal.sample_size) * weight
            n = appraisal.sample_size
            details.append(f"n={n if n is not None else '?'}" + (" (small)" if n is not None and n < 10 else ""))
        else:
            value = getattr(appraisal, item).value
            if value == "not applicable":
                continue
            got = weight if value == "yes" else 0.0
            details.append(f"{NAMES[item]} {'✓' if value == 'yes' else '✗' if value == 'no' else '?'}")
        earned += got
        possible += weight
    return earned, possible, details


def apply_quality(trust: TrustAssessment, appraisal: Appraisal | None, field: FieldFamily,
                  based_on: str | None) -> TrustAssessment:
    """Trust from study design (step 3.2) adjusted by how well THIS study was done."""
    if trust.level == 0 or appraisal is None:                 # retracted, or nothing to appraise
        return trust
    earned, possible, details = score(appraisal, field)
    summary = f"quality {earned:g}/{possible:g}: " + ", ".join(details)

    # 1. the checklist: only with the full text (an abstract hides most items, "?" isn't "no")
    change = 0
    if based_on != "full text":
        note = "checklist not judged (abstract only)"
    elif possible < MIN_APPLICABLE:
        note = "checklist not judged (too few items apply)"
    elif earned / possible >= RAISE_AT:
        change, note = +1, "well-conducted study: trust +1"
    elif earned / possible <= LOWER_AT:
        change, note = -1, "weak methods reporting: trust -1"
    else:
        note = "quality as expected for its design"

    # 2. red flags: count whatever the source, because they are visible even in an abstract
    n = appraisal.sample_size
    if "sample" in QUALITY_POINTS[field] and n is not None and n < VERY_SMALL_SAMPLE:
        change, note = min(change, 0) - 1, f"{note}; very small sample (n={n}): trust -1"

    change = max(-1, min(change, +1))                  # quality moves trust by one step at most
    level = max(1, min(trust.level + change, 5))
    return TrustAssessment(level=level, label=LABELS[level], reasons=trust.reasons + [summary, note])
