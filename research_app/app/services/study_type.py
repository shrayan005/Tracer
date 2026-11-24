"""
Study type: the LLM's guess (screening) checked against the OFFICIAL index tags (step 6.7).

PubMed indexers tag papers ("Randomized Controlled Trial", "Meta-Analysis" ...); Europe PMC and
Semantic Scholar pass those tags on. A specific official tag beats the LLM's guess, which only saw
the first 500 characters of the abstract. "LLM extracts, code checks."

Rules (in order):
  1. a specific tag (meta-analysis > systematic review > randomized trial > case report) -> that type
  2. only the generic "review" tag, and the LLM named a primary-study design -> narrative review
  3. otherwise the LLM's type stays (most papers have no specific tag)
"""

SPECIFIC = (  # (study type, tags that mean it), most specific first
    ("meta_analysis", {"meta-analysis", "metaanalysis"}),
    ("systematic_review", {"systematic review", "systematicreview"}),
    ("randomized_trial", {"randomized controlled trial", "randomizedcontrolledtrial"}),
    ("case_report", {"case reports", "case report", "casereport"}),
)
GENERIC_REVIEW = {"review", "review-article"}          # PubMed / Europe PMC article type
PRIMARY_DESIGNS = {"randomized_trial", "cohort", "case_control", "cross_sectional", "lab_experiment",
                   "case_report", "computational"}


def official_type(pub_types: list[str]) -> str | None:
    tags = {t.lower() for t in pub_types}
    return next((study_type for study_type, names in SPECIFIC if tags & names), None)


def reconcile(llm_type: str, pub_types: list[str]) -> tuple[str, str | None]:
    """(study type to use, note when the index tag changed the LLM's answer)."""
    official = official_type(pub_types)
    if official == "systematic_review" and llm_type == "meta_analysis":
        return llm_type, None     # a meta-analysis IS a systematic review; PubMed often tags only the latter
    if official and official != llm_type:
        return official, f"study type from the index tag ({official}), not the AI's guess ({llm_type})"
    tags = {t.lower() for t in pub_types}
    if official is None and tags & GENERIC_REVIEW and llm_type in PRIMARY_DESIGNS:
        return "narrative_review", f"indexed as a review, not the AI's guess ({llm_type})"
    return llm_type, None
