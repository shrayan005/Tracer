"""Study type: the LLM's guess checked against official index tags (step 6.7). Pure code."""

from app.domain.schemas import PaperRecord, PaperScreening
from app.integrations.sources import europepmc, semantic_scholar
from app.services.dedupe import merge_two
from app.services.research_service import with_official_type
from app.services.study_type import official_type, reconcile


def test_specific_tags_win_most_specific_first():
    assert official_type(["review", "meta-analysis", "systematic review"]) == "meta_analysis"
    assert official_type(["randomized controlled trial", "journal article"]) == "randomized_trial"
    assert official_type(["MetaAnalysis"]) == "meta_analysis"               # Semantic Scholar spelling
    assert official_type(["clinicaltrial", "journal article"]) is None      # not necessarily randomized


def test_reconcile():
    assert reconcile("lab_experiment", ["randomized controlled trial"])[0] == "randomized_trial"
    assert reconcile("randomized_trial", ["review", "journal article"]) == (
        "narrative_review", "indexed as a review, not the AI's guess (randomized_trial)")
    assert reconcile("systematic_review", ["review"]) == ("systematic_review", None)   # generic tag: keep
    assert reconcile("cohort", []) == ("cohort", None)                                  # no tags: keep
    # measured (6.7): PubMed tagged a systematic review + meta-analysis only "systematic review"
    assert reconcile("meta_analysis", ["systematic review", "journal article"]) == ("meta_analysis", None)
    assert reconcile("lab_experiment", ["review-article"])[0] == "narrative_review"


def test_screening_gets_the_checked_type_and_a_note():
    s = PaperScreening(id=0, relevance=8, reason="RCT of creatine.", study_type="randomized_trial")
    paper = PaperRecord(source="europepmc", title="Review of creatine", pub_types=["review"])
    checked = with_official_type(s, paper)
    assert checked.study_type == "narrative_review" and "indexed as a review" in checked.reason
    assert with_official_type(s, PaperRecord(source="x", title="t")) is s          # nothing to check


def test_sources_keep_their_tags_and_merging_unites_them():
    epmc = europepmc.to_record({"title": "T.", "pubTypeList": {"pubType": ["Randomized Controlled Trial",
                                                                           "Journal Article"]}}, "q")
    assert epmc.pub_types == ["randomized controlled trial", "journal article"]
    s2 = semantic_scholar.to_record({"title": "T", "publicationTypes": ["JournalArticle", "ClinicalTrial"]}, "q")
    assert s2.pub_types == ["journalarticle", "clinicaltrial"]
    assert merge_two(epmc, s2).pub_types == ["randomized controlled trial", "journal article",
                                             "journalarticle", "clinicaltrial"]
