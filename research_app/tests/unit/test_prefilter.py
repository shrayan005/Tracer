"""Prefilter ranking: a short title matching one query fully beats a long title touching many words."""

from app.domain.schemas import Candidate, FoundBy, PaperRecord
from app.services.prefilter import best_query_match, keywords, prefilter


def cand(title, abstract="x"):
    return Candidate(paper=PaperRecord(source="openalex", title=title, abstract=abstract),
                     found_by=[FoundBy(source="openalex", query="q")])


def test_best_query_match_is_the_best_share_of_one_query():
    title = keywords(["Acute Creatine Supplementation in Older Men"])
    queries = [keywords(["creatine supplementation older men"]), keywords(["creatine sarcopenia"])]
    assert best_query_match(title, queries) == 1.0
    assert best_query_match(keywords(["Sarcopenia in rats"]), queries) == 0.5
    assert best_query_match(title, []) == 0.0


def test_exact_short_title_beats_verbose_scattered_match():
    queries = ["creatine supplementation older men", "phosphocreatine metabolism aging muscle",
               "muscle protein synthesis creatine aging", "dose response creatine elderly resistance training"]
    exact = cand("Acute Creatine Supplementation in Older Men")
    verbose = cand("Resistance training dose, muscle protein and metabolism in aging",   # no query fully
                   abstract="creatine elderly response muscle aging metabolism synthesis protein training")
    assert prefilter([verbose, exact], queries, keep=1) == [exact]
