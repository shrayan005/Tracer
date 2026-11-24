"""
Cheap pre-filter (pure code, no LLM, instant): keep the most promising candidates
before the expensive LLM screening.

Measured: screening all ~150 candidates needs ~50k tokens, about 6 minutes on Groq's free tier
(8,000 tokens/minute). Pre-filtering to ~40 makes screening take seconds.
"""

import math
import re

from app.domain.schemas import Candidate

STOPWORDS = {"the", "of", "and", "in", "a", "an", "to", "for", "on", "with", "by", "review", "study",
             "analysis", "effects", "effect", "during", "using", "based"}


def keywords(texts: list[str]) -> set[str]:
    """Meaningful lowercase words (3+ letters, not stopwords) from some texts."""
    words: set[str] = set()
    for text in texts:
        words |= {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}
    return words


# Why the best single query counts most (24 Nov 2026, search agent): with ~11 queries (plan + agent) the
# union of their words is ~25 terms, and a raw overlap rewards long titles/abstracts that touch many of
# them. A short exact title like "Acute Creatine Supplementation in Older Men" matched its own query
# fully but ranked 76th. Measured offline on 2 agent searches (creatine gold): gold papers in the top 20
# 5->8 and 4->6, top 40 9->14 and 8->11; without the agent: unchanged (5 and 6).
BEST_QUERY_WEIGHT = 20       # a title holding every word of one query gets +20
OVERLAP_CAP = 12             # the "many scattered words" bonus is capped


def best_query_match(title_words: set[str], query_words: list[set[str]]) -> float:
    """Share (0..1) of the best-matching query's words that appear in the title."""
    return max((len(q & title_words) / len(q) for q in query_words if q), default=0.0)


def prefilter(candidates: list[Candidate], queries: list[str], keep: int = 40) -> list[Candidate]:
    """Rank candidates by cheap signals and keep the top `keep`."""
    terms = keywords(queries)
    query_words = [keywords([q]) for q in queries]

    def score(candidate: Candidate) -> float:
        paper = candidate.paper
        title_words = keywords([paper.title])
        text_words = title_words | keywords([paper.abstract or ""])
        overlap = len(terms & title_words) * 2 + len(terms & text_words)   # title matches count double
        return (BEST_QUERY_WEIGHT * best_query_match(title_words, query_words)
                + min(overlap, OVERLAP_CAP)
                + len(candidate.found_by) * 1.5            # found by more queries/sources
                + math.log1p(paper.citation_count or 0) * 0.5   # some impact, without letting giants dominate
                + (1.0 if paper.abstract else 0.0))        # an abstract makes screening reliable

    return sorted(candidates, key=score, reverse=True)[:keep]
