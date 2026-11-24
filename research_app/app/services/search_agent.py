"""
Search agent: the DECISION step of the agentic search (the loop lives in discovery_service.py).

After a round of searching, the LLM sees what was searched and what came back:
    queries run      + how many papers each found (and how many only that query found)
    top titles       the best-ranked candidates (most-found, most-cited first)
and answers: which studies seem missing (gaps), stop (done) or search these new queries.

"The LLM proposes, code checks": clean_new() drops repeated, too long and extra queries, so a bad
suggestion can cost at most one wasted search, never a runaway loop.
"""

from collections import Counter

from app.domain.schemas import Candidate, SearchReview
from app.integrations.llm.gateway import LlmGateway
from app.services.planning_service import _clean_queries

MAX_TITLES = 25          # ~20 tokens each: 25 titles ~ 500 tokens; all 69 would cost ~3x for little gain
MAX_NEW_QUERIES = 3
MAX_QUERY_WORDS = 5      # databases match words literally: long queries find less

REVIEW_PROMPT = """You review one round of an academic literature search.
You see the queries already run, how many papers each found (and how many ONLY that query found),
and the top titles found so far. Each search returns at most a fixed number of papers: a query
marked "limit reached" was cut off and more matching papers exist. A query that found nothing
no other query found adds nothing new.

Think about which kinds of relevant studies are probably still missing: other sexes or age groups,
other populations, combined treatments, other outcomes, or other wording that studies use for the
same thing. Then either stop or suggest up to 3 new queries.

Rules for new queries:
- each 2-5 words, no quotes or boolean operators, the topic's core term plus the missing angle
- use the plain words study TITLES use, not formal labels: "older women", "young men", "children",
  not "female older adults", "male participants" or "pediatric population". Databases match words
  literally, so one query per wording (e.g. one for women, one for men) finds more than a mixed one
- never repeat or merely rephrase a query that was already run
- never copy a title
Say done=true when the titles already cover the topic's populations and angles well."""


def round_summary(candidates: list[Candidate], queries: list[str]) -> dict[str, tuple[int, int]]:
    """{query: (papers it found, papers ONLY it found)} for one round's merged candidates."""
    found, only = Counter(), Counter()
    for c in candidates:
        by = {f.query for f in c.found_by}
        found.update(by)
        if len(by) == 1:
            only.update(by)
    return {q: (found[q], only[q]) for q in queries}


def clean_new(suggested: list[str], tried: list[str]) -> list[str]:
    """Keep at most 3 short queries that were not run before (case and spacing ignored)."""
    seen = {" ".join(q.lower().split()) for q in tried}
    kept = []
    for q in _clean_queries(suggested, limit=len(suggested) or 1):
        if q.lower() in seen or len(q.split()) > MAX_QUERY_WORDS:
            continue
        kept.append(q)
    return kept[:MAX_NEW_QUERIES]


class SearchAgent:
    def __init__(self, gateway: LlmGateway):
        self.gateway = gateway

    async def review(self, brief: str, summary: dict[str, tuple[int, int]],
                     titles: list[str], limit: int) -> SearchReview:
        """`limit` = papers one search may return: reaching it means the query was cut off."""
        runs = "\n".join(f"- {q!r}: {n} papers{' (limit reached: more exist)' if n >= limit else ''}, "
                         f"{only} found by no other query" for q, (n, only) in summary.items())
        top = "\n".join(f"- {t}" for t in titles[:MAX_TITLES])
        messages = [{"role": "system", "content": REVIEW_PROMPT},
                    {"role": "user", "content": f"Topic: {brief}\n\nQueries run:\n{runs}\n\n"
                                                f"Top {min(len(titles), MAX_TITLES)} titles:\n{top}"}]
        review, _ = await self.gateway.complete_structured(messages, SearchReview,
                                                           purpose="search_review", max_tokens=600)
        review.new_queries = clean_new(review.new_queries, list(summary))   # code checks the LLM
        if not review.new_queries:
            review.done = True          # nothing usable left to search: stopping is the only option
        return review
