"""
Selection: turn screening + trust into ONE score per paper, then pick a varied best few.
Pure code (no LLM, no database): deterministic, explainable, easy to test.

1. Gate:      drop retracted papers and papers that are not relevant enough.
2. Score:     weighted mix of relevance, trust, citations, recency, full text (0..1).
3. Diverse:   MMR picks high scores but skips near-duplicates of papers already picked -> a SHORTLIST.
4. Full text: from the shortlist, papers whose full text we could actually READ come first; abstract-only
              papers only fill places that are left (prefer_full_text, after the full texts were fetched).
"""

import math
from dataclasses import dataclass, field
from datetime import date

from app.domain.schemas import FieldFamily, PaperRecord, PaperScreening, TrustAssessment
from app.services.prefilter import keywords
from app.services.trust import assess_trust, citations_per_year

# Policy as data: change the numbers here, not the logic. They add up to 1.0.
WEIGHTS = {"relevance": 0.50, "trust": 0.25, "citations": 0.10, "recency": 0.10, "full_text": 0.05}
MIN_RELEVANCE = 6            # normal gate
RELAXED_MIN_RELEVANCE = 5    # used only if too few papers pass the normal gate
CITATIONS_PER_YEAR_CAP = 50  # 50+ citations/year counts as "full marks"
RECENCY_YEARS = 15           # a paper this old (or older) gets 0 for recency
MMR_LAMBDA = 0.7             # 1.0 = only score matters, 0.0 = only being different matters


@dataclass
class ScoredPaper:
    index: int                        # position in the input list (so we can find the Candidate again)
    paper: PaperRecord
    screening: PaperScreening
    trust: TrustAssessment
    score: float
    parts: dict[str, float]           # each weighted part, so the report can explain the score


@dataclass
class Selection:
    picked: list[ScoredPaper]
    warnings: list[str] = field(default_factory=list)


def score_paper(paper: PaperRecord, screening: PaperScreening, trust: TrustAssessment,
                today: date | None = None) -> tuple[float, dict[str, float]]:
    today = today or date.today()
    per_year = citations_per_year(paper, today) or 0.0
    age = today.year - paper.year if paper.year else RECENCY_YEARS
    raw = {
        "relevance": screening.relevance / 10,
        "trust": trust.level / 5,
        "citations": min(math.log1p(per_year) / math.log1p(CITATIONS_PER_YEAR_CAP), 1.0),
        "recency": max(0.0, 1 - age / RECENCY_YEARS),
        "full_text": 1.0 if (paper.is_open_access or paper.pdf_url) else 0.0,
    }
    parts = {name: round(WEIGHTS[name] * value, 4) for name, value in raw.items()}
    return round(sum(parts.values()), 4), parts


def similarity(a: PaperRecord, b: PaperRecord) -> float:
    """Jaccard overlap of keywords (0 = nothing shared, 1 = same words). Embeddings replace this in M4."""
    ka = keywords([a.title, a.abstract or ""])
    kb = keywords([b.title, b.abstract or ""])
    if not ka or not kb:
        return 0.0
    return len(ka & kb) / len(ka | kb)


def select_diverse(scored: list[ScoredPaper], k: int, lam: float = MMR_LAMBDA) -> list[ScoredPaper]:
    """MMR: each round, pick the paper with the best  lam*score - (1-lam)*similarity_to_picked."""
    remaining = sorted(scored, key=lambda s: -s.score)
    picked: list[ScoredPaper] = []
    while remaining and len(picked) < k:
        def mmr(s: ScoredPaper) -> float:
            closest = max((similarity(s.paper, p.paper) for p in picked), default=0.0)
            return lam * s.score - (1 - lam) * closest
        best = max(remaining, key=mmr)
        picked.append(best)
        remaining.remove(best)
    return picked


def select_best(papers: list[PaperRecord], screenings: dict[int, PaperScreening],
                field: FieldFamily = "general", k_max: int = 10, k_min: int = 5,
                today: date | None = None) -> Selection:
    """Entry point: trust -> gate -> score -> MMR. `screenings` is the output of ScreeningService.screen."""
    warnings: list[str] = []
    scored: list[ScoredPaper] = []
    for i, screening in screenings.items():
        paper = papers[i]
        trust = assess_trust(paper, screening.study_type, field, today)
        if trust.level == 0:
            continue                                          # retracted: never selected
        score, parts = score_paper(paper, screening, trust, today)
        scored.append(ScoredPaper(i, paper, screening, trust, score, parts))

    passing = [s for s in scored if s.screening.relevance >= MIN_RELEVANCE]
    if len(passing) < k_min:
        passing = [s for s in scored if s.screening.relevance >= RELAXED_MIN_RELEVANCE]
        warnings.append(f"fewer than {k_min} clearly relevant papers; included partly relevant ones "
                        f"(relevance >= {RELAXED_MIN_RELEVANCE})")
    if len(passing) < k_min:
        warnings.append(f"only {len(passing)} relevant papers found; try a broader topic")

    return Selection(picked=select_diverse(passing, k_max), warnings=warnings)


def prefer_full_text(shortlist: list[ScoredPaper], with_text: set[int], k: int) -> Selection:
    """The final `k` papers from a shortlist (best first). `with_text`: indexes of the papers whose full text
    was downloaded AND extracted. Those come first; the best abstract-only papers fill the places left, with a
    warning. The shortlist's order is kept, so the rank still means "best", not "has full text"."""
    readable = [s.index for s in shortlist if s.index in with_text][:k]
    fill = [s.index for s in shortlist if s.index not in with_text][:k - len(readable)]
    chosen = set(readable) | set(fill)
    picked = [s for s in shortlist if s.index in chosen]
    warnings = []
    if fill:
        warnings.append(f"only {len(readable)} of {len(picked)} selected papers have a readable free full text; "
                        f"{len(fill)} are abstract only")
    return Selection(picked=picked, warnings=warnings)
