"""
Finding passages (step 7.6b): meaning search + keyword search, merged.

    meaning   the question's embedding vs every passage's (pgvector cosine distance): finds "1-RM" for "strength"
    keywords  Postgres full text, ANY of the question's words (OR), ranked: finds exact terms like "HMB", "0.3 g"
    merged    reciprocal rank fusion: score = sum of 1 / (60 + rank) over both lists, so a passage that is
              good in EITHER list rises, and one good in both rises most. No weights to tune.

Only the papers the caller may see are searched (a topic's selected papers, or all of a user's).
"""

import asyncio
import re
from dataclasses import dataclass

from sqlalchemy import func, select, text

from app.common.logging import get_logger
from app.common.timing import Timings
from app.domain.errors import ProviderError
from app.integrations.embeddings.gemini import Embedder
from app.repositories.models import PaperChunk
from app.services.discovery_service import UnitOfWork
from app.services.prefilter import keywords

log = get_logger(__name__)
CANDIDATES = 30                  # from each search, before merging
EMBED_TIMEOUT = 90.0             # a question never waits longer than this for its embedding
RRF_K = 60                       # the usual constant: dampens the gap between rank 1 and rank 2
HIGHLIGHT = ("\u0002", "\u0003")  # markers around matched words in snippets (never in paper text)
# measured (7.6b probe): one paper filled every slot ("what dose did the TRIALS use?" -> only 2 reviews),
# and Introduction passages describe OTHER studies. So: at most 3 passages per paper, introductions ranked lower.
PER_PAPER = 3
KIND_WEIGHT = {"introduction": 0.8}


@dataclass
class Hit:
    chunk_id: int
    paper_id: int
    kind: str
    title: str | None
    text: str
    score: float
    meaning_rank: int | None = None   # 1 = best; None = not in that list
    keyword_rank: int | None = None
    snippet: str = ""                 # keyword search: the text around matches, words marked with HIGHLIGHT


def or_query(question: str) -> str | None:
    """'what dose of creatine?' -> 'dose | creatine' for to_tsquery (any word may match; ranking does the rest)."""
    words = sorted(w for w in keywords([question]) if re.fullmatch(r"[a-z0-9]+", w))
    return " | ".join(words) or None


def fuse(meaning: list[int], keyword: list[int]) -> list[tuple[int, float]]:
    """Reciprocal rank fusion of two ranked id lists -> [(id, score)] best first."""
    scores: dict[int, float] = {}
    for ranked in (meaning, keyword):
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class RetrievalService:
    def __init__(self, unit_of_work: UnitOfWork, embedder: Embedder):
        self.unit_of_work = unit_of_work
        self.embedder = embedder

    async def search(self, question: str, paper_ids: list[int], k: int = 8, snippets: bool = False,
                     notes: list[str] | None = None, timings: Timings | None = None) -> list[Hit]:
        """Best passages. If the meaning search is unavailable (embedding quota, outage, slow), fall back to
        keyword search only and say so in `notes`: an answer with a note beats an error (measured 7.7)."""
        if not paper_ids or not question.strip():
            return []
        timings = timings or Timings()       # callers that don't measure (evals, Search page) still work
        vector = None
        with timings.stage("embed"):
            try:
                [vector] = await asyncio.wait_for(self.embedder.embed([question], "query"), timeout=EMBED_TIMEOUT)
            except (ProviderError, TimeoutError) as e:
                log.warning("meaning search unavailable, keywords only: %s", e)
                if notes is not None:
                    notes.append("meaning search was unavailable (embedding service busy): keyword search only")
        query = or_query(question)
        with timings.stage("db_search"):
            async with self.unit_of_work() as session:
                # pgvector 0.8: keep scanning the index until enough rows pass the paper filter
                await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
                meaning: list[int] = []
                if vector is not None:
                    meaning = list(await session.scalars(
                        select(PaperChunk.id).where(PaperChunk.paper_id.in_(paper_ids))
                        .order_by(PaperChunk.embedding.cosine_distance(vector)).limit(CANDIDATES)))
                keyword: list[int] = []
                if query:
                    tsq = func.to_tsquery("english", query)
                    keyword = list(await session.scalars(
                        select(PaperChunk.id).where(PaperChunk.paper_id.in_(paper_ids), PaperChunk.tsv.op("@@")(tsq))
                        .order_by(func.ts_rank_cd(PaperChunk.tsv, tsq).desc()).limit(CANDIDATES)))
                fused = fuse(meaning, keyword)[:CANDIDATES]
                if not fused:
                    return []
                ids = [cid for cid, _ in fused]
                columns = [PaperChunk.id, PaperChunk.paper_id, PaperChunk.kind, PaperChunk.title, PaperChunk.text]
                if snippets and query:
                    columns.append(func.ts_headline("english", PaperChunk.text, func.to_tsquery("english", query),
                                                    f"StartSel={HIGHLIGHT[0]}, StopSel={HIGHLIGHT[1]}, MaxWords=40, "
                                                    "MinWords=15, MaxFragments=2"))
                rows = {row[0]: row for row in await session.execute(select(*columns).where(PaperChunk.id.in_(ids)))}
        hits = []
        for cid, score in fused:
            row = rows[cid]
            hits.append(Hit(chunk_id=cid, paper_id=row[1], kind=row[2], title=row[3], text=row[4],
                            score=score * KIND_WEIGHT.get(row[2], 1.0),
                            meaning_rank=meaning.index(cid) + 1 if cid in meaning else None,
                            keyword_rank=keyword.index(cid) + 1 if cid in keyword else None,
                            snippet=row[5] if len(row) > 5 else row[4][:300]))
        return diversify(sorted(hits, key=lambda h: -h.score), k, PER_PAPER)

    async def indexed(self, paper_ids: list[int]) -> tuple[int, int]:
        """(papers with passages, passages) among these papers."""
        if not paper_ids:
            return 0, 0
        async with self.unit_of_work() as session:
            row = (await session.execute(
                select(func.count(func.distinct(PaperChunk.paper_id)), func.count())
                .where(PaperChunk.paper_id.in_(paper_ids)))).one()
        return int(row[0]), int(row[1])


def diversify(hits: list[Hit], k: int, per_paper: int) -> list[Hit]:
    """The best k hits with at most `per_paper` from one paper (so more papers get a say)."""
    taken: dict[int, int] = {}
    out = []
    for h in hits:
        if taken.get(h.paper_id, 0) < per_paper:
            out.append(h)
            taken[h.paper_id] = taken.get(h.paper_id, 0) + 1
            if len(out) == k:
                break
    return out
