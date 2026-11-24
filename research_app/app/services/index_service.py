"""
Indexing papers for questions (step 7.6a): sections -> passages -> embeddings -> paper_chunks.

    once     a paper is indexed once per index version (embedding model + dimensions + chunker version);
             asking again, or another topic with the same paper, costs nothing
    grouped  papers are embedded in groups of about one request (~90 passages) and SAVED after each group:
             the free tier allows ~90 passages a minute, so a big topic takes minutes, and a rate limit or
             outage halfway never loses the papers already done
    safe     papers that fail (e.g. the embedding API is down) are reported; the run itself never fails
"""

from dataclasses import dataclass, field

from app.common.logging import get_logger
from app.common.progress import Progress, quiet
from app.domain.errors import ProviderError
from app.integrations.embeddings.gemini import Embedder
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.paper_repository import PaperRepository
from app.services.chunking import CHUNKER_VERSION, Passage, chunk_paper
from app.services.discovery_service import UnitOfWork

log = get_logger(__name__)


@dataclass
class IndexResult:
    indexed: int = 0                  # papers (re)indexed now
    reused: int = 0                   # papers already indexed with this version
    empty: int = 0                    # papers with no text at all (no abstract, no full text)
    passages: int = 0                 # passages written now
    failed: list[str] = field(default_factory=list)


class IndexService:
    def __init__(self, unit_of_work: UnitOfWork, embedder: Embedder, group_size: int = 90):
        self.unit_of_work = unit_of_work
        self.embedder = embedder
        self.group_size = group_size                 # passages per group (about one embedding request)

    @property
    def version(self) -> str:
        return f"{self.embedder.name}|chunker-{CHUNKER_VERSION}"

    async def index(self, paper_ids: list[int], *, refresh: bool = False, progress: Progress | None = None) -> IndexResult:
        say = progress or quiet
        result = IndexResult()
        async with self.unit_of_work() as session:
            done = await ChunkRepository(session).versions(paper_ids)
            todo = [pid for pid in dict.fromkeys(paper_ids) if refresh or done.get(pid) != self.version]
            result.reused = len(set(paper_ids)) - len(todo)
            passages = {pid: await self._passages(session, pid) for pid in todo}
        passages = {pid: ps for pid, ps in passages.items() if ps}
        result.empty = len(todo) - len(passages)
        if not passages:
            return result
        total = sum(len(ps) for ps in passages.values())
        await say(f"preparing {len(passages)} papers for questions ({total} passages)...")
        groups = group_papers(passages, self.group_size)
        for n, group in enumerate(groups):
            try:
                vectors = await self.embedder.embed([p.text for pid in group for p in passages[pid]], "document")
            except ProviderError as e:               # stop here; what is saved stays saved
                log.warning("indexing stopped: %s", e)
                result.failed = [f"paper {pid}: {e}" for g in groups[n:] for pid in g]
                return result
            start = 0
            async with self.unit_of_work() as session:   # saved per group: progress survives a later failure
                chunks = ChunkRepository(session)
                for pid in group:
                    count = len(passages[pid])
                    await chunks.replace(pid, passages[pid], vectors[start:start + count], self.version)
                    start += count
                    result.indexed += 1
                    result.passages += count
            if len(groups) > 1 and n < len(groups) - 1:
                await say(f"embedded {result.passages} of {total} passages...")
        return result

    async def _passages(self, session, paper_id: int) -> list[Passage]:
        """The best stored text of a paper (extracted XML over PDF) + its abstract."""
        paper = await PaperRepository(session).get(paper_id)
        if paper is None:
            return []
        documents = DocumentRepository(session)
        usable = [d for d in await documents.for_paper(paper_id) if d.status == "extracted"]
        best = min(usable, key=lambda d: d.content_type != "application/xml") if usable else None
        sections = [(s.kind, s.title, s.text) for s in await documents.sections(best.id)] if best else []
        return chunk_paper(sections, paper.abstract)


def group_papers(passages: dict[int, list[Passage]], size: int) -> list[list[int]]:
    """Paper ids in groups of about `size` passages (a paper is never split; a big paper is its own group)."""
    groups: list[list[int]] = [[]]
    count = 0
    for pid, ps in passages.items():
        if groups[-1] and count + len(ps) > size:
            groups.append([])
            count = 0
        groups[-1].append(pid)
        count += len(ps)
    return [g for g in groups if g]
