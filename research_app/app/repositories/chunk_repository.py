"""Data access for paper_chunks: the passages questions are answered from (step 7.6)."""

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.models import PaperChunk
from app.services.chunking import Passage


class ChunkRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def versions(self, paper_ids: list[int]) -> dict[int, str]:
        """{paper id: index_version} for papers that have passages (one version per paper)."""
        if not paper_ids:
            return {}
        rows = await self.session.execute(
            select(PaperChunk.paper_id, func.min(PaperChunk.index_version))
            .where(PaperChunk.paper_id.in_(paper_ids)).group_by(PaperChunk.paper_id))
        return {paper_id: version for paper_id, version in rows}

    async def replace(self, paper_id: int, passages: list[Passage], vectors: list[list[float]], version: str) -> None:
        """A paper's passages, all new: old ones (another version) are removed in the same transaction."""
        await self.session.execute(delete(PaperChunk).where(PaperChunk.paper_id == paper_id))
        self.session.add_all(PaperChunk(paper_id=paper_id, position=p.position, kind=p.kind, title=p.title,
                                        text=p.text, embedding=v, index_version=version)
                             for p, v in zip(passages, vectors, strict=True))
        await self.session.flush()

    async def count(self, paper_ids: list[int]) -> int:
        return await self.session.scalar(select(func.count()).where(PaperChunk.paper_id.in_(paper_ids))) or 0
