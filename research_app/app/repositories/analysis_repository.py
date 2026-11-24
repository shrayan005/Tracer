"""Data access for paper_analyses: one stored analysis (claims + appraisal) per paper."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.schemas import AnalyzedPaper
from app.repositories.models import PaperAnalysisRow


class AnalysisRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, paper_id: int) -> PaperAnalysisRow | None:
        return await self.session.get(PaperAnalysisRow, paper_id)

    async def get_many(self, paper_ids: list[int]) -> dict[int, PaperAnalysisRow]:
        if not paper_ids:
            return {}
        rows = await self.session.scalars(select(PaperAnalysisRow).where(PaperAnalysisRow.paper_id.in_(paper_ids)))
        return {row.paper_id: row for row in rows}

    async def save(self, paper_id: int, analyzed: AnalyzedPaper, version: str) -> None:
        """Insert or replace this paper's analysis (an upsert)."""
        row = await self.get(paper_id)
        data = analyzed.model_dump(mode="json")
        if row is None:
            self.session.add(PaperAnalysisRow(paper_id=paper_id, version=version, based_on=analyzed.based_on,
                                              analysis=data))
        else:
            row.version, row.based_on, row.analysis = version, analyzed.based_on, data
        await self.session.flush()
