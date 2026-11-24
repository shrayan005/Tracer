"""Data access for documents (stored files) and who may see them."""

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.models import Document, DocumentSection, UserDocument


class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, document_id: int) -> Document | None:
        return await self.session.get(Document, document_id)

    async def get_by_sha256(self, sha256: str) -> Document | None:
        return await self.session.scalar(select(Document).where(Document.sha256 == sha256))

    async def add(self, document: Document) -> Document:
        self.session.add(document)
        await self.session.flush()
        return document

    async def for_paper(self, paper_id: int) -> list[Document]:
        query = select(Document).where(Document.paper_id == paper_id).order_by(Document.id)
        return list(await self.session.scalars(query))

    async def link_user(self, user_id: int, document_id: int, original_filename: str | None = None) -> None:
        """Record that this user uploaded this file (does nothing if already linked)."""
        if await self.session.get(UserDocument, (user_id, document_id)) is None:
            self.session.add(UserDocument(user_id=user_id, document_id=document_id,
                                          original_filename=original_filename))
            await self.session.flush()

    async def replace_sections(self, document_id: int, sections: list[tuple[str, str | None, str]]) -> None:
        """Store a document's extracted text (kind, title, text) in reading order, replacing any old version."""
        await self.session.execute(delete(DocumentSection).where(DocumentSection.document_id == document_id))
        self.session.add_all(DocumentSection(document_id=document_id, position=i, kind=kind, title=title, text=text)
                             for i, (kind, title, text) in enumerate(sections))
        await self.session.flush()

    async def sections(self, document_id: int) -> list[DocumentSection]:
        query = (select(DocumentSection).where(DocumentSection.document_id == document_id)
                 .order_by(DocumentSection.position))
        return list(await self.session.scalars(query))

    async def visible_to(self, user_id: int, document_id: int) -> bool:
        """Open-access files: everyone. Uploads: only users who uploaded them."""
        document = await self.get(document_id)
        if document is None:
            return False
        if document.origin == "open_access":
            return True
        return await self.session.get(UserDocument, (user_id, document_id)) is not None
