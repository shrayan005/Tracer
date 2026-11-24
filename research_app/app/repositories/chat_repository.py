"""Data access for chats and chat_messages (step 7.6b)."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.models import Chat, ChatMessage, Paper, Topic, TopicPaper


class ChatRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, user_id: int, topic_id: int, title: str) -> Chat:
        chat = Chat(user_id=user_id, topic_id=topic_id, title=title[:200])
        self.session.add(chat)
        await self.session.flush()
        return chat

    async def get(self, chat_id: int) -> Chat | None:
        return await self.session.get(Chat, chat_id)

    async def list(self, user_id: int, topic_id: int | None = None, limit: int = 50) -> list[Chat]:
        query = select(Chat).where(Chat.user_id == user_id)
        if topic_id is not None:
            query = query.where(Chat.topic_id == topic_id)
        return list(await self.session.scalars(query.order_by(Chat.updated_at.desc()).limit(limit)))

    async def messages(self, chat_id: int) -> list[ChatMessage]:
        query = select(ChatMessage).where(ChatMessage.chat_id == chat_id).order_by(ChatMessage.id)
        return list(await self.session.scalars(query))

    async def add(self, chat: Chat, role: str, content: str, answer: dict | None = None) -> ChatMessage:
        message = ChatMessage(chat_id=chat.id, role=role, content=content, answer=answer)
        self.session.add(message)
        chat.updated_at = func.now()               # the chat list is sorted by the latest activity
        await self.session.flush()
        return message

    async def delete(self, chat: Chat) -> None:
        await self.session.delete(chat)

    async def user_papers(self, user_id: int, topic_id: int | None = None) -> list[tuple[int, int, int, Paper]]:
        """(paper id, topic id, rank, paper) of every selected paper in the user's topics (newest topic first)."""
        query = (select(TopicPaper.paper_id, TopicPaper.topic_id, TopicPaper.rank, Paper)
                 .join(Paper, Paper.id == TopicPaper.paper_id).join(Topic, Topic.id == TopicPaper.topic_id)
                 .where(Topic.user_id == user_id, TopicPaper.rank.is_not(None)))
        if topic_id is not None:
            query = query.where(TopicPaper.topic_id == topic_id)
        return [tuple(row) for row in await self.session.execute(query.order_by(Topic.id.desc(), TopicPaper.rank))]
