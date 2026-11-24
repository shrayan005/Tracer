"""Data access for the users table."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.models import User


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, user_id: int) -> User | None:
        return await self.session.get(User, user_id)

    async def get_by_email(self, email: str) -> User | None:
        return await self.session.scalar(select(User).where(User.email == email.strip().lower()))

    async def create(self, email: str, password_hash: str, display_name: str | None = None) -> User:
        user = User(email=email.strip().lower(), password_hash=password_hash, display_name=display_name)
        self.session.add(user)
        await self.session.flush()
        return user

    async def get_or_create(self, email: str, display_name: str | None = None) -> User:
        email = email.strip().lower()                    # emails are case-insensitive: store one form
        user = await self.session.scalar(select(User).where(User.email == email))
        if user is None:
            user = User(email=email, display_name=display_name)
            self.session.add(user)
            await self.session.flush()
        return user
