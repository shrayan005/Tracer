"""
Async database engine and sessions.

psycopg 3 supports async itself, so DATABASE_URL (postgresql+psycopg://...) is unchanged.
Every database call is awaited, so the event loop stays free while Postgres works.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    # pool_pre_ping: checks a pooled connection still works before using it
    return create_async_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=get_engine(), expire_on_commit=False)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """
    A unit of work: commit if everything succeeds, roll back if anything fails.

        async with session_scope() as session:
            await PaperRepository(session).add(...)
    """
    session = get_session_factory()()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
