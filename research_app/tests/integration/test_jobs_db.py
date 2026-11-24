"""The database's guarantee (step 7.4): at most ONE active research job per user, even under a race.
Real Postgres; run with: python -m pytest -m integration tests/integration/test_jobs_db.py"""

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from app.dependencies import close_resources
from app.repositories.job_repository import JobRepository
from app.repositories.models import Job, User
from app.repositories.session import session_scope

pytestmark = pytest.mark.integration
EMAIL = "integration-jobs-74@example.invalid"


async def test_a_second_active_research_is_refused_by_the_database():
    try:
        async with session_scope() as s:
            user = User(email=EMAIL)
            s.add(user)
            await s.flush()
            user_id = user.id
            await JobRepository(s).create(user_id=user_id, kind="research", params={"topic": "a"})
        with pytest.raises(IntegrityError, match="uq_jobs_one_active_research"):
            async with session_scope() as s:               # skips the code check: only the index stops it
                await JobRepository(s).create(user_id=user_id, kind="research", params={"topic": "b"})
        async with session_scope() as s:                   # analyze jobs are not limited ...
            await JobRepository(s).create(user_id=user_id, kind="analyze", params={"topic_id": 1})
            first = (await JobRepository(s).active(user_id, "research"))[0]
            await JobRepository(s).finish(first.id, topic_id=None)
        async with session_scope() as s:                   # ... and once the first research ended, a new one is fine
            await JobRepository(s).create(user_id=user_id, kind="research", params={"topic": "c"})
    finally:
        async with session_scope() as s:
            await s.execute(delete(Job).where(Job.user_id.in_(select(User.id).where(User.email == EMAIL))))
            await s.execute(delete(User).where(User.email == EMAIL))
        await close_resources()
