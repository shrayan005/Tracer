"""API test: the real app with real Postgres + Redis, but a fake LLM (no tokens spent)."""

import pytest
import redis.asyncio as aioredis
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.dependencies import get_health_service
from app.domain.errors import ProviderRejected
from app.integrations.llm.gateway import LlmGateway
from app.main import app
from app.services.health_service import HealthService
from tests.fakes import FakeProvider

pytestmark = pytest.mark.integration


async def no_sleep(_seconds):
    pass


def override_with(script):
    """
    Build a HealthService with fresh connections for this test, and close them afterwards.
    Async connections belong to one event loop; TestClient runs the app in its own loop,
    so we create them inside the request (an async generator dependency) and close them after.
    """
    async def dependency():
        engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
        redis_client = aioredis.Redis.from_url(get_settings().redis_url)
        gateway = LlmGateway([FakeProvider("fake", script)], sleep=no_sleep)
        try:
            yield HealthService(async_sessionmaker(engine, expire_on_commit=False), redis_client, gateway)
        finally:
            await redis_client.aclose()
            await engine.dispose()
    return dependency


def test_health_ok():
    app.dependency_overrides[get_health_service] = override_with(["ok"])
    try:
        response = TestClient(app).get("/api/health?deep=true")
    finally:
        app.dependency_overrides.clear()
    body = response.json()
    assert response.status_code == 200, body
    assert body["status"] == "ok"
    assert {c["name"]: c["ok"] for c in body["components"]} == {"database": True, "redis": True, "llm": True}


def test_health_degraded_when_llm_fails():
    app.dependency_overrides[get_health_service] = override_with(
        [ProviderRejected("fake", "bad key", status=401)])
    try:
        response = TestClient(app).get("/api/health?deep=true")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
