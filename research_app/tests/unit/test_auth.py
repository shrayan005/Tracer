"""Login (step 7.3): passwords, sessions, guessing protection, and the HTTP endpoints. Fakes: no Redis, no DB."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from argon2 import PasswordHasher

from app.dependencies import get_auth_service, get_job_service, get_research_service
from app.domain.errors import AuthError, ConflictError, ForbiddenError, InvalidInputError, TooManyAttemptsError
from app.main import app
from app.services import auth_service as auth_module
from app.services.auth_service import AuthService, AuthSettings, token_key

FAST = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)    # real argon2, tiny cost: quick tests
GOOD = "correct horse battery"


class FakeRedis:
    def __init__(self):
        self.data, self.ttl = {}, {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.data[key], self.ttl[key] = value.encode() if isinstance(value, str) else value, ex

    async def delete(self, key):
        self.data.pop(key, None)

    async def expire(self, key, seconds):
        self.ttl[key] = seconds

    async def incr(self, key):
        self.data[key] = int(self.data.get(key, 0)) + 1
        return self.data[key]


USERS: dict[str, SimpleNamespace] = {}


class FakeUsers:
    def __init__(self, session):
        pass

    async def get(self, user_id):
        return next((u for u in USERS.values() if u.id == user_id), None)

    async def get_by_email(self, email):
        return USERS.get(email.strip().lower())

    async def create(self, email, password_hash, display_name=None):
        user = SimpleNamespace(id=len(USERS) + 1, email=email, password_hash=password_hash,
                               display_name=display_name, is_active=True)
        USERS[email] = user
        return user


@asynccontextmanager
async def fake_unit_of_work():
    yield object()


@pytest.fixture(autouse=True)
def fake_users(monkeypatch):
    USERS.clear()
    monkeypatch.setattr(auth_module, "UserRepository", FakeUsers)


def service(redis=None, **settings) -> AuthService:
    return AuthService(fake_unit_of_work, redis or FakeRedis(), AuthSettings(**settings), hasher=FAST)


# ---------- the service ----------

async def test_password_is_stored_only_as_an_argon2_hash():
    await service().register("Ana@Example.com", GOOD, "Ana")
    stored = USERS["ana@example.com"].password_hash               # the email is stored lower-case
    assert stored.startswith("$argon2") and GOOD not in stored


async def test_register_checks_input_and_duplicates():
    s = service()
    with pytest.raises(InvalidInputError, match="valid email"):
        await s.register("not-an-email", GOOD)
    with pytest.raises(InvalidInputError, match="10-128 characters"):
        await s.register("ana@example.com", "short")
    await s.register("ana@example.com", GOOD)
    with pytest.raises(ConflictError, match="already exists"):
        await s.register("ANA@example.com", GOOD)
    with pytest.raises(ForbiddenError, match="registration is closed"):
        await service(registration_open=False).register("bo@example.com", GOOD)


async def test_login_creates_a_session_that_redis_stores_only_hashed():
    redis = FakeRedis()
    s = service(redis, session_days=14)
    await s.register("ana@example.com", GOOD)
    token, user = await s.login("ana@example.com", GOOD, "1.2.3.4")
    assert len(token) >= 40 and token not in str(redis.data)        # only sha256(token) is a key
    assert redis.ttl[token_key(token)] == 14 * 86400
    assert await s.user_id_for(token) == user.id
    assert await s.user_id_for("made-up-token") is None and await s.user_id_for(None) is None
    await s.logout(token)
    assert await s.user_id_for(token) is None                       # logout ends it on the server


async def test_every_failure_says_the_same_thing():
    s = service()
    await s.register("ana@example.com", GOOD)
    for email, password in (("ana@example.com", "wrong password!!"), ("nobody@example.com", GOOD)):
        with pytest.raises(AuthError) as error:
            await s.login(email, password, "1.2.3.4")
        assert str(error.value) == "wrong email or password"        # no hint which emails exist


async def test_an_account_without_a_password_cannot_log_in():
    USERS["local@localhost"] = SimpleNamespace(id=9, email="local@localhost", password_hash=None,
                                               display_name=None, is_active=True)
    with pytest.raises(AuthError):
        await service().login("local@localhost", "", "1.2.3.4")


async def test_too_many_failures_block_that_email_for_a_while():
    redis = FakeRedis()
    s = service(redis, max_failures_per_email=3, failure_window_minutes=15)
    await s.register("ana@example.com", GOOD)
    for _ in range(3):
        with pytest.raises(AuthError):
            await s.login("ana@example.com", "wrong password!!", "1.2.3.4")
    assert redis.ttl["login-fail:email:ana@example.com"] == 15 * 60
    with pytest.raises(TooManyAttemptsError) as blocked:
        await s.login("ana@example.com", GOOD, "1.2.3.4")         # even the right password, for now
    assert blocked.value.retry_after == 15 * 60


async def test_a_successful_login_resets_the_email_counter():
    redis = FakeRedis()
    s = service(redis, max_failures_per_email=3)
    await s.register("ana@example.com", GOOD)
    for _ in range(2):
        with pytest.raises(AuthError):
            await s.login("ana@example.com", "wrong password!!", "1.2.3.4")
    await s.login("ana@example.com", GOOD, "1.2.3.4")
    assert await redis.get("login-fail:email:ana@example.com") is None


# ---------- the HTTP endpoints ----------

@pytest.fixture
async def client():
    auth = service()
    app.dependency_overrides[get_auth_service] = lambda: auth
    app.dependency_overrides[get_research_service] = lambda: SimpleNamespace()
    app.dependency_overrides[get_job_service] = lambda: SimpleNamespace()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.mark.parametrize("method, path", [("GET", "/api/topics"), ("GET", "/api/topics/1"), ("GET", "/api/jobs"),
                                          ("GET", "/api/jobs/1"), ("POST", "/api/research"), ("GET", "/api/auth/me")])
async def test_everything_but_health_and_login_needs_a_session(client, method, path):
    r = await client.request(method, path, json={"topic": "creatine"})
    assert r.status_code == 401 and r.json() == {"detail": "not logged in"}


async def test_register_login_me_logout(client):
    r = await client.post("/api/auth/register", json={"email": "ana@example.com", "password": GOOD,
                                                      "display_name": "Ana"})
    assert r.status_code == 201 and r.json() == {"id": 1, "email": "ana@example.com", "display_name": "Ana"}
    cookie = r.headers["set-cookie"].lower()
    assert "rpa_session=" in cookie and "httponly" in cookie and "samesite=lax" in cookie
    assert "token" not in r.text                                     # the token is never in the body
    assert (await client.get("/api/auth/me")).json()["email"] == "ana@example.com"
    assert (await client.post("/api/auth/logout")).status_code == 204
    assert (await client.get("/api/auth/me")).status_code == 401    # the old cookie no longer works
    r = await client.post("/api/auth/login", json={"email": "ana@example.com", "password": GOOD})
    assert r.status_code == 200 and (await client.get("/api/auth/me")).status_code == 200


async def test_login_errors_over_http(client):
    await client.post("/api/auth/register", json={"email": "ana@example.com", "password": GOOD})
    client.cookies.clear()
    r = await client.post("/api/auth/login", json={"email": "ana@example.com", "password": "wrong password!!"})
    assert r.status_code == 401 and r.json()["detail"] == "wrong email or password"
    assert (await client.post("/api/auth/register", json={"email": "x", "password": GOOD})).status_code == 400
    assert (await client.post("/api/auth/register",
                              json={"email": "ana@example.com", "password": GOOD})).status_code == 409
    for _ in range(5):
        await client.post("/api/auth/login", json={"email": "ana@example.com", "password": "wrong password!!"})
    r = await client.post("/api/auth/login", json={"email": "ana@example.com", "password": GOOD})
    assert r.status_code == 429 and r.headers["retry-after"] == str(15 * 60)


async def test_a_locked_users_session_stops_working(client):
    await client.post("/api/auth/register", json={"email": "ana@example.com", "password": GOOD})
    assert (await client.get("/api/auth/me")).status_code == 200
    USERS["ana@example.com"].is_active = False                       # the admin locks the account
    assert (await client.get("/api/topics")).status_code == 401      # every endpoint, not only /me
