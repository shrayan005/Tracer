"""
Login (step 7.3): email + password, sessions in Redis, a cookie in the browser.

    password    stored only as an argon2 hash (slow on purpose: guessing is expensive). Hashing runs in a
                thread so it never blocks the event loop.
    session     a random token (256 bits) in an httpOnly cookie: JavaScript can't read it, so an XSS bug
                can't steal it. Redis stores only the token's SHA-256, so a Redis leak gives no usable
                tokens. Sessions slide: each use extends them; logout deletes them.
    guessing    failed logins are counted per email and per IP in Redis; too many -> 429 for a while.
    leaks       "wrong email or password" for every failure, and an unknown email still costs one hash
                check, so neither the message nor the timing says which emails have accounts.
"""

import asyncio
import hashlib
import re
import secrets
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.domain.errors import (AuthError, ConflictError, ForbiddenError, InvalidInputError,
                               TooManyAttemptsError)
from app.domain.schemas import UserInfo
from app.repositories.user_repository import UserRepository
from app.services.discovery_service import UnitOfWork

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD, MAX_PASSWORD = 10, 128
WRONG = "wrong email or password"


@dataclass
class AuthSettings:
    session_days: int = 14
    max_failures_per_email: int = 5
    max_failures_per_ip: int = 20
    failure_window_minutes: int = 15
    registration_open: bool = True


def token_key(token: str) -> str:
    return "session:" + hashlib.sha256(token.encode()).hexdigest()


class AuthService:
    def __init__(self, unit_of_work: UnitOfWork, redis, settings: AuthSettings | None = None,
                 hasher: PasswordHasher | None = None):
        self.unit_of_work = unit_of_work
        self.redis = redis
        self.settings = settings or AuthSettings()
        self.hasher = hasher or PasswordHasher()
        self._dummy_hash: str | None = None          # for unknown emails: same work as a real check

    # ---------- passwords ----------

    async def hash_password(self, password: str) -> str:
        check_password(password)
        return await asyncio.to_thread(self.hasher.hash, password)

    async def _verify(self, password_hash: str | None, password: str) -> bool:
        if password_hash is None:                    # unknown email, or an account without a password
            self._dummy_hash = self._dummy_hash or await asyncio.to_thread(self.hasher.hash, "not-a-real-password")
            password_hash = self._dummy_hash
            await asyncio.to_thread(self._safe_verify, password_hash, password)
            return False
        return await asyncio.to_thread(self._safe_verify, password_hash, password)

    def _safe_verify(self, password_hash: str, password: str) -> bool:
        try:
            return self.hasher.verify(password_hash, password)
        except (VerificationError, InvalidHashError):
            return False

    # ---------- accounts ----------

    async def register(self, email: str, password: str, display_name: str | None = None) -> UserInfo:
        if not self.settings.registration_open:
            raise ForbiddenError("registration is closed: ask the admin for an account")
        email = check_email(email)
        password_hash = await self.hash_password(password)
        async with self.unit_of_work() as session:
            users = UserRepository(session)
            if await users.get_by_email(email) is not None:
                raise ConflictError("an account with this email already exists")
            user = await users.create(email, password_hash, display_name)
            return UserInfo(id=user.id, email=user.email, display_name=user.display_name)

    async def set_password(self, email: str, password: str) -> UserInfo:
        """Admin/CLI: give an existing user (e.g. the local CLI user) a password, or reset it."""
        password_hash = await self.hash_password(password)
        async with self.unit_of_work() as session:
            user = await UserRepository(session).get_by_email(email)
            if user is None:
                raise InvalidInputError(f"no user with email {email!r}")
            user.password_hash = password_hash
            return UserInfo(id=user.id, email=user.email, display_name=user.display_name)

    async def get_user(self, user_id: int) -> UserInfo:
        async with self.unit_of_work() as session:
            user = await UserRepository(session).get(user_id)
            if user is None or not user.is_active:
                raise AuthError("not logged in")
            return UserInfo(id=user.id, email=user.email, display_name=user.display_name)

    # ---------- sessions ----------

    async def login(self, email: str, password: str, client_ip: str) -> tuple[str, UserInfo]:
        """(session token, user). Raises AuthError (401) or TooManyAttemptsError (429)."""
        email = email.strip().lower()
        await self._check_not_blocked(email, client_ip)
        async with self.unit_of_work() as session:
            user = await UserRepository(session).get_by_email(email)
            password_hash = user.password_hash if user is not None and user.is_active else None
            info = UserInfo(id=user.id, email=user.email, display_name=user.display_name) if user else None
        if not await self._verify(password_hash, password):
            await self._count_failure(email, client_ip)
            raise AuthError(WRONG)
        await self.redis.delete(f"login-fail:email:{email}")
        token = secrets.token_urlsafe(32)
        await self.redis.set(token_key(token), str(info.id), ex=self.settings.session_days * 86400)
        return token, info

    async def user_id_for(self, token: str | None) -> int | None:
        """The logged-in user's id, or None. Each use slides the session's expiry forward."""
        if not token:
            return None
        key = token_key(token)
        value = await self.redis.get(key)
        if value is None:
            return None
        await self.redis.expire(key, self.settings.session_days * 86400)
        return int(value)

    async def logout(self, token: str | None) -> None:
        if token:
            await self.redis.delete(token_key(token))

    # ---------- guessing protection ----------

    async def _check_not_blocked(self, email: str, ip: str) -> None:
        s = self.settings
        for key, limit in ((f"login-fail:email:{email}", s.max_failures_per_email),
                           (f"login-fail:ip:{ip}", s.max_failures_per_ip)):
            if int(await self.redis.get(key) or 0) >= limit:
                raise TooManyAttemptsError(f"too many failed logins: try again in {s.failure_window_minutes} "
                                           f"minutes", retry_after=s.failure_window_minutes * 60)

    async def _count_failure(self, email: str, ip: str) -> None:
        for key in (f"login-fail:email:{email}", f"login-fail:ip:{ip}"):
            if await self.redis.incr(key) == 1:          # the first failure starts the window
                await self.redis.expire(key, self.settings.failure_window_minutes * 60)


def check_email(email: str) -> str:
    email = email.strip().lower()
    if len(email) > 320 or not EMAIL.match(email):
        raise InvalidInputError("that is not a valid email address")
    return email


def check_password(password: str) -> None:
    if not MIN_PASSWORD <= len(password) <= MAX_PASSWORD:
        raise InvalidInputError(f"the password must be {MIN_PASSWORD}-{MAX_PASSWORD} characters long")
