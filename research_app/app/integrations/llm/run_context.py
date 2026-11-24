"""
Which eval run (or job) and which USER an LLM call belongs to, without passing ids through every service.

A ContextVar is like a global variable that is private to the current async "task tree": everything
awaited inside `with run_scope(42):` (including tasks started with asyncio.gather) sees 42, and code
outside the block sees None. The gateway reads both when it records usage in llm_calls
(run_id: cost per job or eval run; user_id: each user's daily budget, step 7.4).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_RUN_ID: ContextVar[int | None] = ContextVar("llm_run_id", default=None)
_USER_ID: ContextVar[int | None] = ContextVar("llm_user_id", default=None)


def current_run_id() -> int | None:
    return _RUN_ID.get()


def current_user_id() -> int | None:
    return _USER_ID.get()


@contextmanager
def run_scope(run_id: int) -> Iterator[int]:
    token = _RUN_ID.set(run_id)
    try:
        yield run_id
    finally:
        _RUN_ID.reset(token)


@contextmanager
def user_scope(user_id: int | None) -> Iterator[int | None]:
    token = _USER_ID.set(user_id)
    try:
        yield user_id
    finally:
        _USER_ID.reset(token)
