"""
Progress messages from long runs ("screening papers 1-20 ...").

Async (step 7.1): a background job saves every message to the database, which must be awaited.
The CLI passes `printer()`; code that doesn't care passes nothing and gets `quiet`.
"""

from collections.abc import Awaitable, Callable

Progress = Callable[[str], Awaitable[None]]


async def quiet(_message: str) -> None:
    """The default: progress goes nowhere."""


def printer(prefix: str = "  · ") -> Progress:
    """Progress for the command line: print each message."""
    async def say(message: str) -> None:
        print(f"{prefix}{message}")
    return say
