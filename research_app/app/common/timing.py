"""Measure how long each stage of a request takes, and log it as one line.

    timings = Timings()
    with timings.stage("embed"): ...      # how long this block took
    timings.mark("first_token")           # how long since the request started (a moment, not a block)
    timings.log("ask")                    # -> "ask timings: embed=810ms first_token=1900ms total=7300ms"
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager

from app.common.logging import get_logger

log = get_logger(__name__)


class Timings:
    def __init__(self):
        self.start = time.perf_counter()   # perf_counter: a precise stopwatch, not the wall clock
        self.ms: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = round((time.perf_counter() - t0) * 1000)

    def mark(self, name: str) -> None:
        self.ms[name] = round((time.perf_counter() - self.start) * 1000)

    def log(self, label: str) -> None:
        self.mark("total")
        log.info("%s timings: %s", label, " ".join(f"{k}={v}ms" for k, v in self.ms.items()))
