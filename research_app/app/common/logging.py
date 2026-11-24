"""Logging setup: one consistent format for the web server, worker and CLI."""

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level.upper(), format=_FORMAT, force=True)
    # Third-party libraries are noisy at INFO; keep them quieter.
    for noisy in ("httpx", "httpx2", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
