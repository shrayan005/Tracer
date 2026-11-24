"""Server-Sent Events: the text format, and the response settings every stream needs."""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

HEADERS = {"Cache-Control": "no-cache",      # never cache a live stream
           "X-Accel-Buffering": "no"}        # nginx (milestone 8): don't hold events back


def sse(event: str, data: Any) -> str:
    """One event: an `event:` line, a `data:` line (JSON, one line), and a blank line that ends it."""
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json")
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def keep_alive() -> str:
    """A comment line: browsers ignore it, but it shows proxies (and us) that the stream is still alive."""
    return ": still here\n\n"


def retry_after(ms: int) -> str:
    """How long the browser's EventSource waits before reconnecting when the stream ends."""
    return f"retry: {ms}\n\n"


def stream(events: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(events, media_type="text/event-stream", headers=HEADERS)
