"""
Web entry point.

Run (in WSL, from research_app/), each in its own terminal:
    WATCHFILES_FORCE_POLLING=true uvicorn app.main:app --port 8010 --reload
    arq app.worker.WorkerSettings          # the background worker: research and analyze jobs (7.1)
Then open http://localhost:8010/docs (every endpoint, try them in the browser)

Port 8010 because another app on this machine already uses 8000.
WATCHFILES_FORCE_POLLING: WSL can't see file-change events from the Windows drive,
so --reload must poll for changes.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.common.logging import setup_logging
from app.config import get_settings
from app.dependencies import close_resources
from app.domain.errors import (AuthError, ConflictError, ForbiddenError, InvalidInputError, NotFoundError,
                               NotReadyError, TooManyAttemptsError)
from app.presentation.api import (ask_controller, auth_controller, health_controller, jobs_controller,
                                  topics_controller)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Code before `yield` runs when the server starts; after `yield`, when it stops."""
    yield
    await close_resources()      # close database and Redis connections cleanly on shutdown


def add_error_handlers(app: FastAPI) -> None:
    """Service errors -> HTTP status, the same for every endpoint. Another user's topic or job raises
    the same NotFoundError as a missing one, so the 404 reveals nothing about other users."""
    def handler(code: int):
        async def respond(_request: Request, error: Exception) -> JSONResponse:
            return JSONResponse(status_code=code, content={"detail": str(error)})
        return respond

    # NotReadyError is a NotFoundError, but its own handler wins (handlers are matched by the exact class first)
    app.add_exception_handler(NotReadyError, handler(status.HTTP_409_CONFLICT))
    app.add_exception_handler(NotFoundError, handler(status.HTTP_404_NOT_FOUND))
    app.add_exception_handler(ConflictError, handler(status.HTTP_409_CONFLICT))
    app.add_exception_handler(AuthError, handler(status.HTTP_401_UNAUTHORIZED))
    app.add_exception_handler(ForbiddenError, handler(status.HTTP_403_FORBIDDEN))
    app.add_exception_handler(InvalidInputError, handler(status.HTTP_400_BAD_REQUEST))

    async def too_many(_request: Request, error: TooManyAttemptsError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS, content={"detail": str(error)},
                            headers={"Retry-After": str(error.retry_after)})
    app.add_exception_handler(TooManyAttemptsError, too_many)


def create_app() -> FastAPI:
    setup_logging(get_settings().log_level)
    app = FastAPI(title="Research Paper Assistant", version="0.4.0", lifespan=lifespan)
    add_error_handlers(app)
    app.include_router(health_controller.router)
    app.include_router(auth_controller.router)
    app.include_router(jobs_controller.router)
    app.include_router(topics_controller.router)
    app.include_router(ask_controller.router)
    return app


app = create_app()
