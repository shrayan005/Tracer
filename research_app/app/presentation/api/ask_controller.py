"""Questions over a topic's papers, chats, and passage search (step 7.6b); the answer as a live stream (SSE)."""

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.dependencies import get_ask_service, get_job_service, require_user_id
from app.domain.errors import AppError, LlmError
from app.domain.schemas import AskResult, ChatInfo, JobInfo, Readiness, SearchHit
from app.integrations.llm.run_context import user_scope
from app.presentation.api.sse import sse, stream
from app.services.ask_service import AskService
from app.services.job_service import JobService

router = APIRouter(prefix="/api", tags=["questions"])
log = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["What daily dose of creatine did the trials use?"])
    chat_id: int | None = None       # continue this chat (follow-up questions); None = a new chat


@router.post("/topics/{topic_id}/ask", response_model=AskResult)
async def ask(topic_id: int, body: AskRequest, asking: AskService = Depends(get_ask_service),
              jobs: JobService = Depends(get_job_service), user_id: int = Depends(require_user_id)) -> AskResult:
    """Answer from this topic's papers only; every sentence cites a passage whose quote code checked.
    409 if the topic isn't prepared for questions yet; 429 if the daily budget is used up."""
    await jobs.check_budget(user_id)
    with user_scope(user_id):                                   # the LLM call counts toward this user's budget
        return await asking.ask(topic_id, user_id, body.question.strip(), body.chat_id)


@router.post("/topics/{topic_id}/ask/stream", response_class=StreamingResponse,
             responses={200: {"content": {"text/event-stream": {}},
                              "description": "events: stage (searching, reading, writing, checking); while "
                                             "writing: text ({index, text}: a sentence so far), sentence "
                                             "({index, part}: finished and checked; part null = dropped), "
                                             "reset (forget the preview); then done (the AskResult) or "
                                             "error ({detail})"}})
async def ask_stream(topic_id: int, body: AskRequest, asking: AskService = Depends(get_ask_service),
                     jobs: JobService = Depends(get_job_service),
                     user_id: int = Depends(require_user_id)) -> StreamingResponse:
    """The same answer as /ask, but each step is sent the moment it happens (Server-Sent Events).
    Budget (429), ownership (404) and readiness (409) are checked BEFORE the stream starts, so they still
    come back as normal JSON errors. Once streaming, the status is already 200: a failure is an `error` event."""
    await jobs.check_budget(user_id)
    steps = await asking.start_ask(topic_id, user_id, body.question.strip(), body.chat_id)

    async def events() -> AsyncIterator[str]:
        # user_scope INSIDE the generator: FastAPI runs it after this function has returned,
        # so a `with` around the return would already be closed when the LLM is called
        with user_scope(user_id):
            try:
                async for event, data in steps:
                    yield sse(event, data)
            except LlmError:                         # raw provider messages are not for users
                log.exception("answer stream: the LLM failed")
                yield sse("error", {"detail": "the AI service is busy or unavailable right now; please try again in a minute"})
            except AppError as e:                    # our other errors have messages meant for the user
                yield sse("error", {"detail": str(e)})
            except Exception:
                log.exception("answer stream failed")
                yield sse("error", {"detail": "something went wrong while answering; please try again"})

    return stream(events())


@router.get("/topics/{topic_id}/readiness", response_model=Readiness)
async def readiness(topic_id: int, asking: AskService = Depends(get_ask_service),
                    user_id: int = Depends(require_user_id)) -> Readiness:
    return await asking.readiness(topic_id, user_id)


@router.post("/topics/{topic_id}/prepare", response_model=JobInfo, status_code=status.HTTP_202_ACCEPTED)
async def prepare(topic_id: int, asking: AskService = Depends(get_ask_service),
                  jobs: JobService = Depends(get_job_service), user_id: int = Depends(require_user_id)) -> JobInfo:
    """Prepare a topic's papers for questions, as a background job (minutes on Gemini's free tier)."""
    await asking.readiness(topic_id, user_id)                   # 404 if not yours, before anything is queued
    return await jobs.submit("index", {"topic_id": topic_id}, user_id)


@router.get("/chats", response_model=list[ChatInfo])
async def list_chats(topic_id: int | None = None, asking: AskService = Depends(get_ask_service),
                     user_id: int = Depends(require_user_id)) -> list[ChatInfo]:
    return await asking.chats(user_id, topic_id)


@router.get("/chats/{chat_id}", response_model=ChatInfo)
async def get_chat(chat_id: int, asking: AskService = Depends(get_ask_service),
                   user_id: int = Depends(require_user_id)) -> ChatInfo:
    return await asking.chat(chat_id, user_id)


@router.delete("/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: int, asking: AskService = Depends(get_ask_service),
                      user_id: int = Depends(require_user_id)) -> None:
    await asking.delete_chat(chat_id, user_id)


@router.get("/search", response_model=list[SearchHit])
async def search(q: str = Query(min_length=2, max_length=300), topic_id: int | None = None,
                 asking: AskService = Depends(get_ask_service), user_id: int = Depends(require_user_id)) -> list[SearchHit]:
    """Passages across your papers (or one topic's), by meaning and keywords. No LLM."""
    with user_scope(user_id):
        return await asking.search(user_id, q.strip(), topic_id)
