"""
Dependency injection: the ONE place where objects are built and wired together
(the "composition root"). Controllers ask for services with FastAPI's Depends();
tests replace any of these functions with fakes via app.dependency_overrides.
"""

from collections.abc import AsyncIterator
from datetime import timedelta
from functools import lru_cache

import httpx
import redis.asyncio as aioredis
from arq import ArqRedis, create_pool
from arq.connections import RedisSettings
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.logging import get_logger
from app.config import get_settings
from app.domain.errors import AuthError
from app.domain.schemas import LlmCallRecord
from app.integrations.job_events import JobEvents
from app.integrations.llm.gateway import LlmGateway
from app.services.search_agent import SearchAgent
from app.integrations.llm.providers import build_providers
from app.integrations.sources.arxiv import ArxivClient
from app.integrations.sources.base import SourceClient
from app.integrations.sources.europepmc import EuropePmcClient
from app.integrations.sources.openalex import OpenAlexClient
from app.integrations.sources.semantic_scholar import SemanticScholarClient
from app.integrations.sources.unpaywall import UnpaywallClient
from app.integrations.downloader import SafeDownloader
from app.integrations.embeddings.gemini import GeminiEmbedder
from app.integrations.miss_cache import MissCache
from app.integrations.storage.file_storage import FileStorage, LocalFileStorage
from app.repositories.llm_call_repository import LlmCallRepository
from app.repositories.models import EMBEDDING_DIMENSIONS
from app.repositories.session import get_engine, get_session_factory, session_scope
from app.repositories.user_repository import UserRepository
from app.services.analysis_service import AnalysisService
from app.services.ask_service import AskService
from app.services.auth_service import AuthService, AuthSettings
from app.services.connection_service import ConnectionService
from app.services.gap_service import GapService
from app.services.synthesis_service import SynthesisService
from app.services.discovery_service import DiscoveryService
from app.services.extraction_service import ExtractionService
from app.services.fulltext_fetcher import FullTextFetcher
from app.services.fulltext_finder import FullTextFinder
from app.services.fulltext_service import FullTextService
from app.services.health_service import HealthService
from app.services.job_service import JobService
from app.services.index_service import IndexService
from app.services.planning_service import PlanningService
from app.services.research_service import ResearchService
from app.services.retrieval_service import RetrievalService
from app.services.screening_service import ScreeningService
from app.services.summary_service import SummaryService

log = get_logger(__name__)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """One database session per HTTP request; committed at the end, rolled back on error."""
    async with session_scope() as session:
        yield session


@lru_cache
def get_redis() -> aioredis.Redis:
    """One shared async Redis client (it keeps its own connection pool)."""
    return aioredis.Redis.from_url(get_settings().redis_url, socket_timeout=3)


@lru_cache
def get_events_redis() -> aioredis.Redis:
    """Redis for job-change signals (SSE step 6). Its own client: a follower WAITS on its subscription for up
    to 15 s, which the 3 s socket_timeout above would cut off. health_check_interval notices a dead link."""
    return aioredis.Redis.from_url(get_settings().redis_url, socket_connect_timeout=3, health_check_interval=30)


async def save_llm_call(record: LlmCallRecord) -> None:
    """Usage sink: every LLM attempt goes into llm_calls, in its own short transaction."""
    async with session_scope() as session:
        await LlmCallRepository(session).add(record)


async def get_current_user_id() -> int:
    """
    The default local user: who the CLI and the evals act as. The web API uses require_user_id (7.3).
    """
    async with session_scope() as session:
        user = await UserRepository(session).get_or_create(get_settings().default_user_email,
                                                           display_name="Local user")
        return user.id


@lru_cache
def get_judge_gateway() -> LlmGateway:
    """A SEPARATE gateway for the eval judge (milestone 6): only the judge provider, no fallback, so the
    judge can never quietly become the model whose work it is judging. Rate-limited -> it waits."""
    settings = get_settings()
    providers = [p for p in build_providers(settings) if p.name == settings.judge_provider]
    if not providers:
        raise RuntimeError(f"judge provider '{settings.judge_provider}' is not configured (missing API key?)")
    return LlmGateway(providers, usage_sink=save_llm_call, max_retries=settings.judge_max_retries,
                      backoff_base_s=settings.llm_backoff_base_seconds, backoff_max_s=settings.llm_backoff_max_seconds)


@lru_cache
def get_llm_gateway() -> LlmGateway:
    settings = get_settings()
    providers = build_providers(settings)
    log.info("LLM providers (in fallback order): %s", [p.name for p in providers])
    return LlmGateway(
        providers, usage_sink=save_llm_call, max_retries=settings.llm_max_retries,
        backoff_base_s=settings.llm_backoff_base_seconds, backoff_max_s=settings.llm_backoff_max_seconds,
        routes=settings.llm_routes, switch_after_s=settings.llm_switch_after_seconds,
        wait_for=settings.llm_wait_for,
        trace_purposes=set(settings.llm_trace_purposes), trace_max_chars=settings.llm_trace_max_chars,
    )


def get_health_service() -> HealthService:
    return HealthService(get_session_factory(), get_redis(), get_llm_gateway())


# ---------- Milestone 2: academic sources + discovery ----------

@lru_cache
def get_http_client() -> httpx.AsyncClient:
    """ONE shared HTTP connection pool for all academic sources (closed in close_resources)."""
    return httpx.AsyncClient(follow_redirects=True, limits=httpx.Limits(max_connections=20))


KNOWN_SOURCES = ("openalex", "europepmc", "arxiv", "semantic_scholar")


@lru_cache
def get_source_clients() -> dict[str, SourceClient]:
    """
    The academic sources switched on in ENABLED_SOURCES (.env).
    Semantic Scholar also needs its key (it 429s without one).
    A source left out here is invisible to the discovery service: choose_sources skips it.
    """
    settings = get_settings()
    enabled = {name.strip().lower() for name in settings.enabled_sources}
    unknown = enabled - set(KNOWN_SOURCES)
    if unknown:
        log.warning("ENABLED_SOURCES has unknown names %s; known: %s", sorted(unknown), KNOWN_SOURCES)

    http = get_http_client()
    timeout = settings.source_timeout_seconds
    clients: dict[str, SourceClient] = {}
    if "openalex" in enabled:
        clients["openalex"] = OpenAlexClient(http, api_key=settings.openalex_api_key, timeout=timeout)
    if "europepmc" in enabled:
        clients["europepmc"] = EuropePmcClient(http, timeout=timeout)
    if "arxiv" in enabled:
        clients["arxiv"] = ArxivClient(http, timeout=timeout)
    if "semantic_scholar" in enabled:
        if settings.semantic_scholar_api_key:
            clients["semantic_scholar"] = SemanticScholarClient(
                http, api_key=settings.semantic_scholar_api_key, timeout=timeout)
        else:
            log.info("semantic_scholar is enabled but SEMANTIC_SCHOLAR_API_KEY is not set: skipped")
    log.info("academic sources enabled: %s", list(clients))
    return clients


def get_planning_service() -> PlanningService:
    return PlanningService(get_llm_gateway())


def get_discovery_service() -> DiscoveryService:
    settings = get_settings()
    return DiscoveryService(get_source_clients(), get_planning_service(), session_scope,
                            results_per_query=settings.results_per_query,
                            search_timeout_s=settings.search_timeout_seconds,
                            total_budget_s=settings.search_total_budget_seconds,
                            agent=SearchAgent(get_llm_gateway()) if settings.search_agent_rounds > 0 else None,
                            agent_rounds=settings.search_agent_rounds)


# ---------- Milestone 3: screening, summaries, research pipeline ----------

def get_screening_service() -> ScreeningService:
    return ScreeningService(get_llm_gateway())


def get_summary_service() -> SummaryService:
    return SummaryService(get_llm_gateway(), max_parallel=get_settings().summary_max_parallel)


def get_research_service() -> ResearchService:
    settings = get_settings()
    return ResearchService(get_discovery_service(), get_screening_service(), get_summary_service(),
                           session_scope, prefilter_keep=settings.prefilter_keep,
                           screen_first=settings.screen_first, screen_step=settings.screen_step,
                           select_max=settings.select_max, select_min=settings.select_min,
                           fulltext_shortlist=settings.fulltext_shortlist,
                           fulltext=get_fulltext_service() if settings.fulltext_enabled else None,
                           synthesis=get_synthesis_service() if settings.analysis_enabled else None,
                           indexer=get_index_service() if settings.index_enabled else None)


def get_embedder() -> GeminiEmbedder:
    s = get_settings()
    if not s.gemini_api_key:
        raise RuntimeError("questions over papers need GEMINI_API_KEY (Gemini embeddings)")
    return GeminiEmbedder(get_http_client(), s.gemini_api_key, model=s.embedding_model,
                          dimensions=EMBEDDING_DIMENSIONS, usage_sink=save_llm_call)


def get_index_service() -> IndexService:
    return IndexService(session_scope, get_embedder())


def get_retrieval_service() -> RetrievalService:
    return RetrievalService(session_scope, get_embedder())


def get_ask_service() -> AskService:
    return AskService(session_scope, get_retrieval_service(), get_llm_gateway())


def get_synthesis_service() -> SynthesisService:
    return SynthesisService(get_analysis_service(), get_connection_service(), get_gap_service(), session_scope)


# ---------- Milestone 4: full text ----------

def get_fulltext_finder() -> FullTextFinder:
    """Unpaywall (only if UNPAYWALL_EMAIL is set) + Europe PMC (only if EUROPEPMC_FULLTEXT=true)."""
    settings = get_settings()
    http = get_http_client()
    timeout = settings.fulltext_lookup_timeout_seconds
    unpaywall = UnpaywallClient(http, settings.unpaywall_email, timeout=timeout) if settings.unpaywall_email else None
    europepmc = EuropePmcClient(http, timeout=timeout) if settings.europepmc_fulltext else None
    return FullTextFinder(europepmc, unpaywall, lookup_timeout_s=timeout, pmc_xml=settings.europepmc_fulltext)


def get_file_storage() -> FileStorage:
    """Local folder now; an S3/MinIO class with the same 4 methods when deployed (only this line changes)."""
    return LocalFileStorage(get_settings().files_dir)


def get_fulltext_fetcher() -> FullTextFetcher:
    settings = get_settings()
    downloader = SafeDownloader(get_http_client(), max_bytes=settings.download_max_mb * 1024 * 1024,
                                timeout_s=settings.download_timeout_seconds)
    misses = MissCache(get_redis(), ttl_seconds=int(settings.fulltext_miss_ttl_hours * 3600))
    return FullTextFetcher(get_fulltext_finder(), downloader, get_file_storage(), session_scope,
                           max_parallel=settings.download_max_parallel, misses=misses)


def get_extraction_service() -> ExtractionService:
    return ExtractionService(get_file_storage(), session_scope, max_pages=get_settings().extract_max_pages)


def get_analysis_service() -> AnalysisService:
    return AnalysisService(get_llm_gateway(), max_parallel=get_settings().summary_max_parallel)


def get_connection_service() -> ConnectionService:
    return ConnectionService(get_llm_gateway())


def get_gap_service() -> GapService:
    return GapService(get_llm_gateway())


def get_fulltext_service() -> FullTextService:
    return FullTextService(get_fulltext_fetcher(), get_extraction_service(), session_scope)


# ---------- login (step 7.3) ----------

def get_auth_service() -> AuthService:
    s = get_settings()
    return AuthService(session_scope, get_redis(), AuthSettings(
        session_days=s.session_days, max_failures_per_email=s.login_max_failures_per_email,
        max_failures_per_ip=s.login_max_failures_per_ip, failure_window_minutes=s.login_failure_window_minutes,
        registration_open=s.registration_open))


async def require_user_id(request: Request, auth: AuthService = Depends(get_auth_service)) -> int:
    """WHO is calling the web API: the user of the session cookie, or 401. (The CLI and the worker don't use
    this: the CLI acts as the default local user, the worker as the job's owner.)"""
    user_id = await auth.user_id_for(request.cookies.get(get_settings().session_cookie_name))
    if user_id is None:
        raise AuthError("not logged in")
    await auth.get_user(user_id)         # still exists and active? (a locked or deleted user's sessions stop)
    return user_id


# ---------- background jobs (step 7.1) ----------

_arq_pool: ArqRedis | None = None      # created on first use (it needs the running event loop)


async def get_arq_pool() -> ArqRedis:
    global _arq_pool
    if _arq_pool is None:
        _arq_pool = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    return _arq_pool


async def enqueue_job(job_id: int, defer_s: float = 0) -> None:
    """Put a job id on the worker's queue (defer_s: run it no earlier than that many seconds from now).
    The arq id 'job-<id>' stops the same job being queued twice."""
    await (await get_arq_pool()).enqueue_job("run_job", job_id, _job_id=f"job-{job_id}",
                                             _defer_by=timedelta(seconds=defer_s) if defer_s > 0 else None)


def get_job_service() -> JobService:
    return JobService(session_scope, enqueue_job, daily_token_budget=get_settings().daily_token_budget,
                      events=JobEvents(get_events_redis()),     # the worker publishes, the API listens
                      job_timeout_s=get_settings().job_timeout_minutes * 60)


async def close_resources() -> None:
    """
    Close shared connections cleanly (called when the web server stops or the CLI ends).
    Async connections belong to one event loop, so they must be closed before that loop ends.
    """
    global _arq_pool
    await get_engine().dispose()                 # close all pooled database connections
    if _arq_pool is not None:                    # the job queue's Redis connection (7.1)
        await _arq_pool.aclose()
        _arq_pool = None
    if get_redis.cache_info().currsize:          # only if a Redis client was ever created
        await get_redis().aclose()
        get_redis.cache_clear()
    if get_events_redis.cache_info().currsize:   # job-change signals (SSE step 6)
        await get_events_redis().aclose()
        get_events_redis.cache_clear()
    if get_llm_gateway.cache_info().currsize:    # only if the gateway was ever created
        await get_llm_gateway().aclose()         # close the LLM providers' HTTP connections
        get_llm_gateway.cache_clear()
    if get_judge_gateway.cache_info().currsize:  # the eval judge's own gateway (milestone 6)
        await get_judge_gateway().aclose()
        get_judge_gateway.cache_clear()
    if get_http_client.cache_info().currsize:    # only if the HTTP client was ever created
        await get_http_client().aclose()          # close the academic APIs' connections
        get_http_client.cache_clear()
        get_source_clients.cache_clear()          # the clients held the closed HTTP client
