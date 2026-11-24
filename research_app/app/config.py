"""
Application settings, read once from research_app/.env (or real environment variables).

Every other module asks get_settings() instead of reading os.environ itself,
so all configuration lives in one typed, validated place.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent  # research_app/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    # --- Infrastructure ---
    database_url: str
    redis_url: str = "redis://localhost:6379/0"

    # --- Login (step 7.3) ---
    session_days: int = 14                       # a session lasts this long after its LAST use
    session_cookie_name: str = "rpa_session"
    cookie_secure: bool = False                  # True in production (HTTPS only); False for http://localhost
    registration_open: bool = True              # False: only the admin creates accounts (CLI: users add)
    login_max_failures_per_email: int = 5        # then that email is blocked for the window below
    login_max_failures_per_ip: int = 20
    login_failure_window_minutes: int = 15

    # --- Background jobs (milestone 7) ---
    worker_max_jobs: int = 2                     # jobs one worker runs at the same time
    job_timeout_minutes: int = 15                # a job running longer is stopped and marked failed
    # LLM tokens one user may use per day (UTC), checked when a job is submitted; 0 = no limit.
    # Measured: a research ~40-60k tokens, an analysis ~60k (7.1). The free tiers are shared by all users.
    daily_token_budget: int = 250_000
    log_level: str = "INFO"

    # --- LLM providers (a provider without a key is simply skipped) ---
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None

    # Order to try providers in; the gateway falls back left to right.
    # In .env write it as JSON, e.g. LLM_PROVIDER_ORDER=["groq","gemini"]
    llm_provider_order: list[str] = ["groq", "gemini", "openrouter"]
    groq_model: str = "openai/gpt-oss-120b"
    gemini_model: str = "gemini-3.5-flash-lite"
    openrouter_model: str = "inclusionai/ling-3.2-flash:free"
    # Per-task provider order: these are tried FIRST for that purpose, then llm_provider_order.
    # Spreads work over several free-tier limits. In .env as JSON, e.g.
    #   LLM_ROUTES={"summarize":["gemini"],"screen":["groq"]}
    # "connect" reasons across ~30 claims at once: the bigger reasoning model (groq) does it far better
    llm_routes: dict[str, list[str]] = {"summarize": ["gemini"], "analyze": ["gemini"], "connect": ["groq"],
                                        "gaps": ["gemini"], "chat": ["gemini"]}

    # --- LLM reliability ---
    llm_timeout_seconds: float = 60
    llm_max_retries: int = 3       # retries per provider on temporary errors (429, 5xx, timeouts)
    llm_backoff_base_seconds: float = 1.0
    llm_backoff_max_seconds: float = 30.0
    # A provider asks to wait longer than this (e.g. Groq's free-tier 429 "retry after 18s")?
    # Try the next provider now instead; the waiting one stays as a last resort.
    llm_switch_after_seconds: float = 5.0
    # ...except tasks whose quality depends on the routed model: wait up to this long for it first.
    # connect = one call per topic, and a smaller model finds far fewer connections (measured 6.6)
    llm_wait_for: dict[str, float] = {"connect": 60.0}

    # --- Academic sources ---
    # On/off switch: only these sources are used. In .env write it as JSON, e.g.
    #   ENABLED_SOURCES=["openalex","arxiv"]
    # Europe PMC is off by default (23 Nov 2026): it kept failing (HTTP 503, 20 s timeouts) and OpenAlex
    # already lists the same PubMed papers. Add "europepmc" back to search it again.
    # (semantic_scholar additionally needs SEMANTIC_SCHOLAR_API_KEY)
    enabled_sources: list[str] = ["openalex", "arxiv", "semantic_scholar"]
    openalex_api_key: str | None = None          # free key = bigger daily budget
    semantic_scholar_api_key: str | None = None  # optional; that source is skipped without it
    source_timeout_seconds: float = 30
    results_per_query: int = 15                  # papers requested per query, per source
    search_timeout_seconds: float = 20          # time budget for ONE search; slower ones are skipped
    search_total_budget_seconds: float = 30      # deadline for ALL searches of one round together
    search_agent_rounds: int = 2                 # extra search rounds the search agent may add (0 = off)

    # --- Selection (milestone 3) ---
    prefilter_keep: int = 40                     # most candidates that MAY be screened by the LLM
    screen_first: int = 20                       # adaptive screening: first round; stop if enough are relevant
    screen_step: int = 20                        # papers per extra round when the first wasn't enough
    reports_dir: str = "reports"                 # where `research` writes the Markdown reports
    select_max: int = 10                         # most papers in the final selection
    fulltext_shortlist: int = 12                 # best papers whose full text is tried; the final pick
                                                 # prefers those we could read (measured: ~40-60% have one)
    select_min: int = 5                          # fewer than this -> relax the gate + warn
    summary_max_parallel: int = 2                # summary LLM calls at the same time (free tier: keep low)

    # --- Users (milestone 4; real login in milestone 7) ---
    default_user_email: str = "local@localhost"   # the CLI's user; same email as the 4.1b migration

    # --- File storage (milestone 4) ---
    files_dir: str = str(ROOT_DIR / "data" / "files")   # original PDFs/XML, named by SHA-256
    unpaywall_email: str | None = None            # required by Unpaywall; without it, Unpaywall is skipped
    fulltext_lookup_timeout_seconds: float = 10   # per source, per paper; slow sources are skipped
    europepmc_fulltext: bool = False              # Europe PMC XML full texts: best quality, but the service was
                                                  # often down (only 3 of 29 stored full texts came from it)
    download_max_mb: int = 50                     # bigger files are refused while streaming
    download_timeout_seconds: float = 60          # whole download, redirects included
    download_max_parallel: int = 3                # downloads at the same time (be polite to publishers)
    extract_max_pages: int = 100                  # longer PDFs are refused (theses, books)
    fulltext_enabled: bool = True                 # research: get free full texts for the selected papers
    fulltext_miss_ttl_hours: float = 24           # don't retry papers with no full text for this long
    analysis_enabled: bool = True                 # research: claims, quality, connections, gaps (milestone 5)

    # --- Questions over your papers (step 7.6) ---
    index_enabled: bool = True                    # research/analyze also prepare passages for questions
    embedding_model: str = "gemini-embedding-001" # Gemini's embedding API (uses GEMINI_API_KEY)
    # the vector size is fixed by the database column (EMBEDDING_DIMENSIONS in models.py): 768

    # --- Evaluation (milestone 6) ---
    # The judge must not be the model whose work it judges (analysis runs on gemini), and it must not
    # quietly fall back to it: it gets its own gateway with ONLY this provider, which waits when rate-limited.
    judge_provider: str = "groq"
    judge_max_retries: int = 6

    # --- Other APIs ---
    tavily_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse them (cached)."""
    return Settings()
