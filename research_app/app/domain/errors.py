"""
Error types shared by all layers.

Integration code translates library-specific errors (e.g. openai.RateLimitError)
into these, so services never depend on a particular SDK.
"""


class AppError(Exception):
    """Base class for all errors raised by this app."""


# ---------- LLM errors ----------

class LlmError(AppError):
    pass


class ProviderError(LlmError):
    """One LLM provider failed one request."""

    retryable = False

    def __init__(self, provider: str, message: str, status: int | None = None,
                 retry_after: float | None = None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.status = status
        self.retry_after = retry_after  # seconds the provider asked us to wait, if any


class ProviderRateLimited(ProviderError):
    """429: too many requests or quota reached. Waiting may help."""
    retryable = True


class ProviderUnavailable(ProviderError):
    """Timeouts, connection problems, 5xx, empty responses. Usually temporary."""
    retryable = True


class ProviderRejected(ProviderError):
    """4xx other than 429 (bad key, unknown model, bad request). Retrying the same provider won't help."""
    retryable = False


class AllProvidersFailed(LlmError):
    """Every provider failed; holds each provider's last error."""

    def __init__(self, errors: list[ProviderError]):
        summary = "; ".join(str(e) for e in errors) or "no providers configured"
        super().__init__(f"All LLM providers failed: {summary}")
        self.errors = errors


class StreamInterrupted(LlmError):
    """A streamed answer broke off after its first words were shown: too late to switch to another model."""

    def __init__(self, provider: str, message: str):
        super().__init__(f"{provider}: the answer stopped mid-way ({message})")
        self.provider = provider


class StructuredOutputError(LlmError):
    """The model kept returning output that did not match the requested schema."""


# ---------- Academic source errors ----------

class SourceError(AppError):
    """An academic source (OpenAlex, Europe PMC, arXiv, ...) failed."""

    def __init__(self, source: str, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(f"[{source}] {message}")
        self.source = source          # which source failed, e.g. "openalex"
        self.status = status          # HTTP status code if there was one (429, 500, ...)
        self.retryable = retryable    # could trying again later help?


class DownloadError(AppError):
    """A full-text download was refused or failed (blocked address, too big, not a PDF, HTTP error...)."""

    def __init__(self, url: str, reason: str):
        super().__init__(f"{reason} ({url[:120]})")
        self.url = url
        self.reason = reason


# ---------- Lookup errors ----------

class NotFoundError(AppError):
    """Something the caller asked for (e.g. a topic id) does not exist or has no results yet."""


class NotReadyError(NotFoundError):
    """It exists but has no results yet (e.g. a topic still being researched). API: 409, not 404.
    A NotFoundError subclass, so code that only knows 'not found' still handles it."""


class ConflictError(AppError):
    """The request clashes with the current state (e.g. retrying a job that did not fail, or starting a
    second research while one is running). API: 409."""


# ---------- Login and request errors (step 7.3) ----------

class AuthError(AppError):
    """Not logged in, or wrong email/password. API: 401."""


class ForbiddenError(AppError):
    """Logged in (or not), but this is not allowed, e.g. registration is closed. API: 403."""


class InvalidInputError(AppError):
    """A request the service refuses as wrong input (bad email, too short password). API: 400."""


class TooManyAttemptsError(AppError):
    """Too many failed logins for a while. API: 429 with Retry-After."""

    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExceededError(TooManyAttemptsError):
    """A user's daily LLM budget is used up (step 7.4). API: 429 with Retry-After = seconds to midnight UTC."""


class ExtractionError(AppError):
    """A stored file could not be turned into text (scanned PDF, broken file, no article body...)."""
