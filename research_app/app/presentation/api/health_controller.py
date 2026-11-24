"""Health controller: HTTP in, service call, HTTP out. No business logic here."""

from fastapi import APIRouter, Depends, Response, status

from app.dependencies import get_health_service
from app.domain.schemas import HealthReport
from app.services.health_service import HealthService

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthReport)
async def health(response: Response, deep: bool = False,
                 service: HealthService = Depends(get_health_service)) -> HealthReport:
    """Check database, Redis and LLM providers. ?deep=true also makes a tiny real LLM call.

    `async def`: runs on the event loop. Safe because everything below it awaits
    (no blocking calls), so other requests keep being served while we wait.
    """
    report = await service.check(deep=deep)
    if report.status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report
