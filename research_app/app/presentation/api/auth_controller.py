"""Login endpoints (step 7.3). The session token travels ONLY in an httpOnly cookie, never in a JSON body,
so the React app never touches it and an XSS bug can't read it."""

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.dependencies import get_auth_service, require_user_id
from app.domain.schemas import UserInfo
from app.services.auth_service import MAX_PASSWORD, AuthService

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Credentials(BaseModel):
    email: str = Field(max_length=320, examples=["you@example.com"])
    password: str = Field(max_length=MAX_PASSWORD)


class Registration(Credentials):
    display_name: str | None = Field(default=None, max_length=200)


def set_session_cookie(response: Response, token: str) -> None:
    s = get_settings()
    response.set_cookie(
        s.session_cookie_name, token, max_age=s.session_days * 86400,
        httponly=True,          # JavaScript can't read it
        secure=s.cookie_secure,  # HTTPS only in production
        samesite="lax",         # not sent on cross-site POSTs: blocks the classic CSRF attack
        path="/")


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.post("/register", response_model=UserInfo, status_code=status.HTTP_201_CREATED)
async def register(body: Registration, request: Request, response: Response,
                   auth: AuthService = Depends(get_auth_service)) -> UserInfo:
    """Create an account and log in (403 if registration is closed, 409 if the email is taken)."""
    await auth.register(body.email, body.password, body.display_name)
    token, user = await auth.login(body.email, body.password, client_ip(request))
    set_session_cookie(response, token)
    return user


@router.post("/login", response_model=UserInfo)
async def login(body: Credentials, request: Request, response: Response,
                auth: AuthService = Depends(get_auth_service)) -> UserInfo:
    """401 'wrong email or password' for every failure; 429 after too many."""
    token, user = await auth.login(body.email, body.password, client_ip(request))
    set_session_cookie(response, token)
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request, response: Response, auth: AuthService = Depends(get_auth_service)) -> None:
    """End the session on the server AND remove the cookie (works even if already logged out)."""
    name = get_settings().session_cookie_name
    await auth.logout(request.cookies.get(name))
    response.delete_cookie(name, path="/")


@router.get("/me", response_model=UserInfo)
async def me(user_id: int = Depends(require_user_id), auth: AuthService = Depends(get_auth_service)) -> UserInfo:
    """Who is logged in (401 if nobody). The React app calls this on start."""
    return await auth.get_user(user_id)
