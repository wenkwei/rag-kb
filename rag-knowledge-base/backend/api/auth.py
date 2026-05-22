import secrets
import time
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Simple in-memory token store
_tokens: dict[str, dict] = {}

# Demo credentials (configurable via env)
DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin123"


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str
    message: str


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user and return a session token."""
    if request.username != DEMO_USERNAME or request.password != DEMO_PASSWORD:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = secrets.token_hex(32)
    _tokens[token] = {
        "username": request.username,
        "created_at": time.time(),
    }

    return LoginResponse(
        token=token,
        username=request.username,
        message="登录成功",
    )


async def verify_token(authorization: Optional[str] = Header(None)) -> str:
    """Dependency: verify Bearer token and return username."""
    if not authorization:
        raise HTTPException(status_code=401, detail="未提供认证令牌")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="认证令牌格式错误")

    token_data = _tokens.get(token)
    if not token_data:
        raise HTTPException(status_code=401, detail="认证令牌无效或已过期")

    return token_data["username"]
