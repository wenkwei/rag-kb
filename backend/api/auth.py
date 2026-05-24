import json
import secrets
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional

from config import BASE_DIR
from core.audit import add_log

router = APIRouter(prefix="/api/auth", tags=["auth"])

TOKENS_FILE = BASE_DIR / "tokens.json"
TOKEN_TTL = 24 * 3600  # 24 hours

# In-memory token store: token -> {username, role, created_at}
_tokens: dict[str, dict] = {}


def _load_tokens() -> dict[str, dict]:
    """Load persisted tokens from ``tokens.json``."""
    if TOKENS_FILE.exists():
        try:
            with open(TOKENS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_tokens(tokens: dict[str, dict]):
    """Persist tokens dict to ``tokens.json``."""
    with open(TOKENS_FILE, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False)


def _clean_expired_tokens() -> dict[str, dict]:
    """Remove expired tokens and return the cleaned dict.

    Persists the cleaned dict to disk if any tokens were removed.
    """
    global _tokens
    now = time.time()
    before = len(_tokens)
    _tokens = {k: v for k, v in _tokens.items() if now - v.get("created_at", 0) < TOKEN_TTL}
    if len(_tokens) < before:
        _save_tokens(_tokens)
    return _tokens


# Bootstrap: load persisted tokens into memory, discarding expired ones
_tokens = _clean_expired_tokens()


def _load_users() -> list:
    """Load users from ``users.json``."""
    path = BASE_DIR / "users.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_users(users: list):
    """Save users to ``users.json``."""
    path = BASE_DIR / "users.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str
    role: str
    message: str


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user and return a session token with role."""
    users = _load_users()
    matched_user = None
    for u in users:
        if u["username"] == request.username and u["password"] == request.password:
            matched_user = u
            break

    if not matched_user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = secrets.token_hex(32)
    role = matched_user.get("role", "普通用户")
    now = time.time()

    # Update last_login in users.json
    from datetime import datetime
    matched_user["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_users(users)

    _tokens[token] = {
        "username": matched_user["username"],
        "role": role,
        "created_at": now,
    }
    _save_tokens(_tokens)

    add_log(matched_user["username"], "登录", f"用户 {matched_user['username']} 登录系统")

    return LoginResponse(
        token=token,
        username=matched_user["username"],
        role=role,
        message="登录成功",
    )


async def verify_token(authorization: Optional[str] = Header(None)) -> dict:
    """Dependency: verify Bearer token and return ``{username, role}``.

    Checks expiry and cleans up expired tokens on each request.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="未提供认证令牌")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="认证令牌格式错误")

    # Clean expired tokens before looking up
    _clean_expired_tokens()

    token_data = _tokens.get(token)
    if not token_data:
        raise HTTPException(status_code=401, detail="认证令牌无效或已过期")

    return {
        "username": token_data["username"],
        "role": token_data.get("role", "普通用户"),
    }


def require_role(required_role: str):
    """Dependency factory: require a specific role to access an endpoint."""
    async def role_checker(current_user: dict = Depends(verify_token)):
        if current_user["role"] != required_role:
            raise HTTPException(status_code=403, detail=f"需要{required_role}权限")
        return current_user
    return role_checker
