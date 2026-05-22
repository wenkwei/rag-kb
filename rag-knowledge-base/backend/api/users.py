import json
import secrets
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from config import BASE_DIR

router = APIRouter(prefix="/api/users", tags=["users"])

USERS_FILE = BASE_DIR / "users.json"

DEFAULT_USERS = [
    {"id": "u1", "username": "chen_admin", "display_name": "陈经理", "role": "超级管理员", "department": "管理层", "status": "正常", "last_login": "2024-03-22 10:15", "password": "admin123"},
    {"id": "u2", "username": "li_tech", "display_name": "李技术", "role": "系统运营", "department": "IT技术部", "status": "正常", "last_login": "2024-03-21 18:30", "password": "123456"},
    {"id": "u3", "username": "wang_staff", "display_name": "王客服", "role": "普通用户", "department": "客户服务部", "status": "离线", "last_login": "2024-03-15 09:00", "password": "123456"},
]


def _load_users() -> list:
    if USERS_FILE.exists():
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    _save_users(DEFAULT_USERS)
    return list(DEFAULT_USERS)


def _save_users(users: list):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


class UserCreate(BaseModel):
    username: str
    password: str
    display_name: str
    role: str = "普通用户"
    department: str = ""


class UserUpdate(BaseModel):
    display_name: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    status: Optional[str] = None


@router.get("")
async def list_users():
    """List all users (without passwords)."""
    users = _load_users()
    safe_users = []
    for u in users:
        safe_users.append({k: v for k, v in u.items() if k != "password"})
    return {"users": safe_users, "total": len(safe_users)}


@router.post("")
async def create_user(user: UserCreate):
    """Create a new user."""
    users = _load_users()
    if any(u["username"] == user.username for u in users):
        raise HTTPException(status_code=400, detail="用户名已存在")

    new_user = {
        "id": f"u{secrets.token_hex(4)}",
        "username": user.username,
        "display_name": user.display_name,
        "password": user.password,
        "role": user.role,
        "department": user.department,
        "status": "正常",
        "last_login": "-",
    }
    users.append(new_user)
    _save_users(users)
    return {"status": "success", "user": {k: v for k, v in new_user.items() if k != "password"}}


@router.put("/{user_id}")
async def update_user(user_id: str, update: UserUpdate):
    """Update user info."""
    users = _load_users()
    for u in users:
        if u["id"] == user_id:
            update_data = update.model_dump(exclude_none=True)
            u.update(update_data)
            _save_users(users)
            return {"status": "success", "user": {k: v for k, v in u.items() if k != "password"}}
    raise HTTPException(status_code=404, detail="用户不存在")


@router.delete("/{user_id}")
async def delete_user(user_id: str):
    """Delete a user."""
    users = _load_users()
    for i, u in enumerate(users):
        if u["id"] == user_id:
            deleted = users.pop(i)
            _save_users(users)
            return {"status": "success", "deleted": deleted["username"]}
    raise HTTPException(status_code=404, detail="用户不存在")
