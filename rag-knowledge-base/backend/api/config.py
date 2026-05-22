import json
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from config import BASE_DIR, OPENAI_API_KEY

router = APIRouter(prefix="/api/config", tags=["config"])

CONFIG_FILE = BASE_DIR / "rag_config.json"

DEFAULT_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "embedding_model": "text-embedding-3-small",
    "llm_model": "gpt-4o-mini",
    "top_k": 4,
    "temperature": 0,
    "openai_key_configured": bool(OPENAI_API_KEY) and OPENAI_API_KEY != "sk-placeholder",
}


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return dict(DEFAULT_CONFIG)


def _save_config(data: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class ConfigUpdate(BaseModel):
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None


@router.get("")
async def get_config():
    """Get current RAG configuration."""
    return _load_config()


@router.post("")
async def update_config(update: ConfigUpdate):
    """Update RAG configuration (saved to file,生效需重启服务)."""
    config = _load_config()

    update_data = update.model_dump(exclude_none=True)
    for key, value in update_data.items():
        if key in config:
            config[key] = value

    _save_config(config)
    return {"status": "saved", "config": config, "message": "配置已保存，重启服务后生效"}
