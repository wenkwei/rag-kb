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
    # 文档分块
    "chunk_size": 500,
    "chunk_overlap": 50,
    # 向量化
    "embedding_model": "text-embedding-3-small",
    "openai_key_configured": bool(OPENAI_API_KEY) and OPENAI_API_KEY != "sk-placeholder",
    # 检索
    "top_k": 4,
    "temperature": 0,
    "threshold": 0,
    "retrieval_strategy": "hybrid",
    "vector_weight": 0.7,
    "keyword_weight": 0.3,
    # 生成模型
    "llm_provider": "openai",
    "llm_model": "gpt-4o-mini",
    "api_base": "",
    "api_key": "",
    "max_tokens": 2048,
    "system_prompt": "你是一个专业的企业知识库助手，请基于检索到的文档内容回答用户问题。如果文档中没有相关信息，请如实告知。",
    # Rerank
    "rerank_enabled": False,
    "rerank_model": "BAAI/bge-reranker-v2-m3",
    "rerank_top_k": 3,
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
    # 分块
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    # Embedding
    embedding_model: Optional[str] = None
    # 检索
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    threshold: Optional[float] = None
    retrieval_strategy: Optional[str] = None
    vector_weight: Optional[float] = None
    keyword_weight: Optional[float] = None
    # 生成模型
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    # Rerank
    rerank_enabled: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_top_k: Optional[int] = None


@router.get("")
async def get_config():
    """Get current RAG configuration."""
    cfg = _load_config()
    # Mask API key for security
    if cfg.get("api_key"):
        cfg["api_key"] = cfg["api_key"][:4] + "****" + cfg["api_key"][-4:] if len(cfg["api_key"]) > 8 else "****"
    return cfg


@router.post("")
async def update_config(update: ConfigUpdate):
    """Update RAG configuration (saved to file,生效需重启服务)."""
    config = _load_config()

    update_data = update.model_dump(exclude_none=True)

    # If api_key looks masked (contains ****), keep the existing value
    api_key = update_data.get("api_key", "")
    if api_key and "****" in api_key:
        del update_data["api_key"]

    for key, value in update_data.items():
        config[key] = value

    _save_config(config)
    # Reload config to mask key for response
    result = dict(config)
    if result.get("api_key"):
        result["api_key"] = result["api_key"][:4] + "****" + result["api_key"][-4:] if len(result["api_key"]) > 8 else "****"
    return {"status": "saved", "config": result, "message": "配置已保存，重启服务后生效"}
