import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# ── Load saved config (rag_config.json) to override env defaults ──
_CONFIG_FILE = BASE_DIR / "rag_config.json"
_saved = {}
if _CONFIG_FILE.exists():
    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
            _saved = json.load(f)
    except Exception:
        _saved = {}

# Read from: saved config > env var > hardcoded default
def _get(key: str, env_key: str, default):
    # 1) saved config (set via settings page)
    if key in _saved and _saved[key] not in (None, ""):
        return _saved[key]
    # 2) environment variable
    val = os.getenv(env_key)
    if val:
        return val
    # 3) fallback default
    return default

# ── Exported constants ──
OPENAI_API_KEY = _get("api_key", "OPENAI_API_KEY", "sk-placeholder")
OPENAI_BASE_URL = _get("api_base", "OPENAI_BASE_URL", "")
LLM_MODEL = _get("llm_model", "LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = _get("embedding_model", "EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(_get("chunk_size", "CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(_get("chunk_overlap", "CHUNK_OVERLAP", "50"))
K_RETRIEVAL = int(_get("top_k", "K_RETRIEVAL", "4"))
TEMPERATURE = float(_get("temperature", "TEMPERATURE", "0"))

# ── Retrieval & Rerank config ──
RETRIEVAL_STRATEGY = _get("retrieval_strategy", "RETRIEVAL_STRATEGY", "semantic")
VECTOR_WEIGHT = float(_get("vector_weight", "VECTOR_WEIGHT", "0.7"))
KEYWORD_WEIGHT = float(_get("keyword_weight", "KEYWORD_WEIGHT", "0.3"))
_rerank_raw = _get("rerank_enabled", "RERANK_ENABLED", "false")
RERANK_ENABLED = _rerank_raw if isinstance(_rerank_raw, bool) else str(_rerank_raw).lower() == "true"
RERANK_MODEL = _get("rerank_model", "RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_K = int(_get("rerank_top_k", "RERANK_TOP_K", "3"))
