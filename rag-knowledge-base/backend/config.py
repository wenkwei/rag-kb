import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
K_RETRIEVAL = int(os.getenv("K_RETRIEVAL", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
