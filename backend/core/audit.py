import json
from datetime import datetime
from pathlib import Path
from config import BASE_DIR

AUDIT_FILE = BASE_DIR / "audit_logs.json"
MAX_LOGS = 500


def _load_logs() -> list:
    if AUDIT_FILE.exists():
        try:
            with open(AUDIT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_logs(logs: list):
    with open(AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def add_log(user: str, action: str, detail: str):
    """Record an operation log entry."""
    logs = _load_logs()
    logs.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user,
        "action": action,
        "detail": detail,
    })
    if len(logs) > MAX_LOGS:
        logs = logs[-MAX_LOGS:]
    _save_logs(logs)


def get_logs(limit: int = 100) -> list:
    """Return most recent logs, newest first."""
    logs = _load_logs()
    return list(reversed(logs))[:limit]
