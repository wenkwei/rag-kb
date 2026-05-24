from fastapi import APIRouter, Depends, Query

from api.auth import verify_token
from core.audit import get_logs

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("/logs")
async def list_logs(limit: int = Query(100, ge=1, le=500), user: dict = Depends(verify_token)):
    """Return recent operation logs (newest first)."""
    logs = get_logs(limit=limit)
    return {"logs": logs, "total": len(logs)}
