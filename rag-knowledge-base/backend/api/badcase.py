import json
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from config import BASE_DIR

router = APIRouter(prefix="/api/badcases", tags=["badcases"])

BAD_CASES_FILE = BASE_DIR / "bad_cases.json"


def _load_cases() -> list:
    if BAD_CASES_FILE.exists():
        with open(BAD_CASES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_cases(cases: list):
    with open(BAD_CASES_FILE, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)


class BadCaseCreate(BaseModel):
    question: str
    expected_answer: Optional[str] = ""
    actual_answer: Optional[str] = ""
    feedback: str = ""
    category: str = "其他"


class BadCaseUpdate(BaseModel):
    status: Optional[str] = None
    expected_answer: Optional[str] = None
    actual_answer: Optional[str] = None
    note: Optional[str] = None


@router.get("")
async def list_badcases():
    """List all bad cases."""
    cases = _load_cases()
    return {"bad_cases": cases, "total": len(cases)}


@router.get("/stats")
async def badcase_stats():
    """Get bad case statistics."""
    cases = _load_cases()
    total = len(cases)
    pending = sum(1 for c in cases if c.get("status") == "pending")
    resolved = sum(1 for c in cases if c.get("status") == "resolved")
    today = sum(1 for c in cases if c.get("time", "").startswith(datetime.now().strftime("%Y-%m-%d")))
    rate = round(resolved / total * 100, 1) if total > 0 else 0
    return {"total": total, "pending": pending, "resolved": resolved, "today": today, "resolution_rate": rate}


@router.post("")
async def create_badcase(case: BadCaseCreate):
    """Create a new bad case."""
    cases = _load_cases()
    new_case = {
        "id": f"bc{len(cases) + 1}",
        "question": case.question,
        "expected_answer": case.expected_answer,
        "actual_answer": case.actual_answer,
        "feedback": case.feedback,
        "category": case.category,
        "status": "pending",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    cases.append(new_case)
    _save_cases(cases)
    return {"status": "success", "bad_case": new_case}


@router.put("/{case_id}")
async def update_badcase(case_id: str, update: BadCaseUpdate):
    """Update a bad case (e.g., mark as resolved)."""
    cases = _load_cases()
    for c in cases:
        if c["id"] == case_id:
            update_data = update.model_dump(exclude_none=True)
            c.update(update_data)
            _save_cases(cases)
            return {"status": "success", "bad_case": c}
    raise HTTPException(status_code=404, detail="Bad case not found")


@router.delete("/{case_id}")
async def delete_badcase(case_id: str):
    """Delete a bad case."""
    cases = _load_cases()
    for i, c in enumerate(cases):
        if c["id"] == case_id:
            deleted = cases.pop(i)
            _save_cases(cases)
            return {"status": "success", "deleted": deleted["id"]}
    raise HTTPException(status_code=404, detail="Bad case not found")
