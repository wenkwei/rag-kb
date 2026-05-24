import json
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from config import BASE_DIR, K_RETRIEVAL
from rag.retriever import retrieve
from api.evaluation import add_question_to_test_set, _generate_answer, _judge_faithfulness, _judge_correctness

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


class VerifyRequest(BaseModel):
    question: str
    expected_answer: str = ""


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
    root_cause: Optional[str] = None
    add_to_test_set: Optional[bool] = False


@router.post("/verify")
async def verify_badcase(req: VerifyRequest):
    """Run RAG pipeline for a bad case question and return answer + scores."""
    try:
        docs_with_scores = retrieve(req.question, k=K_RETRIEVAL)
        context_str = "\n\n---\n\n".join([d.page_content for d, _ in docs_with_scores]) if docs_with_scores else ""
        llm_answer = _generate_answer(req.question, context_str)
        faithfulness = _judge_faithfulness(req.question, context_str, llm_answer)
        correctness = _judge_correctness(req.expected_answer, llm_answer) if req.expected_answer else 0.0
        sources = [
            {
                "filename": d.metadata.get("filename", "未知文档"),
                "content": d.page_content[:200],
                "chunk_index": d.metadata.get("chunk_index", 0),
                "score": round(float(s), 4),
            }
            for d, s in docs_with_scores
        ]
        return {
            "llm_answer": llm_answer,
            "faithfulness": faithfulness,
            "correctness": correctness,
            "sources": sources,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")


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


@router.get("/stats/categories")
async def badcase_categories():
    """Get bad case statistics grouped by category."""
    cases = _load_cases()
    categories = {}
    for c in cases:
        cat = c.get("category", "未分类")
        if cat not in categories:
            categories[cat] = {"category": cat, "total": 0, "resolved": 0}
        categories[cat]["total"] += 1
        if c.get("status") == "resolved":
            categories[cat]["resolved"] += 1
    result = list(categories.values())
    for cat in result:
        cat["rate"] = round(cat["resolved"] / cat["total"] * 100, 1) if cat["total"] > 0 else 0
    return {"categories": result}


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
    """Update a bad case (e.g., mark as resolved or process with actions)."""
    cases = _load_cases()
    for c in cases:
        if c["id"] == case_id:
            update_data = update.model_dump(exclude_none=True)
            # Remove the control flag before merging
            add_to_test_set = update_data.pop("add_to_test_set", False)
            c.update(update_data)
            # If resolved and flagged, add to evaluation test set
            added_to_test_set = False
            if add_to_test_set and c.get("status") == "resolved":
                try:
                    add_question_to_test_set(c["question"], c.get("expected_answer", ""))
                    added_to_test_set = True
                except Exception:
                    pass
            _save_cases(cases)
            return {
                "status": "success",
                "bad_case": c,
                "added_to_test_set": added_to_test_set,
            }
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
