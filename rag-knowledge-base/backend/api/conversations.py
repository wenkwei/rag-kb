import json
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from config import BASE_DIR

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

CONV_FILE = BASE_DIR / "conversations.json"


def _load_all() -> list:
    if CONV_FILE.exists():
        with open(CONV_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_all(convs: list):
    with open(CONV_FILE, "w", encoding="utf-8") as f:
        json.dump(convs, f, ensure_ascii=False, indent=2)


class MessageItem(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ConversationCreate(BaseModel):
    title: Optional[str] = ""
    messages: list[MessageItem] = []


@router.get("")
async def list_conversations():
    """List all conversations (summary only)."""
    convs = _load_all()
    summaries = []
    for c in convs:
        summaries.append({
            "id": c.get("id"),
            "title": c.get("title", "新对话"),
            "message_count": len(c.get("messages", [])),
            "created_at": c.get("created_at"),
            "updated_at": c.get("updated_at"),
        })
    return {"conversations": summaries, "total": len(summaries)}


@router.get("/stats")
async def conversation_stats():
    """Get conversation statistics."""
    convs = _load_all()
    total = len(convs)
    total_messages = sum(len(c.get("messages", [])) for c in convs)
    return {"total_conversations": total, "total_messages": total_messages}


@router.get("/{conv_id}")
async def get_conversation(conv_id: str):
    """Get a single conversation with all messages."""
    convs = _load_all()
    for c in convs:
        if c["id"] == conv_id:
            return c
    raise HTTPException(status_code=404, detail="Conversation not found")


@router.post("")
async def create_conversation(conv: ConversationCreate):
    """Create a new conversation."""
    convs = _load_all()
    new_conv = {
        "id": f"conv{len(convs) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "title": conv.title or "新对话",
        "messages": [m.model_dump() for m in conv.messages],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    convs.append(new_conv)
    _save_all(convs)
    return {"status": "success", "conversation": new_conv}


@router.delete("/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation."""
    convs = _load_all()
    for i, c in enumerate(convs):
        if c["id"] == conv_id:
            convs.pop(i)
            _save_all(convs)
            return {"status": "success"}
    raise HTTPException(status_code=404, detail="Conversation not found")
