from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import K_RETRIEVAL
from rag.qa_chain import qa_with_sources

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    k: int = Field(default=K_RETRIEVAL, ge=1, le=20, description="检索返回的切片数量")
    threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="相似度阈值，低于此值的过滤掉")


class ChatResponse(BaseModel):
    answer: str
    sources: list


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question and get an answer with cited sources."""
    try:
        result = qa_with_sources(request.question, k=request.k, threshold=request.threshold)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")
