import threading
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel

from config import UPLOAD_DIR
from core.loader import load_document
from core.processor import process_document, delete_chunks_by_filename, get_all_documents
from db.chroma_client import get_collection

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}

# In-memory upload processing status: filename -> status dict
_upload_status: dict[str, dict] = {}
_upload_lock = threading.Lock()


def _bg_process(file_path: Path, filename: str):
    """Background task: load, chunk, embed a document."""
    try:
        _upload_status[filename] = {"status": "processing", "progress": "正在解析文档..."}
        result = process_document(str(file_path), filename)
        _upload_status[filename] = {
            "status": "done",
            "chunk_count": result["chunk_count"],
            "text_length": result["text_length"],
        }
    except Exception as e:
        _upload_status[filename] = {"status": "error", "detail": str(e)}
        # Clean up the uploaded file on failure
        if file_path.exists():
            file_path.unlink()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Upload a document and process it asynchronously.

    Returns immediately with ``{"status": "pending"}``. Poll
    ``GET /api/documents/upload/status/{filename}`` for completion.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 PDF、TXT、DOCX 和 MD 文件")

    # Prevent duplicate processing of the same filename
    with _upload_lock:
        existing = _upload_status.get(file.filename, {})
        if existing.get("status") in ("pending", "processing"):
            raise HTTPException(status_code=409, detail=f"文件 '{file.filename}' 正在处理中，请勿重复上传")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(content)

    _upload_status[file.filename] = {"status": "pending", "progress": "等待处理..."}

    if background_tasks:
        background_tasks.add_task(_bg_process, file_path, file.filename)
    else:
        # Fallback: process synchronously (should not happen with FastAPI)
        _bg_process(file_path, file.filename)

    return {
        "filename": file.filename,
        "size": len(content),
        "status": "pending",
    }


@router.get("/upload/status/{filename:path}")
async def get_upload_status(filename: str):
    """Poll the processing status of an uploaded document."""
    status = _upload_status.get(filename)
    if status is None:
        raise HTTPException(status_code=404, detail=f"文件 '{filename}' 没有上传记录")

    return {"filename": filename, **status}


@router.get("")
async def list_documents():
    """List all indexed documents with chunk counts."""
    documents = get_all_documents()
    return {"documents": documents, "total": len(documents)}


@router.delete("/{filename:path}")
async def delete_document(filename: str):
    """Delete a document and all its vector chunks."""
    file_path = UPLOAD_DIR / filename

    chunks_removed = delete_chunks_by_filename(filename)

    if file_path.exists():
        file_path.unlink()

    # Clean up upload status
    _upload_status.pop(filename, None)

    if chunks_removed == 0 and not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文档 '{filename}' 不存在")

    return {
        "deleted": True,
        "filename": filename,
        "chunks_removed": chunks_removed,
    }


class ChunkItem(BaseModel):
    chunk_index: int
    content: str
    filename: str


@router.get("/{filename:path}/content")
async def get_document_content(filename: str):
    """Extract and return the raw text content of an uploaded document."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文档 '{filename}' 不存在")

    try:
        text = load_document(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取文档内容失败: {str(e)}")

    return {"filename": filename, "content": text, "size": len(text)}


@router.get("/{filename:path}/chunks")
async def get_document_chunks(filename: str):
    """Get all chunks for a specific document."""
    collection = get_collection()
    results = collection.get(where={"filename": filename})

    if not results.get("ids"):
        raise HTTPException(status_code=404, detail=f"文档 '{filename}' 不存在或没有切片")

    chunks: List[ChunkItem] = []
    for i, meta in enumerate(results["metadatas"]):
        if meta and meta.get("filename") == filename:
            chunks.append(ChunkItem(
                chunk_index=meta.get("chunk_index", i),
                content=results["documents"][i] if results.get("documents") else "",
                filename=filename,
            ))

    chunks.sort(key=lambda c: c.chunk_index)
    return {"filename": filename, "total_chunks": len(chunks), "chunks": chunks}
