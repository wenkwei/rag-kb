from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from config import UPLOAD_DIR
from core.processor import process_document, delete_chunks_by_filename, get_all_documents

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF/TXT document, automatically process and index it."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 PDF 和 TXT 文件")

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        chunk_count = process_document(str(file_path), file.filename)
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

    return {
        "filename": file.filename,
        "size": len(content),
        "chunk_count": chunk_count,
        "status": "success",
    }


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

    if chunks_removed == 0 and not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文档 '{filename}' 不存在")

    return {
        "deleted": True,
        "filename": filename,
        "chunks_removed": chunks_removed,
    }
