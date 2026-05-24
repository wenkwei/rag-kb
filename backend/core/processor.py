import os
from datetime import datetime
from pathlib import Path
from typing import Union, List

from langchain_core.documents import Document

from config import UPLOAD_DIR
from core.loader import load_document
from core.chunker import split_text
from db.chroma_client import get_vector_store, get_collection


def process_document(file_path: Union[str, Path], filename: str) -> dict:
    """Process a document: extract text, chunk, embed, and store in Chroma.

    Args:
        file_path: Path to the uploaded file.
        filename: Original filename for metadata tracking.

    Returns:
        dict with keys:
            - chunk_count: int, number of chunks created (0 if extraction failed)
            - text_length: int, length of extracted text
    """
    raw_text = load_document(file_path)
    text_length = len(raw_text)
    chunks = split_text(raw_text)

    if not chunks:
        return {"chunk_count": 0, "text_length": text_length}

    # Remove existing chunks for this filename (supports re-indexing)
    delete_chunks_by_filename(filename)

    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "filename": filename,
                "chunk_index": i,
                "source": str(file_path),
            },
        )
        documents.append(doc)

    vector_store = get_vector_store()
    # Batch embedding requests (API gateway typically limits to 64 per batch)
    vector_store.add_documents(documents, batch_size=64)

    return {"chunk_count": len(chunks), "text_length": text_length}


def delete_chunks_by_filename(filename: str) -> int:
    """Delete all vector chunks associated with a filename.

    Returns:
        Number of chunks removed.
    """
    collection = get_collection()
    existing = collection.get(where={"filename": filename})
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def _format_size(size_bytes: int) -> str:
    """Format byte size into human-readable string (e.g. '2.4 MB', '156 KB')."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _get_file_meta(filename: str) -> dict:
    """Read file size and last modified time from the uploads directory.

    Returns dict with 'size' (human-readable) and 'updated' (formatted datetime).
    Both default to None if the file is not found on disk.
    """
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        return {"size": None, "updated": None}

    stat = file_path.stat()
    size = _format_size(stat.st_size)
    updated = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
    return {"size": size, "updated": updated}


def get_all_documents() -> List[dict]:
    """Get list of all indexed documents with their chunk counts and file metadata.

    Returns:
        List of dicts with keys: filename, chunk_count, size, updated
    """
    collection = get_collection()
    all_data = collection.get()
    filenames = set()

    for meta in all_data.get("metadatas", []):
        if meta and "filename" in meta:
            filenames.add(meta["filename"])

    result = []
    for fn in sorted(filenames):
        docs = collection.get(where={"filename": fn})
        meta = _get_file_meta(fn)
        result.append({
            "filename": fn,
            "chunk_count": len(docs.get("ids", [])),
            "size": meta["size"],
            "updated": meta["updated"],
        })
    return result
