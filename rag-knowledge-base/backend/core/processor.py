from pathlib import Path
from typing import Union, List

from langchain_core.documents import Document

from core.loader import load_document
from core.chunker import split_text
from db.chroma_client import get_vector_store, get_collection


def process_document(file_path: Union[str, Path], filename: str) -> int:
    """Process a document: extract text, chunk, embed, and store in Chroma.

    Args:
        file_path: Path to the uploaded file.
        filename: Original filename for metadata tracking.

    Returns:
        Number of chunks created. Returns 0 if no text could be extracted.
    """
    raw_text = load_document(file_path)
    chunks = split_text(raw_text)

    if not chunks:
        return 0

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
    vector_store.add_documents(documents)

    return len(chunks)


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


def get_all_documents() -> List[dict]:
    """Get list of all indexed documents with their chunk counts.

    Returns:
        List of dicts with keys: filename, chunk_count
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
        result.append({
            "filename": fn,
            "chunk_count": len(docs.get("ids", [])),
        })
    return result
