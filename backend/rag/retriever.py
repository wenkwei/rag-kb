from typing import List, Tuple, Optional

import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from db.chroma_client import get_vector_store, get_collection
from config import RETRIEVAL_STRATEGY, VECTOR_WEIGHT, KEYWORD_WEIGHT

# ── BM25 index cache ──
_bm25_index: Optional[BM25Okapi] = None
_bm25_doc_count: int = 0
_bm25_corpus: List[dict] = []


def _build_bm25_index():
    """Build or rebuild the BM25 keyword index from all Chroma documents.

    The index is lazily built on first hybrid query and automatically
    rebuilt when the document count in Chroma changes.
    """
    global _bm25_index, _bm25_doc_count, _bm25_corpus

    collection = get_collection()
    current_count = collection.count()

    if _bm25_index is not None and current_count == _bm25_doc_count:
        return

    all_data = collection.get()
    _bm25_corpus = []
    tokenized_corpus = []

    for i, doc_text in enumerate(all_data.get("documents", []) or []):
        meta = (all_data.get("metadatas") or [{}])[i] or {}
        _bm25_corpus.append({"content": doc_text, "metadata": meta})
        tokenized_corpus.append(list(jieba.cut(doc_text)))

    _bm25_index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    _bm25_doc_count = current_count


def _keyword_search(query: str, k: int) -> List[Tuple[Document, float]]:
    """BM25 keyword search with jieba Chinese tokenization."""
    _build_bm25_index()

    if _bm25_index is None or _bm25_index.doc_count == 0:
        return []

    query_tokens = list(jieba.cut(query))
    scores = _bm25_index.get_scores(query_tokens)

    # Top-k indices sorted by score descending
    top_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:k]

    results: List[Tuple[Document, float]] = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        doc = Document(
            page_content=_bm25_corpus[idx]["content"],
            metadata=_bm25_corpus[idx]["metadata"],
        )
        results.append((doc, float(scores[idx])))

    return results


def _rrf_fusion(
    vector_results: List[Tuple[Document, float]],
    keyword_results: List[Tuple[Document, float]],
    k: int = 60,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> List[Tuple[Document, float]]:
    """Reciprocal Rank Fusion merging semantic and keyword results.

    Uses standard RRF formula: score = weight / (k + rank).
    Deduplicates by content prefix to avoid double-counting the same chunk.
    """
    merged: dict = {}

    for rank, (doc, _) in enumerate(vector_results):
        key = doc.page_content[:120]
        rrf_score = vector_weight / (k + rank + 1)
        if key in merged:
            merged[key][1] += rrf_score
        else:
            merged[key] = [doc, rrf_score]

    for rank, (doc, _) in enumerate(keyword_results):
        key = doc.page_content[:120]
        rrf_score = keyword_weight / (k + rank + 1)
        if key in merged:
            merged[key][1] += rrf_score
        else:
            merged[key] = [doc, rrf_score]

    sorted_items = sorted(merged.values(), key=lambda x: x[1], reverse=True)
    return [(doc, score) for doc, score in sorted_items]


def retrieve(query: str, k: int = 4) -> List[Tuple[Document, float]]:
    """Retrieve relevant document chunks using the configured strategy.

    Strategies:
    - ``"semantic"``: Pure vector similarity search (Chroma default).
    - ``"hybrid"``: Semantic + BM25 keyword fused via RRF.

    Args:
        query: The search query string.
        k: Number of results to return.

    Returns:
        List of ``(Document, relevance_score)`` tuples.
    """
    vector_store = get_vector_store()

    if RETRIEVAL_STRATEGY == "hybrid":
        vector_results = vector_store.similarity_search_with_relevance_scores(
            query, k=k * 2
        )
        keyword_results = _keyword_search(query, k=k * 2)

        fused = _rrf_fusion(
            vector_results,
            keyword_results,
            vector_weight=VECTOR_WEIGHT,
            keyword_weight=KEYWORD_WEIGHT,
        )
        return fused[:k]

    # Default: pure semantic search
    return vector_store.similarity_search_with_relevance_scores(query, k=k)
