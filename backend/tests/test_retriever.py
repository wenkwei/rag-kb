"""Tests for rag.retriever (retrieve with semantic and hybrid strategies)."""

import sys
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest
from langchain_core.documents import Document


@pytest.fixture(autouse=True)
def clear_bm25_cache():
    """Clear BM25 cache between tests."""
    from rag import retriever
    retriever._bm25_index = None
    retriever._bm25_doc_count = 0
    retriever._bm25_corpus = []


@patch("rag.retriever.get_vector_store")
@patch("rag.retriever.get_collection")
@patch("rag.retriever.RETRIEVAL_STRATEGY", "semantic")
def test_retrieve_semantic(mock_get_collection, mock_get_vector_store, mock_vector_store):
    """Semantic strategy delegates to Chroma similarity_search."""
    mock_get_vector_store.return_value = mock_vector_store
    from rag.retriever import retrieve

    results = retrieve("test query", k=2)
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc, _ in results)
    mock_vector_store.similarity_search_with_relevance_scores.assert_called_once_with("test query", k=2)


@patch("rag.retriever.get_vector_store")
@patch("rag.retriever.get_collection")
@patch("rag.retriever.RETRIEVAL_STRATEGY", "hybrid")
@patch("rag.retriever._keyword_search")
def test_retrieve_hybrid(mock_keyword_search, mock_get_collection, mock_get_vector_store, mock_vector_store):
    """Hybrid strategy returns RRF-fused results."""
    mock_get_vector_store.return_value = mock_vector_store
    mock_keyword_search.return_value = [
        (Document(page_content="keyword result"), 1.0),
    ]
    from rag.retriever import retrieve

    results = retrieve("test query", k=2)
    assert len(results) > 0
    mock_vector_store.similarity_search_with_relevance_scores.assert_called_once()
    mock_keyword_search.assert_called_once()


@patch("rag.retriever.get_vector_store")
@patch("rag.retriever.get_collection")
@patch("rag.retriever.RETRIEVAL_STRATEGY", "hybrid")
def test_retrieve_hybrid_empty_keyword(mock_get_collection, mock_get_vector_store, mock_vector_store):
    """Hybrid still returns vector results when keyword returns nothing."""
    mock_get_vector_store.return_value = mock_vector_store
    from rag.retriever import retrieve

    results = retrieve("test query", k=2)
    assert len(results) > 0
