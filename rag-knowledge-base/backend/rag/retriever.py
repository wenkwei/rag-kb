from typing import List, Tuple

from langchain_core.documents import Document

from db.chroma_client import get_vector_store


def retrieve(query: str, k: int = 4) -> List[Tuple[Document, float]]:
    """Semantic search: retrieve top-k relevant document chunks for a query.

    Uses the Chroma vector store's similarity search with relevance scores.
    Higher scores indicate more relevant results.

    Args:
        query: The search query string.
        k: Number of results to return (default 4).

    Returns:
        List of (Document, relevance_score) tuples.
    """
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    return results
