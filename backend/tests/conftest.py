"""Shared fixtures for RAG backend tests."""
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


@pytest.fixture
def tmp_txt_file(tmp_path: Path) -> Path:
    """Create a temporary UTF-8 text file."""
    path = tmp_path / "test.txt"
    path.write_text("这是测试内容。Hello world.", encoding="utf-8")
    return path


@pytest.fixture
def tmp_empty_txt(tmp_path: Path) -> Path:
    """Create an empty text file."""
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    return path


@pytest.fixture
def tmp_unsupported_file(tmp_path: Path) -> Path:
    """Create a file with an unsupported extension."""
    path = tmp_path / "test.xyz"
    path.write_text("unsupported", encoding="utf-8")
    return path


@pytest.fixture
def mock_chroma_collection():
    """Mock a Chroma collection with sample documents."""
    mock = MagicMock()
    mock.count.return_value = 2
    mock.get.return_value = {
        "ids": ["id1", "id2"],
        "documents": ["今天天气怎么样", "RAG系统的检索原理"],
        "metadatas": [
            {"filename": "doc1.txt", "chunk_index": 0},
            {"filename": "doc2.txt", "chunk_index": 0},
        ],
    }
    return mock


@pytest.fixture
def mock_vector_store():
    """Mock a Chroma vector store returning sample results."""
    mock = MagicMock()
    mock.similarity_search_with_relevance_scores.return_value = [
        (Document(page_content="RAG系统的检索原理", metadata={"filename": "doc2.txt", "chunk_index": 0}), 0.85),
        (Document(page_content="今天天气怎么样", metadata={"filename": "doc1.txt", "chunk_index": 0}), 0.72),
    ]
    return mock


@pytest.fixture
def sample_docs_with_scores() -> List[Tuple[Document, float]]:
    """Sample retrieval results for rerank tests."""
    return [
        (Document(page_content="Python是一种编程语言", metadata={"filename": "a.txt", "chunk_index": 0}), 0.9),
        (Document(page_content="Java也是一种编程语言", metadata={"filename": "b.txt", "chunk_index": 0}), 0.8),
        (Document(page_content="HTML是标记语言", metadata={"filename": "c.txt", "chunk_index": 0}), 0.7),
    ]
