"""Tests for rag.qa_chain (qa_with_sources and _rerank)."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest
from langchain_core.documents import Document


class TestRerank:
    def test_disabled_by_default(self, sample_docs_with_scores):
        """When RERANK_ENABLED is False, _rerank returns docs unchanged."""
        with patch("rag.qa_chain.RERANK_ENABLED", False):
            from rag.qa_chain import _rerank
            result = _rerank("query", sample_docs_with_scores, top_k=2)
            assert result == sample_docs_with_scores

    def test_empty_input(self):
        with patch("rag.qa_chain.RERANK_ENABLED", True):
            from rag.qa_chain import _rerank
            assert _rerank("query", [], top_k=3) == []

    @patch("rag.qa_chain.requests.post")
    def test_api_fallback(self, mock_post, sample_docs_with_scores):
        """When rerank API fails, original order is preserved."""
        mock_post.side_effect = Exception("API error")
        with patch("rag.qa_chain.RERANK_ENABLED", True):
            from rag.qa_chain import _rerank
            result = _rerank("query", sample_docs_with_scores, top_k=3)
            assert result == sample_docs_with_scores

    @patch("rag.qa_chain.requests.post")
    def test_api_success(self, mock_post, sample_docs_with_scores):
        """Successful rerank API call reorders documents by relevance."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.95},  # HTML was last, now first
                {"index": 0, "relevance_score": 0.80},
                {"index": 1, "relevance_score": 0.60},
            ]
        }
        mock_post.return_value = mock_resp

        with patch("rag.qa_chain.RERANK_ENABLED", True):
            with patch("rag.qa_chain.RERANK_TOP_K", 3):
                from rag.qa_chain import _rerank
                result = _rerank("markup language", sample_docs_with_scores, top_k=3)
                assert len(result) == 3
                # The HTML doc was index 2, should come first now
                assert "HTML" in result[0][0].page_content


class TestQaWithSources:
    @patch("rag.qa_chain.retrieve")
    @patch("rag.qa_chain.RERANK_ENABLED", False)
    def test_basic_qa_flow(self, mock_retrieve):
        """qa_with_sources returns answer and sources."""
        from langchain_core.prompts import ChatPromptTemplate

        mock_chain = MagicMock()
        mock_chain.invoke.return_value.content = "RAG是检索增强生成技术。"
        mock_retrieve.return_value = [
            (Document(page_content="RAG是检索增强生成", metadata={"filename": "doc.txt", "chunk_index": 0}), 0.9),
        ]

        with patch.object(ChatPromptTemplate, "__or__", return_value=mock_chain):
            from rag.qa_chain import qa_with_sources
            result = qa_with_sources("什么是RAG？", k=1)

        assert result["answer"] == "RAG是检索增强生成技术。"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["filename"] == "doc.txt"

    @patch("rag.qa_chain.retrieve")
    @patch("rag.qa_chain.RERANK_ENABLED", False)
    def test_no_results(self, mock_retrieve):
        """When no documents are retrieved, returns a helpful message."""
        mock_retrieve.return_value = []
        from rag.qa_chain import qa_with_sources

        result = qa_with_sources("nothing", k=1)
        assert "暂无相关内容" in result["answer"]
        assert result["sources"] == []
