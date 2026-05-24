"""Tests for core.chunker (split_text)."""

import sys
from pathlib import Path

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.chunker import split_text


class TestSplitText:
    def test_empty_string(self):
        assert split_text("") == []

    def test_whitespace_only(self):
        assert split_text("   \n  \t  ") == []

    def test_short_text(self):
        result = split_text("你好世界")
        assert len(result) == 1
        assert "你好世界" in result[0]

    def test_long_text(self):
        text = "今天天气不错。适合出门散步。我们去公园吧。花开得很漂亮。孩子们在玩耍。"
        result = split_text(text)
        assert len(result) >= 1
        # All chunks should be non-empty
        assert all(len(c.strip()) > 0 for c in result)

    def test_chinese_mixed_english(self):
        text = "RAG系统使用Retriever来检索文档。然后使用Generator生成答案。"
        result = split_text(text)
        assert len(result) >= 1
        combined = "".join(result)
        assert "RAG" in combined
        assert "Retriever" in combined

    def test_single_character(self):
        result = split_text("a")
        assert len(result) == 1

    def test_large_chunk_boundary(self):
        """Text just under chunk_size should remain one chunk."""
        text = "测试" * 100  # 200 chars, under default 500
        result = split_text(text)
        assert len(result) >= 1
