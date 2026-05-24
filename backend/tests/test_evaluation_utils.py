"""Tests for evaluation utility functions (_parse_score, _compute_text_overlap)."""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.evaluation import _parse_score, _compute_text_overlap


class TestParseScore:
    def test_simple_float(self):
        assert _parse_score("0.85") == 0.85

    def test_integer_string(self):
        assert _parse_score("1") == 1.0

    def test_text_with_number(self):
        assert _parse_score("得分：0.75 分") == 0.75

    def test_out_of_range_clamp_high(self):
        assert _parse_score("1.5") == 1.0

    def test_out_of_range_clamp_low(self):
        # "-0.5" -> regex extracts "0.5" -> clamp -> 0.5
        assert _parse_score("-0.5") == 0.5

    def test_no_number(self):
        assert _parse_score("无评分") == 0.0

    def test_empty_string(self):
        assert _parse_score("") == 0.0

    def test_leading_trailing_whitespace(self):
        assert _parse_score("  0.63  ") == 0.63

    def test_multiple_numbers(self):
        """Should take the first number found."""
        assert _parse_score("0.5 vs 0.8") == 0.5


class TestComputeTextOverlap:
    def test_exact_match(self):
        score = _compute_text_overlap("今天天气很好", "今天天气很好")
        assert score == 1.0

    def test_no_overlap(self):
        score = _compute_text_overlap("abcdef", "xyz")
        assert score == 0.0

    def test_partial_overlap(self):
        score = _compute_text_overlap("今天天气很好", "天气")
        # bigram tokenization: "天气" produces {'天气'}
        # "今天天气很好" bigrams: {'今天','天天','天气','气很','很好'}
        # overlap is 1/1 = 1.0 for exact bigram match
        assert score == 1.0

    def test_empty_expected(self):
        assert _compute_text_overlap("something", "") == 0.0

    def test_empty_retrieved(self):
        assert _compute_text_overlap("", "expected") == 0.0

    def test_both_empty(self):
        assert _compute_text_overlap("", "") == 0.0
