"""
tests/test_eval_metrics.py

Unit tests for eval/metrics.py's pure parsing helpers (_extract_float,
_extract_int). These parse free-form LLM judge output into clamped
numeric scores — exactly the kind of "LLM returned something slightly
off-format" edge case worth testing directly, since the rest of the
eval harness depends on these never raising or returning out-of-range
values regardless of what the judge model actually outputs.

These tests need no LLM call, no mocking — pure string/regex logic.

Run with:
    pytest tests/test_eval_metrics.py -v
"""

from eval.metrics import _extract_float, _extract_int


class TestExtractFloat:
    def test_plain_decimal(self):
        assert _extract_float("0.85") == 0.85

    def test_decimal_embedded_in_sentence(self):
        assert _extract_float("The score is 0.7 based on analysis") == 0.7

    def test_upper_bound_exact(self):
        assert _extract_float("1.0") == 1.0

    def test_lower_bound_exact(self):
        assert _extract_float("Faithfulness: 0.0") == 0.0

    def test_no_number_returns_default(self):
        assert _extract_float("garbage no number here") == 0.0

    def test_custom_default_used_when_no_number(self):
        assert _extract_float("no number", default=0.5) == 0.5

    def test_value_above_one_is_clamped(self):
        assert _extract_float("2.5") == 1.0

    def test_negative_sign_is_stripped_not_clamped_to_zero(self):
        # The regex only captures digits/decimal point, so a leading minus
        # sign is dropped rather than producing a negative number — this
        # means "-0.5" parses as 0.5, NOT 0.0. Worth knowing explicitly
        # since it's a real, if minor, quirk of this simple parser.
        assert _extract_float("-0.5") == 0.5

    def test_integer_without_decimal_point(self):
        assert _extract_float("1") == 1.0


class TestExtractInt:
    def test_plain_integer(self):
        assert _extract_int("4") == 4

    def test_integer_embedded_in_sentence(self):
        assert _extract_int("Rating: 5/5") == 5

    def test_value_above_max_is_clamped(self):
        assert _extract_int("10") == 5

    def test_value_below_min_is_clamped(self):
        assert _extract_int("0") == 1

    def test_no_number_returns_default(self):
        assert _extract_int("no number") == 3

    def test_custom_bounds_respected(self):
        assert _extract_int("99", min_val=1, max_val=10) == 10
        assert _extract_int("0", min_val=1, max_val=10) == 1
