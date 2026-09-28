"""A flag's wording and its place among its neighbours: one answer for every renderer."""

import math

import pytest

from dataeval_flow._blocks import Flag
from dataeval_flow._blocks._flags import ordered, percentile_text, tag_text

pytestmark = pytest.mark.required


def _flag(
    name: str = "brightness",
    direction: str = "upper",
    value: float = 0.99,
    bound: float = 0.84,
    percentile: float = 99.95,
) -> Flag:
    return Flag(name=name, value=value, direction=direction, bound=bound, percentile=percentile, mean=0.52, std=0.11)  # type: ignore[arg-type]


class TestOrder:
    def test_a_cell_lists_its_flags_by_name(self):
        """Nothing ranks one flag above another: how far past its limit a value lies doesn't say it's worse."""
        flags = [_flag("entropy", "lower"), _flag("contrast"), _flag("brightness"), _flag("blur", "lower")]
        assert [flag.name for flag in ordered(flags)] == ["blur", "brightness", "contrast", "entropy"]


class TestText:
    def test_an_upper_flag_reads_as_its_value_over_its_limit(self):
        assert tag_text(_flag()) == "brightness 0.99 > 0.84"

    def test_a_lower_flag_reads_as_its_value_under_its_limit(self):
        assert tag_text(_flag("entropy", "lower", value=1.2, bound=3.1)) == "entropy 1.2 < 3.1"

    def test_a_flag_whose_limit_is_unknown_reads_as_its_value(self):
        """An issue from a DataEval that predates the limit columns: the value still shows."""
        assert tag_text(_flag("blur", value=0.1, bound=math.nan)) == "blur 0.1"

    @pytest.mark.parametrize(
        ("percentile", "expected"),
        [(98.43, "p98.4"), (99.953, "p99.95"), (0.031, "p0.03"), (99.5, "p99.5"), (50.0, "p50"), (math.nan, "p?")],
    )
    def test_the_percentile_keeps_the_digits_that_tell_the_ends_apart(self, percentile, expected):
        """One decimal, two within 1% of either end, and no trailing zeros."""
        assert percentile_text(percentile) == expected
