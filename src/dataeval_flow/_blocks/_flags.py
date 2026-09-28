"""A flag's wording and its order among its neighbours, shared so that every renderer agrees.

A flag is a fact: a value against the limit it crossed. Nothing here grades how far past its limit
a value lies. A limit may be a plain value rather than a percentile rule, and a larger distance
doesn't mean a worse item, so flags are listed by name and read as the comparison itself.
"""

__all__ = ["flags_in", "ordered", "percentile_text", "tag_text"]

import math
from collections.abc import Sequence

from dataeval_flow._blocks._draw import fmt_num
from dataeval_flow._blocks._models import Cell, Flag


def flags_in(value: Cell) -> list[Flag]:
    """A flags cell's flags; none where the cell holds something else, or nothing."""
    if not isinstance(value, list):
        return []
    return [flag for flag in value if isinstance(flag, Flag)]


def ordered(flags: Sequence[Flag]) -> list[Flag]:
    """A cell's flags as every renderer lists them: by name."""
    return sorted(flags, key=lambda flag: flag.name)


def percentile_text(percentile: float) -> str:
    """``p99.95``: one decimal, or two within 1% of either end, where values are told apart; ``p?`` when unknown.

    The tail is rounded so that p99.9 sits 0.1 from the end rather than just under it: ``100 - 99.9`` is
    ``0.0999…`` in floating point.
    """
    if not math.isfinite(percentile):
        return "p?"
    tail = round(min(percentile, 100.0 - percentile), 9)
    decimals = 2 if tail < 1 else 1
    text = f"{percentile:.{decimals}f}".rstrip("0").rstrip(".")
    return f"p{text}"


def tag_text(flag: Flag) -> str:
    """``brightness 0.99 > 0.84``: the measurement, its value, and the limit it crossed.

    ``<`` for a lower limit. A flag whose limit is unknown reads as its value alone.
    """
    if not math.isfinite(flag.bound):
        return f"{flag.name} {fmt_num(flag.value)}"
    sign = ">" if flag.direction == "upper" else "<"
    return f"{flag.name} {fmt_num(flag.value)} {sign} {fmt_num(flag.bound)}"
