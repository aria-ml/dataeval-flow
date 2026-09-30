"""How a check reads a threshold: past it warns, and ``None`` judges nothing (spec §9.2)."""

__all__ = ["exceeds", "unjudged"]

from typing import Literal

Severity = Literal["ok", "info", "warning"]


def exceeds(value: float, limit: float | None) -> bool:
    """Whether `value` is past `limit`; never where the limit is ``None``, which judges nothing."""
    return limit is not None and value > limit


def unjudged(limit: float | None, severity: Severity) -> Severity:
    """`severity`, or ``info`` where `limit` is ``None``: a finding nothing judged is information."""
    return "info" if limit is None else severity
