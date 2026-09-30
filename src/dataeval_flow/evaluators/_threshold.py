"""The threshold forms a config file can write: DataEval's ``ThresholdLike``, less the objects it cannot build.

``outliers`` passes them to DataEval as they are; ``chunking:`` resolves them with DataEval's
``resolve_threshold``, because ``chunked()`` takes a ``Threshold`` object.
"""

__all__ = ["ThresholdSpec"]

# One bound, or a (lower, upper) pair.
Bounds = float | tuple[float | None, float | None]
# Limits a bound is clipped to, as (lower, upper).
Limits = tuple[float | None, float | None]
# DataEval's ``ThresholdLike``, less the ``Threshold`` objects a config cannot build: a
# method name, bounds, or ``[method, bounds]``, ``[method, bounds, limits]``, ``[bounds, limits]``.
ThresholdSpec = (
    str | Bounds | tuple[str, Bounds | None] | tuple[str, Bounds | None, Limits] | tuple[Bounds | None, Limits]
)
