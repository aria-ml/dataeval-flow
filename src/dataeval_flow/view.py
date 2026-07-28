"""View convenience builder wrapping DataEval."""

__all__ = ["build_view", "build_selection"]

from typing import TYPE_CHECKING, TypeVar

import dataeval.data as ddata
from dataeval.data import View
from dataeval.protocols import AnnotatedDataset

if TYPE_CHECKING:
    from dataeval_flow.config.schemas import ViewOperation

T = TypeVar("T")


def build_view(dataset: AnnotatedDataset[T], operations: list["ViewOperation"]) -> View[T]:
    """Build a dataset view pipeline from config.

    Pass-through to :mod:`dataeval.data` - no custom logic.

    Parameters
    ----------
    dataset : MaiteDataset
        Input dataset to wrap with view operations.
    operations : list[ViewOperation]
        View operations from config.

    Returns
    -------
    View
        Dataset wrapped with the configured operations.

    Example
    -------
    >>> from dataeval_flow.config import ViewOperation
    >>> operations = [
    ...     ViewOperation(type="Limit", params={"size": 10000}),
    ...     ViewOperation(type="ClassFilter", params={"classes": [0, 1, 2]}),
    ... ]
    >>> filtered = build_view(dataset, operations)
    """
    ops = []
    for op in operations:
        operation_cls = getattr(ddata, op.type, None)
        if operation_cls is None:
            raise ValueError(f"Unknown view operation type: '{op.type}'. Check dataeval.data docs.")
        ops.append(operation_cls(**op.params))

    return View(dataset, operations=ops)


# Deprecated alias retained for backward compatibility; use build_view instead.
build_selection = build_view
