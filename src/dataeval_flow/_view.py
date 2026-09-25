"""View convenience builder wrapping DataEval."""

__all__ = ["build_view"]

import typing
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

import dataeval.data as ddata
from dataeval.data import View
from dataeval.protocols import AnnotatedDataset

if TYPE_CHECKING:
    from dataeval_flow.config._schemas import ViewOperation

T = TypeVar("T")


def _admits_tuple(annotation: Any) -> bool:
    """Whether *annotation* accepts a tuple, looking inside unions and optionals."""
    if annotation is None:
        return False
    if annotation is tuple or typing.get_origin(annotation) is tuple:
        return True
    return any(_admits_tuple(arg) for arg in typing.get_args(annotation))


def _coerce_tuple_params(operation_cls: type, params: Mapping[str, Any]) -> dict[str, Any]:
    """Convert list values to tuples where the operation's signature asks for one.

    YAML and JSON have no tuple type, so a config can only ever supply a list. The
    operations that validate their arguments as tuples -- ``Resize(size=(h, w))``
    and ``Crop(region=(x0, y0, x1, y1))`` -- reject a list outright, which puts them
    out of reach of every config-driven view unless the conversion happens here.

    Only parameters whose annotation admits a tuple are converted, so the operations
    that take an ordinary sequence (``ClassFilter(classes=...)``,
    ``Indices(indices=...)``) receive exactly what the config gave them.
    """
    try:
        hints = typing.get_type_hints(operation_cls.__init__)
    except (NameError, TypeError):  # pragma: no cover — unresolvable annotations
        return dict(params)

    return {
        name: tuple(value) if isinstance(value, list) and _admits_tuple(hints.get(name)) else value
        for name, value in params.items()
    }


def build_view(dataset: AnnotatedDataset[T], operations: list["ViewOperation"]) -> View[T]:
    """Build a dataset view pipeline from config.

    Pass-through to :mod:`dataeval.data`, except that list parameters are converted
    to tuples where an operation's signature requires one — see
    :func:`_coerce_tuple_params`.

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
        ops.append(operation_cls(**_coerce_tuple_params(operation_cls, op.params)))

    return View(dataset, operations=ops)
