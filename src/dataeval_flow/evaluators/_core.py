"""A dataeval.core function's plain result, shaped as DataEval's Output so Flow keeps and serializes it the same way."""

__all__ = ["CoreOutput", "execution"]

from collections.abc import Mapping
from datetime import datetime
from typing import Any


class CoreOutput:
    """A core function's result: ``data()`` returns it, ``meta()`` records the call as DataEval would."""

    def __init__(self, data: Mapping[str, Any], meta: Any) -> None:
        self._data = dict(data)
        self._meta = meta

    def data(self) -> dict[str, Any]:
        """The result, as plain data."""
        return self._data

    def meta(self) -> Any:
        """The call's ``ExecutionMetadata``."""
        return self._meta


def execution(name: str, started: datetime, seconds: float, arguments: Mapping[str, Any]) -> Any:
    """DataEval's ``ExecutionMetadata`` for a core function Flow called."""
    import dataeval
    from dataeval.types import ExecutionMetadata

    return ExecutionMetadata(
        name=name,
        execution_time=started,
        execution_duration=seconds,
        arguments=dict(arguments),
        state={},
        version=dataeval.__version__,
    )
