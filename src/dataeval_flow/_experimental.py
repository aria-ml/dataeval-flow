"""Marking what is experimental: it may change or be removed in any release without a deprecation cycle.

A step type is marked by setting ``experimental = True`` on its class, and warns once per process when a chain runs
it. A public function or class is marked with :func:`experimental`, and warns once, on its first call or
instantiation.
"""

__all__ = ["ExperimentalWarning", "experimental", "warn_experimental", "warn_if_experimental"]

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

from dataeval.exceptions import ExperimentalWarning as _DataEvalExperimentalWarning

F = TypeVar("F", bound=Callable[..., Any])


class ExperimentalWarning(_DataEvalExperimentalWarning):
    """Issued when an experimental part of Flow is used.

    Experimental parts may change or be removed in any release without a deprecation cycle. A subclass of DataEval's
    ``ExperimentalWarning``, so a filter on DataEval's also covers Flow's.
    """


_WARNED: set[str] = set()


def warn_experimental(subject: str, *, stacklevel: int = 3) -> None:
    """Warn once per process that `subject` is experimental, such as ``The 'parity' evaluator``.

    The default `stacklevel` points at the caller of the function that calls this one.
    """
    if subject in _WARNED:
        return
    _WARNED.add(subject)
    message = f"{subject} is experimental: it may change or be removed in any release without a deprecation cycle."
    warnings.warn(message, ExperimentalWarning, stacklevel=stacklevel)


def warn_if_experimental(step: type[Any]) -> None:
    """Warn once per process that `step`'s type is experimental, where it is.

    `step` is a ``Step`` class. A preset's is typed ``type[Preset]``, a mixin that declares neither ``name`` nor
    ``kind``, though every preset class is a ``Workflow`` and so a ``Step``.
    """
    if step.experimental:
        warn_experimental(f"The '{step.name}' {step.kind}")


def experimental(target: F) -> F:
    """Mark a public function or class experimental: it warns once, on its first call or instantiation."""
    subject = f"'{target.__qualname__}'"
    if isinstance(target, type):
        init = target.__init__

        @functools.wraps(init)
        def warned_init(self: Any, *args: Any, **kwargs: Any) -> None:
            warn_experimental(subject)
            init(self, *args, **kwargs)

        target.__init__ = warned_init
        return target  # type: ignore[return-value]

    @functools.wraps(target)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warn_experimental(subject)
        return target(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
