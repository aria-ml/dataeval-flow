"""Runtime discovery of available transforms and selection classes.

Introspects torchvision.transforms.v2 and dataeval.selection to provide
dropdown options and parameter schemas for the builder TUI.
"""

from __future__ import annotations

import collections.abc
import inspect
import types
import typing
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, get_args, get_origin


@dataclass
class ParamInfo:
    """Describes a single constructor parameter for a transform or selection class."""

    name: str
    type_hint: str  # human-readable type string
    required: bool
    default: Any = None
    choices: list[str] = field(default_factory=list)  # for Literal types


_PRIMITIVE_NAMES: dict[type, str] = {int: "int", float: "float", bool: "bool", str: "str"}


def _simplify_union(annotation: Any) -> tuple[str, list[str]]:
    """Simplify a Union or Optional type annotation."""
    args = [a for a in get_args(annotation) if a is not type(None)]
    if len(args) == 1:
        return _simplify_type(args[0])
    parts = []
    for a in args:
        t, _ = _simplify_type(a)
        parts.append(t)
    return " | ".join(dict.fromkeys(parts)), []


def _simplify_type(annotation: Any) -> tuple[str, list[str]]:
    """Convert a type annotation to a human-readable string + optional choices."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return "any", []

    if get_origin(annotation) is typing.Literal:
        return "select", [str(v) for v in get_args(annotation)]

    origin = get_origin(annotation)
    if origin is typing.Union or isinstance(annotation, types.UnionType):
        return _simplify_union(annotation)

    if annotation in _PRIMITIVE_NAMES:
        return _PRIMITIVE_NAMES[annotation], []

    if origin in (list, tuple, collections.abc.Sequence):
        inner_args = get_args(annotation)
        if inner_args:
            inner_t, choices = _simplify_type(inner_args[0])
            if choices:
                return "select", choices
            return f"list[{inner_t}]", []
        return "list", []

    return getattr(annotation, "__name__", str(annotation)), []


def _introspect_params(cls: type) -> list[ParamInfo]:
    """Extract constructor parameters from a class."""
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return []

    params: list[ParamInfo] = []
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        # Skip *args/**kwargs
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        required = param.default is inspect.Parameter.empty
        default = None if required else param.default
        type_hint, choices = _simplify_type(param.annotation)

        params.append(
            ParamInfo(
                name=pname,
                type_hint=type_hint,
                required=required,
                default=default,
                choices=choices,
            )
        )

    return params


@lru_cache(maxsize=1)
def list_transforms() -> list[str]:
    """Return sorted names of available torchvision.transforms.v2 classes."""
    from torchvision.transforms import v2

    skip = {
        "Transform",
        "Compose",
        "Identity",
        "RandomApply",
        "RandomChoice",
        "RandomOrder",
        "AutoAugmentPolicy",
        "InterpolationMode",
        "Lambda",
    }
    names = []
    for name, obj in inspect.getmembers(v2):
        if (
            inspect.isclass(obj)
            and not name.startswith("_")
            and name[0].isupper()
            and name not in skip
            and callable(obj)
        ):
            names.append(name)
    return sorted(set(names))


@lru_cache(maxsize=1)
def list_selection_classes() -> list[str]:
    """Return sorted names of available dataeval.selection classes."""
    from dataeval import selection
    from dataeval.selection._select import Selection

    skip = {"Selection", "Subselection", "SelectionStage", "Select"}
    names = []
    for name, obj in inspect.getmembers(selection):
        if (
            inspect.isclass(obj)
            and not name.startswith("_")
            and name not in skip
            and issubclass(obj, Selection)
            and obj is not Selection
        ):
            names.append(name)
    return sorted(names)


@lru_cache(maxsize=64)
def get_transform_params(name: str) -> list[ParamInfo]:
    """Get parameter info for a torchvision.transforms.v2 class."""
    from torchvision.transforms import v2

    cls = getattr(v2, name, None)
    if cls is None:
        return []
    return _introspect_params(cls)


@lru_cache(maxsize=64)
def get_selection_params(name: str) -> list[ParamInfo]:
    """Get parameter info for a dataeval.selection class."""
    from dataeval import selection

    cls = getattr(selection, name, None)
    if cls is None:
        return []
    return _introspect_params(cls)
