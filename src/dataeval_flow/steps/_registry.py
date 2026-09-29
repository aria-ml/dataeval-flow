"""The transform registry: the built-in table and the `dataeval_flow.transforms` entry points."""

__all__ = ["TRANSFORMS", "get_transform", "list_transforms"]

from typing import Any

from dataeval_flow._registry import Registry
from dataeval_flow.steps._step import Transform

_BUILTINS: dict[str, str] = {}

TRANSFORMS: Registry[Transform[Any]] = Registry(
    kind="transform",
    group="dataeval_flow.transforms",
    base=lambda: Transform,
    builtins=_BUILTINS,
)


def get_transform(name: str) -> type[Transform[Any]]:
    """The dataset transform registered as `name`, built-in or plugin.

    Raises
    ------
    ValueError
        When nothing is registered under `name`, or the plugin registered under it failed to load.
    """
    return TRANSFORMS.get(name)


def list_transforms() -> list[type[Transform[Any]]]:
    """Every installed dataset transform, sorted by name."""
    return TRANSFORMS.list()
