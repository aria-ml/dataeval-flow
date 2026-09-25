"""The transform registry: the built-in table and the `dataeval_flow.transforms` entry points."""

from typing import Any

from dataeval_flow._registry import Registry
from dataeval_flow.config.transforms._base import Transform

__all__ = ["TRANSFORMS", "get_transform", "list_transforms", "resolve_step"]

_BUILTINS = {
    "ToRGB": "dataeval_flow.config.transforms._to_rgb:ToRGB",
}


def _torchvision_transform(name: str) -> Any:
    """What ``torchvision.transforms.v2`` holds as `name`, or ``None``; ``None`` too without torchvision."""
    try:
        from torchvision.transforms import v2
    except ImportError:
        return None
    return getattr(v2, name, None)


def _takes_no_torchvision_name(name: str, cls: type[Transform]) -> str | None:  # noqa: ARG001
    """A transform may not take a torchvision name: installing a plugin must not change what a step means."""
    if _torchvision_transform(name) is None:
        return None
    return (
        f"its name {name!r} is taken by `torchvision.transforms.v2.{name}`: a step {name!r} means torchvision's "
        "transform, and installing a plugin must not change that. Choose a name torchvision does not use."
    )


TRANSFORMS: Registry[Transform] = Registry(
    kind="transform",
    group="dataeval_flow.transforms",
    base=lambda: Transform,
    builtins=_BUILTINS,
    check=_takes_no_torchvision_name,
)


def get_transform(name: str) -> type[Transform]:
    """The transform registered as `name`, built-in or plugin.

    Only registered transforms resolve here; a step may also name a ``torchvision.transforms.v2`` transform.

    Parameters
    ----------
    name : str
        The transform's name: its YAML ``step:`` value, e.g. ``"ToRGB"``.

    Returns
    -------
    type[Transform]
        The transform class, whose ``name`` and ``description`` describe it. Flow builds its instances.

    Raises
    ------
    ValueError
        When nothing is registered under `name`, or the plugin registered under it failed to load.
    """
    return TRANSFORMS.get(name)


def list_transforms() -> list[type[Transform]]:
    """Every installed transform, built-in or plugin, sorted by name. torchvision's are not listed.

    A plugin that failed to load is left out; :func:`get_transform` raises its error.

    Returns
    -------
    list[type[Transform]]
        The transform classes.
    """
    return TRANSFORMS.list()


def resolve_step(name: str) -> Any:
    """The transform class a preprocessing step's ``step:`` names: a registered one, else torchvision's.

    No registered transform takes a torchvision name (the registry refuses a plugin that tries), so a name
    torchvision has always means torchvision's transform.

    Raises
    ------
    ValueError
        When a plugin claiming `name` was refused (the error is the reason, e.g. its import failure), or when
        neither the registry nor ``torchvision.transforms.v2`` has `name`.
    """
    if name in TRANSFORMS.names():
        return TRANSFORMS.get(name)
    if (torchvision := _torchvision_transform(name)) is not None:
        return torchvision
    if (problem := TRANSFORMS.problem(name)) is not None:
        raise ValueError(problem)
    raise ValueError(
        f"Unknown transform: '{name}'. Must be a registered transform "
        f"({', '.join(TRANSFORMS.names())}) or a torchvision.transforms.v2 transform."
    )
