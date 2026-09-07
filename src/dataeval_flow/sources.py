"""Resolve a source to the datasets it reads, merging where it names operands."""

__all__ = ["MergeConfigError", "flatten_source"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataeval_flow.config import SourceConfig

#: How deep a source may nest merges. A merge of merges is legitimate; a chain this long
#: is a config that has lost track of itself.
_MAX_MERGE_DEPTH = 8


class MergeConfigError(ValueError):
    """A `merge:` that cannot be resolved — a cycle, or nesting past the depth bound."""


def flatten_source(name: str, pool: "Sequence[SourceConfig] | None") -> "list[SourceConfig]":
    """Return the leaf sources *name* reads, in the order they are concatenated.

    A source naming a `dataset` is its own single operand. A source naming a `merge`
    expands to its operands' leaves, depth first, so a merge of merges flattens to one
    ordered list.

    Parameters
    ----------
    name : str
        Source to resolve.
    pool : Sequence[SourceConfig] or None
        The config's `sources:` pool.

    Returns
    -------
    list[SourceConfig]
        Leaf sources, each naming a dataset.

    Raises
    ------
    MergeConfigError
        If a source merges itself through any path, or nests merges past the depth bound.
    ValueError
        If *name* or any operand is not in *pool*.
    """
    return _flatten(name, pool, path=[])


def _flatten(
    name: str,
    pool: "Sequence[SourceConfig] | None",
    path: "list[str]",
) -> "list[SourceConfig]":
    """Expand one source, carrying the path walked so far to report a cycle."""
    from dataeval_flow.workflow.orchestrator import _resolve_by_name

    if name in path:
        walked = " -> ".join([*path, name])
        if len(path) == 1:
            raise MergeConfigError(f"Source '{name}' merges itself: {walked}.")
        raise MergeConfigError(f"Source '{name}' merges itself through: {walked}.")
    if len(path) >= _MAX_MERGE_DEPTH:
        raise MergeConfigError(
            f"Source '{path[0]}' nests merges more than {_MAX_MERGE_DEPTH} deep. "
            "Flatten the config: name the leaf sources in one `merge`."
        )

    source: SourceConfig = _resolve_by_name(pool, name, "source")
    if source.merge is None:
        return [source]

    operands: list[SourceConfig] = []
    for operand in source.merge:
        operands.extend(_flatten(operand, pool, [*path, name]))
    return operands
