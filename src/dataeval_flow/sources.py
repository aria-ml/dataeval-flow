"""Resolve a source to the datasets it reads, merging where it names operands."""

__all__ = ["MergeConfigError", "ResolvedSource", "SourceOperand", "flatten_source", "resolve_source"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any

    from dataeval.protocols import AnnotatedDataset
    from pydantic import BaseModel

    from dataeval_flow.config import PipelineConfig, SourceConfig, ViewConfig

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


@dataclass(frozen=True)
class SourceOperand:
    """One leaf source inside a source, with the dataset it reads."""

    source: "SourceConfig"
    dataset_config: "BaseModel"
    view_config: "ViewConfig | None"
    raw: "AnnotatedDataset[Any]"
    label_source: str | None
    cache_key: str

    def view(self) -> "AnnotatedDataset[Any]":
        """Return this operand's dataset with its own view applied.

        Build it here rather than storing it, so a source that is never merged pays
        nothing: a `Relabel` validates the dataset the moment its view is built.
        """
        from dataeval_flow.view import build_view

        if self.view_config is None:
            return self.raw
        return build_view(self.raw, list(self.view_config.operations))


@dataclass(frozen=True)
class ResolvedSource:
    """A source resolved to the dataset a task reads, and the operands behind it."""

    name: str
    operands: tuple[SourceOperand, ...]
    dataset: "AnnotatedDataset[Any]"
    view_config: "ViewConfig | None"
    cache_name: str
    cache_key: str

    @property
    def is_merged(self) -> bool:
        """Whether this source concatenates more than one operand."""
        return len(self.operands) > 1

    @property
    def label_sources(self) -> tuple[str | None, ...]:
        """Where each operand's labels came from, in merge order."""
        return tuple(operand.label_source for operand in self.operands)

    def realized(self) -> "AnnotatedDataset[Any]":
        """Return the dataset with this source's own view applied.

        A task gets :attr:`dataset` and :attr:`view_config` separately and lets the
        workflow apply the view. Use this where you need the finished corpus, such as
        writing it out.
        """
        from dataeval_flow.view import build_view

        if self.view_config is None:
            return self.dataset
        return build_view(self.dataset, list(self.view_config.operations))


def resolve_source(
    name: str,
    config: "PipelineConfig",
    data_dir: "Path | None" = None,
) -> ResolvedSource:
    """Resolve a source to the dataset it reads.

    A source naming a `dataset` hands that dataset back as loaded, with its view left
    for the workflow to apply. A source naming a `merge` loads every operand, applies
    each operand's own view, and concatenates the results — each operand must be
    conformed before it is merged, because `merge_datasets` refuses operands whose
    `index2label` differ.

    Parameters
    ----------
    name : str
        Source to resolve.
    config : PipelineConfig
        Pipeline holding the `sources:`, `datasets:` and `views:` pools.
    data_dir : Path or None
        Root a relative dataset path resolves against.

    Returns
    -------
    ResolvedSource
        The dataset, its operands, and the cache identity for the pair.
    """
    from dataeval.data import merge_datasets

    from dataeval_flow.workflow.orchestrator import _resolve_by_name

    source: SourceConfig = _resolve_by_name(config.sources, name, "source")
    leaves = flatten_source(name, config.sources)
    operands = tuple(_load_operand(leaf, config, data_dir) for leaf in leaves)

    if source.merge is None:
        return ResolvedSource(
            name=name,
            operands=operands,
            dataset=operands[0].raw,
            view_config=operands[0].view_config,
            cache_name=operands[0].dataset_config.name,  # pyright: ignore[reportAttributeAccessIssue]
            cache_key=operands[0].cache_key,
        )

    merged = merge_datasets([operand.view() for operand in operands])
    view_config = _view_of(source, config)
    return ResolvedSource(
        name=name,
        operands=operands,
        dataset=merged,
        view_config=view_config,
        cache_name=name,
        cache_key="merge:" + "|".join(operand.cache_key for operand in operands),
    )


def _load_operand(
    leaf: "SourceConfig",
    config: "PipelineConfig",
    data_dir: "Path | None",
) -> SourceOperand:
    """Load one leaf source's dataset and resolve its view."""
    from dataeval_flow.dataset import resolve_dataset
    from dataeval_flow.workflow.orchestrator import _resolve_by_name

    if leaf.dataset is None:  # pragma: no cover — flatten_source returns only leaves
        raise ValueError(f"Source '{leaf.name}' names no dataset.")

    dataset_config = _resolve_by_name(config.datasets, leaf.dataset, "dataset")
    resolved = resolve_dataset(dataset_config, data_dir=data_dir)
    return SourceOperand(
        source=leaf,
        dataset_config=dataset_config,
        view_config=_view_of(leaf, config),
        raw=resolved.dataset,
        label_source=resolved.label_source,
        cache_key=resolved.cache_key,
    )


def _view_of(source: "SourceConfig", config: "PipelineConfig") -> "ViewConfig | None":
    """Resolve a source's own view, or None where it names none."""
    from dataeval_flow.workflow.orchestrator import _resolve_by_name

    if source.view is None:
        return None
    return _resolve_by_name(config.views, source.view, "view")
