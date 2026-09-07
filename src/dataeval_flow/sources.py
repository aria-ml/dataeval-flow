"""Resolve a source to the datasets it reads, merging where it names operands.

A source names either one dataset or a `merge` of other sources. Resolving one flattens
that to its leaf datasets, loads each, applies the view it is conformed under, and
concatenates the results into the corpus a task reads. `label_space_records` reports the
vocabulary each of those views conformed its labels to, one record per view, so a result
carries the label space it was produced under.
"""

__all__ = [
    "MergeConfigError",
    "ResolvedSource",
    "SourceOperand",
    "flatten_source",
    "label_space_records",
    "resolve_source",
]

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any

    from dataeval.protocols import AnnotatedDataset
    from pydantic import BaseModel

    from dataeval_flow.config import PipelineConfig, SourceConfig, ViewConfig
    from dataeval_flow.config.schemas import LabelSpaceRecord
    from dataeval_flow.workflow import ResolvedOntology

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
        cache_key="merge:" + "|".join(_operand_key(operand) for operand in operands),
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


def _operand_key(operand: SourceOperand) -> str:
    """Return an operand's cache identity, covering the view it is merged under.

    An operand's view is applied before the merge, so it is part of the corpus rather
    than something the workflow applies later. Fold it into the key, or narrowing one
    operand's view would serve the previous corpus's cached embeddings.
    """
    view = operand.view_config.model_dump_json() if operand.view_config is not None else "none"
    return f"{operand.cache_key}+view:{view}"


def _view_of(source: "SourceConfig", config: "PipelineConfig") -> "ViewConfig | None":
    """Resolve a source's own view, or None where it names none."""
    from dataeval_flow.workflow.orchestrator import _resolve_by_name

    if source.view is None:
        return None
    return _resolve_by_name(config.views, source.view, "view")


def _relabel_params(view_config: "ViewConfig | None") -> "Mapping[str, Any] | None":
    """Return the params of the last Relabel in a view, or None where it holds none.

    The last one wins, because a later Relabel rewrites what an earlier one produced.
    """
    if view_config is None:
        return None
    relabels = [op.params for op in view_config.operations if op.type == "Relabel"]
    return relabels[-1] if relabels else None


def label_space_records(
    resolved_sources: "Sequence[ResolvedSource]",
    ontology: "ResolvedOntology | None",
) -> "list[LabelSpaceRecord]":
    """Build one record per view that conformed labels.

    One record per operand rather than one per result: a merge applies a different
    ``class_remap`` per operand against one shared target, so a single record would have
    to union the mappings — and a union hashes to a value no audit ever produced. A merged
    source's own view can then coarsen that shared target again, so it gets its own record
    under the source's own name.
    """
    from dataeval_flow.label_space import ontology_digest

    # `ResolvedOntology.source` is a source label: a pool entry's name, a resolved path, or
    # `inline`. Read it only when the ontology loaded — a failed load leaves the label set
    # and the ontology None, and recording the label then would claim a vocabulary nothing
    # was conformed to.
    name = ontology.source if ontology is not None and ontology.ontology is not None else None
    ids = list(ontology.ontology.ids) if ontology is not None and ontology.ontology is not None else []
    digest_of_ontology = ontology_digest(ids)

    records: list[LabelSpaceRecord] = []
    for rs in resolved_sources:
        for operand in rs.operands:
            params = _relabel_params(operand.view_config)
            if params is not None:
                records.append(_label_space_record(operand.source.name, params, name, digest_of_ontology))
        # A merged source's own view can carry a further Relabel — one that collapses the
        # operands' shared target into a coarser one. Only a merged source's own view is
        # recorded here: a plain source's view_config is the same object as its one
        # operand's, already recorded above, and recording it again would duplicate it.
        if rs.is_merged:
            merge_params = _relabel_params(rs.view_config)
            if merge_params is not None:
                records.append(_label_space_record(rs.name, merge_params, name, digest_of_ontology))
    return records


def _label_space_record(
    source_name: str,
    params: "Mapping[str, Any]",
    ontology_name: str | None,
    digest_of_ontology: str,
) -> "LabelSpaceRecord":
    """Build one record from a Relabel's params."""
    from dataeval_flow.config.schemas import LabelSpaceRecord
    from dataeval_flow.label_space import label_space_digest

    class_remap = {str(k): str(v) for k, v in dict(params.get("class_remap") or {}).items()}
    target = _relabel_target(params.get("target"), class_remap)
    return LabelSpaceRecord(
        source=source_name,
        ontology=ontology_name,
        ontology_digest=digest_of_ontology if ontology_name else None,
        class_remap=class_remap,
        target=target,
        digest=label_space_digest(ontology=digest_of_ontology, class_remap=class_remap, target=target),
    )


def _relabel_target(declared: "Any", class_remap: "Mapping[str, str]") -> list[str]:
    """Return the target vocabulary a Relabel actually ran with.

    Mirrors `dataeval.data.Relabel`: a mapping is an `index -> label` vocabulary — indices
    become positions, and a gap a sparse mapping leaves is backfilled with `""` so it
    cannot collide with the dense list a smaller mapping produces. A sequence is taken as
    given, including an explicit empty one. Omitted (`None`), the vocabulary is the
    remap's distinct values, first seen first. The mapping case is handled explicitly —
    letting a dict fall through to the sequence branch would iterate its keys and silently
    produce the wrong vocabulary.
    """
    if isinstance(declared, Mapping):
        indexed = {int(k): str(v) for k, v in declared.items()}
        width = max(indexed) + 1 if indexed else 0
        return [indexed.get(i, "") for i in range(width)]
    if declared is not None:
        return [str(v) for v in declared]
    return list(dict.fromkeys(class_remap.values()))
