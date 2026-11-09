"""The ``feasibility`` evaluator: DataEval's Feasibility over one classification source, its embeddings and labels."""

__all__ = ["FeasibilityEvaluator"]

import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from dataeval.data import View
from dataeval.performance import Feasibility, FeasibilityOutput

from dataeval_flow._input_spec import InputKind
from dataeval_flow._view import root_indices
from dataeval_flow.evaluators._evaluator import Evaluator
from dataeval_flow.evaluators._fields import dataeval_arguments, require
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.performance._feasibility_config import FeasibilityConfig
from dataeval_flow.evaluators.performance._feasibility_groups import Grouping, group_items
from dataeval_flow.evaluators.performance._feasibility_result import PopulatedFeasibilityOutput
from dataeval_flow.evaluators.quality._config import DuplicatesConfig
from dataeval_flow.evaluators.quality._evaluator import find_duplicates
from dataeval_flow.evaluators.quality._leakage import item_column


class FeasibilityEvaluator(Evaluator[FeasibilityConfig, FeasibilityOutput]):
    """``feasibility``: whether the labeled items of one source demonstrate that a model can reach each target, per
    DataEval's Feasibility."""

    name: ClassVar[str] = "feasibility"
    title: ClassVar[str] = "Feasibility"
    description: ClassVar[str] = (
        "Whether the data demonstrates each performance target reachable, rules it out under stated assumptions, or "
        "does neither, with the proof behind every answer (DataEval Feasibility)."
    )
    experimental: ClassVar[bool] = True  # DataEval's Feasibility is experimental
    dataeval_class: ClassVar[type] = Feasibility
    # `from_stats` is `Duplicates`' (the hash groups), and `evaluate`'s `groups` is the factor the units are made from.
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {
        InputKind.DATASET: "evaluate",
        InputKind.EMBEDDINGS: "evaluate",
        InputKind.LABELS: "evaluate",
        InputKind.STATS: "from_stats",
        InputKind.METADATA: "evaluate",
    }
    output_extras: ClassVar[tuple[str, ...]] = (
        "items",
        "labeled",
        "absent_classes",
        "groups",
        "largest_group",
        "duplicate_merges",
    )
    reads_factors: ClassVar[bool] = False

    def run(self, config: FeasibilityConfig, inputs: Sequence[EvaluatorInputs]) -> FeasibilityOutput:
        """Refuse a source whose items are not independent or whose labels cannot support a claim, then decide each
        target on the source's labeled items, grouped into the units they are independent as.

        Flow's extractor produced the embeddings, so DataEval is not given one. Items are one unit when a view repeats
        one (the items of the dataset at the bottom of the source's views, ``root_indices``, which is what DataEval
        itself assumes of a view over a plain dataset and what it would refuse to assume of a derived one), when they
        share the value of ``groups``, or when they are exact or near duplicates.

        Raises
        ------
        ValueError
            When the source is a derived dataset (one whose root has a ``source``, such as ``DetectionCrops``), is
            empty, has no labeled item, or has labeled items in fewer than two classes; or ``groups`` names a factor
            the source lacks or one a labeled item has no value for.
        """
        (source,) = inputs
        dataset = require(source.dataset, "its dataset", source.source)
        embeddings = require(source.embeddings, "embeddings", source.source)
        labels = require(source.labels, "labels", source.source)
        names = dict(source.index2label or {})
        _refuse(dataset, labels, names, source.source)
        grouping = _group(config, source, dataset)
        _require_aggregation(config.target, grouping, len(dataset), source.source)
        arguments = dataeval_arguments(config)
        arguments.pop("target")
        arguments.pop("groups", None)
        allowances = arguments.pop("allowances", None)
        measured = Feasibility(**arguments).evaluate(
            dataset,
            target=config.target,
            embeddings=embeddings,
            groups=grouping.units,
            allowances=None if allowances is None else allowances.model_dump(exclude_none=True),
        )
        observed = {int(label) for label in labels}
        output = PopulatedFeasibilityOutput(
            determinations=measured.determinations,
            evidence=measured.evidence,
            items=len(dataset),
            labeled=len(labels),
            absent_classes=[name for index, name in sorted(names.items()) if index not in observed],
            groups=grouping.groups,
            largest_group=grouping.largest_group,
            duplicate_merges=grouping.duplicate_merges,
        )
        object.__setattr__(output, "_meta", measured.meta())  # DataEval's record of the call, as `set_metadata` sets it
        return output


def _group(config: FeasibilityConfig, source: EvaluatorInputs, dataset: Any) -> Grouping:
    """The source's items as the independent units they are: a view's repeats, a factor's values and duplicates."""
    factor = None if config.groups is None else _factor(config.groups, source, len(dataset))
    duplicates: list[Sequence[int]] = []
    if config.group_near_duplicates:
        found = find_duplicates(DuplicatesConfig(), [source]).items
        duplicates = [*found.exact, *(members for members, _ in found.near)]
    return group_items(root_indices(dataset), factor, duplicates)


def _factor(name: str, source: EvaluatorInputs, count: int) -> list[Any]:
    """The value of the metadata factor `name` for each of the source's `count` items, ``None`` for an unlabeled one.

    A labeled item with no value is refused: an item of unknown source is not shown independent of any other, and
    counting it alone would claim it is.
    """
    metadata = require(source.metadata, "metadata", source.source)
    # The per-item frame keeps the factor as recorded, so an ID that nearly every item holds apart, which the factors
    # DataEval keeps drop, still groups: the scene or recording an item came from is that kind of column.
    frame, found = item_column(metadata, name, source.source)
    column = frame[found].to_list()
    items = frame["item_index"].to_list()
    absent = sum(value is None or (isinstance(value, float) and math.isnan(value)) for value in column)
    if absent:
        raise ValueError(
            f"Source '{source.source}' has no value of `{name}` for {absent} of its {len(column)} labeled items, so "
            "they cannot be placed in a group: an item of unknown source is not shown independent of any other. Fill "
            "the factor in, or leave `groups` unset."
        )
    factor: list[Any] = [None] * count
    for item, value in zip(items, column, strict=True):
        factor[item] = value
    return factor


def _require_aggregation(target: Mapping[str, float], grouping: Grouping, items: int, source: str) -> None:
    """Raise when items share a unit and a target does not say how to average over it, which DataEval refuses too, but
    without saying why the source is grouped."""
    bare = [key for key in target if "|" not in key]
    if grouping.largest_group > 1 and bare:
        write = ", ".join(f"`{key}|sequence_mean`" for key in bare)
        raise ValueError(
            f"The {items:,} items of source '{source}' fall into {grouping.groups:,} independent units, the largest of "
            f"{grouping.largest_group:,} items, because items that a view repeats, that share the `groups` factor or "
            f"that duplicate each other are one unit. A target on units of more than one item must say how to average "
            f"over a unit: write {write}, which averages the metric over the units."
        )


def _refuse(dataset: Any, labels: Any, names: Mapping[int, str], source: str) -> None:
    """Raise for a source no target can be judged on, naming it and saying why."""
    root = dataset.root if isinstance(dataset, View) else dataset
    if getattr(root, "source", None) is not None:
        raise ValueError(
            f"Source '{source}' is a derived dataset, a {type(root).__name__} over another dataset: its items share a "
            "source and are not independent, and `feasibility` cannot tell which share one, so it judges no target on "
            "it."
        )
    if len(dataset) == 0:
        raise ValueError(f"Source '{source}' has no items, so there is nothing to judge a target on.")
    if len(labels) == 0:
        raise ValueError(
            f"Source '{source}' has no labeled item among its {len(dataset)}: only labeled items count, and "
            "`feasibility` judges a target on them."
        )
    observed = np.unique(labels)
    if len(observed) < 2:
        only = names.get(int(observed[0]), str(int(observed[0])))
        raise ValueError(
            f"Source '{source}' has labeled items in only one class, '{only}': with fewer than two observed classes, "
            "any target is met by answering that class every time, so there is insufficient data to judge one."
        )
