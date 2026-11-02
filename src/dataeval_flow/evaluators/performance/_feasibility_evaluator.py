"""The ``feasibility`` evaluator: DataEval's Feasibility over one classification source, its embeddings and labels."""

__all__ = ["FeasibilityEvaluator"]

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
from dataeval_flow.evaluators.performance._feasibility_result import PopulatedFeasibilityOutput


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
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {
        InputKind.DATASET: "evaluate",
        InputKind.EMBEDDINGS: "evaluate",
        InputKind.LABELS: "evaluate",
    }
    output_extras: ClassVar[tuple[str, ...]] = ("items", "labeled", "absent_classes")
    reads_factors: ClassVar[bool] = False

    def run(self, config: FeasibilityConfig, inputs: Sequence[EvaluatorInputs]) -> FeasibilityOutput:
        """Refuse a source whose items are not independent or whose labels cannot support a claim, then decide each
        target on the source's labeled items, each item its own unit unless a view repeats it.

        Flow's extractor produced the embeddings, so DataEval is not given one. The units are the items of the dataset
        at the bottom of the source's views (``root_indices``), which is what DataEval itself assumes of a view over a
        plain dataset and what it would refuse to assume of a derived one.

        Raises
        ------
        ValueError
            When the source is a derived dataset (one whose root has a ``source``, such as ``DetectionCrops``), is
            empty, has no labeled item, or has labeled items in fewer than two classes.
        """
        (source,) = inputs
        dataset = require(source.dataset, "its dataset", source.source)
        embeddings = require(source.embeddings, "embeddings", source.source)
        labels = require(source.labels, "labels", source.source)
        names = dict(source.index2label or {})
        _refuse(dataset, labels, names, source.source)
        arguments = dataeval_arguments(config)
        arguments.pop("target")
        allowances = arguments.pop("allowances", None)
        measured = Feasibility(**arguments).evaluate(
            dataset,
            target=config.target,
            embeddings=embeddings,
            groups=root_indices(dataset),
            allowances=None if allowances is None else allowances.model_dump(exclude_none=True),
        )
        observed = {int(label) for label in labels}
        output = PopulatedFeasibilityOutput(
            determinations=measured.determinations,
            evidence=measured.evidence,
            items=len(dataset),
            labeled=len(labels),
            absent_classes=[name for index, name in sorted(names.items()) if index not in observed],
        )
        object.__setattr__(output, "_meta", measured.meta())  # DataEval's record of the call, as `set_metadata` sets it
        return output


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
