"""The scope evaluators: DataEval's Representation, Coverage and Prioritize.

``run`` is the only code here that calls DataEval.
"""

__all__ = ["RepresentationEvaluator"]

from collections.abc import Mapping, Sequence
from typing import ClassVar

from dataeval.scope import Representation, RepresentationOutput

from dataeval_flow._input_spec import InputKind
from dataeval_flow.evaluators._evaluator import Evaluator
from dataeval_flow.evaluators._fields import dataeval_arguments, require
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.scope._config import RepresentationConfig


class RepresentationEvaluator(Evaluator[RepresentationConfig, RepresentationOutput]):
    """``scope.representation``: which of an ontology's classes fall short of their share, per DataEval's
    Representation."""

    name: ClassVar[str] = "scope.representation"
    description: ClassVar[str] = "Class counts against an ontology's leaves (DataEval Representation)"
    dataeval_class: ClassVar[type] = Representation
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.LABELS: "evaluate"}
    output_extras: ClassVar[tuple[str, ...]] = ("leaf_coverage", "total_deficit", "violations", "dark_branches")

    def run(self, config: RepresentationConfig, inputs: Sequence[EvaluatorInputs]) -> RepresentationOutput:
        """Count the source's labels against the task's ontology, or one synthesized from its ``index2label``."""
        from dataeval_flow.workflows._ontology import synthesize_ontology

        (source,) = inputs
        labels = require(source.labels, "labels", source.source)
        if len(labels) == 0:
            raise ValueError(
                f"Source '{source.source}' has no labels: its dataset's targets carry none, and scope.representation "
                "counts class labels."
            )
        index2label = dict(source.index2label or {})
        ontology = source.ontology if source.ontology is not None else synthesize_ontology(index2label)[0]
        return Representation(ontology, **dataeval_arguments(config)).evaluate(labels, index2label=index2label)
