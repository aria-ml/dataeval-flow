"""Configs for the ``scope`` evaluators: DataEval's Representation, Coverage and Prioritize.

Field names are DataEval's argument names. An unset field is not passed, so DataEval's own default applies.
"""

__all__ = ["RepresentationConfig"]

from typing import Annotated, Any, ClassVar

from pydantic import Field

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.evaluators._base import EvaluatorConfig
from dataeval_flow.evaluators.scope._result import RepresentationResult


class RepresentationConfig(EvaluatorConfig[RepresentationResult]):
    """Config for ``scope.representation``, DataEval's Representation.

    Counts the source's class labels against an ontology's leaves, and lists what to acquire for every leaf to reach
    its expected share. Needs a dataset with labels. With no ``ontology``, a flat one is synthesized from the
    dataset's ``index2label``, which can only name the classes the dataset declares.

    Every parameter, its DataEval argument and its unset behaviour is listed in the Evaluator Catalog
    (``reference/evaluators``), and ``dataeval-flow evaluators scope.representation`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: representation
            type: scope.representation
            ontology: vehicles
            expected: {truck: 0.2}
    """

    type: str = Field(
        default="scope.representation",
        description="The evaluator type this entry configures: `scope.representation`.",
    )
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.LABELS}), sources=SourceCount.ONE)

    ontology: dict[str, Any] | str | None = Field(
        default=None,
        description=(
            "The label space to count against: a name under the top-level `ontologies:` key, a path to a serialized "
            "RDF artifact resolved against the data root, or a nested mapping of concept to children. Unset "
            "synthesizes a flat ontology from the dataset's `index2label`, which can only name the classes the "
            "dataset declares."
        ),
    )
    expected: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]] | None = Field(
        default=None,
        description=(
            "Class name to its minimum expected share of the dataset, a fraction in [0, 1]. Named classes take this "
            "floor as their target in place of the uniform share, and are checked in `violations`; the others keep "
            "the uniform target. Unset expects a uniform share for every leaf."
        ),
    )
