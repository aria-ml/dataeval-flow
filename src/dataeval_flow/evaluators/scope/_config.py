"""Configs for the ``scope`` evaluators: DataEval's Representation, Coverage and Prioritize.

Field names are DataEval's argument names. An unset field is not passed, so DataEval's own default applies.
"""

__all__ = ["CoverageConfig", "LabelAlignmentConfig", "LabelAlignmentResult", "PrioritizeConfig", "RepresentationConfig"]

from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field

from dataeval_flow._alignment import LabelAlignmentOutput
from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.evaluators._base import EvaluatorConfig
from dataeval_flow.evaluators._result import EvaluatorResult
from dataeval_flow.evaluators.scope._result import CoverageResult, PrioritizeResult, RepresentationResult


class RepresentationConfig(EvaluatorConfig[RepresentationResult]):
    """Config for ``representation``, DataEval's Representation.

    Counts the source's class labels against an ontology's leaves, and lists what to acquire for every leaf to reach
    its expected share. Needs a dataset with labels. With no ``ontology``, a flat one is synthesized from the
    dataset's ``index2label``, which can only name the classes the dataset declares.

    Every parameter, its DataEval argument and its unset behaviour is listed in the Evaluator Catalog
    (``reference/evaluators``), and ``dataeval-flow evaluators representation`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: representation
            type: representation
            ontology: vehicles
            expected: {truck: 0.2}
    """

    type: str = Field(
        default="representation",
        description="The evaluator type this entry configures: `representation`.",
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


class CoverageConfig(EvaluatorConfig[CoverageResult]):
    """Config for ``coverage``, DataEval's Coverage.

    Finds the items in sparse regions of the embedding space, which the rest of the data does not cover, and breaks
    the coverage down by class. Needs an extractor on the task. Uses the dataset's labels where there is one per item;
    without them it runs over every item as one class, ``0``, and logs a warning.

    Every parameter, its DataEval argument and its unset behaviour is listed in the Evaluator Catalog
    (``reference/evaluators``), and ``dataeval-flow evaluators coverage`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: coverage
            type: coverage
            num_observations: 30
    """

    type: str = Field(default="coverage", description="The evaluator type this entry configures: `coverage`.")
    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.EMBEDDINGS, InputKind.LABELS}), sources=SourceCount.ONE
    )

    method: Literal["naive", "adaptive"] | None = Field(
        default=None,
        description=(
            "How the coverage radius is set: `naive`, a fixed analytic radius, or `adaptive`, a cutoff on the "
            "`percent` most sparsely neighbored items. Unset uses DataEval's default (`adaptive`)."
        ),
    )
    num_observations: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Neighbors an item needs within the radius to count as covered, fewer than the source's items; 20 to 50 is "
            "typical. Unset uses DataEval's default (20)."
        ),
    )
    percent: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of items to flag as uncovered, for the `adaptive` method only. Unset uses DataEval's default "
            "(0.01)."
        ),
    )
    min_class_samples: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Items a class needs for its dispersion, isotropy and near-duplicate fraction to be assessed; smaller "
            "classes are reported with `assessable` false. Unset uses DataEval's default (20)."
        ),
    )
    isotropy_min_samples: int | None = Field(
        default=None,
        gt=0,
        description="Items a class needs for its isotropy to be reported. Unset uses one more than the embedding size.",
    )
    near_duplicate_factor: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "A nearest-neighbor pair counts as a near duplicate when its distance is below this fraction of the "
            "class's median nearest-neighbor distance. Unset uses DataEval's default (0.5)."
        ),
    )


class PrioritizeConfig(EvaluatorConfig[PrioritizeResult]):
    """Config for ``prioritize``, DataEval's Prioritize.

    Ranks the first source's items from easiest to hardest, or the reverse, by where they sit in the embedding space.
    A second source is the reference: the items are then ranked relative to it, as when choosing what to label next
    beside data already labeled. Needs an extractor on the task.

    Every parameter, its DataEval argument and its unset behaviour is listed in the Evaluator Catalog
    (``reference/evaluators``), and ``dataeval-flow evaluators prioritize`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: next_to_label
            type: prioritize
            order: hard_first

        tasks:
          - name: rank_unlabeled
            evaluator: next_to_label
            sources: [unlabeled, labeled]
            extractor: bovw_ext
    """

    type: str = Field(default="prioritize", description="The evaluator type this entry configures: `prioritize`.")
    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.EMBEDDINGS, InputKind.LABELS}), sources=SourceCount.ONE_OR_TWO
    )

    method: Literal["knn", "kmeans_distance", "kmeans_complexity", "hdbscan_distance", "hdbscan_complexity"] | None = (
        Field(
            default=None,
            description=(
                "How items are ranked: `knn`, by distance to their nearest neighbors; `kmeans_distance` or "
                "`hdbscan_distance`, by distance to their cluster's center; `kmeans_complexity` or "
                "`hdbscan_complexity`, by weighted sampling over the cluster structure. Unset uses DataEval's "
                "default (`knn`)."
            ),
        )
    )
    k: int | None = Field(
        default=None, gt=0, description="Nearest neighbors, for `knn`. Unset uses the square root of the item count."
    )
    c: int | None = Field(
        default=None,
        gt=0,
        description="Clusters, for the clustering methods. Unset uses the square root of the item count.",
    )
    n_init: int | Literal["auto"] | None = Field(
        default=None,
        description=(
            "K-means initializations, for the `kmeans_*` methods: a count, or `auto`. Unset uses DataEval's default "
            "(`auto`)."
        ),
    )
    max_cluster_size: int | None = Field(
        default=None, gt=0, description="Largest cluster, for the `hdbscan_*` methods. Unset leaves it unbounded."
    )
    order: Literal["easy_first", "hard_first"] | None = Field(
        default=None,
        description=(
            "Sort direction: `easy_first` puts prototypical items first, `hard_first` challenging ones. Unset uses "
            "DataEval's default (`easy_first`)."
        ),
    )
    policy: Literal["difficulty", "stratified", "class_balanced"] | None = Field(
        default=None,
        description=(
            "How the ranking is reordered: `difficulty` keeps it, `stratified` balances it across difficulty bins, and "
            "`class_balanced` across class labels, which needs a dataset with one label per item. Unset uses "
            "DataEval's default (`difficulty`)."
        ),
    )
    num_bins: int | None = Field(
        default=None,
        gt=0,
        description="Difficulty bins, for the `stratified` policy. Unset uses DataEval's default (50).",
    )


class LabelAlignmentResult(EvaluatorResult[LabelAlignmentOutput]):
    """The result of a ``label-alignment`` run; ``output`` is a
    :class:`~dataeval_flow.evaluators.scope.LabelAlignmentOutput`.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        The alignment: ``data()`` is the ``LabelAlignment`` as JSON (``mergeability``, ``correspondences``,
        ``unaligned_source``, ``unaligned_target``, ``class_remap``, ``paste_remap``, ``target_vocabulary``,
        ``ambiguous_labels``, ``label_space_digest``). ``output.alignment`` is the same, as a pydantic model, and
        ``output.ontology`` the resolved ontology a ``conform`` step relabels onto.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """


class LabelAlignmentConfig(EvaluatorConfig[LabelAlignmentResult]):
    """Config for ``label-alignment``: how a Dataset's class names align to an ontology.

    Wraps ``dataeval.core.label_alignment``, with the remap written as labels to paste, the target vocabulary, and the
    label-space digest a conformed corpus carries. A ``conform`` step applies it.
    """

    type: str = Field(
        default="label-alignment",
        description="The evaluator type this entry configures: `label-alignment`.",
    )
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.LABELS}), sources=SourceCount.ONE)
    ontology: dict[str, Any] | str = Field(
        description="The ontology to align to: a name from `ontologies:`, a path, or an inline hierarchy."
    )
    threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="DataEval's `threshold`: the lowest confidence a fuzzy match keeps."
    )
