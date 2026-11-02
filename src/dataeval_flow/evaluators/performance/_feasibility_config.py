"""Config for the ``feasibility`` evaluator: DataEval's Feasibility, whether the data demonstrates a performance target.

Field names are DataEval's argument names. An unset field is not passed, so DataEval's own default applies, except
``alpha``, which is stated explicitly so the result's ``resolved_config`` records the error budget the call spent.
"""

__all__ = ["AllowancesConfig", "FeasibilityConfig"]

import math
import re
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.evaluators._base import EvaluatorConfig
from dataeval_flow.evaluators.performance._feasibility_result import FeasibilityResult

# DataEval's `metric[:class][/scope][|aggregation]` for the metrics a classification dataset takes without a
# false-positive limit. `recall` needs a limit a key cannot carry, and `@iou` is for detection.
_TARGET_KEY = re.compile(
    r"(accuracy|ap)(:[^@/|]+)?(/(full_input|fixed_representation|partition_restricted))?(\|sequence_mean)?"
)


class AllowancesConfig(BaseModel):
    """The assumptions a ``feasibility`` run is told to make, which widen the scope at which a target can be ruled out.

    Each is **your assertion**, recorded as assumed in the result, not something DataEval measures. Each must be a
    justified bound: ``beta: 0`` asserts that no model reading the representation beats one fixed answer per cell of
    DataEval's partition of it.

    Example YAML::

        allowances: {beta: 0, gamma: 0}
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    beta: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description=(
            "Cell heterogeneity, a scalar: how much better than one fixed answer per cell a model reading the "
            "representation is asserted to do. `0` asserts it does no better. Your assertion, recorded as assumed."
        ),
    )
    gamma: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description=(
            "Information loss, a scalar, only with `beta`: how much a model reading the whole input could gain over "
            "one reading the representation. `0` asserts that the representation loses nothing such a model could use. "
            "Your assertion, recorded as assumed."
        ),
    )

    @model_validator(mode="after")
    def _gamma_needs_beta(self) -> Self:
        if self.gamma is not None and self.beta is None:
            raise ValueError("`gamma` is allowed only together with `beta`: DataEval adds it on top of `beta`.")
        return self


class FeasibilityConfig(EvaluatorConfig[FeasibilityResult]):
    """Config for ``feasibility``, DataEval's Feasibility.

    For each target, such as ``accuracy: 0.9``, decides whether the labeled items of the one source demonstrate that
    a model can reach it, rule it out under recorded assumptions, or neither, and keeps the evidence behind each
    answer. One call is one ledger: ``alpha`` is the error budget shared by every bound of every target, so adding a
    target widens every bound, and choosing among several calls on one dataset (several extractors, say) needs
    ``alpha`` divided by their number. For that reason a task's ``matrix:`` may vary only ``sources``.

    Only the labeled items count, and the claims are about them. A source that is a derived dataset, such as
    ``DetectionCrops``, is refused, because its items share a source image and are not independent; so is a source with
    fewer than two classes among its labeled items. The task's extractor must not have seen the labels, which Flow
    cannot check.

    Every parameter, its DataEval argument and its unset behavior is listed in the Evaluator Catalog
    (``reference/evaluators``), and ``dataeval-flow evaluators feasibility`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: feasibility
            type: feasibility
            target: {accuracy: 0.9}
            allowances: {beta: 0, gamma: 0}
    """

    type: str = Field(default="feasibility", description="The evaluator type this entry configures: `feasibility`.")
    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.DATASET, InputKind.EMBEDDINGS, InputKind.LABELS}),
        sources=SourceCount.ONE,
        dataset_kinds=frozenset({"classification"}),
    )
    matrix_refusal: ClassVar[str | None] = (
        "each run is a separate call over the same data, and choosing among runs needs `alpha` divided by their "
        "number, which Flow cannot know you mean; only separate populations (`sources`) make separate claims."
    )

    target: dict[str, float] = Field(
        min_length=1,
        description=(
            "The performance targets: `metric[:class][/scope][|aggregation]` mapped to the threshold the model "
            "should reach, in (0, 1], such as `{accuracy: 0.9}`. The metric is `accuracy`, or `ap` (macro, or for "
            "`ap:<class>`); the scope, which defaults to `full_input`, is `fixed_representation` or "
            "`partition_restricted` to judge models that read less; `|sequence_mean` averages accuracy over groups "
            "of items. A recall target needs a false-positive limit and is not offered. Adding a target widens "
            "every bound, because `alpha` is shared among them."
        ),
    )
    alpha: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        allow_inf_nan=False,
        description=(
            "The error budget of the whole call, shared by every bound of every target: all of them hold together "
            "with probability at least 1 - `alpha`. Comparing `k` extractors or models on one dataset, or keeping the "
            "best of `k` runs, needs `alpha / k` per call."
        ),
    )
    panel_size: int | None = Field(
        default=None,
        ge=2,
        description="Labels per local panel, at least 2. Unset uses DataEval's provisional default (5).",
    )
    n_cells: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Cells of the partition the local panels pool within. Unset follows the number of audit units (10 "
            "below 20,000)."
        ),
    )
    builtin_witnesses: bool | None = Field(
        default=None,
        description=(
            "Whether to fit and score the built-in k-NN and linear-probe models. Unset uses DataEval's default (true)."
        ),
    )
    seed: int | None = Field(
        default=None,
        description=(
            "The seed of the splits, the partition and the panels, which makes a run repeatable. Unset uses the "
            "pipeline's `seed`, else 0."
        ),
    )
    allowances: AllowancesConfig | None = Field(
        default=None,
        description=(
            "Assumptions asserted to rule a target out at a broader scope: `beta`, then `gamma` on top of it. Without "
            "them, and without repeated labels, which Flow does not take, a target at the default scope can only be "
            "shown, not ruled out. They are your assertions, recorded as assumed; each must be a justified bound. "
            "`beta: 0` asserts that no model reading the representation beats one fixed answer per cell."
        ),
    )

    @field_validator("target")
    @classmethod
    def _targets(cls, value: dict[str, float]) -> dict[str, float]:
        """Keys in DataEval's grammar, for the metrics offered, and thresholds in (0, 1], checked here because DataEval
        parses them against the data."""
        for key, threshold in value.items():
            if not _TARGET_KEY.fullmatch(key):
                raise ValueError(
                    f"`{key}` is not a target key: write `accuracy` or `ap`, then optionally `:<class>` (for `ap`), "
                    "`/<scope>` (`full_input`, `fixed_representation` or `partition_restricted`) and `|sequence_mean`."
                )
            if not (math.isfinite(threshold) and 0 < threshold <= 1):
                raise ValueError(f"The threshold of `{key}` must be a number in (0, 1]; got {threshold}.")
        return value
