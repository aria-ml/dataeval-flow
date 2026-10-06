"""The ``data-coverage`` preset: coverage and completeness on the crops, labels and metadata on the source, and the
class worklist (coverage spec §4.2)."""

__all__ = ["DataCoverageWorkflow", "coverage_evaluators", "embedding_steps", "factor_steps", "gap_steps"]

from typing import TYPE_CHECKING, Any, ClassVar

from dataeval_flow.evaluators.bias import BalanceConfig, DiversityConfig, FactorSummaryConfig
from dataeval_flow.evaluators.quality import LabelHealthConfig
from dataeval_flow.evaluators.scope import CompletenessConfig, CoverageConfig, RepresentationConfig
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import Preset, PresetChain
from dataeval_flow.workflows.data_coverage._config import DataCoverageConfig

if TYPE_CHECKING:
    from dataeval_flow.workflows.audit._config import AuditConfig

    Covered = DataCoverageConfig | AuditConfig
    """The entries whose settings hold the coverage steps' blocks, and their checks'."""


def coverage_evaluators(config: "Covered") -> list[Any]:
    """The entries the coverage steps name: `coverage`, `completeness`, `factor-summary`, `balance` and `diversity`."""
    return [
        CoverageConfig(name="coverage", **config.coverage.model_dump()),
        CompletenessConfig(name="completeness"),
        FactorSummaryConfig(name="factor-summary", metadata=config.metadata),
        BalanceConfig(name="balance", metadata=config.metadata),
        DiversityConfig(name="diversity", method=config.diversity.method, metadata=config.metadata),
    ]


def embedding_steps(config: "Covered", source: str, *, completeness: bool = True) -> list[dict[str, Any]]:
    """`crops`, `source` with its detections cropped and other kinds passed through, then `coverage` (optional) with
    `class-coverage`, and `uncovered-items` under `naive` coverage; and, where `completeness`, `completeness`
    (optional) with `dimensional-completeness`."""
    limits = config.checks
    steps: list[dict[str, Any]] = [
        {
            "name": "crops",
            "transform": "wrap",
            "input": source,
            "wrapper": "DetectionCrops",
            "params": config.wrap.params.model_dump(),
            "other_kinds": "pass",
        },
        {"name": "coverage", "evaluator": "coverage", "input": "crops", "optional": True},
        {
            "name": "class-coverage",
            "check": "class-coverage",
            "input": "coverage",
            **limits.class_coverage.model_dump(),
        },
    ]
    if config.coverage.method == "naive":
        steps.append(
            {
                "name": "uncovered-items",
                "check": "uncovered-items",
                "input": "coverage",
                **limits.uncovered_items.model_dump(),
            }
        )
    if completeness:
        steps += [
            {"name": "completeness", "evaluator": "completeness", "input": "crops", "optional": True},
            {
                "name": "dimensional-completeness",
                "check": "dimensional-completeness",
                "input": "completeness",
                **limits.dimensional_completeness.model_dump(),
            },
        ]
    return steps


def factor_steps(source: str) -> list[dict[str, Any]]:
    """`factor-summary`, `balance` (optional) and `diversity` (optional) on `source`."""
    return [
        {"name": "factor-summary", "evaluator": "factor-summary", "input": source},
        {"name": "balance", "evaluator": "balance", "input": source, "optional": True},
        {"name": "diversity", "evaluator": "diversity", "input": source, "optional": True},
    ]


def gap_steps(config: "Covered", source: str) -> list[dict[str, Any]]:
    """`factor-gaps` (optional) on `source` with `factor-coverage-gaps`; none where `factor-gaps` is false."""
    if config.factor_gaps is False:
        return []
    return [
        {
            "name": "factor-gaps",
            "combine": "factor-gaps",
            "input": source,
            "balance": "balance",
            "optional": True,
            "metadata": config.metadata,
            **config.factor_gaps.model_dump(),
        },
        {
            "name": "factor-coverage-gaps",
            "check": "factor-coverage-gaps",
            "input": "factor-gaps",
            **config.checks.factor_coverage_gaps.model_dump(),
        },
    ]


class DataCoverageWorkflow(Preset, Workflow[DataCoverageConfig, ChainResult]):
    """Judges the task's one source, ``data``: how its embeddings cover their space, and its labels and metadata.

    The settings expand to, in legacy's finding order:

    - ``crops`` (``wrap``, ``DetectionCrops``, passing other kinds through), then ``coverage`` (optional) with
      ``class-coverage`` and, under ``naive`` coverage, ``uncovered-items``; and ``completeness``
      (optional) with ``dimensional-completeness``, where ``completeness`` is set;
    - ``label-health`` and ``class-imbalance``;
    - ``factor-summary``, ``balance`` and ``diversity`` (optional), and ``factor-gaps`` (optional) with
      ``factor-coverage-gaps``, where ``factor-gaps`` is set;
    - ``representation`` (with no ontology, optional) and ``class-shortfall``.

    The embedding steps are skipped with "requires an extractor" when the task names none. It makes no Dataset, so it
    declares no outputs; ontology analysis is ``label-space``'s.
    """

    name: ClassVar[str] = "data-coverage"
    title: ClassVar[str] = "Data Coverage"
    description: ClassVar[str] = (
        "Judges how a Dataset's embeddings cover their space, its class balance and metadata gaps, and what to acquire "
        "per class; detections are cropped first."
    )
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

    @classmethod
    def chain(cls, config: DataCoverageConfig) -> PresetChain:
        """The crops' steps, then the source's."""
        evaluators = [
            *coverage_evaluators(config),
            LabelHealthConfig(name="label-health", metadata=config.metadata),
            RepresentationConfig(name="representation", expected=config.representation.expected),
        ]
        steps = [
            *embedding_steps(config, "data", completeness=config.completeness),
            {"name": "label-health", "evaluator": "label-health", "input": "data"},
            {
                "name": "class-imbalance",
                "check": "class-imbalance",
                "input": "label-health",
                **config.checks.class_imbalance.model_dump(),
            },
            *factor_steps("data"),
            *gap_steps(config, "data"),
            {"name": "representation", "evaluator": "representation", "input": "data", "optional": True},
            {"name": "class-shortfall", "check": "class-shortfall", "input": "representation"},
        ]
        return PresetChain(steps=steps, evaluators=evaluators)
