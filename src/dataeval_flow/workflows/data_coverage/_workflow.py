"""The ``data-coverage`` preset: coverage and completeness on the crops, labels and metadata on the source, and the
class worklist (coverage spec §4.2)."""

__all__ = ["DataCoverageWorkflow"]

from typing import Any, ClassVar

from dataeval_flow.evaluators.bias import BalanceConfig, DiversityConfig, FactorSummaryConfig
from dataeval_flow.evaluators.quality import LabelHealthConfig
from dataeval_flow.evaluators.scope import CompletenessConfig, CoverageConfig, RepresentationConfig
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import Preset, PresetChain
from dataeval_flow.workflows.data_coverage._config import DataCoverageConfig


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
        limits = config.checks
        evaluators: list[Any] = [
            CoverageConfig(name="coverage", **config.coverage.model_dump()),
            CompletenessConfig(name="completeness"),
            LabelHealthConfig(name="label-health", metadata=config.metadata),
            FactorSummaryConfig(name="factor-summary", metadata=config.metadata),
            BalanceConfig(name="balance", metadata=config.metadata),
            DiversityConfig(name="diversity", method=config.diversity.method, metadata=config.metadata),
            RepresentationConfig(name="representation", expected=config.representation.expected),
        ]
        steps: list[dict[str, Any]] = [
            {
                "name": "crops",
                "transform": "wrap",
                "input": "data",
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
        if config.completeness:
            steps += [
                {"name": "completeness", "evaluator": "completeness", "input": "crops", "optional": True},
                {
                    "name": "dimensional-completeness",
                    "check": "dimensional-completeness",
                    "input": "completeness",
                    **limits.dimensional_completeness.model_dump(),
                },
            ]
        steps += [
            {"name": "label-health", "evaluator": "label-health", "input": "data"},
            {
                "name": "class-imbalance",
                "check": "class-imbalance",
                "input": "label-health",
                **limits.class_imbalance.model_dump(),
            },
            {"name": "factor-summary", "evaluator": "factor-summary", "input": "data"},
            {"name": "balance", "evaluator": "balance", "input": "data", "optional": True},
            {"name": "diversity", "evaluator": "diversity", "input": "data", "optional": True},
        ]
        if config.factor_gaps is not False:
            steps += [
                {
                    "name": "factor-gaps",
                    "combine": "factor-gaps",
                    "input": "data",
                    "balance": "balance",
                    "optional": True,
                    "metadata": config.metadata,
                    **config.factor_gaps.model_dump(),
                },
                {
                    "name": "factor-coverage-gaps",
                    "check": "factor-coverage-gaps",
                    "input": "factor-gaps",
                    **limits.factor_coverage_gaps.model_dump(),
                },
            ]
        steps += [
            {"name": "representation", "evaluator": "representation", "input": "data", "optional": True},
            {"name": "class-shortfall", "check": "class-shortfall", "input": "representation"},
        ]
        return PresetChain(steps=steps, evaluators=evaluators)
