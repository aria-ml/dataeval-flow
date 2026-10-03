"""The ``data-coverage`` preset: coverage and completeness on the crops, labels and metadata on the source, and the
class worklist (coverage spec §4.2)."""

__all__ = ["DataCoverageWorkflow"]

from typing import Any, ClassVar

from dataeval_flow.evaluators.bias import BalanceConfig, DiversityConfig, MetadataSummaryConfig
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
      ``class-coverage`` and, under ``naive`` coverage, ``uncovered`` (``uncovered-rate``); and ``completeness``
      (optional) with ``completeness-check`` (``completeness-score``), where ``completeness`` is set;
    - ``labels`` (``label-health``) and ``labels-check`` (``class-imbalance``);
    - ``summary`` (``metadata-summary``), ``balance`` and ``diversity`` (optional), and ``gaps`` (``factor-gaps``,
      optional) with ``gaps-check`` (``coverage-gaps``), where ``gaps`` is set;
    - ``worklist`` (``representation`` with no ontology, optional) and ``shortfall`` (``class-shortfall``).

    The embedding steps are skipped with "requires an extractor" when the task names none. It makes no Dataset, so it
    declares no outputs; ontology analysis is ``label-space``'s.
    """

    name: ClassVar[str] = "data-coverage"
    title: ClassVar[str] = "Data Coverage"
    description: ClassVar[str] = (
        "Judges how a Dataset's embeddings cover their space, its class balance and metadata gaps, and what to acquire "
        "per class; detections are cropped first"
    )
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

    @classmethod
    def chain(cls, config: DataCoverageConfig) -> PresetChain:
        """The crops' steps, then the source's."""
        limits = config.health_thresholds
        evaluators: list[Any] = [
            CoverageConfig(name="coverage", **config.coverage.model_dump()),
            CompletenessConfig(name="completeness"),
            LabelHealthConfig(name="labels", metadata=config.metadata),
            MetadataSummaryConfig(name="summary", metadata=config.metadata),
            BalanceConfig(name="balance", metadata=config.metadata),
            DiversityConfig(name="diversity", method=config.diversity, metadata=config.metadata),
            RepresentationConfig(name="worklist", expected=config.expected),
        ]
        steps: list[dict[str, Any]] = [
            {
                "name": "crops",
                "transform": "wrap",
                "input": "data",
                "wrapper": "DetectionCrops",
                "params": config.crops.model_dump(),
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
                    "name": "uncovered",
                    "check": "uncovered-rate",
                    "input": "coverage",
                    **limits.uncovered_rate.model_dump(),
                }
            )
        if config.completeness:
            steps += [
                {"name": "completeness", "evaluator": "completeness", "input": "crops", "optional": True},
                {
                    "name": "completeness-check",
                    "check": "completeness-score",
                    "input": "completeness",
                    **limits.completeness_score.model_dump(),
                },
            ]
        steps += [
            {"name": "labels", "evaluator": "labels", "input": "data"},
            {
                "name": "labels-check",
                "check": "class-imbalance",
                "input": "labels",
                **limits.class_imbalance.model_dump(),
            },
            {"name": "summary", "evaluator": "summary", "input": "data"},
            {"name": "balance", "evaluator": "balance", "input": "data", "optional": True},
            {"name": "diversity", "evaluator": "diversity", "input": "data", "optional": True},
        ]
        if config.gaps is not None:
            steps += [
                {
                    "name": "gaps",
                    "combine": "factor-gaps",
                    "input": "data",
                    "balance": "balance",
                    "optional": True,
                    "metadata": config.metadata,
                    **config.gaps.model_dump(),
                },
                {"name": "gaps-check", "check": "coverage-gaps", "input": "gaps", **limits.coverage_gaps.model_dump()},
            ]
        steps += [
            {"name": "worklist", "evaluator": "worklist", "input": "data", "optional": True},
            {"name": "shortfall", "check": "class-shortfall", "input": "worklist"},
        ]
        return PresetChain(steps=steps, evaluators=evaluators)
