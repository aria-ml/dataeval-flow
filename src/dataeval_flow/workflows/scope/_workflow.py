"""The ``scope`` preset: coverage and completeness on the crops, and the class worklist on the source
(coverage spec §4.2)."""

__all__ = ["ScopeWorkflow", "coverage_evaluators", "embedding_steps"]

from typing import TYPE_CHECKING, Any, ClassVar

from dataeval_flow.evaluators.scope import CompletenessConfig, CoverageConfig, RepresentationConfig
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import Preset, PresetChain
from dataeval_flow.workflows.scope._config import ScopeConfig

if TYPE_CHECKING:
    from dataeval_flow.workflows.audit._config import AuditConfig

    Covered = ScopeConfig | AuditConfig
    """The entries whose settings hold the coverage steps' blocks, and their checks'."""


def coverage_evaluators(config: "Covered") -> list[Any]:
    """The entries the embedding steps name: `coverage` and `completeness`."""
    return [CoverageConfig(name="coverage", **config.coverage.model_dump()), CompletenessConfig(name="completeness")]


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


class ScopeWorkflow(Preset, Workflow[ScopeConfig, ChainResult]):
    """Judges the task's one source, ``data``: how its embeddings cover their space, and what to acquire per class.

    The settings expand to, in legacy's finding order:

    - ``crops`` (``wrap``, ``DetectionCrops``, passing other kinds through), then ``coverage`` (optional) with
      ``class-coverage`` and, under ``naive`` coverage, ``uncovered-items``; and ``completeness``
      (optional) with ``dimensional-completeness``, where ``completeness`` is set;
    - ``representation`` (with no ontology, optional) and ``class-shortfall``.

    The embedding steps are skipped with "requires an extractor" when the task names none. It makes no Dataset, so it
    declares no outputs; ontology analysis is ``taxonomy``'s, and class balance and the metadata factors are
    ``bias``'s.
    """

    name: ClassVar[str] = "scope"
    title: ClassVar[str] = "Scope"
    description: ClassVar[str] = (
        "Judges how a Dataset's embeddings cover their space, and what to acquire per class; detections are cropped "
        "first."
    )
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

    @classmethod
    def chain(cls, config: ScopeConfig) -> PresetChain:
        """The crops' steps, then the source's."""
        evaluators = [
            *coverage_evaluators(config),
            RepresentationConfig(name="representation", expected=config.representation.expected),
        ]
        steps = [
            *embedding_steps(config, "data", completeness=config.completeness),
            {"name": "representation", "evaluator": "representation", "input": "data", "optional": True},
            {"name": "class-shortfall", "check": "class-shortfall", "input": "representation"},
        ]
        return PresetChain(steps=steps, evaluators=evaluators)
