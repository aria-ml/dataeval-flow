"""The ``bias`` preset: how the class labels and the metadata factors relate in one source."""

__all__ = ["BiasWorkflow", "bias_evaluators", "factor_steps", "gap_steps"]

from typing import TYPE_CHECKING, Any, ClassVar

from dataeval_flow.evaluators.bias import BalanceConfig, DiversityConfig, FactorSummaryConfig, ParityConfig
from dataeval_flow.evaluators.quality import LabelHealthConfig
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import Preset, PresetChain
from dataeval_flow.workflows.bias._config import BiasConfig

if TYPE_CHECKING:
    from dataeval_flow.workflows.audit._config import AuditConfig

    Biased = BiasConfig | AuditConfig
    """The entries whose settings hold the factor steps' blocks, and their checks'."""


def bias_evaluators(config: "Biased") -> list[Any]:
    """The entries the factor steps name: `factor-summary`, `balance` and `diversity`."""
    return [
        FactorSummaryConfig(name="factor-summary", metadata=config.metadata),
        BalanceConfig(name="balance", metadata=config.metadata),
        DiversityConfig(name="diversity", method=config.diversity.method, metadata=config.metadata),
    ]


def factor_steps(source: str) -> list[dict[str, Any]]:
    """`factor-summary`, `balance` (optional) and `diversity` (optional) on `source`."""
    return [
        {"name": "factor-summary", "evaluator": "factor-summary", "input": source},
        {"name": "balance", "evaluator": "balance", "input": source, "optional": True},
        {"name": "diversity", "evaluator": "diversity", "input": source, "optional": True},
    ]


def gap_steps(config: "Biased", source: str) -> list[dict[str, Any]]:
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


class BiasWorkflow(Preset, Workflow[BiasConfig, ChainResult]):
    """Judges the task's one source, ``data``: its class balance, and how its metadata factors relate to the class.

    The settings expand to:

    - ``label-health`` and ``class-imbalance``;
    - ``factor-summary``, ``balance`` (optional) with ``shortcut-risk``, ``parity`` (optional) with ``factor-parity``,
      and ``diversity`` (optional);
    - ``factor-gaps`` (optional) with ``factor-coverage-gaps``, unless ``factor-gaps`` is false.

    It reads labels and metadata only, so it needs no extractor. It makes no Dataset, so it declares no outputs.
    """

    name: ClassVar[str] = "bias"
    title: ClassVar[str] = "Bias"
    description: ClassVar[str] = (
        "Judges a Dataset's class balance and how its metadata factors relate to the class: shortcuts, association and "
        "under-represented combinations."
    )
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

    @classmethod
    def chain(cls, config: BiasConfig) -> PresetChain:
        """The labels' steps, then the factors'."""
        c = config.checks
        evaluators = [
            LabelHealthConfig(name="label-health", metadata=config.metadata),
            *bias_evaluators(config),
            ParityConfig(name="parity", metadata=config.metadata),
        ]
        steps = [
            {"name": "label-health", "evaluator": "label-health", "input": "data"},
            {
                "name": "class-imbalance",
                "check": "class-imbalance",
                "input": "label-health",
                **c.class_imbalance.model_dump(),
            },
            *factor_steps("data"),
            {"name": "shortcut-risk", "check": "shortcut-risk", "input": "balance", **c.shortcut_risk.model_dump()},
            {"name": "parity", "evaluator": "parity", "input": "data", "optional": True},
            {"name": "factor-parity", "check": "factor-parity", "input": "parity", **c.factor_parity.model_dump()},
            *gap_steps(config, "data"),
        ]
        return PresetChain(steps=steps, evaluators=evaluators)
