"""The ``prioritization`` preset: each pool ranked against a reference, and the top of each ranking kept
(spec §10.9)."""

__all__ = ["PrioritizationWorkflow"]

from typing import Any, ClassVar

from dataeval_flow.evaluators.scope import PrioritizationConfig
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import Preset, PresetChain
from dataeval_flow.workflows.prioritization._config import PrioritizationWorkflowConfig


class PrioritizationWorkflow(Preset, Workflow[PrioritizationWorkflowConfig, ChainResult]):
    """Ranks each pool against the reference, and keeps the top of each ranking.

    The task's first source is ``reference``; every later one is an element of ``pools``. The settings expand to:

    - ``prioritization``: each pool ranked against the reference;
    - ``selected`` (``select``): the first ``n``, or ``fraction``, of each pool's ranking; all of it when neither is
      set.

    Each step runs once per pool. Run as a step of a custom workflow, ``<step>.selected`` reads the selection, a list
    keyed by pool; a ``quality`` step ahead of it, over the reference and over the pools, ranks clean data.
    """

    name: ClassVar[str] = "prioritization"
    title: ClassVar[str] = "Prioritization"
    description: ClassVar[str] = "Ranks each pool against a reference for labeling, and keeps the top."
    slots: ClassVar[tuple[str | InputSlot, ...]] = (
        "reference",
        InputSlot.model_validate({"name": "pools", "list": True}),
    )
    outputs: ClassVar[tuple[Port, ...]] = (Port("selected", DataType.DATASET),)

    @classmethod
    def chain(cls, config: PrioritizationWorkflowConfig) -> PresetChain:
        """The ranking, and the selection."""
        evaluators = [PrioritizationConfig(name="prioritization", **config.prioritization.model_dump())]
        amount: dict[str, Any] = (
            {"n": config.select.n} if config.select.n is not None else {"fraction": config.select.fraction or 1.0}
        )
        steps: list[dict[str, Any]] = [
            {"name": "prioritization", "evaluator": "prioritization", "input": ["pools", "reference"]},
            {"name": "selected", "transform": "select", "input": "pools", "ranking": "prioritization", **amount},
        ]
        return PresetChain(steps=steps, evaluators=evaluators)
