"""The ``ood-detection`` preset: each test source scored against the reference by each detector, the detectors'
agreement, and the metadata behind what they flagged (ood-detection spec §4)."""

__all__ = ["OODDetectionWorkflow"]

from typing import Any, ClassVar

from dataeval_flow.steps._port import Port
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.steps.checks._drift import evaluator_heading
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import Preset, PresetChain
from dataeval_flow.workflows.ood_detection._config import OODDetectionConfig, evaluator_entry


class OODDetectionWorkflow(Preset, Workflow[OODDetectionConfig, ChainResult]):
    """Scores each test source against the reference with each detector, and combines their flags.

    The task's first source is ``reference``; every later one is an element of ``tests``. Per detector, the settings
    expand to ``<detector>`` (its evaluator, with its own ``extractor``) and ``<detector>-check`` (``ood``). Then:

    - ``ood-union`` combines every detector's flags;
    - with two detectors or more, ``ood-agreement`` judges them;
    - ``factor-predictors`` and ``factor-deviation`` explain them, both optional; ``false`` drops a step.

    Every step runs once per test source.
    """

    name: ClassVar[str] = "ood-detection"
    title: ClassVar[str] = "OOD Detection"
    description: ClassVar[str] = (
        "Flags each test source's images unlike the reference, by each detector and by their agreement, with the "
        "metadata behind them."
    )
    slots: ClassVar[tuple[str | InputSlot, ...]] = (
        "reference",
        InputSlot.model_validate({"name": "tests", "list": True}),
    )
    outputs: ClassVar[tuple[Port, ...]] = ()

    @classmethod
    def chain(cls, config: OODDetectionConfig) -> PresetChain:
        """Each detector and its check, then the agreement and its check, then the factor steps."""
        ood = config.checks.ood.model_dump()
        evaluators: list[Any] = []
        steps: list[dict[str, Any]] = []
        for detector in config.detectors:
            name = detector.name
            extractor = getattr(detector, "extractor", None)
            own = {"extractor": extractor} if extractor is not None else {}
            evaluators.append(evaluator_entry(detector))
            steps += [
                {"name": name, "evaluator": name, "input": ["reference", "tests"], **own},
                # Each check names its detector, so a finding is titled by it even when its run made nothing to judge.
                {"name": f"{name}-check", "check": "ood", "input": name, "subject": evaluator_heading(detector), **ood},
            ]
        names = [detector.name for detector in config.detectors]
        steps.append({"name": "ood-union", "combine": "ood-union", "input": names})
        if len(names) > 1:
            agreement = config.checks.ood_agreement.model_dump()
            steps.append({"name": "ood-agreement", "check": "ood-agreement", "input": "ood-union", **agreement})
        policies = {key: value for key, value in (("metadata", config.metadata), ("stats", config.stats)) if value}
        factors = {"ood": "ood-union", "reference": "reference", "input": "tests", "optional": True, **policies}
        if config.factor_predictors is not False:
            steps.append({"name": "factor-predictors", "combine": "factor-predictors", **factors})
        if config.factor_deviation is not False:
            steps.append(
                {
                    "name": "factor-deviation",
                    "combine": "factor-deviation",
                    **factors,
                    "max_items": config.factor_deviation.max_items,
                }
            )
        return PresetChain(steps=steps, evaluators=evaluators)
