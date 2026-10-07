"""The ``shift`` preset: each test source tested against the reference by each drift and OOD detector, drift by class
where `classwise` names it, and the OOD detectors' agreement and the metadata behind what they flagged."""

__all__ = ["ShiftWorkflow"]

from typing import Any, ClassVar

from dataeval_flow.steps._port import Port
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.steps.checks._drift import evaluator_heading
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import Preset, PresetChain
from dataeval_flow.workflows.shift._config import ShiftConfig, evaluator_entry, is_drift


class ShiftWorkflow(Preset, Workflow[ShiftConfig, ChainResult]):
    """Tests each test source against the reference with each detector, by class where ``classwise`` says, and
    combines the OOD detectors' flags.

    The task's first source is ``reference``; every later one is an element of ``tests``. Per detector, in list order,
    the settings expand to ``<detector>`` (its evaluator, with its own ``extractor``) and ``<detector>-check``
    (``drift`` or ``ood``, by its family). Each drift detector ``classwise`` maps then adds ``<detector>-by-class``
    (its evaluator, unchunked, with the mapped ``by:``) and ``<detector>-by-class-check``. Where the list holds OOD
    detectors:

    - ``ood-union`` combines their flags;
    - with two or more, ``ood-agreement`` judges them;
    - ``factor-predictors`` and ``factor-deviation`` explain them, both optional; ``false`` drops a step.

    A list of drift detectors alone builds what drift-monitoring built, and OOD detectors alone what ood-detection
    built. Every step runs once per test source.
    """

    name: ClassVar[str] = "shift"
    title: ClassVar[str] = "Shift"
    description: ClassVar[str] = (
        "Tests each incoming source against a reference for drift and for out-of-distribution images, with the "
        "metadata behind them."
    )
    slots: ClassVar[tuple[str | InputSlot, ...]] = (
        "reference",
        InputSlot.model_validate({"name": "tests", "list": True}),
    )
    outputs: ClassVar[tuple[Port, ...]] = ()

    @classmethod
    def chain(cls, config: ShiftConfig) -> PresetChain:
        """Each detector and its check, then each classwise run and its check, then the OOD agreement and factors."""
        drift_limits = config.checks.drift.model_dump()
        ood_limits = config.checks.ood.model_dump()
        evaluators: list[Any] = []
        steps: list[dict[str, Any]] = []
        by_class: list[dict[str, Any]] = []
        for detector in config.detectors:
            name = detector.name
            # Each check names its detector, so a finding is titled by it even when its run made nothing to judge.
            subject = evaluator_heading(detector)
            entry = evaluator_entry(detector)
            extractor = getattr(detector, "extractor", None)
            own = {"extractor": extractor} if extractor is not None else {}
            evaluators.append(entry)
            check, limits = ("drift", drift_limits) if is_drift(detector) else ("ood", ood_limits)
            steps += [
                {"name": name, "evaluator": name, "input": ["reference", "tests"], **own},
                {"name": f"{name}-check", "check": check, "input": name, "subject": subject, **limits},
            ]
            by = config.classwise.get(name)
            if by is None:
                continue
            unchunked = name
            if entry.chunking is not None:
                unchunked = f"{name}-unchunked"
                evaluators.append(entry.model_copy(update={"name": unchunked, "chunking": None}))
            by_class += [
                {
                    "name": f"{name}-by-class",
                    "evaluator": unchunked,
                    "input": ["reference", "tests"],
                    "by": by.model_dump(),
                    "optional": True,
                    **own,
                },
                {
                    "name": f"{name}-by-class-check",
                    "check": "drift",
                    "input": f"{name}-by-class",
                    "by": "predicted" if by.predicted is not None else "class",
                    "subject": subject,
                    **drift_limits,
                },
            ]
        return PresetChain(steps=[*steps, *by_class, *_ood_steps(config)], evaluators=evaluators)


def _ood_steps(config: ShiftConfig) -> list[dict[str, Any]]:
    """The OOD detectors' union, their agreement where there are two or more, and the factor steps; none without OOD
    detectors."""
    names = [detector.name for detector in config.detectors if not is_drift(detector)]
    if not names:
        return []
    steps: list[dict[str, Any]] = [{"name": "ood-union", "combine": "ood-union", "input": names}]
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
    return steps
