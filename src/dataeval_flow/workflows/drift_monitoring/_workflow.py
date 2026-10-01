"""The ``drift-monitoring`` preset: each test source tested against the reference by each detector, and by class
where `classwise` names it (spec §10.11)."""

__all__ = ["DriftMonitoringWorkflow"]

from typing import Any, ClassVar

from dataeval_flow.steps._port import Port
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.steps.checks._drift import evaluator_heading
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import Preset, PresetChain
from dataeval_flow.workflows.drift_monitoring._config import DriftMonitoringConfig, evaluator_entry


class DriftMonitoringWorkflow(Preset, Workflow[DriftMonitoringConfig, ChainResult]):
    """Tests each test source against the reference with each detector, and by class where ``classwise`` says.

    The task's first source is ``reference``; every later one is an element of ``tests``. Per detector, the settings
    expand to ``<detector>`` (its evaluator) and ``<detector>-check`` (``drift``). Each detector ``classwise`` maps
    then adds ``<detector>-classes`` (its evaluator, unchunked, with the mapped ``by:``) and
    ``<detector>-classes-check`` (``drift``, with the bare kind, ``class`` or ``predicted``). A detector's own
    ``extractor`` goes on its evaluate steps. Every step runs once per test source.
    """

    name: ClassVar[str] = "drift-monitoring"
    title: ClassVar[str] = "Drift Monitoring"
    description: ClassVar[str] = "Tests each incoming source for drift from a reference, whole, by chunk and by class"
    slots: ClassVar[tuple[str | InputSlot, ...]] = (
        "reference",
        InputSlot.model_validate({"name": "tests", "list": True}),
    )
    outputs: ClassVar[tuple[Port, ...]] = ()

    @classmethod
    def chain(cls, config: DriftMonitoringConfig) -> PresetChain:
        """Each detector and its check, then each classwise run and its check, in detector order."""
        limits = config.health_thresholds.drift.model_dump()
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
            steps += [
                {"name": name, "evaluator": name, "input": ["reference", "tests"], **own},
                {"name": f"{name}-check", "check": "drift", "input": name, "subject": subject, **limits},
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
                    "name": f"{name}-classes",
                    "evaluator": unchunked,
                    "input": ["reference", "tests"],
                    "by": by.model_dump(),
                    "optional": True,
                    **own,
                },
                {
                    "name": f"{name}-classes-check",
                    "check": "drift",
                    "input": f"{name}-classes",
                    "by": "predicted" if by.predicted is not None else "class",
                    "subject": subject,
                    **limits,
                },
            ]
        return PresetChain(steps=[*steps, *by_class], evaluators=evaluators)
