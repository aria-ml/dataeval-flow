"""Presets: workflow types whose settings expand to a chain of steps, which Flow runs as a custom workflow's."""

__all__ = ["NextSteps", "Preset", "PresetChain", "Record", "ReportGroup", "expand_preset", "preset_of"]

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from dataeval_flow._kind import is_abstract
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._workflow import CustomWorkflowConfig, InputSlot, StepEntry

if TYPE_CHECKING:
    from dataeval_flow._chain._nodes import Node, NodeList
    from dataeval_flow.evaluators._base import EvaluatorConfig
    from dataeval_flow.steps._result import ChainResult
    from dataeval_flow.workflows._context import WorkflowContext


@dataclass(frozen=True)
class ReportGroup:
    """A heading of a chain's report, with the check types whose findings it holds and the evaluate step types shown
    under it as evidence."""

    heading: str
    checks: tuple[str, ...]
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class Record:
    """A section that records what a chain read: one column per bound source, with rows from the run's metadata and
    from the results of the step types named in `steps`."""

    title: str
    steps: tuple[str, ...] = ()


@dataclass(frozen=True)
class NextSteps:
    """What to do about each warning, by check type, and about each check left unassessed, by a fragment of its
    reason. Reasons match in order; a reason no fragment matches is quoted as it stands."""

    by_check: Mapping[str, str] = field(default_factory=dict)
    by_reason: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PresetChain:
    """What one preset entry's settings expand to.

    Attributes
    ----------
    steps
        The steps, in the order they run: each a ``StepEntry``, or the mapping a config file would write for one.
    evaluators
        The evaluator entries the steps name. They are found before the pipeline's own ``evaluators:``, so a
        preset's steps run with its settings whatever the pipeline defines.
    """

    steps: Sequence[StepEntry | Mapping[str, Any]]
    evaluators: Sequence["EvaluatorConfig[Any]"] = ()
    outputs: Mapping[str, str] = field(default_factory=dict)
    """Where each declared output is read in the chain, by output name: an address such as `split.train`. An output
    the map leaves out is the step of its own name, with one output."""
    reference: str | None = None
    """The slot whose Dataset every other Dataset's metadata is encoded like (audit spec §9.3); a metadata policy's
    `reference_split` names a source in its place. ``None`` encodes each Dataset on its own draw."""
    groups: Sequence[ReportGroup] = ()
    """The report's headings, in order, each with the check types it holds (audit spec §9.4)."""
    record: Record | None = None
    """The report's section recording what the chain read; ``None`` draws none."""
    blocking: Sequence[str] | None = None
    """The check types whose unaccepted warning makes the chain not ready (audit spec §6); ``None`` gives no verdict."""
    accepted: Mapping[str, str] = field(default_factory=dict)
    """Why each check type's warning is accepted, by check type: an accepted warning neither blocks nor counts as an
    unaccepted warning in the verdict."""
    next_steps: NextSteps = field(default_factory=NextSteps)
    """What the report says to do about each warning and each check left unassessed."""


class Preset:
    """A workflow type whose settings expand to a chain of steps, which Flow runs as it runs a custom workflow's.

    Mix it in ahead of the workflow base, with ``ChainResult`` as the result:
    ``class DataCleaningWorkflow(Preset, Workflow[DataCleaningConfig, ChainResult])``. The config class stays the
    type's settings. Declare:

    - ``slots``: what the steps call the task's sources, in the order a task names them. The last may be a list slot,
      ``InputSlot.model_validate({"name": "pools", "list": True})``, which takes every source left, keyed by name.
      With ``empty: <reason>`` it may bind no source, and every check over it then reports that reason as not assessed,
      unless the check judges an empty list;
    - ``outputs``: the Datasets a custom workflow may read when it runs the preset as a step. Each is read where
      :meth:`chain`'s ``outputs`` maps it, or else from the step of the chain named after it, which has one output; it
      is a list where that address is one;
    - :meth:`chain`: the steps an entry's settings expand to, and the evaluator entries they name.

    As a task, the result's ``type`` is the preset's type id and its steps are the chain's. As a step of a custom
    workflow, named ``cleaning`` say, the chain is spliced in: its steps run as ``cleaning/<step>``, and
    ``cleaning.<output>`` reads each declared output.
    """

    slots: ClassVar[tuple[str | InputSlot, ...]]
    outputs: ClassVar[tuple[Port, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Require ``slots`` on a concrete preset, and outputs that are single Datasets."""
        super().__init_subclass__(**kwargs)
        if is_abstract(cls):
            return
        if not getattr(cls, "slots", ()):
            raise TypeError(f"{cls.__name__} is a preset, so it must declare `slots`: what its steps call its inputs.")
        if any(isinstance(slot, InputSlot) and slot.is_list for slot in cls.slots[:-1]):
            raise TypeError(
                f"{cls.__name__} declares a list slot before its last: only a preset's last slot may take a list of "
                "sources."
            )
        wrong = [port.name for port in cls.outputs if port.type is not DataType.DATASET or port.is_list]
        if wrong:
            raise TypeError(
                f"{cls.__name__} declares outputs {', '.join(wrong)}, which are not Datasets: a preset's outputs are "
                "the Datasets its steps make."
            )

    @classmethod
    def slot_names(cls) -> tuple[str, ...]:
        """What the steps call each input, in the order a task names them."""
        return tuple(slot if isinstance(slot, str) else slot.name for slot in cls.slots)

    @classmethod
    @abstractmethod
    def chain(cls, config: Any) -> PresetChain:
        """The steps `config`'s settings expand to, and the evaluator entries they name."""

    @classmethod
    def preflight(cls, config: Any, inputs: "Mapping[str, Node | NodeList]") -> None:
        """Refuse `config`'s run over `inputs`, the Datasets bound to its slots, by raising ``GraphError``, before any
        step runs. The default refuses nothing."""

    @classmethod
    def output_ports(cls) -> tuple[Port, ...]:
        """The Datasets a custom workflow may read when it runs this preset as a step."""
        return cls.outputs

    def run(self, config: Any, context: "WorkflowContext") -> "ChainResult":  # noqa: ARG002
        """Refused: Flow runs a preset's chain of steps, through ``run_task`` or ``run``."""
        raise TypeError(
            f"{type(self).__name__} is a preset: Flow runs its chain of steps, through `run_task` or `run`, not "
            "through `Workflow.run`."
        )


def preset_of(config: object) -> "type[Preset] | None":
    """The preset `config`'s workflow type is; ``None`` for a custom workflow, an evaluator, or a type with a ``run``
    of its own."""
    from dataeval_flow.workflows._base import WorkflowConfig
    from dataeval_flow.workflows._registry import get_workflow

    if not isinstance(config, WorkflowConfig):
        return None
    try:
        implementation = get_workflow(config.type)
    except ValueError:
        return None
    return implementation if issubclass(implementation, Preset) else None


def expand_preset(config: Any, preset: type[Preset]) -> tuple[CustomWorkflowConfig, tuple["EvaluatorConfig[Any]", ...]]:
    """`config`'s chain, as a custom workflow named after the entry, and the evaluator entries its steps name."""
    chain = preset.chain(config)
    workflow = CustomWorkflowConfig.model_validate(
        {"name": config.name, "inputs": list(preset.slots), "steps": list(chain.steps)}
    )
    return workflow, tuple(chain.evaluators)
