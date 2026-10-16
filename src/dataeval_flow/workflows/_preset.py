"""Presets: workflow types whose settings expand to a chain of steps, which Flow runs as a custom workflow's."""

__all__ = [
    "NextSteps",
    "Preflight",
    "Preset",
    "PresetChain",
    "Record",
    "ReportGroup",
    "Structural",
    "StructuralChoice",
    "expand_preset",
    "preset_of",
    "structural_choices",
]

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from dataeval_flow._kind import is_abstract
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._workflow import CustomWorkflowConfig, InputSlot, StepEntry

if TYPE_CHECKING:
    from dataeval_flow._chain._nodes import Node, NodeList
    from dataeval_flow.evaluators._base import EvaluatorConfig


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
class Structural:
    """A setting of a preset's that decides which steps its chain has, not only how they run (preset decomposition
    spec §4). A decomposed copy keeps the steps the setting chose; this tells an editor what it chose and why."""

    setting: str
    """Its path in the entry, dotted, as a config file writes it: `ontology`, `coverage.method`, `factor-gaps`."""
    rule: str
    """What the setting decides, in a sentence: the steps its values add or leave out, any other setting it works
    with, and any step whose type it switches."""
    steps: tuple[str, ...] = ()
    """The chain's steps a value of it adds or leaves out, by name. Empty where the steps follow from a collection,
    such as one pair per detector, and `rule` explains them instead."""
    unset: Any = None
    """What leaving it unset stands for, where that differs from the `null` it reads as: `coverage.method` unset is
    DataEval's `adaptive`."""


@dataclass(frozen=True)
class StructuralChoice:
    """What one structural setting chose for one entry."""

    setting: str
    value: Any
    """The setting's value as the entry serializes it, its default where the entry omits it."""
    written: bool
    """Whether the entry writes the setting, rather than taking its default."""
    effective: Any
    """What the value stands for: `Structural.unset` where the value is `null` and the preset names one."""
    included: tuple[str, ...]
    """The declaration's steps this entry's chain has."""
    excluded: tuple[str, ...]
    """The declaration's steps this entry's chain leaves out."""
    rule: str


@dataclass(frozen=True)
class Preflight:
    """What a chain refuses before any step runs, judged on the Datasets bound to its inputs."""

    nonempty: bool = False
    """Refuse an input, or an element of a list input, that holds no items."""
    one_kind: bool = False
    """Refuse inputs of different kinds, such as classification beside object detection."""


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
    """Why each check type's, or check step's, warning is accepted: an accepted warning neither blocks nor counts as an
    unaccepted warning in the verdict, but an acceptance whose check warned leaves the verdict ready with caveats."""
    next_steps: NextSteps = field(default_factory=NextSteps)
    """What the report says to do about each warning and each check left unassessed."""
    preflight: Preflight = field(default_factory=Preflight)
    """What the chain refuses before any step runs. A refusal no field here declares belongs in
    :meth:`Preset.preflight`, which no custom workflow can carry."""


class Preset:
    """A workflow type whose settings expand to a chain of steps, which Flow runs as it runs a custom workflow's. Every
    workflow type is one.

    Mix it in ahead of the workflow base, with ``ChainResult`` as the result:
    ``class QualityWorkflow(Preset, Workflow[QualityConfig, ChainResult])``. The config class stays the
    type's settings. Declare:

    - ``slots``: what the steps call the task's sources, in the order a task names them. The last may be a list slot,
      ``InputSlot.model_validate({"name": "pools", "list": True})``, which takes every source left, keyed by name.
      With ``empty: <reason>`` it may bind no source, and every check over it then reports that reason as not assessed,
      unless the check judges an empty list;
    - ``outputs``: the Datasets a custom workflow may read when it runs the preset as a step. Each is read where
      :meth:`chain`'s ``outputs`` maps it, or else from the step of the chain named after it, which has one output; it
      is a list where that address is one;
    - ``structure``: the settings that decide which steps :meth:`chain` makes, each a :class:`Structural`; ``()``
      when none does. Left unset, the preset can't be decomposed into a custom workflow;
    - :meth:`chain`: the steps an entry's settings expand to, and the evaluator entries they name.

    As a task, the result's ``type`` is the preset's type id and its steps are the chain's. As a step of a custom
    workflow, named ``cleaning`` say, the chain is spliced in: its steps run as ``cleaning/<step>``, and
    ``cleaning.<output>`` reads each declared output.
    """

    slots: ClassVar[tuple[str | InputSlot, ...]]
    outputs: ClassVar[tuple[Port, ...]] = ()
    structure: ClassVar[tuple[Structural, ...] | None] = None
    """The settings that decide which steps the chain has (preset decomposition spec §4): one :class:`Structural`
    per such setting, or ``()`` where none does. ``None``, the default, means the preset hasn't declared it, and an
    editor can't decompose it into a custom workflow until it does."""

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
        step runs, after the chain's declared :class:`Preflight`. The default refuses nothing. Declare a refusal in
        the chain's ``preflight`` where it can be: a preset overriding this hook cannot be decomposed into a custom
        workflow."""

    @classmethod
    def output_ports(cls) -> tuple[Port, ...]:
        """The Datasets a custom workflow may read when it runs this preset as a step."""
        return cls.outputs


def preset_of(config: object) -> "type[Preset] | None":
    """The preset `config`'s workflow type is; ``None`` for a custom workflow, an evaluator, or an unknown type."""
    from dataeval_flow.workflows._base import WorkflowConfig
    from dataeval_flow.workflows._registry import get_workflow

    if not isinstance(config, WorkflowConfig):
        return None
    try:
        implementation = get_workflow(config.type)
    except ValueError:
        return None
    return implementation if issubclass(implementation, Preset) else None


def structural_choices(config: Any, preset: type[Preset]) -> tuple[StructuralChoice, ...] | None:
    """What each of `preset`'s structural settings chose for entry `config`; ``None`` where `preset` declares no
    structure."""
    if preset.structure is None:
        return None
    names = {_name_of(step) for step in preset.chain(config).steps}
    dumped = config.model_dump(mode="json", by_alias=True)
    choices: list[StructuralChoice] = []
    for declared in preset.structure:
        value, written = _setting(config, dumped, declared.setting)
        effective = declared.unset if value is None and declared.unset is not None else value
        choices.append(
            StructuralChoice(
                setting=declared.setting,
                value=value,
                written=written,
                effective=effective,
                included=tuple(step for step in declared.steps if step in names),
                excluded=tuple(step for step in declared.steps if step not in names),
                rule=declared.rule,
            )
        )
    return tuple(choices)


def _setting(config: Any, dumped: Mapping[str, Any], setting: str) -> tuple[Any, bool]:
    """`setting`'s value in `dumped`, `config` serialized, and whether `config` writes it at every level of its path."""
    model: Any = config
    value: Any = dumped
    written = True
    for segment in setting.split("."):
        value = value.get(segment) if isinstance(value, Mapping) else None
        if not isinstance(model, BaseModel):
            written, model = False, None
            continue
        fields = type(model).model_fields
        name = next((name for name, info in fields.items() if segment in (name, info.alias)), None)
        written = written and name in model.model_fields_set
        model = getattr(model, name) if name is not None else None
    return value, written


def _name_of(step: StepEntry | Mapping[str, Any]) -> str:
    """The name of `step`, written as a mapping or a ``StepEntry``."""
    return step["name"] if isinstance(step, Mapping) else step.name


def expand_preset(config: Any, preset: type[Preset]) -> tuple[CustomWorkflowConfig, tuple["EvaluatorConfig[Any]", ...]]:
    """`config`'s chain, as a custom workflow named after the entry that declares what the chain declares, and the
    evaluator entries its steps name. Report groups and next steps keep only the check types the steps run."""
    chain = preset.chain(config)
    ran = {check for step in chain.steps if (check := _check_of(step)) is not None}
    declared: dict[str, Any] = {}
    groups = [
        {"heading": group.heading, "checks": checks, **({"evidence": list(group.evidence)} if group.evidence else {})}
        for group in chain.groups
        if (checks := [check for check in group.checks if check in ran])
    ]
    if groups:
        declared["groups"] = groups
    if chain.reference is not None:
        declared["reference"] = chain.reference
    if chain.record is not None:
        declared["record"] = {"title": chain.record.title, "steps": list(chain.record.steps)}
    if chain.blocking is not None:
        declared["verdict"] = {"blocking": list(chain.blocking), "accepted": dict(chain.accepted)}
    by_check = {check: text for check, text in chain.next_steps.by_check.items() if check in ran}
    if by_check or chain.next_steps.by_reason:
        declared["next_steps"] = {"by_check": by_check, "by_reason": dict(chain.next_steps.by_reason)}
    if chain.preflight != Preflight():
        declared["preflight"] = {"nonempty": chain.preflight.nonempty, "one_kind": chain.preflight.one_kind}
    if getattr(config, "ontology", None) is not None:
        declared["ontology"] = config.ontology
    workflow = CustomWorkflowConfig.model_validate(
        {"name": config.name, "inputs": list(preset.slots), "steps": list(chain.steps), **declared}
    )
    named = {step.evaluator for step in workflow.steps}
    return workflow, tuple(evaluator for evaluator in chain.evaluators if evaluator.name in named)


def _check_of(step: StepEntry | Mapping[str, Any]) -> str | None:
    """The check type `step` runs, written as a mapping or a ``StepEntry``; ``None`` for any other kind."""
    return step.get("check") if isinstance(step, Mapping) else step.check
