"""A custom workflow: named inputs, and the steps that read them, as a config file writes them."""

__all__ = ["KIND_KEYS", "CustomWorkflowConfig", "InputSlot", "StepEntry"]

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    StringConstraints,
    field_validator,
    model_serializer,
    model_validator,
)

from dataeval_flow.config._schemas._refs import pool_ref
from dataeval_flow.steps._address import STEP_NAME_PATTERN
from dataeval_flow.steps._by import ByConfig
from dataeval_flow.steps._step import StepConfig, StepKind

if TYPE_CHECKING:
    from dataeval_flow.config._definitions import Definition
    from dataeval_flow.workflows._preset import PresetChain

KIND_KEYS: tuple[StepKind, ...] = ("evaluator", "workflow", "transform", "combine", "check")


class InputSlot(BaseModel):
    """One input of a custom workflow, which a task binds to a source."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)

    name: str = Field(pattern=STEP_NAME_PATTERN, description="What the steps call this input.")
    is_list: bool = Field(
        default=False,
        alias="list",
        description="Whether it binds every source left after the single inputs, as a list keyed by source name.",
    )

    empty: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "For a list input, the reason its checks give when a task binds it no source, which it then may, such as "
            "`no evaluation split given`; a check that judges an empty list judges it instead. Unset, a list input "
            "binds at least one source."
        ),
    )

    @model_validator(mode="after")
    def _empty_on_a_list(self) -> "InputSlot":
        if self.empty is not None and not self.is_list:
            raise ValueError(
                f"Input '{self.name}' binds one source, so it takes no `empty:`: only a list input may bind none."
            )
        return self

    @model_serializer(mode="plain")
    def _as_written(self) -> Any:
        if not self.is_list:
            return self.name
        return {"name": self.name, "list": True, **({"empty": self.empty} if self.empty is not None else {})}


class StepEntry(BaseModel):
    """One step of a custom workflow, as written: what runs, what it reads, and its settings.

    Name exactly one of ``evaluator``, ``workflow``, ``transform``, ``combine`` and ``check``. An evaluator or
    workflow step takes its settings from the pool entry it names. A transform, combine or check step writes its
    settings beside it, and its type's config validates them.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    name: str = Field(pattern=STEP_NAME_PATTERN, description="The step's name, which is the address of its output.")
    evaluator: str | None = Field(
        default=None, description="An `evaluators:` entry to run.", json_schema_extra=pool_ref("evaluators")
    )
    workflow: str | None = Field(
        default=None,
        description="A `workflows:` entry with a `type:` to run.",
        json_schema_extra=pool_ref("workflows"),
    )
    transform: str | None = Field(
        default=None, description="A registered transform to run, with its settings beside it."
    )
    combine: str | None = Field(default=None, description="A registered combine to run, with its settings beside it.")
    check: str | None = Field(default=None, description="A registered check to run, with its thresholds beside it.")
    input: str | list[str] | None = Field(default=None, description="The address, or addresses, the step reads.")
    extractor: str | None = Field(
        default=None,
        description="An `extractors:` entry to embed with, instead of the task's.",
        json_schema_extra=pool_ref("extractors"),
    )
    optional: bool = Field(
        default=False, description="Whether a failure is recorded as a skip that does not fail the task."
    )
    by: ByConfig | None = Field(
        default=None,
        description=(
            "Run this evaluate step or check once per key inside one Output: `class`, `predicted`, or with settings, "
            "`{class: {groups: ..., min_items: ...}}`."
        ),
    )

    pairs: bool = Field(
        default=False,
        description=(
            "Run once per unordered pair of the elements of the one list `input` names, in list order, keyed `a_vs_b`: "
            "for a step that reads two Datasets through one input."
        ),
    )

    _config: StepConfig | None = PrivateAttr(default=None)

    @property
    def kind(self) -> StepKind:
        """Which kind of step this is: the one kind key it names."""
        return next(key for key in KIND_KEYS if getattr(self, key) is not None)

    @property
    def target(self) -> str:
        """What the kind key names: a pool entry or a registered type."""
        return cast(str, getattr(self, self.kind))

    @property
    def settings(self) -> dict[str, Any]:
        """The settings written beside the step, other than its common keys."""
        return dict(self.model_extra or {})

    @property
    def config(self) -> StepConfig | None:
        """An inline step's validated config: a transform's, combine's or check's; ``None`` otherwise."""
        return self._config

    @model_validator(mode="after")
    def _one_kind_and_its_settings(self) -> "StepEntry":
        named = [key for key in KIND_KEYS if getattr(self, key) is not None]
        if len(named) != 1:
            which = "none" if not named else len(named)
            raise ValueError(f"Step '{self.name}' names {which} of {', '.join(KIND_KEYS)}; name exactly one.")
        kind, target = named[0], cast(str, getattr(self, named[0]))
        if self.by is not None and kind not in ("evaluator", "check"):
            raise ValueError(
                f"Step '{self.name}' is a {kind}, which takes no `by:`: only evaluate steps and checks do."
            )
        if self.by is not None and kind == "check" and self.by.keys != type(self.by.keys)():
            raise ValueError(
                f"Step '{self.name}' is a check, whose `by:` takes no settings: its keys come from its input."
            )
        if kind in ("evaluator", "workflow") and self.settings:
            keys = ", ".join(sorted(self.settings))
            raise ValueError(
                f"Step '{self.name}' runs {kind} '{target}', whose settings live in its `{kind}s:` entry; "
                f"move {keys} to it."
            )
        if kind in ("transform", "combine", "check"):
            from dataeval_flow.steps._registry import inline_registry

            impl = inline_registry(kind).get(target)
            data = {**({"input": self.input} if self.input is not None else {}), **self.settings}
            self._config = cast(StepConfig, impl.config_type.model_validate(data))
            if self.by is not None and len(impl.input_ports()) != 1:
                raise ValueError(f"Step '{self.name}': `by:` maps a check over one input, and `{target}` reads more.")
        return self

    @model_serializer(mode="wrap")
    def _as_written(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        written: dict[str, Any] = {"name": self.name, self.kind: self.target}
        if self.input is not None:
            written["input"] = self.input
        if self.extractor is not None:
            written["extractor"] = self.extractor
        if self.optional:
            written["optional"] = True
        if self.pairs:
            written["pairs"] = True
        if self.by is not None:
            written["by"] = data["by"]
        written.update({key: data[key] for key in self.settings if key in data})
        return written


_Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ReportGroupConfig(BaseModel):
    """A heading of a custom workflow's report, the check types whose findings it holds, and the step types shown
    under it as evidence."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    heading: str = Field(min_length=1, description="The heading, such as a question its checks answer.")
    checks: list[str] = Field(
        min_length=1, description="The check types whose findings it holds, as the steps' `check:` names them."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="The step types whose results are shown under it, after its findings, such as `diversity`.",
    )

    @model_serializer(mode="wrap")
    def _as_written(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if not self.evidence:
            data.pop("evidence", None)
        return data


class RecordConfig(BaseModel):
    """A report section recording what the workflow read: a column per bound source, with rows from the run's
    metadata and from the results of the step types it names."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    title: _Text = Field(description="The section's title, such as `What was audited`.")
    steps: list[str] = Field(
        default_factory=list, description="The step types whose results give rows, such as `content-digest`."
    )


class VerdictConfig(BaseModel):
    """The workflow's verdict over its checks: ready, ready with caveats, or not ready."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    blocking: list[str] = Field(
        default_factory=list,
        description="The check types whose unaccepted warning makes the verdict not ready. Empty, none does.",
    )
    accepted: dict[str, _Text] = Field(
        default_factory=dict,
        description=(
            "Why each warning is accepted, by check type, by check step for all its runs, or by `step[element]` for "
            "one run of a step that runs once per element of a list: an accepted warning can't make the verdict not "
            "ready, but it still leaves it ready with caveats."
        ),
    )


class NextStepsConfig(BaseModel):
    """What the report says to do about each warning and each check left unassessed."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    by_check: dict[str, _Text] = Field(default_factory=dict, description="By check type, what to do about its warning.")
    by_reason: dict[str, _Text] = Field(
        default_factory=dict,
        description=(
            "By a fragment of a not-assessed reason, what to do about the checks it left unassessed. Fragments match "
            "in order; a reason none matches is quoted as it stands."
        ),
    )


class PreflightConfig(BaseModel):
    """What the workflow refuses before any step runs."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    nonempty: bool = Field(default=False, description="Refuse an input, or a list input's element, with no items.")
    one_kind: bool = Field(default=False, description="Refuse inputs of different kinds of data.")


class DerivationConfig(BaseModel):
    """Where a custom workflow's steps came from: the preset entry they were decomposed from, as it stood then.

    Historical only. Nothing reads it to run, it names no entry an editor must keep, and the preset may since have
    changed or gone; decomposing again makes a new workflow rather than updating this one.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    type: str = Field(description="The preset's workflow type, such as `audit`.")
    entry: str = Field(description="The preset entry's name when it was decomposed; not a reference to it.")
    settings: dict[str, Any] = Field(
        default_factory=dict, description="The entry's settings, defaults filled in, as it was decomposed."
    )
    steps: list[str] | None = Field(
        default=None,
        description="For steps inlined into a parent workflow, the names they were given; unset, the whole workflow.",
    )
    flow_version: str | None = Field(default=None, description="The Flow version that expanded the preset.")

    @model_serializer(mode="wrap")
    def _as_written(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return {key: value for key, value in handler(self).items() if value is not None}


class CustomWorkflowConfig(BaseModel):
    """A workflow built from steps: its named inputs, and the steps that read them in order.

    A task runs it with ``workflow: <name>`` and binds its ``sources:`` to the inputs in order. The last input may
    be a list (``{name: ..., list: true}``), binding every remaining source keyed by source name. With
    ``empty: <reason>`` it may bind no source, and every check over it then reports that reason as not assessed, unless
    the check judges an empty list. A step names what it reads by address (spec §4): an input, an earlier step, one of
    an earlier step's outputs, or one element of a list.

    Examples
    --------
    >>> from dataeval_flow.steps import CustomWorkflowConfig
    >>> workflow = CustomWorkflowConfig.model_validate({
    ...     "name": "dupes_only",
    ...     "inputs": ["data"],
    ...     "steps": [{"name": "dupes", "evaluator": "dupes", "input": "data"}],
    ... })
    >>> print(workflow.to_yaml())  # doctest: +NORMALIZE_WHITESPACE
    workflows:
    - name: dupes_only
      inputs:
      - data
      steps:
      - name: dupes
        evaluator: dupes
        input: data
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str = Field(description="What tasks call this workflow.")
    description: str | None = Field(default=None, description="One line on what the workflow is for.")
    inputs: list[InputSlot] = Field(min_length=1, description="The inputs a task binds its sources to, in order.")
    steps: list[StepEntry] = Field(min_length=1, description="The steps, in the order they run.")
    groups: list[ReportGroupConfig] = Field(
        default_factory=list,
        description=(
            "The report's headings, in order, each holding the findings of the check types it names, as a preset's "
            "report does. The findings of check types no heading names follow them."
        ),
    )
    ontology: dict[str, Any] | str | None = Field(
        default=None,
        description=(
            "Label space the workflow's labels are read under, recorded in the result envelope's `label_space` as a "
            "workflow type's is. Name an entry under the top-level `ontologies:` key, or give a path to a serialized "
            "RDF artifact resolved against the data root; a nested mapping of concept to children is read as an "
            "inline hierarchy."
        ),
        json_schema_extra=pool_ref("ontologies", or_path=True),
    )
    reference: str | None = Field(
        default=None,
        description=(
            "An input taking one source, whose Dataset every other Dataset's metadata is encoded like. Unset, each "
            "Dataset is encoded on its own."
        ),
    )
    record: RecordConfig | None = Field(default=None, description="A report section recording what the run read.")
    verdict: VerdictConfig | None = Field(
        default=None, description="Give a verdict over the checks. Unset, the workflow gives none of its own."
    )
    next_steps: NextStepsConfig | None = Field(
        default=None, description="What the report says to do about each warning and each check left unassessed."
    )
    preflight: PreflightConfig | None = Field(
        default=None, description="What the workflow refuses before any step runs."
    )
    derived_from: list[DerivationConfig] = Field(
        default_factory=list,
        description=(
            "The preset entries the steps were decomposed from, as they were then: historical, never a reference, and "
            "ignored when the workflow runs."
        ),
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def _slot_shorthand(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [{"name": item} if isinstance(item, str) else item for item in value]
        return value

    @model_validator(mode="after")
    def _names(self) -> "CustomWorkflowConfig":
        slots = [slot.name for slot in self.inputs]
        for name in slots:
            if slots.count(name) > 1:
                raise ValueError(f"Workflow '{self.name}' names input '{name}' more than once.")
        if any(slot.is_list for slot in self.inputs[:-1]):
            raise ValueError(f"Workflow '{self.name}': Only the last input may be a list.")
        seen: set[str] = set()
        for step in self.steps:
            if step.name in slots:
                raise ValueError(f"Workflow '{self.name}': step '{step.name}' shares its name with an input.")
            if step.name in seen:
                raise ValueError(f"Workflow '{self.name}' has more than one step named '{step.name}'.")
            seen.add(step.name)
        return self

    @model_validator(mode="after")
    def _groups_name_checks(self) -> "CustomWorkflowConfig":
        """Refuse a group naming a check type no step runs, listing the ones the steps run."""
        ran = {step.check for step in self.steps if step.check is not None}
        for group in self.groups:
            for check in group.checks:
                if check not in ran:
                    raise ValueError(
                        f"Workflow '{self.name}': group '{group.heading}' names check `{check}`, which no step runs. "
                        f"Its steps' checks: {', '.join(sorted(ran)) or 'none'}."
                    )
        return self

    @model_validator(mode="after")
    def _declarations_name_what_runs(self) -> "CustomWorkflowConfig":
        """Refuse a `reference` naming no input taking one source, a verdict or next step naming a check the steps
        do not run, and next steps by check without a verdict to list the warnings they advise on."""
        singles = [slot.name for slot in self.single_slots]
        if self.reference is not None and self.reference not in singles:
            raise ValueError(
                f"Workflow '{self.name}': `reference` names `{self.reference}`, which is no input taking one source. "
                f"Its inputs taking one source: {', '.join(singles) or 'none'}."
            )
        entries = [step for step in self.steps if step.check is not None]
        checks = {cast(str, step.check) for step in entries}
        check_steps = {step.name for step in entries}
        listed = ", ".join(sorted(checks)) or "none"
        verdict = self.verdict or VerdictConfig()
        for key in verdict.blocking:
            if key not in checks:
                raise ValueError(
                    f"Workflow '{self.name}': `blocking` names `{key}`, which no step runs. Its steps' "
                    f"checks: {listed}."
                )
        for key in verdict.accepted:
            step = key.split("[", 1)[0]
            if key in checks or (step in check_steps and (key == step or key.endswith("]"))):
                continue
            raise ValueError(
                f"Workflow '{self.name}': `accepted` names `{key}`, which is no check type or check step of it. Its "
                f"checks: {listed}. Its check steps: {', '.join(sorted(check_steps)) or 'none'}."
            )
        for key in self.next_steps.by_check if self.next_steps is not None else ():
            if key not in checks:
                raise ValueError(
                    f"Workflow '{self.name}': `next_steps` names check `{key}`, which no step runs. Its "
                    f"steps' checks: {listed}."
                )
        if self.next_steps is not None and self.next_steps.by_check and self.verdict is None:
            raise ValueError(
                f"Workflow '{self.name}': `next_steps.by_check` advises on the warnings a verdict lists, so it needs "
                "`verdict:`; add `verdict: {}` to give one that blocks on nothing."
            )
        return self

    @model_serializer(mode="wrap")
    def _as_written(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        written: dict[str, Any] = {"name": self.name}
        if self.description is not None:
            written["description"] = self.description
        written["inputs"] = data["inputs"]
        written["steps"] = data["steps"]
        if self.groups:
            written["groups"] = data["groups"]
        for key in ("ontology", "reference", "record", "verdict", "next_steps", "preflight"):
            if getattr(self, key) is not None:
                written[key] = data[key]
        if self.derived_from:
            written["derived_from"] = data["derived_from"]
        return written

    @property
    def single_slots(self) -> list[InputSlot]:
        """The inputs that bind one source each."""
        return [slot for slot in self.inputs if not slot.is_list]

    @property
    def list_slot(self) -> InputSlot | None:
        """The last input when it binds every remaining source, else ``None``."""
        return self.inputs[-1] if self.inputs[-1].is_list else None

    def declared_chain(self) -> "PresetChain | None":
        """What this workflow declares beyond its steps, as a preset's chain declares it: its report's headings,
        reference, record, verdict, next steps and preflight. ``None`` where it declares none of them."""
        if not (self.groups or self.reference or self.record or self.verdict or self.next_steps or self.preflight):
            return None
        from dataeval_flow.workflows._preset import NextSteps, Preflight, PresetChain, Record, ReportGroup

        steps, preflight = self.next_steps, self.preflight
        return PresetChain(
            steps=self.steps,
            groups=tuple(ReportGroup(g.heading, tuple(g.checks), tuple(g.evidence)) for g in self.groups),
            reference=self.reference,
            record=Record(self.record.title, tuple(self.record.steps)) if self.record is not None else None,
            blocking=tuple(self.verdict.blocking) if self.verdict is not None else None,
            accepted=dict(self.verdict.accepted) if self.verdict is not None else {},
            next_steps=NextSteps(dict(steps.by_check), dict(steps.by_reason)) if steps is not None else NextSteps(),
            preflight=Preflight(preflight.nonempty, preflight.one_kind) if preflight is not None else Preflight(),
        )

    def binding_problem(self, source_count: int) -> str | None:
        """Why a task naming `source_count` sources cannot run this workflow, or ``None``."""
        singles = [slot.name for slot in self.single_slots]
        wanted = " and ".join(f"one source for '{name}'" for name in singles)
        slot = self.list_slot
        if slot is None:
            fits = source_count == len(singles)
        else:
            least = 0 if slot.empty is not None else 1
            phrase = f"at least one for '{slot.name}'" if least else f"any number for '{slot.name}'"
            wanted = (f"{wanted} and " if wanted else "") + phrase
            fits = source_count >= len(singles) + least
        return None if fits else f"takes {wanted}, but the task names {source_count}."

    def to_yaml(self, definitions: "Sequence[Definition]" = ()) -> str:
        """This workflow as a config fragment: each of `definitions` in its section, then a ``workflows:`` list holding
        this entry, each as written.

        `definitions` takes what :func:`~dataeval_flow.run`'s does: the evaluator entries, views, policies and other
        named entries the steps refer to. With them, the fragment loads and runs on its own.

        Raises
        ------
        TypeError
            When a definition is none of the types `definitions` takes.
        """
        return yaml.safe_dump(self._fragment({}, definitions, caller="to_yaml()"), sort_keys=False)

    def save(self, path: str | Path, definitions: "Sequence[Definition]" = ()) -> Path:
        """Write this workflow, and each of `definitions`, into the config fragment at `path`.

        Each entry replaces the entry of its name in its section, or is added after them. `definitions` takes what
        :func:`~dataeval_flow.run`'s does: the evaluator entries, views, policies and other named entries the steps
        refer to, so the file holds a block that loads and runs on new data on its own. Each is written with its name,
        its type and the settings that differ from their defaults.

        The file's other keys and entries are kept, in place. A file that does not exist is created. A file that holds
        keys but no pipeline section is refused. PyYAML keeps no comments, so a rewritten file loses them.

        Returns
        -------
        Path
            `path`.

        Raises
        ------
        ValueError
            When `path` holds YAML that is not a pipeline config fragment.
        TypeError
            When a definition is none of the types `definitions` takes. Nothing is written.
        """
        from dataeval_flow.config._models import top_level_keys

        path = Path(path)
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
        data = {} if data is None else data
        if not isinstance(data, Mapping) or (data and top_level_keys().isdisjoint(data)):
            raise ValueError(f"{path} is not a pipeline config fragment; refusing to rewrite it.")
        fragment = self._fragment(data, definitions, caller="save()")
        path.write_text(yaml.safe_dump(fragment, sort_keys=False), encoding="utf-8")
        return path

    def _fragment(self, data: Mapping[str, Any], definitions: "Sequence[Definition]", *, caller: str) -> dict[str, Any]:
        """`data` with each of `definitions` in its section, then this workflow in ``workflows:``, each written as a
        config file writes it and replacing the entry of its name."""
        from dataeval_flow.config._definitions import definition_pools, entry_as_written

        fragment = dict(data)
        for section, entries in definition_pools(definitions, caller=caller).items():
            for entry in entries:
                fragment[section] = _replaced(fragment.get(section), entry_as_written(entry))
        fragment["workflows"] = _replaced(fragment.get("workflows"), self.model_dump(mode="json"))
        return fragment


def _replaced(entries: Any, entry: dict[str, Any]) -> list[Any]:
    """`entries` with `entry` in place of the one of its name, or after them."""
    items = list(entries or [])
    names = [item.get("name") if isinstance(item, Mapping) else None for item in items]
    if entry["name"] in names:
        items[names.index(entry["name"])] = entry
    else:
        items.append(entry)
    return items
