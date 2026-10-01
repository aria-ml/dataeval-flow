"""A custom workflow: named inputs, and the steps that read them, as a config file writes them."""

__all__ = ["KIND_KEYS", "CustomWorkflowConfig", "InputSlot", "StepEntry"]

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from dataeval_flow.steps._address import STEP_NAME_PATTERN
from dataeval_flow.steps._by import ByConfig, ClassKeys
from dataeval_flow.steps._step import StepConfig, StepKind

if TYPE_CHECKING:
    from dataeval_flow.config._definitions import Definition

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

    @model_serializer(mode="plain")
    def _as_written(self) -> Any:
        return {"name": self.name, "list": True} if self.is_list else self.name


class StepEntry(BaseModel):
    """One step of a custom workflow, as written: what runs, what it reads, and its settings.

    Name exactly one of ``evaluator``, ``workflow``, ``transform``, ``combine`` and ``check``. An evaluator or
    workflow step takes its settings from the pool entry it names. A transform, combine or check step writes its
    settings beside it, and its type's config validates them.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    name: str = Field(pattern=STEP_NAME_PATTERN, description="The step's name, which is the address of its output.")
    evaluator: str | None = Field(default=None, description="An `evaluators:` entry to run.")
    workflow: str | None = Field(default=None, description="A `workflows:` entry with a `type:` to run.")
    transform: str | None = Field(
        default=None, description="A registered transform to run, with its settings beside it."
    )
    combine: str | None = Field(default=None, description="A registered combine to run, with its settings beside it.")
    check: str | None = Field(default=None, description="A registered check to run, with its thresholds beside it.")
    input: str | list[str] | None = Field(default=None, description="The address, or addresses, the step reads.")
    extractor: str | None = Field(
        default=None, description="An `extractors:` entry to embed with, instead of the task's."
    )
    optional: bool = Field(
        default=False, description="Whether a failure is recorded as a skip that does not fail the task."
    )
    by: ByConfig | None = Field(
        default=None,
        description=(
            "Run this evaluate step or check once per class, or group of classes, inside one Output: `class`, or "
            "`{class: {groups: ..., min_items: ...}}`."
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
        if self.by is not None and kind == "check" and self.by.class_ != ClassKeys():
            raise ValueError(
                f"Step '{self.name}' is a check, whose `by: class` takes no settings: its keys come from its input."
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
                raise ValueError(
                    f"Step '{self.name}': `by: class` maps a check over one input, and `{target}` reads more."
                )
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
        if self.by is not None:
            written["by"] = data["by"]
        written.update({key: data[key] for key in self.settings if key in data})
        return written


class CustomWorkflowConfig(BaseModel):
    """A workflow built from steps: its named inputs, and the steps that read them in order.

    A task runs it with ``workflow: <name>`` and binds its ``sources:`` to the inputs in order. The last input may
    be a list (``{name: ..., list: true}``), binding every remaining source keyed by source name. A step names what
    it reads by address (spec §4): an input, an earlier step, one of an earlier step's outputs, or one element of a
    list.

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

    @model_serializer(mode="wrap")
    def _as_written(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        written: dict[str, Any] = {"name": self.name}
        if self.description is not None:
            written["description"] = self.description
        written["inputs"] = data["inputs"]
        written["steps"] = data["steps"]
        return written

    @property
    def single_slots(self) -> list[InputSlot]:
        """The inputs that bind one source each."""
        return [slot for slot in self.inputs if not slot.is_list]

    @property
    def list_slot(self) -> InputSlot | None:
        """The last input when it binds every remaining source, else ``None``."""
        return self.inputs[-1] if self.inputs[-1].is_list else None

    def binding_problem(self, source_count: int) -> str | None:
        """Why a task naming `source_count` sources cannot run this workflow, or ``None``."""
        singles = [slot.name for slot in self.single_slots]
        wanted = " and ".join(f"one source for '{name}'" for name in singles)
        if self.list_slot is not None:
            wanted = (f"{wanted} and " if wanted else "") + f"at least one for '{self.list_slot.name}'"
            fits = source_count >= len(singles) + 1
        else:
            fits = source_count == len(singles)
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
