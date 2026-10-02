"""What a matrix key names: its canonical target, what it may not vary, the keys and list indices it walks through an
entry, and what a run reads (task-matrix spec §3.4)."""

__all__ = ["POOLS", "Target", "as_written", "at", "canonical", "reads", "set_override", "walk"]

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.config._schemas._task import TaskConfig

POOLS = ("evaluators", "workflows", "extractors")
# The keys that say which entry an entry is: varying one would change what every other key names.
_IDENTITY = {"evaluators": ("name", "type"), "workflows": ("name", "type", "inputs"), "extractors": ("name", "model")}
# A custom workflow step's kind and wiring, which every run of a matrix shares.
_WIRING = ("name", "evaluator", "workflow", "transform", "combine", "check", "input", "by", "optional", "extractor")
_ITEM_IDS = ("name", "type", "model")


@dataclass(frozen=True)
class Target:
    """The setting a key names, in one spelling: a task field (``pool`` ``"task"``), or a path in a pool entry."""

    pool: str
    entry: str | None
    path: tuple[str, ...]

    def __str__(self) -> str:
        return ".".join([self.pool, *([self.entry] if self.entry else []), *self.path])

    def covers(self, other: "Target") -> bool:
        """Whether `other` names this setting or one inside it, compared segment by segment."""
        return (self.pool, self.entry) == (other.pool, other.entry) and other.path[: len(self.path)] == self.path


def _entry(pipeline: "PipelineConfig", pool: str, name: str | None) -> Any:
    return next((entry for entry in getattr(pipeline, pool) or () if entry.name == name), None)


def _custom_workflow(task: "TaskConfig", pipeline: "PipelineConfig") -> Any:
    from dataeval_flow.steps._workflow import CustomWorkflowConfig

    entry = _entry(pipeline, "workflows", task.workflow) if task.kind == "workflow" else None
    return entry if isinstance(entry, CustomWorkflowConfig) else None


def canonical(key: str, task: "TaskConfig", pipeline: "PipelineConfig") -> Target:
    """The one target `key` names, whichever spelling it uses; raises ``ValueError`` where it names nothing, or a
    setting a matrix may not vary."""
    segments = tuple(key.split("."))
    if segments in (("sources",), ("extractor",)):
        return Target("task", None, segments)
    head = segments[0]
    if head in POOLS:
        if len(segments) < 3:
            raise ValueError(f"`{key}` names no setting: write `{head}.<name>.<setting>`")
        if _entry(pipeline, head, segments[1]) is None:
            raise ValueError(f"`{key}`: `{head}:` has no entry named `{segments[1]}`")
        target = Target(head, segments[1], segments[2:])
    elif head == "steps":
        if _custom_workflow(task, pipeline) is None:
            raise ValueError(f"`{key}`: task '{task.name}' runs no custom workflow, so it has no steps to vary")
        target = Target("workflows", task.workflow, segments)
    else:
        target = Target("evaluators" if task.kind == "evaluator" else "workflows", task.workflow, segments)
    _refuse_structural(key, target, pipeline)
    return target


def _refuse_structural(key: str, target: Target, pipeline: "PipelineConfig") -> None:
    from dataeval_flow.config.extractors._base import _InstanceExtractorConfig
    from dataeval_flow.steps._workflow import CustomWorkflowConfig

    entry = _entry(pipeline, target.pool, target.entry)
    head = target.path[0]
    if isinstance(entry, CustomWorkflowConfig) and head == "steps":
        if len(target.path) < 3:
            raise ValueError(f"`{key}` names a step, not a setting: write `steps.<step>.<setting>`")
        step = next((step for step in entry.steps if step.name == target.path[1]), None)
        if step is None:
            raise ValueError(f"`{key}`: workflow '{entry.name}' has no step named `{target.path[1]}`")
        if step.kind in ("evaluator", "workflow"):
            rest = ".".join(target.path[2:])
            raise ValueError(
                f"`{key}`: step '{step.name}' runs the {step.kind} entry `{step.target}`, whose settings live there: "
                f"vary `{step.kind}s.{step.target}.{rest}`"
            )
        if target.path[2] in _WIRING:
            raise ValueError(f"`{key}` varies a step's `{target.path[2]}`, which wires the chain every run shares")
        if step.kind == "transform" and step.target == "export" and target.path[2] == "to":
            raise ValueError(
                f"`{key}` varies where export step '{step.name}' writes, which a matrix may not: each run already "
                "writes under `datasets/<to>/run-<n>/`"
            )
        return
    if head in _IDENTITY[target.pool]:
        raise ValueError(
            f"`{key}` varies `{head}`, which says which entry this is: a matrix varies settings, not identities"
        )
    if isinstance(entry, _InstanceExtractorConfig):
        raise ValueError(f"`{key}`: extractor '{entry.name}' was given as an object, so a matrix can't vary it")


def _field(model: BaseModel, segment: str) -> tuple[str | None, str] | None:
    """The attribute a segment names on `model` and the key a dump writes it under; ``None`` where it names none."""
    for name, info in type(model).model_fields.items():
        if segment in (name, info.alias):
            return name, info.serialization_alias or info.alias or name
    if segment in (model.model_extra or {}):
        return None, segment
    return None


def _owner(model: BaseModel, segment: str) -> BaseModel:
    """Where `segment` lives on `model`: an inline step's validated config, for a setting not written on the step and
    so left at its default; `model` itself otherwise."""
    from dataeval_flow.steps._workflow import StepEntry

    if isinstance(model, StepEntry) and model.config is not None and _field(model, segment) is None:
        return model.config
    return model


def _name(item: Any) -> Any:
    return item.get("name") if isinstance(item, Mapping) else getattr(item, "name", None)


def _child(value: Any, key: str | int) -> Any:
    if isinstance(value, BaseModel):
        value = _owner(value, str(key))
        found = _field(value, str(key))
        if found is None:
            return None
        attribute, _ = found
        return getattr(value, attribute) if attribute else (value.model_extra or {})[str(key)]
    if isinstance(value, Mapping):
        return value.get(key)
    return value[key]


def walk(root: BaseModel, path: Sequence[str], where: str) -> tuple[tuple[str | int, ...], Any]:
    """The dump keys and list indices `path` walks through `root`, and the value at its end.

    Fields are matched by name or alias, an inline step's settings on its config where the step does not write them,
    mapping keys as they are (an absent one is added), and list items by their ``name``. Raises ``ValueError`` naming
    the segment that names nothing, a path through an unset value, and a list item's ``name``.
    """
    keys: list[str | int] = []
    node: Any = root
    for depth, segment in enumerate(path):
        shown, parent = ".".join(path[: depth + 1]), ".".join(path[:depth])
        if isinstance(node, BaseModel):
            node = _owner(node, segment)
            found = _field(node, segment)
            if found is None:
                raise ValueError(f"{where} has no setting `{shown}`")
            keys.append(found[1])
            node = _child(node, segment)
        elif isinstance(node, Mapping):
            keys.append(segment)
            node = node.get(segment)
        elif isinstance(node, Sequence) and not isinstance(node, str | bytes):
            names = [_name(item) for item in node]
            if segment not in names:
                listed = ", ".join(f"`{name}`" for name in names if name is not None) or "no named items"
                raise ValueError(f"`{parent}` in {where} has no item named `{segment}` (it has {listed})")
            if depth + 1 < len(path) and path[depth + 1] == "name":
                raise ValueError(
                    f"`{shown}.name` renames an item, which says which item it is: a matrix varies settings, not "
                    "identities"
                )
            keys.append(names.index(segment))
            node = node[names.index(segment)]
        else:
            raise ValueError(f"`{parent}` in {where} is unset, so `{shown}` has nothing to set; vary `{parent}` whole")
    return tuple(keys), node


def as_written(value: Any) -> Any:
    """`value` as a config file writes it: models dumped by alias with only the fields set, plus the name, type and
    model that say which entry each is, so a dump validates back to the same entry."""
    if isinstance(value, BaseModel):
        dump = value.model_dump(by_alias=True, exclude_unset=True)
        for key in _ITEM_IDS:
            if key in type(value).model_fields and key not in dump:
                dump[key] = getattr(value, key)
        return dump
    if isinstance(value, Mapping):
        return {key: as_written(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [as_written(item) for item in value]
    return value


def set_override(dump: dict[str, Any], root: BaseModel, keys: Sequence[str | int], value: Any) -> None:
    """Set `value` at `keys` in `dump`, the written form of `root`, filling in a level left at its default from
    `root`, and storing a copy of `value`, so the matrix as written is never written into. A whole list item keeps the
    matched item's name; a value giving another is refused."""
    node: Any = dump
    source: Any = root
    for key in keys[:-1]:
        source = _child(source, key)
        if isinstance(node, dict) and key not in node:
            node[key] = as_written(source)
        node = node[key]
    last = keys[-1]
    if isinstance(last, int):
        name = _name(_child(source, last))
        if not isinstance(value, Mapping) or value.get("name", name) != name:
            raise ValueError(f"a whole item replacing `{name}` must keep its name `{name}`")
        value = {**value, "name": name}
    node[last] = copy.deepcopy(value)


def at(root: Any, keys: Sequence[str | int]) -> Any:
    """The value at `keys` in `root`, as :func:`walk` returned them."""
    node = root
    for key in keys:
        node = _child(node, key)
    return node


def reads(task: "TaskConfig", pipeline: "PipelineConfig") -> set[tuple[str, str]]:
    """Every pool entry `task` reads, as ``(pool, name)``: the entry it names, its extractor, a custom workflow's step
    entries and their extractors, and the extractors a preset entry's chain names, through nested presets."""
    from dataeval_flow.steps._workflow import CustomWorkflowConfig
    from dataeval_flow.workflows._preset import expand_preset, preset_of

    found = {("evaluators" if task.kind == "evaluator" else "workflows", task.workflow)}
    if task.extractor is not None:
        found.add(("extractors", task.extractor))

    def visit(entry: Any) -> None:
        if isinstance(entry, CustomWorkflowConfig):
            for step in entry.steps:
                if step.extractor:
                    found.add(("extractors", step.extractor))
                if step.kind == "evaluator":
                    found.add(("evaluators", step.target))
                elif step.kind == "workflow" and ("workflows", step.target) not in found:
                    found.add(("workflows", step.target))
                    visit(_entry(pipeline, "workflows", step.target))
        elif entry is not None and (preset := preset_of(entry)) is not None:
            chain, _ = expand_preset(entry, preset)
            found.update(("extractors", step.extractor) for step in chain.steps if step.extractor)

    if task.kind == "workflow":
        visit(_entry(pipeline, "workflows", task.workflow))
    return found
