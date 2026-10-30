"""The step catalog: every registered step described as data, so a palette needs to import none of them (spec §8)."""

__all__ = [
    "AddressGrammar",
    "PortEntry",
    "SameDataset",
    "Shape",
    "StepCatalog",
    "StepCatalogEntry",
    "class_path",
    "dataset_kinds",
    "input_shape",
    "list_steps",
    "port_entry",
]

import sys
import types
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field

from dataeval_flow.steps._address import ADDRESS_PATTERN, STEP_NAME_PATTERN
from dataeval_flow.steps._port import DATASET_KINDS, DataType, Port
from dataeval_flow.steps._step import StepKind

Shape = Literal["scalar", "array", "map_keys"]


class PortEntry(BaseModel):
    """One port of a step, as the catalog describes it."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    port: str = Field(description="The config field that feeds an input, or an output's name.")
    type: DataType = Field(description="What flows along it: `dataset`, `output`, `export` or `findings`.")
    kinds: list[str] | None = Field(
        default=None,
        description=(
            'On a Dataset port, the Dataset kinds it takes or makes, or `["any"]`; `null` on a port that carries '
            "no Dataset."
        ),
    )
    is_list: bool = Field(default=False, alias="list", description="Whether it takes or gives a whole keyed list.")
    count: str | None = Field(default=None, description="How many addresses a Dataset port takes, such as `1+`.")
    derives: list[str] = Field(default_factory=list, description="What Flow derives from an evaluator's Dataset port.")
    classes: list[str] = Field(
        default_factory=list, description="The output classes an output port takes or gives, by import path."
    )
    shape: Shape | None = Field(
        default=None,
        description=(
            "On an input, how the field named for the port holds its addresses: one (`scalar`), several in order "
            "(`array`), or the keys of a mapping whose values are each binding's settings (`map_keys`). `null` on an "
            "output."
        ),
    )


class SameDataset(BaseModel):
    """An input whose Output must have been computed on the step's own Dataset."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    port: str = Field(description="The input reading the Output.")
    on: Literal["input"] = Field(default="input", description="The input holding the Dataset it must be computed on.")
    first_only: bool = Field(
        default=False, description="Whether only the first address of `on` counts, as for a ranking with a reference."
    )


class StepCatalogEntry(BaseModel):
    """One step type."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    kind: StepKind = Field(description="`evaluator`, `transform`, `combine`, `check` or `workflow`.")
    type: str = Field(description="The name a step uses under its kind key.")
    title: str = Field(description="Its friendly name, such as `K-Fold`; its `type` where it declares none.")
    description: str = Field(description="One line on what it does.")
    origin: str = Field(description="The distribution that registered it: `dataeval-flow` for a built-in.")
    experimental: bool = Field(
        default=False,
        description="Whether it may change or be removed in any release without a deprecation cycle.",
    )
    inputs: list[PortEntry] = Field(description="Its input ports.")
    outputs: list[PortEntry] = Field(description="Its output ports.")
    judges: list[str] = Field(
        default_factory=list, description="On a check, the evaluator and combine types whose Outputs it judges."
    )
    judged_by: list[str] = Field(
        default_factory=list, description="On an evaluator or combine, the check types that judge its Output."
    )
    same_dataset: list[SameDataset] = Field(
        default_factory=list, description="The inputs whose Outputs must have been computed on this step's Dataset."
    )
    config_schema: dict[str, Any] = Field(description="The JSON Schema of its settings.")


class AddressGrammar(BaseModel):
    """How an address names a chain input, a step's output, or one element of a list."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name_pattern: str = Field(default=STEP_NAME_PATTERN, description="What a step's or input's name may be.")
    address_pattern: str = Field(default=ADDRESS_PATTERN, description="What an address may be.")
    rules: dict[str, str] = Field(
        default_factory=lambda: {
            "single_output": "A step with one output stores it at its name: `clean`.",
            "multiple_outputs": "A step with several stores each at `name.output`: `split.train`.",
            "element": "`[key]` names one element of a list: `kfold.train[0]`.",
        },
        description="What each form of address names.",
    )


STEP_SETTINGS: dict[str, Any] = {
    "evaluator": {"pool": "evaluators"},
    "workflow": {"pool": "workflows"},
    "transform": "inline",
    "combine": "inline",
    "check": "inline",
}


class StepCatalog(BaseModel):
    """Every step Flow can chain, and what produced this description."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    format: Literal[1] = Field(default=1, description="The document's format version, raised when its shape breaks.")
    flow_version: str = Field(description="The dataeval-flow version that produced it.")
    dataeval_version: str = Field(description="The DataEval version installed beside it.")
    data_types: list[str] = Field(description="What can flow along a port.")
    dataset_kinds: list[str] = Field(description="The Dataset kinds a port may name.")
    steps: list[StepCatalogEntry] = Field(description="Every step, by kind, then name.")
    address_grammar: AddressGrammar = Field(default_factory=AddressGrammar, description="How addresses are written.")
    step_settings: dict[str, Any] = Field(
        default_factory=lambda: dict(STEP_SETTINGS),
        description=(
            "Where each kind's settings live: a pooled kind's step names an entry of that pool under its kind key; an "
            "inline kind's settings sit beside its kind key."
        ),
    )
    classes: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Each Output class a port names, by import path, to its ancestors among them: a port taking `A` accepts an "
            "Output of class `C` when `C` is `A` or `A` is among `C`'s ancestors. A port naming none accepts any."
        ),
    )


def class_path(cls: type) -> str:
    """Where `cls` imports from: the shortest package enclosing its module that exports it, else that module."""
    parts = cls.__module__.split(".")
    for end in range(1, len(parts)):
        package = sys.modules.get(".".join(parts[:end]))  # imported already: a module's packages import before it
        if package is not None and getattr(package, cls.__qualname__, None) is cls:
            return f"{package.__name__}.{cls.__qualname__}"
    return f"{cls.__module__}.{cls.__qualname__}"


def dataset_kinds(port: Port) -> list[str] | None:
    """The kinds a Dataset port names, ``["any"]`` when it names none; ``None`` for a port carrying no Dataset."""
    if port.type != DataType.DATASET:
        return None
    return sorted(port.kinds) if port.kinds is not None else ["any"]


def port_entry(port: Port, shape: Shape | None = None) -> PortEntry:
    return PortEntry(
        port=port.name,
        type=port.type,
        kinds=dataset_kinds(port),
        list=port.is_list,  # by its alias, the name a type checker gives the constructor's parameter
        count=str(port.count) if port.count is not None else None,
        derives=sorted(str(kind) for kind in port.derives),
        classes=[class_path(cls) for cls in port.classes],
        shape=shape,
    )


def _held(annotation: Any) -> set[str]:
    """What an annotation lets a field hold: ``{"map"}``, ``{"list"}`` or ``{"text"}``, a union's members each."""
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return set().union(*(_held(member) for member in get_args(annotation)))
    held = origin or annotation
    if isinstance(held, type) and issubclass(held, Mapping):
        return {"map"}
    if isinstance(held, type) and issubclass(held, Sequence) and held is not str:
        return {"list"}
    return {"text"}


def input_shape(cls: Any, port: Port) -> Shape:
    """How `cls`'s step holds `port`'s addresses (builder spec §4.2): in the field named for the port.

    A pooled step reads `input:`, a string or a list, so a port taking several sources is written as a list. An inline
    step's field says by its annotation: a mapping's keys, a list, or one address.
    """
    if cls.kind in ("evaluator", "workflow"):
        return "array" if port.count is not None and port.count.allows(2) else "scalar"
    field = cls.config_type.model_fields.get(port.name)
    if field is None:
        # ponytail: a malformed plugin step names a port with no field; `port_addresses` reads it as no address.
        return "scalar"
    held = _held(field.annotation)
    if "map" in held:
        return "map_keys"
    return "array" if "list" in held else "scalar"


def _ancestry(steps: Sequence[Any]) -> dict[str, list[str]]:
    """Each Output class a port of `steps` names, to its ancestors among those classes."""
    named = {cls for step in steps for port in (*step.input_ports(), *step.output_ports()) for cls in port.classes}
    return {
        class_path(cls): sorted(class_path(other) for other in named if other is not cls and issubclass(cls, other))
        for cls in sorted(named, key=class_path)
    }


def _feeds(producer: Any, check: Any) -> bool:
    """Whether an Output `producer` gives is of a class one of `check`'s ports takes."""
    made = [cls for port in producer.output_ports() if port.type == DataType.OUTPUT for cls in port.classes]
    taken = [cls for port in check.input_ports() if port.type == DataType.OUTPUT for cls in port.classes]
    return any(issubclass(out, into) for out in made for into in taken)


def list_steps(*, plugins: bool = True) -> StepCatalog:
    """Every registered step as data: kind, name, description, origin, ports and settings schema.

    Custom workflows are config, not step types, so they do not appear.

    Parameters
    ----------
    plugins : bool
        Whether to include steps installed by other packages. Without them, only dataeval-flow's own.

    Returns
    -------
    StepCatalog
        The catalog, ordered by kind, then name.
    """
    import dataeval

    from dataeval_flow import __version__
    from dataeval_flow.evaluators._registry import EVALUATORS
    from dataeval_flow.steps._registry import CHECKS, COMBINES, TRANSFORMS
    from dataeval_flow.workflows._registry import WORKFLOWS

    found = [
        (registry, cls)
        for registry in (EVALUATORS, TRANSFORMS, COMBINES, CHECKS, WORKFLOWS)  # in kind order
        for cls in registry.list(plugins=plugins)
    ]
    producers = [cls for _, cls in found if cls.kind in ("evaluator", "combine")]
    checks = [cls for _, cls in found if cls.kind == "check"]
    judges = {check.name: sorted({p.name for p in producers if _feeds(p, check)}) for check in checks}
    judged_by = {(p.kind, p.name): sorted({c.name for c in checks if _feeds(p, c)}) for p in producers}
    entries = [
        StepCatalogEntry(
            kind=cls.kind,
            type=cls.name,
            title=cls.title,
            description=cls.description,
            origin=registry.origin(cls.name),
            experimental=cls.experimental,
            inputs=[port_entry(port, input_shape(cls, port)) for port in cls.input_ports()],
            outputs=[port_entry(port) for port in cls.output_ports()],
            same_dataset=[
                SameDataset(port=name, first_only=getattr(cls, "same_node_first_input", False))
                for name in getattr(cls, "same_node", ())
            ],
            config_schema=cls.config_type.model_json_schema(),
            judges=judges.get(cls.name, []) if cls.kind == "check" else [],
            judged_by=judged_by.get((cls.kind, cls.name), []),
        )
        for registry, cls in found
    ]
    return StepCatalog(
        flow_version=__version__,
        dataeval_version=dataeval.__version__,
        data_types=[str(member) for member in DataType],
        dataset_kinds=list(DATASET_KINDS),
        steps=entries,
        classes=_ancestry([cls for _, cls in found]),
    )
