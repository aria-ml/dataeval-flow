"""Pydantic schema introspection for dynamic form generation.

Walks Pydantic model fields and produces ``FieldDescriptor`` objects that
the TUI can render as appropriate widgets (inputs, selects, checkboxes, etc.)
without hard-coding knowledge of each workflow's parameter model.
"""

from __future__ import annotations

import collections.abc
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefinedType


class FieldKind(Enum):
    """Widget hint for the TUI layer."""

    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    NESTED = "nested"
    LIST = "list"


@dataclass
class FieldDescriptor:
    """Metadata extracted from a single Pydantic field."""

    name: str
    kind: FieldKind
    description: str
    required: bool
    default: Any = None
    choices: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    nested_model: type[BaseModel] | None = None
    item_descriptors: list[FieldDescriptor] | None = None  # for lists of nested models
    discriminator: str | None = None  # for discriminated unions
    union_variants: dict[str, type[BaseModel]] | None = None  # discriminator value -> model


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Strip ``Optional[X]`` / ``X | None`` and return (inner_type, is_optional)."""
    import types

    origin = get_origin(annotation)
    if origin is typing.Union or isinstance(annotation, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
        return annotation, False
    return annotation, False


def _extract_literal_values(annotation: Any) -> list[str] | None:
    """Return literal values if annotation is ``Literal[...]``."""
    if get_origin(annotation) is typing.Literal:
        return [str(v) for v in get_args(annotation)]
    return None


def _is_pydantic_model(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _extract_constraints(field_info: FieldInfo) -> dict[str, Any]:
    """Pull numeric constraints (ge, le, gt, lt) from field metadata."""
    constraints: dict[str, Any] = {}
    for meta in field_info.metadata:
        for attr in ("ge", "le", "gt", "lt", "min_length", "max_length"):
            val = getattr(meta, attr, None)
            if val is not None:
                constraints[attr] = val
    return constraints


def _resolve_discriminated_union(
    annotation: Any,
    field_info: FieldInfo,
) -> tuple[str | None, dict[str, type[BaseModel]] | None]:
    """Detect discriminated unions and return (discriminator_field, {value: model})."""
    disc = field_info.discriminator
    if disc is None:
        return None, None

    disc_field = disc if isinstance(disc, str) else None
    if disc_field is None:
        return None, None

    # The annotation is Annotated[Union[A, B, ...], Field(discriminator=...)]
    inner = annotation
    if get_origin(annotation) is typing.Annotated:
        inner = get_args(annotation)[0]

    import types as _types

    union_args = get_args(inner) if (get_origin(inner) is typing.Union or isinstance(inner, _types.UnionType)) else []
    variants: dict[str, type[BaseModel]] = {}
    for variant in union_args:
        if _is_pydantic_model(variant):
            disc_info = variant.model_fields.get(disc_field)
            if disc_info and disc_info.default is not None:
                variants[str(disc_info.default)] = variant

    return disc_field, variants if variants else None


_PRIMITIVE_KINDS: dict[type, FieldKind] = {
    bool: FieldKind.BOOL,
    int: FieldKind.INT,
    float: FieldKind.FLOAT,
}


def _introspect_list_field(
    inner: Any,
    name: str,
    description: str,
    required: bool,
    default: Any,
    constraints: dict[str, Any],
) -> FieldDescriptor | None:
    """Attempt to classify a list/Sequence annotation, returning a descriptor or None."""
    list_args = get_args(inner)
    if list_args:
        list_inner = list_args[0]
        list_literals = _extract_literal_values(list_inner)
        if list_literals:
            return FieldDescriptor(
                name=name,
                kind=FieldKind.MULTI_SELECT,
                description=description,
                required=required,
                default=default,
                choices=list_literals,
                constraints=constraints,
            )
        if _is_pydantic_model(list_inner):
            return FieldDescriptor(
                name=name,
                kind=FieldKind.LIST,
                description=description,
                required=required,
                default=default,
                nested_model=list_inner,
                constraints=constraints,
            )
    # Check for list of discriminated union
    if list_args and get_origin(list_args[0]) is typing.Annotated:
        ann_args = get_args(list_args[0])
        union_type = ann_args[0]
        field_meta = [a for a in ann_args[1:] if isinstance(a, FieldInfo)]
        import types as _types

        if field_meta and (get_origin(union_type) is typing.Union or isinstance(union_type, _types.UnionType)):
            d_field, d_variants = _resolve_discriminated_union(list_args[0], field_meta[0])
            if d_field and d_variants:
                return FieldDescriptor(
                    name=name,
                    kind=FieldKind.LIST,
                    description=description,
                    required=required,
                    default=default,
                    discriminator=d_field,
                    union_variants=d_variants,
                    constraints=constraints,
                )
    return None


def introspect_model(model: type[BaseModel]) -> list[FieldDescriptor]:
    """Walk a Pydantic model and return field descriptors for each field."""
    descriptors: list[FieldDescriptor] = []

    for name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        if annotation is None:
            continue

        inner, is_optional = _unwrap_optional(annotation)

        required = field_info.is_required()
        raw_default = field_info.default
        default = None if required or isinstance(raw_default, PydanticUndefinedType) else raw_default
        description = field_info.description or ""
        constraints = _extract_constraints(field_info)

        # Check for discriminated union first
        disc_field, union_variants = _resolve_discriminated_union(annotation, field_info)
        if disc_field and union_variants:
            descriptors.append(
                FieldDescriptor(
                    name=name,
                    kind=FieldKind.NESTED,
                    description=description,
                    required=required,
                    default=default,
                    discriminator=disc_field,
                    union_variants=union_variants,
                )
            )
            continue

        # Literal -> select
        literal_vals = _extract_literal_values(inner)
        if literal_vals:
            descriptors.append(
                FieldDescriptor(
                    name=name,
                    kind=FieldKind.SELECT,
                    description=description,
                    required=required,
                    default=str(default) if default is not None else None,
                    choices=literal_vals,
                    constraints=constraints,
                )
            )
            continue

        # list[...] / Sequence[...]
        if get_origin(inner) in (list, collections.abc.Sequence):
            list_desc = _introspect_list_field(inner, name, description, required, default, constraints)
            if list_desc:
                descriptors.append(list_desc)
                continue
            # Plain list
            descriptors.append(
                FieldDescriptor(
                    name=name,
                    kind=FieldKind.LIST,
                    description=description,
                    required=required,
                    default=default,
                    constraints=constraints,
                )
            )
            continue

        # Nested BaseModel
        if _is_pydantic_model(inner):
            sub_descs = introspect_model(inner)
            expandable_kinds = {FieldKind.BOOL, FieldKind.INT, FieldKind.FLOAT, FieldKind.STRING, FieldKind.SELECT}
            expandable = sub_descs and all(d.kind in expandable_kinds and not d.required for d in sub_descs)
            descriptors.append(
                FieldDescriptor(
                    name=name,
                    kind=FieldKind.NESTED,
                    description=description,
                    required=required,
                    default=default,
                    nested_model=inner,
                    constraints=constraints,
                    item_descriptors=sub_descs if expandable else None,
                )
            )
            continue

        # Primitives
        kind = _PRIMITIVE_KINDS.get(inner, FieldKind.STRING)
        descriptors.append(
            FieldDescriptor(
                name=name,
                kind=kind,
                description=description,
                required=required,
                default=default,
                constraints=constraints,
            )
        )

    return descriptors
