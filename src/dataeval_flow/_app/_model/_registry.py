"""Variant registry, section constants, cross-reference overlays, and field descriptor helpers.

Auto-built by introspecting :class:`~dataeval_flow.config._models.PipelineConfig`
so that new schema variants (e.g. a new workflow type) are picked up automatically.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from dataeval_flow._app._model._introspect import (
    FieldDescriptor,
    FieldKind,
    introspect_model,
)
from dataeval_flow.config._models import PipelineConfig

__all__ = [
    "CROSS_REFS",
    "MULTI_REF_FIELDS",
    "SECTION_KEYS",
    "SECTION_MODELS",
    "SECTIONS",
    "STEP_BUILDER_SECTIONS",
    "VARIANT_REGISTRY",
    "WORKFLOW_SKIP_FIELDS",
    "get_discriminator_field",
    "get_fields",
    "get_model_for_variant",
    "get_variant_choices",
]

# ---------------------------------------------------------------------------
# Section ordering
# ---------------------------------------------------------------------------

SECTIONS: list[tuple[str, str]] = [
    ("datasets", "Datasets"),
    ("selections", "Selections"),
    ("sources", "Sources"),
    ("preprocessors", "Preprocessors"),
    ("extractors", "Extractors"),
    ("workflows", "Workflows"),
    ("tasks", "Tasks"),
]

SECTION_KEYS: list[str] = [s[0] for s in SECTIONS]

# ---------------------------------------------------------------------------
# Variant registry building
# ---------------------------------------------------------------------------


def _get_literal_value(model: type[BaseModel], field_name: str) -> str | None:
    """Return the ``Literal`` default for *field_name*, or ``None``."""
    field_info = model.model_fields.get(field_name)
    if field_info is None:
        return None
    ann = field_info.annotation
    if get_origin(ann) is Literal:
        args = get_args(ann)
        if args:
            return str(args[0])
    if field_info.default is not None:
        return str(field_info.default)
    return None


def _unwrap_sequence_inner(annotation: Any) -> Any | None:
    """Unwrap ``Optional[Sequence[X]]`` and return *X*, or ``None``."""
    args = get_args(annotation)
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == 1:
        annotation = non_none[0]
    origin = get_origin(annotation)
    if not (origin is Sequence or (origin is not None and issubclass(origin, Sequence))):
        return None
    inner_args = get_args(annotation)
    return inner_args[0] if inner_args else None


def _extract_discriminated_variants(inner: Any) -> tuple[str, dict[str, type[BaseModel]]] | None:
    """If *inner* is an ``Annotated`` discriminated union, return ``(disc, variants)``."""
    if get_origin(inner) is Union:
        for ua in get_args(inner):
            if get_origin(ua) is Annotated:
                inner = ua
                break
    if get_origin(inner) is not Annotated:
        return None
    annotated_args = get_args(inner)
    union_type, field_meta = annotated_args[0], annotated_args[1]
    disc = getattr(field_meta, "discriminator", None)
    if not disc or not get_args(union_type):
        return None
    variants: dict[str, type[BaseModel]] = {}
    for member in get_args(union_type):
        if isinstance(member, type) and issubclass(member, BaseModel):
            val = _get_literal_value(member, disc)
            if val:
                variants[val] = member
    return (disc, variants) if variants else None


def _build_registries() -> tuple[dict[str, tuple[str, dict[str, type[BaseModel]]]], dict[str, type[BaseModel]]]:
    """Introspect :class:`PipelineConfig` to derive the variant and section-model registries."""
    variant_registry: dict[str, tuple[str, dict[str, type[BaseModel]]]] = {}
    section_models: dict[str, type[BaseModel]] = {}

    for name, field_info in PipelineConfig.model_fields.items():
        inner = _unwrap_sequence_inner(field_info.annotation)
        if inner is None:
            continue
        result = _extract_discriminated_variants(inner)
        if result is not None:
            variant_registry[name] = result
        elif isinstance(inner, type) and issubclass(inner, BaseModel):
            section_models[name] = inner

    return variant_registry, section_models


VARIANT_REGISTRY, SECTION_MODELS = _build_registries()

# ---------------------------------------------------------------------------
# Cross-reference overlays
# ---------------------------------------------------------------------------

CROSS_REFS: dict[str, dict[str, str]] = {
    "sources": {"dataset": "datasets", "selection": "selections"},
    "tasks": {"workflow": "workflows", "extractor": "extractors"},
    "extractors": {"preprocessor": "preprocessors"},
}

MULTI_REF_FIELDS: dict[str, dict[str, str]] = {
    "tasks": {"sources": "sources"},
}

# ---------------------------------------------------------------------------
# Step-builder sections (preprocessors, selections)
# ---------------------------------------------------------------------------

STEP_BUILDER_SECTIONS: dict[str, dict[str, str]] = {
    "preprocessors": {
        "step_key": "step",
        "list_fn": "list_transforms",
        "params_fn": "get_transform_params",
    },
    "selections": {
        "step_key": "type",
        "list_fn": "list_selection_classes",
        "params_fn": "get_selection_params",
    },
}

WORKFLOW_SKIP_FIELDS = frozenset({"name", "type", "mode"})

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def get_variant_choices(section: str) -> list[str] | None:
    """Return discriminator choices for *section*, or ``None`` if not discriminated."""
    if section in VARIANT_REGISTRY:
        _, variants = VARIANT_REGISTRY[section]
        return list(variants)
    return None


def get_discriminator_field(section: str) -> str | None:
    """Return the discriminator field name for *section*, or ``None``."""
    if section in VARIANT_REGISTRY:
        return VARIANT_REGISTRY[section][0]
    return None


def get_model_for_variant(section: str, variant_value: str) -> type[BaseModel] | None:
    """Return the concrete Pydantic model for a variant value."""
    if section in VARIANT_REGISTRY:
        _, variants = VARIANT_REGISTRY[section]
        return variants.get(variant_value)
    return SECTION_MODELS.get(section)


# ---------------------------------------------------------------------------
# Field descriptors with cross-ref overlay
# ---------------------------------------------------------------------------


def _apply_cross_refs(
    descriptors: list[FieldDescriptor],
    section: str,
    state: Any,
) -> None:
    cross_refs = CROSS_REFS.get(section, {})
    multi_refs = MULTI_REF_FIELDS.get(section, {})
    for desc in descriptors:
        if desc.name in cross_refs:
            ref_section = cross_refs[desc.name]
            desc.choices = state.names(ref_section)
            if desc.kind not in (FieldKind.MULTI_SELECT,):
                desc.kind = FieldKind.SELECT
        if desc.name in multi_refs:
            ref_section = multi_refs[desc.name]
            desc.choices = state.names(ref_section)
            desc.kind = FieldKind.MULTI_SELECT


def _build_skip_set(section: str, *, skip_name: bool) -> set[str]:
    disc_field = get_discriminator_field(section)
    skip: set[str] = set()
    if skip_name:
        skip.add("name")
    if disc_field:
        skip.add(disc_field)
    if section == "workflows":
        skip.update(WORKFLOW_SKIP_FIELDS)
    if section in STEP_BUILDER_SECTIONS:
        skip.add("steps")
    return skip


def get_fields(
    section: str,
    variant_value: str | None,
    state: Any,
    *,
    skip_name: bool = True,
) -> list[FieldDescriptor]:
    """Return field descriptors for creating/editing an item.

    For discriminated sections, *variant_value* selects the concrete model.
    Cross-reference fields get their ``choices`` populated from *state*.

    The ``name`` field and the discriminator field are excluded by default
    (the caller handles those separately).
    """
    if section in VARIANT_REGISTRY:
        if variant_value is None:
            return []
        model = get_model_for_variant(section, variant_value)
    else:
        model = SECTION_MODELS.get(section)

    if model is None:
        return []

    descriptors = introspect_model(model)
    _apply_cross_refs(descriptors, section, state)
    skip = _build_skip_set(section, skip_name=skip_name)
    return [d for d in descriptors if d.name not in skip]
