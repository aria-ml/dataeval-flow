"""Item assembly, field collection, step-builder validation/coercion, and action constants.

Pure functions shared by both the TUI and CLI for building config item
dicts, collecting/coercing field values, and validating step parameters.
No UI or state dependencies.
"""

from __future__ import annotations

import json
from typing import Any

from dataeval_flow._app._model._coerce import (
    _split_type_alternatives,
    coerce_field_value,
    coerce_value,
    validate_value,
)
from dataeval_flow._app._model._introspect import FieldDescriptor
from dataeval_flow._app._model._registry import (
    STEP_BUILDER_SECTIONS,
    get_discriminator_field,
)

__all__ = [
    "DELETE_SENTINEL",
    "SKIP",
    "build_item_dict",
    "coerce_step_params",
    "collect_field_value",
    "collect_multi_select_value",
    "diagnose_collect_failure",
    "finalize_item",
    "validate_step_params",
]

# Sentinel indicating a field value should be omitted from the result dict.
SKIP = object()

# ---------------------------------------------------------------------------
# Sentinel for delete actions
# ---------------------------------------------------------------------------

DELETE_SENTINEL = "__DELETE__"


# ---------------------------------------------------------------------------
# Item assembly helpers
# ---------------------------------------------------------------------------


def finalize_item(section: str, item: dict[str, Any]) -> dict[str, Any]:
    """Apply section-specific defaults to an assembled item dict.

    Currently handles tasks defaulting to ``enabled=True``.
    """
    if section == "tasks":
        item.setdefault("enabled", True)
    return item


def build_item_dict(
    section: str,
    name: str,
    variant_value: str | None,
    field_values: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a config item dict from its component parts.

    Combines name, discriminator, and field values into a single dict,
    then applies section-specific defaults via :func:`finalize_item`.
    """
    item: dict[str, Any] = {"name": name}
    disc_field = get_discriminator_field(section)
    if disc_field and variant_value:
        item[disc_field] = variant_value
    item.update(field_values)
    return finalize_item(section, item)


# ---------------------------------------------------------------------------
# Step-builder validation and coercion (pure functions)
# ---------------------------------------------------------------------------


def validate_step_params(params: list[Any], values: dict[str, str | bool | None]) -> list[str]:
    """Validate step parameter values. Returns a list of error messages.

    *params* is a list of ``ParamInfo``-like objects with ``name``,
    ``type_hint``, ``required``, and ``choices`` attributes.
    *values* maps param names to raw string values (or ``None``).
    Params whose name is not in *values* are skipped (widget not found).
    """
    errors: list[str] = []
    for p in params:
        if p.name not in values:
            continue
        val = values.get(p.name)
        if p.choices:
            if p.required and not val:
                errors.append(f"'{p.name}' is required (select a value).")
        elif p.type_hint == "bool":
            pass  # booleans are always valid
        else:
            if p.required and not val:
                errors.append(f"'{p.name}' is required.")
            elif val:
                val_str = str(val)
                alternatives = _split_type_alternatives(p.type_hint)
                if not any(validate_value(val_str, alt) for alt in alternatives):
                    expected = " or ".join(alternatives)
                    errors.append(f"'{p.name}' must be {expected} (got '{val_str}').")
    return errors


def coerce_step_params(params: list[Any], values: dict[str, str | bool | None]) -> dict[str, Any]:
    """Coerce raw string values for step parameters. Returns the result dict.

    *params* is a list of ``ParamInfo``-like objects.
    *values* maps param names to raw string values (or ``None``).
    Params whose name is not in *values* are skipped (widget not found).
    """
    result: dict[str, Any] = {}
    for p in params:
        if p.name not in values:
            continue
        val = values.get(p.name)
        if p.choices:
            if val:
                result[p.name] = val
        elif p.type_hint == "bool":
            bool_val = values.get(p.name)
            if isinstance(bool_val, bool):
                default = p.default if isinstance(p.default, bool) else False
                if bool_val != default:
                    result[p.name] = bool_val
        else:
            if val:
                result[p.name] = coerce_value(str(val), p.type_hint)
    return result


# ---------------------------------------------------------------------------
# Field value collection (pure logic, no widget access)
# ---------------------------------------------------------------------------


def collect_field_value(desc: FieldDescriptor, raw: str) -> Any:
    """Coerce a raw string from a widget into a typed value.

    Returns :data:`SKIP` if the value is empty/should be omitted.
    """
    if not raw:
        return SKIP
    return coerce_field_value(raw, desc)


def collect_json_value(raw: str) -> Any:
    """Parse a raw JSON string. Returns :data:`SKIP` if empty."""
    if not raw:
        return SKIP
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def collect_multi_select_value(
    selected: list[str],
    section: str,
    field_name: str,
) -> Any:
    """Process a multi-select result. Returns :data:`SKIP` if empty.

    For task sources, a single-element list is unwrapped to a plain string.
    """
    if not selected:
        return SKIP
    if len(selected) == 1 and section == "tasks" and field_name == "sources":
        return selected[0]
    return selected


def collect_bool_value(value: bool, default: Any) -> Any:
    """Return *value* if it differs from *default*, otherwise :data:`SKIP`."""
    resolved_default = default if isinstance(default, bool) else False
    return value if value != resolved_default else SKIP


# ---------------------------------------------------------------------------
# Validation error diagnosis (pure logic)
# ---------------------------------------------------------------------------


def diagnose_collect_failure(
    section: str,
    name: str,
    steps: list[dict[str, Any]],
    descriptors: list[FieldDescriptor],
    result: dict[str, Any] | None,
) -> str:
    """Return a human-readable error message when ``_collect_raw`` returns None."""
    if not name:
        return "Name is required."
    if section in STEP_BUILDER_SECTIONS and not steps:
        return "Add at least one step."
    disc_field = get_discriminator_field(section)
    if disc_field:
        return f"{disc_field.replace('_', ' ')} is required."
    missing = [d.name for d in descriptors if d.required and d.name not in (result or {})]
    if missing:
        return f"Required: {', '.join(missing)}"
    return "Invalid input."
