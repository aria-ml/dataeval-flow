"""Value coercion and validation for the configuration builder.

Pure functions that convert string inputs to typed values based on
type hints or field descriptors.  No UI or state dependencies.
"""

from __future__ import annotations

import json
from typing import Any

from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind

__all__ = [
    "coerce_field_value",
    "coerce_value",
    "validate_value",
]


def _split_type_alternatives(type_hint: str) -> list[str]:
    """Split a type hint like ``"int | str"`` into alternatives."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in type_hint:
        if ch in ("(", "["):
            depth += 1
            current.append(ch)
        elif ch in (")", "]"):
            depth -= 1
            current.append(ch)
        elif ch == "|" and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    parts.append("".join(current).strip())
    return [p for p in parts if p]


def validate_value(value: str, hint: str) -> bool:
    """Check whether *value* is compatible with *hint*."""
    h = hint.strip()
    if h in ("any", "str", "string"):
        return True
    if h == "bool":
        return value.lower() in ("true", "false", "1", "0", "yes", "no")
    if h == "int":
        try:
            int(value)
            return True
        except ValueError:
            return False
    if h == "float":
        try:
            float(value)
            return True
        except ValueError:
            return False
    if h.startswith("list") or h.startswith("tuple"):
        try:
            parsed = json.loads(value)
            return isinstance(parsed, (list, tuple))
        except (ValueError, TypeError):
            return False
    return True


_COERCE_SCALAR: dict[str, type] = {"int": int, "float": float}


def _try_scalar_coercion(value: str, alternatives: list[str]) -> tuple[bool, Any]:
    """Try int/float coercion. Returns ``(matched, result)``."""
    for alt in alternatives:
        converter = _COERCE_SCALAR.get(alt)
        if converter is not None:
            try:
                return True, converter(value)
            except (ValueError, TypeError):
                continue
    return False, value


def coerce_value(value: str, type_hint: str) -> Any:
    """Coerce a string *value* according to *type_hint*.

    JSON parsing is only attempted for complex types (list, dict).
    """
    if not value:
        return value
    alternatives = [a.strip() for a in _split_type_alternatives(type_hint)]

    if {"str", "string"} & set(alternatives):
        matched, result = _try_scalar_coercion(value, alternatives)
        return result if matched else value

    matched, result = _try_scalar_coercion(value, alternatives)
    if matched:
        return result
    if "bool" in alternatives:
        return value.lower() in ("true", "1", "yes")

    try:
        return json.loads(value)
    except (ValueError, TypeError):
        pass
    return value


def coerce_field_value(value: str, desc: FieldDescriptor) -> Any:
    """Coerce *value* using a FieldDescriptor's kind."""
    if not value:
        return value
    if desc.kind == FieldKind.INT:
        return int(value)
    if desc.kind == FieldKind.FLOAT:
        return float(value)
    if desc.kind == FieldKind.BOOL:
        return value.lower() in ("true", "1", "yes")
    if desc.kind in (FieldKind.LIST, FieldKind.NESTED):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value
