"""Param form helpers for step-builder modals (transforms/selections)."""

from __future__ import annotations

import logging
from typing import Any

from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Checkbox, Input, Label, Select

from dataeval_flow._app._model._item import coerce_step_params, validate_step_params
from dataeval_flow._app._screens._base import _select_value

_log = logging.getLogger(__name__)


def build_param_form(container: Vertical, params: list[Any], prefix: str) -> None:
    """Populate *container* with widgets for each parameter."""

    container.remove_children()
    if not params:
        return
    for p in params:
        suffix = "" if p.required else " (optional)"
        label_text = f"{p.name} [{p.type_hint}]{suffix}:"
        widget_id = f"{prefix}-{p.name}"

        if p.choices:
            container.mount(Label(label_text))
            options = [(c, c) for c in p.choices]
            has_default = p.default is not None and str(p.default) in p.choices
            container.mount(
                Select(
                    options,
                    id=widget_id,
                    allow_blank=not p.required,
                    **({"value": str(p.default)} if has_default else {}),  # type: ignore
                    **({"prompt": "(default)"} if not p.required and not has_default else {}),
                )
            )
        elif p.type_hint == "bool":
            default_val = p.default if isinstance(p.default, bool) else False
            container.mount(Checkbox(label_text, value=default_val, id=widget_id))
        else:
            container.mount(Label(label_text))
            placeholder = str(p.default) if p.default is not None else ""
            container.mount(Input(placeholder=placeholder, id=widget_id))


def _read_param_values(container: Vertical, params: list[Any], prefix: str) -> dict[str, str | bool | None]:
    """Read raw values from param form widgets into a plain dict."""
    values: dict[str, str | bool | None] = {}
    for p in params:
        widget_id = f"{prefix}-{p.name}"
        try:
            if p.choices:
                sel = container.query_one(f"#{widget_id}", Select)
                values[p.name] = _select_value(sel) or None
            elif p.type_hint == "bool":
                cb = container.query_one(f"#{widget_id}", Checkbox)
                values[p.name] = cb.value
            else:
                inp = container.query_one(f"#{widget_id}", Input)
                values[p.name] = inp.value.strip() or None
        except NoMatches:
            _log.warning("Widget %s not found — skipping param '%s'", widget_id, p.name)
    return values


def validate_param_form(container: Vertical, params: list[Any], prefix: str) -> list[str]:
    """Validate the param form. Returns list of error messages."""
    values = _read_param_values(container, params, prefix)
    return validate_step_params(params, values)


def collect_param_form(container: Vertical, params: list[Any], prefix: str) -> dict[str, Any]:
    """Read values from a dynamic param form."""
    values = _read_param_values(container, params, prefix)
    return coerce_step_params(params, values)
