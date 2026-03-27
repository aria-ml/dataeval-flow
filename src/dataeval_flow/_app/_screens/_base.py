"""Base modal class and shared utilities for modal dialogs."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from textual.app import ComposeResult, ScreenStackError
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Select

from dataeval_flow._app._model._item import DELETE_SENTINEL
from dataeval_flow._app._screens._pathpicker import PathPickerScreen

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def _select_value(sel: Select) -> str:
    v = sel.value
    if v is Select.BLANK or v is Select.NULL or v is None:
        return ""
    return str(v)


# ---------------------------------------------------------------------------
# Shared modal CSS
# ---------------------------------------------------------------------------

_MODAL_CSS = """
.modal-dialog {
    width: 70;
    height: auto;
    max-height: 85%;
    border: round $accent 40%;
    background: $surface;
    padding: 1 2;
}

.modal-dialog Vertical {
    height: auto;
}

SectionModal,
ComponentModal {
    align: center middle;
}

.modal-dialog Label {
    margin: 1 0 0 0;
    text-style: bold;
}

.modal-dialog Input {
    margin: 0 0 1 0;
}

.modal-dialog Select {
    margin: 0 0 1 0;
}

.modal-button-bar {
    height: auto;
    padding: 1 0 0 0;
    align: right middle;
}

.modal-button-bar Button {
    margin: 0 0 0 1;
    min-width: 10;
}

.browse-row {
    height: auto;
}

.browse-row Input {
    width: 1fr;
}

.browse-row Button {
    width: auto;
    min-width: 10;
}

.conditional-field {
    height: auto;
}

.hidden {
    display: none;
}

.step-row {
    height: auto;
    padding: 0 1;
    margin: 0 0 0 1;
}

.step-row:hover {
    background: $accent 8%;
}

.step-row Static {
    width: 1fr;
}

.step-row Button {
    width: auto;
    min-width: 4;
}

#modal-step-list {
    height: auto;
    max-height: 15;
    margin: 0 0 1 0;
}

#modal-params-form {
    height: auto;
}

.step-section-title {
    text-style: bold;
    padding: 1 0 0 0;
}

.nested-group {
    margin: 0 0 1 1;
    padding: 0 0 0 1;
    border-left: tall $accent 40%;
}
"""


# ---------------------------------------------------------------------------
# ComponentModal — base class
# ---------------------------------------------------------------------------


class ComponentModal(ModalScreen[dict | str | None]):
    """Base modal for creating/editing pipeline components."""

    CSS = _MODAL_CSS
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        existing: dict[str, Any] | None = None,
        data_dir: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._existing = existing
        self._data_dir = data_dir

    @property
    def is_edit_mode(self) -> bool:
        return self._existing is not None

    def _collect_raw(self) -> dict[str, Any] | None:
        raise NotImplementedError

    def _check_dirty(self) -> bool:
        raise NotImplementedError

    def _collect(self) -> dict[str, Any] | None:
        raise NotImplementedError

    def _update_ok_state(self) -> None:
        try:
            btn = self.query_one("#btn-modal-ok", Button)
            btn.disabled = not self._check_dirty()
        except NoMatches:
            pass

    def compose_buttons(self) -> ComposeResult:
        with Horizontal(classes="modal-button-bar"):
            yield Button("OK", id="btn-modal-ok", variant="primary", disabled=True)
            yield Button("Cancel", id="btn-modal-cancel")
            if self.is_edit_mode:
                yield Button("Delete", id="btn-modal-delete", variant="error")

    def _safe_dismiss(self, result: dict | str | None) -> None:
        with contextlib.suppress(ScreenStackError):
            self.dismiss(result)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id or ""
        if btn == "btn-modal-ok":
            result = self._collect()
            if result is not None:
                self._safe_dismiss(result)
        elif btn == "btn-modal-cancel":
            self._safe_dismiss(None)
        elif btn == "btn-modal-delete":
            self._safe_dismiss(DELETE_SENTINEL)

    def on_input_changed(self, _event: Input.Changed) -> None:
        self._update_ok_state()

    def on_select_changed(self, _event: Select.Changed) -> None:
        self._update_ok_state()

    def on_checkbox_changed(self, _event: Checkbox.Changed) -> None:
        self._update_ok_state()

    def action_cancel(self) -> None:
        with contextlib.suppress(ScreenStackError):
            self.dismiss(None)

    def _browse_for_input(self, input_id: str) -> None:
        try:
            current = self.query_one(f"#{input_id}", Input).value.strip() or "."
        except NoMatches:
            current = "."

        data_dir = self._data_dir

        def _on_result(result: str | None) -> None:
            if result is None:
                return
            # Auto-relativize absolute paths against data_dir for portability
            if Path(result).is_absolute() and data_dir is not None:
                from dataeval_flow.config._paths import relativize_to_data_dir

                try:
                    result = relativize_to_data_dir(result, data_dir)
                except ValueError:
                    _log.warning("Browsed path '%s' is not under data root '%s'", result, data_dir)
            with contextlib.suppress(NoMatches):
                self.query_one(f"#{input_id}", Input).value = result

        self.app.push_screen(PathPickerScreen(start_path=current, mode="folder"), callback=_on_result)
