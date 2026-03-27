"""ModelModal — flattened model creation/editing."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Button, Input, Label, Select, Static

from dataeval_flow._app._screens._base import ComponentModal, _select_value
from dataeval_flow._app._viewmodel._model_vm import MODEL_TYPES, ModelViewModel


class ModelModal(ComponentModal):
    """Create or edit a model entry (flattened — no nested extractor).

    Delegates business logic to :class:`ModelViewModel`.
    """

    def __init__(self, existing: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(existing, **kwargs)
        self._vm = ModelViewModel(existing)

    def compose(self) -> ComposeResult:
        title = "Edit Model" if self.is_edit_mode else "New Model"
        with VerticalScroll(classes="modal-dialog"):
            yield Static(f"[bold]{title}[/bold]")
            yield Label("Name:")
            yield Input(id="md-model-name", placeholder="e.g. resnet50")
            yield Label("Type:")
            yield Select(
                [(t, t) for t in MODEL_TYPES],
                id="md-model-type",
                prompt="Select type",
            )
            with Vertical(id="md-model-path-group", classes="conditional-field hidden"):
                yield Label("Model Path:")
                with Horizontal(classes="browse-row"):
                    yield Input(id="md-model-path", placeholder="e.g. ./models/resnet50.onnx")
                    yield Button("Browse", id="btn-browse-md-model-path")
            with Vertical(id="md-model-vocab-group", classes="conditional-field hidden"):
                yield Label("Vocab Size:")
                yield Input(id="md-model-vocab", placeholder="2048")
            yield from self.compose_buttons()

    def on_mount(self) -> None:
        if self._existing:
            try:
                self.query_one("#md-model-name", Input).value = self._existing.get("name", "")
                model_type = self._existing.get("type", "")
                if model_type:
                    self.query_one("#md-model-type", Select).value = model_type
                if self._existing.get("model_path"):
                    self.query_one("#md-model-path", Input).value = self._existing["model_path"]
                if self._existing.get("vocab_size"):
                    self.query_one("#md-model-vocab", Input).value = str(self._existing["vocab_size"])
            except NoMatches:
                pass
        self._toggle_fields()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-browse-md-model-path":
            self._browse_for_input("md-model-path")
            return
        super().on_button_pressed(event)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "md-model-type":
            self._toggle_fields()
        super().on_select_changed(event)

    def _toggle_fields(self) -> None:
        model_type = _select_value(self.query_one("#md-model-type", Select))
        path_group = self.query_one("#md-model-path-group")
        vocab_group = self.query_one("#md-model-vocab-group")
        if ModelViewModel.needs_path(model_type):
            path_group.remove_class("hidden")
        else:
            path_group.add_class("hidden")
        if ModelViewModel.needs_vocab(model_type):
            vocab_group.remove_class("hidden")
        else:
            vocab_group.add_class("hidden")

    def _collect_raw(self) -> dict[str, Any] | None:
        return self._vm.build_result(
            name=self.query_one("#md-model-name", Input).value.strip(),
            model_type=_select_value(self.query_one("#md-model-type", Select)),
            model_path=self.query_one("#md-model-path", Input).value.strip(),
            vocab_size_str=self.query_one("#md-model-vocab", Input).value.strip(),
        )

    def _check_dirty(self) -> bool:
        return self._vm.check_dirty(self._collect_raw())

    def _collect(self) -> dict[str, Any] | None:
        result = self._collect_raw()
        if result is None:
            self.app.notify(ModelViewModel.validation_message(), severity="error")
        return result
