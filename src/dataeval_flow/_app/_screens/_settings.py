"""Settings modal for execution context (F10).

These settings are *not* part of the pipeline JSON/YAML config.
They control how tasks are executed (paths, resource limits, etc.)
and are ephemeral to the dashboard session (or persisted separately).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from dataeval_flow._app._screens._pathpicker import PathPickerScreen

__all__ = ["ExecutionSettings", "SettingsModal"]

_CSS = """
SettingsModal {
    align: center middle;
}

#settings-dialog {
    width: 70;
    height: auto;
    max-height: 80%;
    border: round $accent 40%;
    background: $surface;
    padding: 1 2;
}

#settings-title {
    text-style: bold;
    margin: 0 0 1 0;
}

.settings-label {
    margin: 1 0 0 0;
}

.settings-row {
    height: auto;
    margin: 0;
}

.settings-row Input {
    width: 1fr;
}

.settings-row Button {
    width: auto;
    min-width: 10;
    margin: 0 0 0 1;
}

.settings-hint {
    color: $text-muted;
    margin: 0 0 0 1;
}

#settings-buttons {
    height: auto;
    align: right middle;
    margin: 1 0 0 0;
}
"""


@dataclass
class ExecutionSettings:
    """Execution context settings — not part of pipeline config."""

    data_dir: str = ""
    cache_dir: str = ""
    output_dir: str = ""


class SettingsModal(ModalScreen[ExecutionSettings | None]):
    """Modal for editing execution settings (F10)."""

    CSS = _CSS
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, current: ExecutionSettings, **kw: Any) -> None:
        super().__init__(**kw)
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-dialog"):
            yield Static("[bold]Execution Settings[/bold]", id="settings-title", markup=True)
            yield Static(
                "[dim]These control how tasks run — not saved to pipeline config.[/dim]",
                markup=True,
            )

            yield Label("Data directory", classes="settings-label")
            with Horizontal(classes="settings-row"):
                yield Input(
                    value=self._current.data_dir,
                    placeholder="$DATAEVAL_DATA or working directory",
                    id="input-data-dir",
                )
                yield Button("Browse", id="btn-browse-data-dir")
            yield Static("[dim]Root for resolving relative dataset paths[/dim]", classes="settings-hint", markup=True)

            yield Label("Cache directory", classes="settings-label")
            with Horizontal(classes="settings-row"):
                yield Input(
                    value=self._current.cache_dir,
                    placeholder="(disabled — no caching)",
                    id="input-cache-dir",
                )
                yield Button("Browse", id="btn-browse-cache-dir")
            yield Static(
                "[dim]Disk cache for embeddings, metadata, stats[/dim]",
                classes="settings-hint",
                markup=True,
            )

            yield Label("Output directory", classes="settings-label")
            with Horizontal(classes="settings-row"):
                yield Input(
                    value=self._current.output_dir,
                    placeholder="(console only — no file output)",
                    id="input-output-dir",
                )
                yield Button("Browse", id="btn-browse-output-dir")
            yield Static(
                "[dim]Where to write result.json / result.txt on export[/dim]",
                classes="settings-hint",
                markup=True,
            )

            with Vertical(id="settings-buttons"):
                yield Button("Save", id="btn-settings-save", variant="primary")
                yield Button("Cancel", id="btn-settings-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn-settings-save":
            self.dismiss(self._collect())
        elif btn_id == "btn-settings-cancel":
            self.dismiss(None)
        elif btn_id == "btn-browse-data-dir":
            self._browse_for("input-data-dir")
        elif btn_id == "btn-browse-cache-dir":
            self._browse_for("input-cache-dir")
        elif btn_id == "btn-browse-output-dir":
            self._browse_for("input-output-dir")

    def _browse_for(self, input_id: str) -> None:
        """Open a folder picker and write the result into the input field."""
        current_value = self.query_one(f"#{input_id}", Input).value.strip()
        start = current_value if current_value else "."

        def _on_picked(result: str | None, _input_id: str = input_id) -> None:
            if result is not None:
                self.query_one(f"#{_input_id}", Input).value = result

        self.app.push_screen(PathPickerScreen(start_path=start, mode="folder"), callback=_on_picked)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _collect(self) -> ExecutionSettings:
        return ExecutionSettings(
            data_dir=self.query_one("#input-data-dir", Input).value.strip(),
            cache_dir=self.query_one("#input-cache-dir", Input).value.strip(),
            output_dir=self.query_one("#input-output-dir", Input).value.strip(),
        )

    @staticmethod
    def to_paths(settings: ExecutionSettings) -> tuple[Path | None, Path | None, Path | None]:
        """Convert settings strings to Path | None triples (data, cache, output)."""
        data = Path(settings.data_dir) if settings.data_dir else None
        cache = Path(settings.cache_dir) if settings.cache_dir else None
        output = Path(settings.output_dir) if settings.output_dir else None
        return data, cache, output
