# ---------------------------------------------------------------------------
# Path picker (file/folder selection modal)
# ---------------------------------------------------------------------------


import contextlib
from typing import Any

from textual.app import ComposeResult, ScreenStackError
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Static


class PathPickerScreen(ModalScreen[str | None]):
    """Unified file/folder picker.

    - ``"file"``: selects files and directories (for loading configs).
    - ``"folder"``: selects directories only (for browse buttons / save).
    """

    CSS = """
    PathPickerScreen { align: center middle; }
    #pp-dialog { width: 70; height: 30; border: round $accent 40%; background: $surface; padding: 1 2; }
    #pp-tree { height: 1fr; margin: 1 0; }
    #pp-selected { height: auto; margin: 0 0 1 0; }
    #pp-buttons { height: auto; align: right middle; }
    """
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, start_path: str = ".", mode: str = "folder", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._start_path = start_path
        self._selected_path = start_path
        self._mode = mode

    def compose(self) -> ComposeResult:
        title = "Select File" if self._mode == "file" else "Select Directory"
        with Vertical(id="pp-dialog"):
            yield Static(f"[bold]{title}[/bold]")
            yield DirectoryTree(self._start_path, id="pp-tree")
            yield Static(f"Selected: {self._start_path}", id="pp-selected")
            with Horizontal(id="pp-buttons"):
                yield Button("Select", id="btn-pp-select", variant="primary")
                yield Button("Cancel", id="btn-pp-cancel")

    def _update_selected(self, path: str) -> None:
        self._selected_path = path
        with contextlib.suppress(NoMatches):
            self.query_one("#pp-selected", Static).update(f"Selected: {self._selected_path}")

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        if self._mode == "file":
            self._update_selected(str(event.path))

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self._update_selected(str(event.path))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-pp-select":
            self.dismiss(self._selected_path)
        elif event.button.id == "btn-pp-cancel":
            self.dismiss(None)

    def action_cancel(self) -> None:
        with contextlib.suppress(ScreenStackError):
            self.dismiss(None)
