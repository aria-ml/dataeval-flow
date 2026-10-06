"""Result detail modal for the dashboard.

Full-screen modal showing a single task's ``Result``: its metadata, then its report body as the text report
renders it.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from dataeval_flow._app._viewmodel._result_vm import ResultViewModel

__all__ = ["ErrorDetailModal", "ResultDetailModal"]

# Plain-text markers emitted by summary_line() → Rich-markup replacements
_MARKER_COLORS: list[tuple[str, str]] = [
    ("  [!!]", "  [bold red]\\[!!][/bold red]"),
    ("  [ok]", "  [green]\\[ok][/green]"),
    ("  [..]", "  [blue]\\[..][/blue]"),
]


def _colorize_marker(line: str) -> str:
    """A plain summary line as Rich markup: its trailing severity marker colored, the rest escaped.

    A title or brief can carry the user's own strings, such as a class name, so a bracket in one is
    shown as written rather than read as markup.
    """
    for plain, colored in _MARKER_COLORS:
        if line.endswith(plain):
            return escape(line[: -len(plain)]) + colored
    return escape(line)


_CSS = """
ResultDetailModal {
    align: center middle;
}

#rd-dialog {
    width: 90%;
    height: 90%;
    border: round $accent 40%;
    background: $surface;
    padding: 1 2;
}

#rd-scroll {
    height: 1fr;
    overflow-x: auto;
}

#rd-title {
    text-style: bold;
    margin: 0 0 1 0;
}

.rd-metadata {
    margin: 0 0 1 0;
    color: $text-muted;
}

.rd-finding-detail {
    padding: 0 1 0 2;
    height: auto;
    /* The report's 40-column minimum plus the padding: a narrower modal scrolls it sideways. */
    min-width: 43;
    background: $surface;
}

.rd-separator {
    height: 1;
    margin: 1 0;
    background: $accent 15%;
}

#rd-buttons {
    height: auto;
    align: right middle;
    margin: 1 0 0 0;
}
"""


class ResultDetailModal(ModalScreen[None]):
    """Full-screen modal showing detailed results for a single task."""

    CSS = _CSS
    BINDINGS = [("escape", "close", "Close")]

    def __init__(self, task_name: str, result: Any, **kw: Any) -> None:
        super().__init__(**kw)
        self._task_name = task_name
        self._result = result
        self._rvm = ResultViewModel(result)

    def compose(self) -> ComposeResult:
        with Vertical(id="rd-dialog"):
            yield Static(f"[bold]Result: {escape(self._task_name)}[/bold]", id="rd-title", markup=True)
            with VerticalScroll(id="rd-scroll"):
                yield from self._compose_content()
            with Vertical(id="rd-buttons"):
                yield Button("Close", id="btn-rd-close", variant="primary")

    def _compose_content(self) -> ComposeResult:
        for line in self._rvm.metadata_lines():
            yield Static(f"[dim]{escape(line)}[/dim]", classes="rd-metadata", markup=True)
        yield Static("", classes="rd-separator")
        # An evaluator's determinations, a workflow's steps or a matrix's runs, as the text report renders them.
        yield Static(self._rvm.output_text(), classes="rd-finding-detail", markup=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-rd-close":
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Error detail modal (for failed tasks)
# ---------------------------------------------------------------------------

_ERROR_CSS = """
ErrorDetailModal {
    align: center middle;
}

#ed-dialog {
    width: 80%;
    height: auto;
    max-height: 70%;
    border: round $error 50%;
    background: $surface;
    padding: 1 2;
}

#ed-title {
    text-style: bold;
    color: $error;
    margin: 0 0 1 0;
}

#ed-error {
    height: auto;
    max-height: 40;
    padding: 1;
    background: $boost;
    margin: 1 0;
}

#ed-buttons {
    height: auto;
    align: right middle;
    margin: 1 0 0 0;
}
"""


class ErrorDetailModal(ModalScreen[None]):
    """Modal showing error details for a failed task."""

    CSS = _ERROR_CSS
    BINDINGS = [("escape", "close", "Close")]

    def __init__(self, task_name: str, error: str, **kw: Any) -> None:
        super().__init__(**kw)
        self._task_name = task_name
        self._error = error

    def compose(self) -> ComposeResult:
        with Vertical(id="ed-dialog"):
            yield Static(
                f"[bold]FAILED: {escape(self._task_name)}[/bold]",
                id="ed-title",
                markup=True,
            )
            yield Static(self._error, id="ed-error", markup=False)
            with Vertical(id="ed-buttons"):
                yield Button("Close", id="btn-ed-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-ed-close":
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)
