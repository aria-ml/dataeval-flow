"""Result detail modal for the dashboard.

Full-screen modal showing metadata, finding summaries, and expandable
detail sections for a single task's ``WorkflowResult``.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Static

from dataeval_flow._app._viewmodel._result_vm import ResultViewModel

__all__ = ["ErrorDetailModal", "ResultDetailModal"]

_SEVERITY_MARKUP: dict[str, str] = {
    "ok": "[green][ok][/green]",
    "info": "[blue][..][/blue]",
    "warning": "[bold red][!!][/bold red]",
}

# Plain-text markers emitted by _summary_line() → Rich-markup replacements
_MARKER_COLORS: list[tuple[str, str]] = [
    ("  [!!]", "  [bold red]\\[!!][/bold red]"),
    ("  [ok]", "  [green]\\[ok][/green]"),
    ("  [..]", "  [blue]\\[..][/blue]"),
]


def _colorize_marker(line: str) -> str:
    """Replace the trailing plain-text severity marker with a colored Rich-markup version."""
    for plain, colored in _MARKER_COLORS:
        if line.endswith(plain):
            return line[: -len(plain)] + colored
    return line


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
}

#rd-title {
    text-style: bold;
    margin: 0 0 1 0;
}

.rd-metadata {
    margin: 0 0 1 0;
    color: $text-muted;
}

.rd-summary-line {
    height: auto;
    padding: 0 1;
}

.rd-health {
    margin: 1 0;
    text-style: bold;
}

.rd-finding-header {
    height: 3;
    padding: 0 1;
    margin: 1 0 0 0;
    background: $boost;
    content-align: left middle;
}

.rd-finding-header:focus {
    background: $accent 20%;
    color: $text;
    border-left: tall $accent;
}

.rd-finding-detail {
    padding: 0 1 0 2;
    height: auto;
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


class _FindingHeader(Static):
    """Clickable finding header — toggles detail expansion."""

    can_focus = True

    def __init__(self, content: str, finding_idx: int, **kw: Any) -> None:
        super().__init__(content, **kw)
        self.finding_idx = finding_idx


class ResultDetailModal(ModalScreen[None]):
    """Full-screen modal showing detailed results for a single task."""

    CSS = _CSS
    BINDINGS = [("escape", "close", "Close")]

    def __init__(self, task_name: str, result: Any, **kw: Any) -> None:
        super().__init__(**kw)
        self._task_name = task_name
        self._result = result
        self._rvm = ResultViewModel(result)
        self._expanded_findings: set[int] = set()
        self._gen: int = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="rd-dialog"):
            yield Static(f"[bold]Result: {self._task_name}[/bold]", id="rd-title", markup=True)
            with VerticalScroll(id="rd-scroll"):
                yield from self._compose_content()
            with Vertical(id="rd-buttons"):
                yield Button("Close", id="btn-rd-close", variant="primary")

    def _compose_content(self) -> ComposeResult:
        # Metadata
        for line in self._rvm.metadata_lines():
            yield Static(f"[dim]{line}[/dim]", classes="rd-metadata", markup=True)

        yield Static("", classes="rd-separator")

        # Summary
        yield Static("[bold]SUMMARY[/bold]", markup=True)
        summaries = self._rvm.finding_summaries()
        for idx, _fs in enumerate(summaries):
            line = self._rvm.finding_summary_markup(idx)
            line = _colorize_marker(line)
            yield Static(f"  {line}", classes="rd-summary-line", markup=True)

        # Health
        health = self._rvm.health_line()
        if "warning" in health.lower():
            yield Static(f"[bold red]  {health}[/bold red]", classes="rd-health", markup=True)
        else:
            yield Static(f"[green]  {health}[/green]", classes="rd-health", markup=True)

        yield Static("", classes="rd-separator")

        # Finding detail sections
        gen = self._gen
        for idx, fs in enumerate(summaries):
            expanded = idx in self._expanded_findings
            arrow = "\u25bc" if expanded else "\u25b6"
            marker = _SEVERITY_MARKUP.get(fs.severity, _SEVERITY_MARKUP["info"])
            header = _FindingHeader(
                f"{arrow} DETAIL: {fs.title}  {marker}",
                finding_idx=idx,
                classes="rd-finding-header",
                id=f"rd-fh-{gen}-{idx}",
                markup=True,
            )
            yield header

            if expanded:
                # Try DataTable for tabular findings
                table_data = self._rvm.finding_table_data(idx)
                if table_data is not None:
                    headers, rows = table_data
                    dt = DataTable(id=f"rd-dt-{gen}-{idx}")
                    yield dt
                    # DataTable columns/rows added in _populate_tables
                else:
                    # Rich markup detail
                    detail_text = self._rvm.finding_detail_markup(idx)
                    if detail_text.strip():
                        yield Static(detail_text, classes="rd-finding-detail", markup=False)

    def on_mount(self) -> None:
        """Populate DataTable widgets after initial compose."""
        self._populate_tables()

    def _populate_tables(self) -> None:
        """Fill DataTable widgets with data for expanded findings."""
        gen = self._gen
        for idx in self._expanded_findings:
            table_data = self._rvm.finding_table_data(idx)
            if table_data is not None:
                headers, rows = table_data
                try:
                    dt = self.query_one(f"#rd-dt-{gen}-{idx}", DataTable)
                    for h in headers:
                        dt.add_column(h)
                    for row in rows:
                        dt.add_row(*row)
                except NoMatches:
                    pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-rd-close":
            self.dismiss(None)

    def on_click(self, event: Any) -> None:
        widget = event.widget
        while widget is not None:
            if isinstance(widget, _FindingHeader):
                self._toggle_finding(widget.finding_idx)
                event.stop()
                return
            if isinstance(widget, VerticalScroll):
                break
            widget = widget.parent

    async def _on_key(self, event: Any) -> None:
        if event.key == "enter":
            focused = self.app.focused
            if isinstance(focused, _FindingHeader):
                self._toggle_finding(focused.finding_idx)
                event.stop()
                event.prevent_default()
                return
        await super()._on_key(event)

    def _toggle_finding(self, idx: int) -> None:
        if idx in self._expanded_findings:
            self._expanded_findings.discard(idx)
        else:
            self._expanded_findings.add(idx)
        self._rebuild_content()

    def _rebuild_content(self) -> None:
        self._gen += 1
        try:
            scroll = self.query_one("#rd-scroll", VerticalScroll)
        except NoMatches:
            return
        scroll.remove_children()
        for widget in self._compose_content():
            scroll.mount(widget)
        self._populate_tables()

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
                f"[bold]FAILED: {self._task_name}[/bold]",
                id="ed-title",
                markup=True,
            )
            yield Static(self._error, id="ed-error")
            with Vertical(id="ed-buttons"):
                yield Button("Close", id="btn-ed-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-ed-close":
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)
