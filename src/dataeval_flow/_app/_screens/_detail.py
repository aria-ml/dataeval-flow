"""Result detail modal for the dashboard.

Full-screen modal showing metadata, finding summaries, and expandable
detail sections for a single task's ``WorkflowResult``.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Static

from dataeval_flow._app._viewmodel._result_vm import ResultViewModel, table_data
from dataeval_flow._blocks import Block, Table
from dataeval_flow._blocks._table import shared_widths
from dataeval_flow._blocks._text import MIN_WIDTH, Frame, render_text

__all__ = ["ErrorDetailModal", "ResultDetailModal"]

_SEVERITY_MARKUP: dict[str, str] = {
    "ok": "[green][ok][/green]",
    "info": "[blue][..][/blue]",
    "warning": "[bold red][!!][/bold red]",
}

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


class _FindingHeader(Static):
    """Clickable finding header — toggles detail expansion."""

    can_focus = True

    def __init__(self, content: str, finding_idx: int, **kw: Any) -> None:
        super().__init__(content, **kw)
        self.finding_idx = finding_idx


class _BlockText(Static):
    """Blocks drawn as text at this widget's width, and drawn again whenever that width changes.

    *layouts* are the column widths shared by the finding's tables with identical columns, so tables
    read against each other line up even when a native table sits between them.
    """

    def __init__(
        self, blocks: Sequence[Block], layouts: Mapping[tuple[Any, ...], tuple[int, ...]] | None = None, **kw: Any
    ) -> None:
        super().__init__("", markup=False, **kw)
        self._blocks = list(blocks)
        self._layouts = dict(layouts or {})
        self._drawn_at = 0
        self.text = ""

    def on_resize(self, _event: events.Resize) -> None:
        width = self.content_size.width
        if width and width != self._drawn_at:
            self._drawn_at = width
            frame = Frame(width=max(width, MIN_WIDTH), depth=2, layouts=self._layouts)
            self.text = "\n".join(render_text(self._blocks, frame))
            self.update(self.text)


class _BlockTable(DataTable[Text]):
    """A data table as a native ``DataTable``, each cell printed as the text report prints it.

    Headers and cells are plain ``Text``: a class or split name is the user's own, so a bracket in it
    is shown as written rather than read as markup.
    """

    def __init__(self, table: Table, **kw: Any) -> None:
        super().__init__(**kw)
        self._table = table

    def on_mount(self) -> None:
        headers, rows = table_data(self._table)
        self.add_columns(*(Text(header) for header in headers))
        for row in rows:
            # A cell may hold several lines, such as one factor per line: a row as tall as its
            # tallest cell shows them all rather than only the first.
            self.add_row(*(Text(cell) for cell in row), height=None)


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
            yield Static(f"[bold]Result: {escape(self._task_name)}[/bold]", id="rd-title", markup=True)
            with VerticalScroll(id="rd-scroll"):
                yield from self._compose_content()
            with Vertical(id="rd-buttons"):
                yield Button("Close", id="btn-rd-close", variant="primary")

    def _compose_content(self) -> ComposeResult:
        # Metadata
        for line in self._rvm.metadata_lines():
            yield Static(f"[dim]{escape(line)}[/dim]", classes="rd-metadata", markup=True)

        yield Static("", classes="rd-separator")

        if self._rvm.shows_output:
            # An evaluator's determinations, or a custom workflow's steps: no SUMMARY, health or severity markers.
            yield Static(self._rvm.output_text(), classes="rd-finding-detail", markup=False)
            return

        # Summary
        yield Static("[bold]SUMMARY[/bold]", markup=True)
        summaries = self._rvm.finding_summaries()
        for idx, _fs in enumerate(summaries):
            line = _colorize_marker(self._rvm.finding_summary_markup(idx))
            yield Static(textwrap.indent(line, "  "), classes="rd-summary-line", markup=True)

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
                f"{arrow} DETAIL: {escape(fs.title)}  {marker}",
                finding_idx=idx,
                classes="rd-finding-header",
                id=f"rd-fh-{gen}-{idx}",
                markup=True,
            )
            yield header

            if expanded:
                # Each data table as a native DataTable; every other run of blocks as text.
                layouts = shared_widths(self._rvm.finding_blocks(idx))
                for position, segment in enumerate(self._rvm.finding_segments(idx)):
                    if isinstance(segment, Table):
                        yield _BlockTable(segment, id=f"rd-dt-{gen}-{idx}-{position}")
                    else:
                        yield _BlockText(segment, layouts, classes="rd-finding-detail")

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
