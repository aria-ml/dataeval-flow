"""Result pane — rebuild, incremental append/update."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Static

from dataeval_flow._app._panes._widgets import PaneWidget, ResultCard, uid

if TYPE_CHECKING:
    from textual.app import App

    from dataeval_flow._app._viewmodel._builder_vm import BuilderViewModel
else:
    App = object


class ResultPaneMixin(App):
    """Mixin providing result pane rebuild logic for FlowApp.

    Expects the host class to also inherit from ``App`` and to provide ``_vm``.
    """

    if TYPE_CHECKING:
        _vm: BuilderViewModel

    def _rebuild_result_pane(self) -> None:
        """Full rebuild of the result pane (used on undo/redo/load/new)."""
        try:
            container = self.query_one("#result-pane", VerticalScroll)
        except NoMatches:
            return

        container.remove_children()
        executions = [e for e in self._vm.all_executions() if e.status in ("completed", "failed")]
        count = len(executions)
        header_text = f"[bold]Results[/bold] [dim]({count})[/dim]" if count else "[bold]Results[/bold]"
        container.mount(Static(header_text, classes="result-pane-header", id=uid("rh"), markup=True))

        if not executions:
            container.mount(Static("[dim]  (no results yet)[/dim]"))
            return

        for entry in executions:
            snippet = self._render_result_snippet(entry)
            card = ResultCard(
                snippet,
                task_name=entry.task_name,
                classes="result-card",
                id=uid("rc"),
                markup=True,
            )
            container.mount(card)

    def _render_result_snippet(self, entry: Any) -> str:
        """Render a result card snippet for a single execution entry."""
        from dataeval_flow._app._screens._detail import _colorize_marker
        from dataeval_flow._app._viewmodel._result_vm import ResultViewModel

        if entry.status == "completed" and entry.result is not None:
            rvm = ResultViewModel(entry.result)
            summary = rvm.summary_line()
            warnings = rvm.warning_count()
            severity_tag = " [bold red][!!][/bold red]" if warnings else " [green][ok][/green]"
            lines = [f"[bold]{entry.task_name}[/bold] — {summary}{severity_tag}"]
            for fi in range(rvm.finding_count()):
                fline = _colorize_marker(rvm.finding_summary_markup(fi))
                lines.append(f"  {fline}")
            return "\n".join(lines)
        error_brief = (entry.error or "unknown error")[:60]
        return f"[bold]{entry.task_name}[/bold] — [bold red]FAILED[/bold red]: {error_brief}"

    def _append_or_update_result(self, task_name: str) -> None:
        """Add or update a single result card without full rebuild."""
        entry = self._vm.task_execution(task_name)
        if entry is None or entry.status not in ("completed", "failed"):
            return

        snippet = self._render_result_snippet(entry)

        # Try to update existing card (re-run of same task)
        try:
            container = self.query_one("#result-pane", VerticalScroll)
        except NoMatches:
            return
        for child in container.children:
            if isinstance(child, ResultCard) and child.task_name == task_name:
                child.update(snippet)
                self._update_result_header()
                return

        # Remove "no results yet" placeholder
        for child in list(container.children):
            is_placeholder = isinstance(child, Static) and not isinstance(child, PaneWidget)
            if is_placeholder and "no results" in str(getattr(child, "content", "")):
                child.remove()
                break

        # Append new card
        card = ResultCard(
            snippet,
            task_name=task_name,
            classes="result-card",
            id=uid("rc"),
            markup=True,
        )
        container.mount(card)
        self._update_result_header()

    def _update_result_header(self) -> None:
        """Update the result pane header count without rebuilding."""
        executions = [e for e in self._vm.all_executions() if e.status in ("completed", "failed")]
        count = len(executions)
        header_text = f"[bold]Results[/bold] [dim]({count})[/dim]" if count else "[bold]Results[/bold]"
        try:
            container = self.query_one("#result-pane", VerticalScroll)
        except NoMatches:
            return
        for child in container.children:
            if isinstance(child, Static) and "result-pane-header" in (child.classes or set()):
                child.update(header_text)
                break
