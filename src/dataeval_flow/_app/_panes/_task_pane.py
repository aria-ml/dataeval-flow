"""Task pane — rebuild and targeted card updates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Static

from dataeval_flow._app._panes._widgets import TaskCard, TaskPaneHeader, uid
from dataeval_flow._app._viewmodel._rendering import snippet_task_with_execution

if TYPE_CHECKING:
    from textual.app import App

    from dataeval_flow._app._viewmodel._builder_vm import BuilderViewModel
else:
    App = object


class TaskPaneMixin(App):
    """Mixin providing task pane rebuild logic for FlowApp.

    Expects the host class to also inherit from ``App`` and to provide ``_vm``.
    """

    if TYPE_CHECKING:
        _vm: BuilderViewModel

    def _rebuild_task_pane(self) -> None:
        """Full rebuild of the task pane (structural changes only)."""
        try:
            container = self.query_one("#task-pane", VerticalScroll)
        except NoMatches:
            return

        focused = self.focused
        restore_index: int | None = None
        if isinstance(focused, TaskCard):
            restore_index = focused.fc_index

        container.remove_children()
        tasks = self._vm.items("tasks")
        header = TaskPaneHeader(
            f"[bold]Tasks[/bold] [dim]({len(tasks)})[/dim]",
            classes="task-pane-header",
            id=uid("th"),
            markup=True,
        )
        container.mount(header)

        if not tasks:
            container.mount(Static("[dim]  (no tasks defined)[/dim]"))
        else:
            for idx, task in enumerate(tasks):
                task_name = task.get("name", "")
                execution = self._vm.task_execution(task_name)
                snippet = snippet_task_with_execution(task, execution)
                card = TaskCard(
                    snippet,
                    index=idx,
                    task_name=task_name,
                    classes="task-card",
                    id=uid("tc"),
                    markup=True,
                )
                container.mount(card)

        if restore_index is not None:
            target = next(
                (w for w in container.children if isinstance(w, TaskCard) and w.fc_index == restore_index), None
            )
            if target is not None:
                self.set_focus(target)

    def _update_task_card_by_name(self, task_name: str) -> None:
        """Update a single task card's content without rebuilding the pane."""
        try:
            container = self.query_one("#task-pane", VerticalScroll)
        except NoMatches:
            return
        for child in container.children:
            if isinstance(child, TaskCard) and child.task_name == task_name:
                execution = self._vm.task_execution(task_name)
                task = next((t for t in self._vm.items("tasks") if t.get("name") == task_name), None)
                if task is not None:
                    child.update(snippet_task_with_execution(task, execution))
                return
        # Card not found — structural change happened, full rebuild
        self._rebuild_task_pane()
