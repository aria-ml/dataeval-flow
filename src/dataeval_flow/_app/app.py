"""Three-pane dashboard application for DataEval Flow.

Layout:
    config sidebar (left) | task pane (center-top) / result pane (center-bottom)

Config editing uses the same ``SectionModal`` as before.  Task execution
and result viewing are layered on in subsequent phases.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Footer, Header, Static

from dataeval_flow._app._panes import (
    PANE_IDS,
    CfgItem,
    CfgSectionHeader,
    ConfigPaneMixin,
    PaneWidget,
    ResultCard,
    ResultPaneMixin,
    TaskCard,
    TaskPaneHeader,
    TaskPaneMixin,
)
from dataeval_flow._app._screens import ExecutionSettings, PathPickerScreen, SectionModal, SettingsModal
from dataeval_flow._app._viewmodel._builder_vm import BuilderViewModel
from dataeval_flow._app._viewmodel._rendering import snippet_task_with_execution

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
#dashboard {
    height: 1fr;
}

/* --- Config sidebar --- */
#config-pane {
    width: 36;
    min-width: 28;
    border-right: tall $surface-lighten-2;
}

.cfg-section-header {
    height: 3;
    padding: 0 1;
    margin: 1 0 0 0;
    background: $boost;
    content-align: left middle;
    text-style: bold;
}
.cfg-section-header:focus {
    background: $accent 20%;
    color: $text;
    border-left: tall $accent;
}

.cfg-item {
    padding: 0 1 0 2;
    height: auto;
    background: $surface;
}
.cfg-item:focus {
    background: $accent 12%;
    border-left: tall $accent;
}

/* --- Main (right) area --- */
#main-pane {
    width: 1fr;
}

/* Task pane (center-top) */
#task-pane {
    height: 1fr;
    min-height: 5;
    border-bottom: tall $surface-lighten-2;
}

.task-pane-header {
    height: 3;
    padding: 0 1;
    margin: 1 0 0 0;
    background: $boost;
    content-align: left middle;
    text-style: bold;
}
.task-pane-header:focus {
    background: $accent 20%;
    color: $text;
    border-left: tall $accent;
}

.task-card {
    padding: 0 1 0 2;
    height: auto;
    background: $surface;
}
.task-card:focus {
    background: $accent 12%;
    border-left: tall $accent;
}

/* Result pane (center-bottom) */
#result-pane {
    height: 2fr;
    min-height: 5;
}

.result-pane-header {
    height: 3;
    padding: 0 1;
    background: $boost;
    content-align: left middle;
    text-style: bold;
}

.result-card {
    padding: 0 1 0 2;
    height: auto;
    background: $surface;
}
.result-card:focus {
    background: $accent 12%;
    border-left: tall $accent;
}
"""


# ---------------------------------------------------------------------------
# Loading screen
# ---------------------------------------------------------------------------


class LoadingScreen(ModalScreen[None]):
    CSS = """
    LoadingScreen { align: center middle; }
    #loading-dialog {
        width: 44; height: 5; border: round $accent 40%;
        background: $surface; padding: 1 2;
        content-align: center middle; text-align: center;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="loading-dialog"):
            yield Static("[dim]Loading...[/dim]", id="loading-message")

    def update_message(self, msg: str) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#loading-message", Static).update(f"[dim]{msg}[/dim]")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


_FLOW_DARK = Theme(
    name="flow-dark",
    primary="#0178D4",
    secondary="#004578",
    accent="#04B5D4",
    warning="#ffa62b",
    error="#ba3c5b",
    success="#4EBF71",
    foreground="#e0e0e0",
    dark=True,
)


class FlowApp(ConfigPaneMixin, TaskPaneMixin, ResultPaneMixin, App):
    """DataEval Flow interactive dashboard."""

    TITLE = "DataEval Flow Dashboard"
    CSS = CSS
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "blur"),
        ("a", "add_item", "Add"),
        ("r", "run", "Run"),
        ("ctrl+n", "new_config", "New"),
        ("ctrl+o", "open_config", "Open"),
        ("ctrl+s", "save_config", "Save"),
        ("ctrl+e", "export_results", "Export"),
        ("ctrl+z", "undo", "Undo"),
        ("ctrl+y", "redo", "Redo"),
        ("f10", "open_settings", "Settings"),
        ("f12", "save_config_as", "SaveAs"),
        Binding("enter", "activate_item", "Open", show=False),
        Binding("space", "space_item", "Toggle", show=False),
        Binding("delete", "delete_item", "Del", show=False),
    ]

    def __init__(
        self,
        config_path: str | Path | None = None,
        data_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.register_theme(_FLOW_DARK)
        self.theme = "flow-dark"
        self._initial_config_path = Path(config_path) if config_path else None
        self._vm = BuilderViewModel(config_path)
        self._settings = ExecutionSettings(
            data_dir=str(data_dir) if data_dir else "",
            cache_dir=str(cache_dir) if cache_dir else "",
        )
        self._editing_category: str = ""
        self._editing_index: int = -1

    @property
    def _data_dir(self) -> Path | None:
        return Path(self._settings.data_dir) if self._settings.data_dir else None

    @property
    def _cache_dir(self) -> Path | None:
        return Path(self._settings.cache_dir) if self._settings.cache_dir else None

    @property
    def _output_dir(self) -> Path | None:
        return Path(self._settings.output_dir) if self._settings.output_dir else None

    # ==================================================================
    # Compose
    # ==================================================================

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="dashboard"):
            with VerticalScroll(id="config-pane"):
                yield Static("[dim]Loading...[/dim]")
            with Vertical(id="main-pane"):
                with VerticalScroll(id="task-pane"):
                    yield Static("[dim]Loading...[/dim]")
                with VerticalScroll(id="result-pane"):
                    yield Static("[dim]  (no results yet)[/dim]")
        yield Footer()

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def on_mount(self) -> None:
        self._loading_screen = LoadingScreen()
        self.push_screen(self._loading_screen)
        self.run_worker(self._deferred_init_worker, thread=True)

    def _deferred_init_worker(self) -> None:
        log = logging.getLogger(__name__)
        loading = self._loading_screen

        try:
            self.call_from_thread(loading.update_message, "Loading workflows...")
            from dataeval_flow.workflow import list_workflows

            list_workflows()

            self.call_from_thread(loading.update_message, "Loading transforms...")
            from dataeval_flow._app._model._discover import list_transforms

            list_transforms()

            self.call_from_thread(loading.update_message, "Loading selections...")
            from dataeval_flow._app._model._discover import list_selection_classes

            list_selection_classes()

            initial = self._initial_config_path
            if initial:
                self.call_from_thread(loading.update_message, "Loading config...")

                def _load_initial() -> None:
                    success, msg = self._vm.load_file(initial)
                    self.sub_title = self._vm.config_file_path
                    self.notify(msg, severity="information" if success else "error")

                self.call_from_thread(_load_initial)
        except Exception:
            log.exception("Failed during deferred initialization")
            self.call_from_thread(self.notify, "Initialization failed — check logs.", severity="error")
        finally:

            def _dismiss() -> None:
                with contextlib.suppress(ScreenStackError):
                    self._loading_screen.dismiss(None)
                self._rebuild_all()

            self.call_from_thread(_dismiss)

    def action_blur(self) -> None:
        if self.focused is not None:
            self.set_focus(None)

    # ==================================================================
    # Pane-aware navigation
    # ==================================================================

    def _get_pane_widgets(self, pane_id: str) -> list[PaneWidget]:
        """Return all focusable PaneWidget children inside a pane."""
        try:
            container = self.query_one(f"#{pane_id}", VerticalScroll)
        except NoMatches:
            return []
        return [c for c in container.children if isinstance(c, PaneWidget)]

    def _focus_within_pane(self, pane_id: str, direction: int) -> None:
        """Move focus up/down within a single pane."""
        items = self._get_pane_widgets(pane_id)
        if not items:
            return
        focused = self.focused
        current_idx: int | None = None
        for i, w in enumerate(items):
            if w is focused:
                current_idx = i
                break
        if current_idx is None:
            self.set_focus(items[0] if direction > 0 else items[-1])
        else:
            new_idx = (current_idx + direction) % len(items)
            self.set_focus(items[new_idx])

    def _cycle_pane(self, direction: int) -> None:
        """Tab to the next/prev pane, focusing the first widget in it."""
        focused = self.focused
        current_pane_idx = 0
        if isinstance(focused, PaneWidget) and focused.pane in PANE_IDS:
            current_pane_idx = PANE_IDS.index(focused.pane)

        for offset in range(1, len(PANE_IDS) + 1):
            next_idx = (current_pane_idx + direction * offset) % len(PANE_IDS)
            items = self._get_pane_widgets(PANE_IDS[next_idx])
            if items:
                self.set_focus(items[0])
                return

    # ==================================================================
    # Rebuild orchestration
    # ==================================================================

    def _rebuild_all(self) -> None:
        """Full rebuild of all three panes (undo/redo/load/new)."""
        self._rebuild_config_pane()
        self._rebuild_task_pane()
        self._rebuild_result_pane()

    # ==================================================================
    # Key actions (config editing — delegate to ViewModel)
    # ==================================================================

    def _focused_category(self) -> str | None:
        focused = self.focused
        if isinstance(focused, (CfgSectionHeader, CfgItem)):
            return focused.fc_category
        if isinstance(focused, (TaskPaneHeader, TaskCard)):
            return "tasks"
        return None

    def action_undo(self) -> None:
        success, msg = self._vm.undo()
        if not success:
            self.notify(msg, severity="warning")
            return
        self._rebuild_all()
        self.notify(msg)

    def action_redo(self) -> None:
        success, msg = self._vm.redo()
        if not success:
            self.notify(msg, severity="warning")
            return
        self._rebuild_all()
        self.notify(msg)

    def action_add_item(self) -> None:
        category = self._focused_category()
        if category:
            self._open_modal(category, None, -1)

    def action_activate_item(self) -> None:
        """Enter key — context-sensitive open/edit/run/view."""
        focused = self.focused
        if isinstance(focused, CfgItem):
            existing = self._vm.get_item(focused.fc_category, focused.fc_index)
            if existing:
                self._open_modal(focused.fc_category, existing, focused.fc_index)
        elif isinstance(focused, CfgSectionHeader):
            self._open_modal(focused.fc_category, None, -1)
        elif isinstance(focused, TaskCard):
            self.action_run_task()
        elif isinstance(focused, TaskPaneHeader):
            self._open_modal("tasks", None, -1)
        elif isinstance(focused, ResultCard):
            self._view_result(focused)

    def action_space_item(self) -> None:
        """Space key — context-sensitive toggle."""
        focused = self.focused
        if isinstance(focused, TaskCard):
            self._toggle_task(focused)

    def _toggle_task(self, card: TaskCard) -> None:
        desc = self._vm.toggle_task(card.fc_index)
        if desc:
            task = self._vm.get_item("tasks", card.fc_index)
            if task:
                execution = self._vm.task_execution(task.get("name", ""))
                card.update(snippet_task_with_execution(task, execution))

    def _view_result(self, card: ResultCard) -> None:
        execution = self._vm.task_execution(card.task_name)
        if execution is None:
            return
        if execution.status == "failed":
            from dataeval_flow._app._screens._detail import ErrorDetailModal

            self.push_screen(ErrorDetailModal(card.task_name, execution.error or "Unknown error"))
            return
        if execution.result is None:
            return
        from dataeval_flow._app._screens._detail import ResultDetailModal

        self.push_screen(ResultDetailModal(card.task_name, execution.result))

    def action_delete_item(self) -> None:
        focused = self.focused
        category: str | None = None
        index: int = -1
        if isinstance(focused, CfgItem):
            category = focused.fc_category
            index = focused.fc_index
        elif isinstance(focused, TaskCard):
            category = "tasks"
            index = focused.fc_index

        if category is None or index < 0:
            return

        outcome = self._vm.delete_item(category, index)
        if outcome is None:
            return
        description, warnings = outcome
        self.notify(f"{description}. ctrl+z to undo.")
        for w in warnings:
            self.notify(w, severity="warning")
        if category == "tasks":
            self._rebuild_task_pane()
        else:
            self._rebuild_config_section(category)

    def action_open_settings(self) -> None:
        """Open execution settings modal (F10)."""

        def _on_result(result: ExecutionSettings | None) -> None:
            if result is not None:
                self._settings = result
                self.notify("Settings updated.")

        self.push_screen(SettingsModal(self._settings), callback=_on_result)

    # ==================================================================
    # Task execution
    # ==================================================================

    def action_run_task(self) -> None:
        """Run the focused task."""
        focused = self.focused
        if not isinstance(focused, TaskCard):
            self.notify("Focus a task card to run it.", severity="warning")
            return
        task = self._vm.get_item("tasks", focused.fc_index)
        if task is None:
            return
        task_name = task.get("name", "")
        if not task_name:
            return

        try:
            config = self._vm.build_pipeline_config()
        except (ValueError, TypeError) as e:
            self.notify(f"Config validation failed: {e}", severity="error")
            return

        self._vm.mark_task_running(task_name)
        execution = self._vm.task_execution(task_name)
        focused.update(snippet_task_with_execution(task, execution))
        self.notify(f"Running task '{task_name}'...")

        self.run_worker(
            lambda: self._execute_task_worker(task_name, config),
            thread=True,
        )

    def action_run_all(self) -> None:
        """Run all enabled tasks sequentially."""
        tasks = self._vm.items("tasks")
        enabled = [t for t in tasks if t.get("enabled", True)]
        if not enabled:
            self.notify("No enabled tasks to run.", severity="warning")
            return

        try:
            config = self._vm.build_pipeline_config()
        except (ValueError, TypeError) as e:
            self.notify(f"Config validation failed: {e}", severity="error")
            return

        task_names = [t.get("name", "") for t in enabled if t.get("name")]
        self.notify(f"Running {len(task_names)} task(s)...")

        self.run_worker(
            lambda: self._execute_all_worker(task_names, config),
            thread=True,
        )

    def _execute_task_worker(self, task_name: str, config: Any) -> None:
        """Worker thread: run a single task and update state."""
        from dataeval_flow.workflow.orchestrator import _run_single_task

        try:
            task_cfg = next(t for t in config.tasks if t.name == task_name)
            result = _run_single_task(task_cfg, config, data_dir=self._data_dir, cache_dir=self._cache_dir)

            def _on_done() -> None:
                self._vm.mark_task_completed(task_name, result)
                self._update_task_card_by_name(task_name)
                self._append_or_update_result(task_name)
                elapsed = result.metadata.execution_time_s
                time_str = f" ({elapsed:.1f}s)" if elapsed is not None else ""
                self.notify(f"Task '{task_name}' completed{time_str}.")

            self.call_from_thread(_on_done)
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)

            def _on_fail() -> None:
                self._vm.mark_task_failed(task_name, error_msg)
                self._update_task_card_by_name(task_name)
                self._append_or_update_result(task_name)
                self.notify(f"Task '{task_name}' failed: {error_msg}", severity="error")

            self.call_from_thread(_on_fail)

    def _execute_all_worker(self, task_names: list[str], config: Any) -> None:
        """Worker thread: run multiple tasks sequentially."""
        from dataeval_flow.workflow.orchestrator import _run_single_task

        for task_name in task_names:

            def _mark_running(name: str = task_name) -> None:
                self._vm.mark_task_running(name)
                self._update_task_card_by_name(name)

            self.call_from_thread(_mark_running)

            try:
                task_cfg = next(t for t in config.tasks if t.name == task_name)
                result = _run_single_task(task_cfg, config, data_dir=self._data_dir, cache_dir=self._cache_dir)

                def _on_done(name: str = task_name, res: Any = result) -> None:
                    self._vm.mark_task_completed(name, res)
                    self._update_task_card_by_name(name)
                    self._append_or_update_result(name)

                self.call_from_thread(_on_done)
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)

                def _on_fail(name: str = task_name, err: str = error_msg) -> None:
                    self._vm.mark_task_failed(name, err)
                    self._update_task_card_by_name(name)
                    self._append_or_update_result(name)

                self.call_from_thread(_on_fail)

        def _on_all_done() -> None:
            completed = len([entry for entry in self._vm.all_executions() if entry.status == "completed"])
            failed = len([entry for entry in self._vm.all_executions() if entry.status == "failed"])
            self.notify(f"Run all: {completed} completed, {failed} failed.")

        self.call_from_thread(_on_all_done)

    def action_export_results(self) -> None:
        """Export all completed results to a directory."""
        if not self._vm.completed_results():
            self.notify("No results to export.", severity="warning")
            return

        if self._output_dir:
            success, msg = self._vm.export_results(self._output_dir)
            self.notify(msg, severity="information" if success else "warning")
            return

        start = self._default_browse_path()

        def _on_result(result: str | None) -> None:
            if result is not None:
                success, msg = self._vm.export_results(Path(result))
                self.notify(msg, severity="information" if success else "warning")

        self.push_screen(PathPickerScreen(start_path=start, mode="folder"), callback=_on_result)

    # ==================================================================
    # File operations (delegate to ViewModel)
    # ==================================================================

    def _default_browse_path(self) -> str:
        """Starting path for file pickers: config parent dir > data_dir > cwd."""
        if self._vm.config_file_path:
            p = Path(self._vm.config_file_path)
            return str(p.parent if p.is_file() else p)
        if self._data_dir:
            return str(self._data_dir)
        return "."

    def action_new_config(self) -> None:
        msg = self._vm.new_config()
        self._rebuild_all()
        self.sub_title = ""
        self.notify(msg)

    def action_open_config(self) -> None:
        start = self._default_browse_path()

        def _on_result(result: str | None) -> None:
            if result is not None:
                success, msg = self._vm.load_file(Path(result))
                self.sub_title = self._vm.config_file_path
                self.notify(msg, severity="information" if success else "error")
                if success:
                    self._rebuild_all()

        self.push_screen(PathPickerScreen(start_path=start, mode="file"), callback=_on_result)

    def action_save_config(self) -> None:
        if self._vm.config_file_path:
            success, msg = self._vm.save_file(Path(self._vm.config_file_path))
            self.sub_title = self._vm.config_file_path
            self.notify(msg, severity="information" if success else "warning")
        else:
            self.action_save_config_as()

    def action_save_config_as(self) -> None:
        start = self._default_browse_path()

        def _on_folder(result: str | None) -> None:
            if result is not None:
                save_path = str(Path(result) / "params.yaml") if Path(result).is_dir() else result
                success, msg = self._vm.save_file(Path(save_path))
                self.sub_title = self._vm.config_file_path
                self.notify(msg, severity="information" if success else "warning")

        self.push_screen(PathPickerScreen(start_path=start, mode="file"), callback=_on_folder)

    # ==================================================================
    # Click routing
    # ==================================================================

    def on_click(self, event: Any) -> None:
        widget = event.widget
        while widget is not None:
            if isinstance(widget, CfgItem):
                existing = self._vm.get_item(widget.fc_category, widget.fc_index)
                if existing:
                    self._open_modal(widget.fc_category, existing, widget.fc_index)
                event.stop()
                return
            if isinstance(widget, TaskCard):
                if hasattr(event, "x") and event.x < 4:
                    self._toggle_task(widget)
                else:
                    self.set_focus(widget)
                    self.action_run_task()
                event.stop()
                return
            if isinstance(widget, ResultCard):
                self._view_result(widget)
                event.stop()
                return
            if isinstance(widget, (VerticalScroll, Vertical, Horizontal)):
                break
            widget = widget.parent

    # ==================================================================
    # Modal launching
    # ==================================================================

    def _open_modal(self, category: str, existing: dict[str, Any] | None, index: int) -> None:
        self._editing_category = category
        self._editing_index = index

        sec_vm = self._vm.create_section_vm(category, existing)
        modal = SectionModal(
            section=category,
            existing=existing,
            section_vm=sec_vm,
            data_dir=self._data_dir,
        )
        self.push_screen(modal, callback=self._on_modal_result)

    def _on_modal_result(self, result: dict | str | None) -> None:
        category = self._editing_category
        index = self._editing_index

        outcome = self._vm.apply_result(category, index, result)
        if outcome is None:
            return

        description, warnings = outcome
        self.notify(description)
        for w in warnings:
            self.notify(w, severity="warning")
        if category == "tasks":
            self._rebuild_task_pane()
        else:
            self._rebuild_config_section(category)


def run_builder(
    config_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """Launch the interactive dashboard."""
    app = FlowApp(config_path=config_path, data_dir=data_dir, cache_dir=cache_dir)
    app.run()
