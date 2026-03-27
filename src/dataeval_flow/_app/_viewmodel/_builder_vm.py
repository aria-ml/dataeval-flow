"""ViewModel for the main builder application.

Orchestrates ConfigState, ExecutionState, UndoStack, and file I/O.
No UI dependencies.  The View (FlowApp) calls methods here and
translates return values into notifications and widget updates.
"""

from __future__ import annotations

import json as json_mod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dataeval_flow._app._model._execution import ExecutionState, TaskExecution
from dataeval_flow._app._model._item import DELETE_SENTINEL
from dataeval_flow._app._model._registry import SECTIONS
from dataeval_flow._app._model._state import ConfigState
from dataeval_flow._app._model._undo import UndoStack
from dataeval_flow._app._viewmodel._rendering import _item_to_yaml_snippet, _snippet_task
from dataeval_flow._app._viewmodel._section_vm import SectionViewModel

if TYPE_CHECKING:
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.workflow import WorkflowResult

__all__ = ["BuilderViewModel"]


class BuilderViewModel:
    """ViewModel for the main builder application."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._state = ConfigState()
        self._execution = ExecutionState()
        self.history = UndoStack()
        self.config_file_path: str = str(config_path) if config_path else ""

    # -- Queries -----------------------------------------------------------

    @property
    def sections(self) -> list[tuple[str, str]]:
        """Return the ordered list of (key, title) section pairs."""
        return SECTIONS

    def section_data(self) -> list[tuple[str, str, list[dict[str, Any]]]]:
        """Return ``(key, title, items)`` for every section."""
        return [(key, title, self._state.items(key)) for key, title in SECTIONS]

    def get_item(self, section: str, index: int) -> dict[str, Any] | None:
        """Return item at *index* in *section*, or ``None``."""
        return self._state.get(section, index)

    def items(self, section: str) -> list[dict[str, Any]]:
        """Return a shallow copy of items in *section*."""
        return self._state.items(section)

    def names(self, section: str) -> list[str]:
        """Return the names of all items in *section*."""
        return self._state.names(section)

    def count(self, section: str) -> int:
        """Return the number of items in *section*."""
        return self._state.count(section)

    def is_empty(self) -> bool:
        """Return ``True`` if all sections are empty."""
        return self._state.is_empty()

    def to_dict(self) -> dict[str, Any]:
        """Export state as a plain dict suitable for YAML serialization."""
        return self._state.to_dict()

    def validate_item(self, section: str, data: dict[str, Any]) -> list[str]:
        """Validate a single item dict. Returns a list of error strings."""
        return self._state.validate_item(section, data)

    def validate_all(self) -> list[str]:
        """Validate the full config. Returns errors."""
        return self._state.validate_all()

    # -- Rendering ---------------------------------------------------------

    def item_snippet(self, category: str, item: dict[str, Any]) -> str:
        """Render a config item as a Rich-markup snippet string."""
        return _item_to_yaml_snippet(category, item)

    def task_snippet(self, task: dict[str, Any]) -> str:
        """Render a task item snippet (used for inline toggle updates)."""
        return _snippet_task(task)

    # -- Factory -----------------------------------------------------------

    def create_section_vm(
        self,
        section: str,
        existing: dict[str, Any] | None = None,
    ) -> SectionViewModel:
        """Create a SectionViewModel bound to this builder's state."""
        return SectionViewModel(section, existing, self._state)

    # -- Undo / Redo -------------------------------------------------------

    def snapshot(self, description: str) -> None:
        """Save a state snapshot for undo."""
        self.history.push(self._state.snapshot(), description)

    def undo(self) -> tuple[bool, str]:
        """Undo the last action. Returns ``(success, message)``."""
        entry = self.history.undo(self._state.snapshot())
        if entry is None:
            return False, "Nothing to undo."
        self._state.restore(entry.state)
        return True, f"Undone: {entry.description}"

    def redo(self) -> tuple[bool, str]:
        """Redo the last undone action. Returns ``(success, message)``."""
        entry = self.history.redo(self._state.snapshot())
        if entry is None:
            return False, "Nothing to redo."
        self._state.restore(entry.state)
        return True, f"Redone: {entry.description}"

    # -- Item mutations ----------------------------------------------------

    def toggle_task(self, index: int) -> str | None:
        """Toggle a task's enabled state. Returns description or None."""
        task = self._state.get("tasks", index)
        if task is None:
            return None
        toggling_to = "off" if task.get("enabled", True) else "on"
        desc = f"Toggle task '{task.get('name', '?')}' {toggling_to}"
        self.snapshot(desc)
        task["enabled"] = not task.get("enabled", True)
        return desc

    def delete_item(self, category: str, index: int) -> tuple[str, list[str]] | None:
        """Delete an item. Snapshots before mutation.

        Returns ``(description, warnings)`` or None.
        """
        existing = self._state.get(category, index)
        if existing is None:
            return None
        desc = f"Delete {category[:-1]} '{existing.get('name', '?')}'"
        self.snapshot(desc)
        return self._state.apply_modal_result(category, index, DELETE_SENTINEL)

    def apply_result(self, category: str, index: int, result: dict | str | None) -> tuple[str, list[str]] | None:
        """Apply a modal result (add/update/delete). Snapshots before mutation.

        Returns ``(description, warnings)`` or None.
        """
        if result is None:
            return None

        if result == DELETE_SENTINEL:
            return self.delete_item(category, index)

        # Derive description before mutating
        name = result.get("name", "?") if isinstance(result, dict) else "?"
        action = "Update" if index >= 0 else "Add"
        self.snapshot(f"{action} {category[:-1]} '{name}'")
        return self._state.apply_modal_result(category, index, result)

    # -- File I/O ----------------------------------------------------------

    def new_config(self) -> str:
        """Reset to empty config. Returns status message."""
        if not self._state.is_empty():
            self.snapshot("New config (clear)")
        self._state = ConfigState()
        self.config_file_path = ""
        return "New config."

    def load_file(self, path: Path) -> tuple[bool, str]:
        """Load config from *path*. Returns ``(success, message)``."""
        if not path.exists():
            return False, f"File not found: {path}"
        self.snapshot("Load config file")
        try:
            warning = self._state.load_file(path)
        except (ValueError, TypeError, OSError) as e:
            return False, f"Failed to load config: {e}"
        self.config_file_path = str(path)
        msg = f"Loaded config from {path}"
        if warning:
            msg += f" ({warning})"
        return True, msg

    def save_file(self, path: Path, *, disable_tasks: bool = False) -> tuple[bool, str]:
        """Save config to *path*. Returns ``(success, message)``.

        When *disable_tasks* is ``True``, all tasks are written with
        ``enabled: false`` (for programmatic use).
        """
        if self._state.is_empty():
            return False, "Nothing to save."
        if disable_tasks:
            # Temporarily disable all tasks for export
            tasks = self._state.items("tasks")
            original_states = [(i, t.get("enabled", True)) for i, t in enumerate(tasks)]
            for task in tasks:
                task["enabled"] = False
            try:
                self._state.save_file(path)
            finally:
                # Restore original enabled states
                for idx, enabled in original_states:
                    t = self._state.get("tasks", idx)
                    if t is not None:
                        t["enabled"] = enabled
        else:
            self._state.save_file(path)
        self.config_file_path = str(path)
        return True, f"Saved to {path.resolve()}"

    # -- Config building ---------------------------------------------------

    def build_pipeline_config(self) -> PipelineConfig:
        """Validate and return a ``PipelineConfig`` from current state.

        Raises ``ValueError`` if the config is invalid.
        """
        return self._state.to_pipeline_config()

    # -- Execution ---------------------------------------------------------

    def task_execution(self, name: str) -> TaskExecution | None:
        """Return execution state for task *name*, or ``None``."""
        return self._execution.get(name)

    def mark_task_running(self, name: str) -> TaskExecution:
        """Mark a task as running."""
        return self._execution.mark_running(name)

    def mark_task_completed(self, name: str, result: WorkflowResult[Any, Any]) -> TaskExecution:
        """Mark a task as completed with its result."""
        return self._execution.mark_completed(name, result)

    def mark_task_failed(self, name: str, error: str) -> TaskExecution:
        """Mark a task as failed."""
        return self._execution.mark_failed(name, error)

    def clear_task_execution(self, name: str | None = None) -> None:
        """Clear execution state.  ``None`` clears all."""
        self._execution.clear(name)

    def all_executions(self) -> list[TaskExecution]:
        """Return all execution entries."""
        return self._execution.entries()

    def completed_results(self) -> list[tuple[str, WorkflowResult[Any, Any]]]:
        """Return ``(name, result)`` for all completed tasks."""
        return self._execution.completed_results()

    def export_results(self, output_dir: Path) -> tuple[bool, str]:
        """Write result.json and result.txt for all completed tasks."""
        results = self._execution.completed_results()
        if not results:
            return False, "No completed results to export."

        output_dir.mkdir(parents=True, exist_ok=True)

        merged: dict[str, dict] = {}
        text_parts: list[str] = []
        for name, result in results:
            merged[name] = result.to_dict()
            text_parts.append(result.report(detailed=True))

        (output_dir / "result.json").write_text(json_mod.dumps(merged, indent=2), encoding="utf-8")
        (output_dir / "result.txt").write_text("\n".join(text_parts), encoding="utf-8")
        return True, f"Exported {len(results)} result(s) to {output_dir}"
