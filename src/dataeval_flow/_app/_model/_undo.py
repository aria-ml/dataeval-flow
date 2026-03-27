"""Undo/redo stack for the configuration builder.

Pure data structure with no UI dependency.  Uses full state snapshots
for simplicity and correctness.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "UndoEntry",
    "UndoStack",
]


@dataclass
class UndoEntry:
    """A snapshot of builder state paired with a human-readable description."""

    state: dict[str, list[dict[str, Any]]]
    description: str


@dataclass
class UndoStack:
    """Fixed-depth undo/redo history using full state snapshots."""

    _undo: list[UndoEntry] = field(default_factory=list)
    _redo: list[UndoEntry] = field(default_factory=list)
    max_depth: int = 50

    def push(self, state: dict[str, list[dict[str, Any]]], description: str) -> None:
        self._undo.append(UndoEntry(state=copy.deepcopy(state), description=description))
        if len(self._undo) > self.max_depth:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, current_state: dict[str, list[dict[str, Any]]]) -> UndoEntry | None:
        if not self._undo:
            return None
        entry = self._undo.pop()
        self._redo.append(UndoEntry(state=copy.deepcopy(current_state), description=entry.description))
        return entry

    def redo(self, current_state: dict[str, list[dict[str, Any]]]) -> UndoEntry | None:
        if not self._redo:
            return None
        entry = self._redo.pop()
        self._undo.append(UndoEntry(state=copy.deepcopy(current_state), description=entry.description))
        return entry

    @property
    def can_undo(self) -> bool:
        return len(self._undo) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo) > 0
