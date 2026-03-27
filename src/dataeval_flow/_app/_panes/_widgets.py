"""Focusable widget classes shared across panes."""

from __future__ import annotations

from typing import Any

from textual.widgets import Static

# Pane IDs used for Tab-cycling between panes
PANE_IDS = ("config-pane", "task-pane", "result-pane")

_CONFIG_SECTIONS = [
    ("datasets", "Datasets"),
    ("selections", "Selections"),
    ("sources", "Sources"),
    ("preprocessors", "Preprocessors"),
    ("extractors", "Extractors"),
    ("workflows", "Workflows"),
]

SECTION_TITLES: dict[str, str] = dict(_CONFIG_SECTIONS)

# ---------------------------------------------------------------------------
# Global unique-ID factory — avoids Textual DuplicateIds on rebuild
# ---------------------------------------------------------------------------

_next_uid: int = 0


def uid(prefix: str) -> str:
    """Return a globally unique widget ID like ``prefix-42``."""
    global _next_uid
    _next_uid += 1
    return f"{prefix}-{_next_uid}"


# ---------------------------------------------------------------------------
# Focusable pane widgets
# ---------------------------------------------------------------------------


class PaneWidget(Static):
    """Base for focusable widgets that know which pane they live in."""

    can_focus = True
    pane: str = ""

    def key_down(self) -> None:
        self.app._focus_within_pane(self.pane, 1)  # type: ignore[attr-defined]  # noqa: SLF001

    def key_up(self) -> None:
        self.app._focus_within_pane(self.pane, -1)  # type: ignore[attr-defined]  # noqa: SLF001

    def key_tab(self) -> None:
        self.app._cycle_pane(1)  # type: ignore[attr-defined]  # noqa: SLF001

    def key_shift_tab(self) -> None:
        self.app._cycle_pane(-1)  # type: ignore[attr-defined]  # noqa: SLF001

    async def _on_key(self, event: Any) -> None:
        if event.key in ("tab", "shift+tab", "down", "up"):
            event.stop()
            event.prevent_default()
        await super()._on_key(event)


class CfgSectionHeader(PaneWidget):
    pane = "config-pane"

    def __init__(self, content: str, category: str, **kw: Any) -> None:
        super().__init__(content, **kw)
        self.fc_category = category


class CfgItem(PaneWidget):
    pane = "config-pane"

    def __init__(self, content: str, category: str, index: int, **kw: Any) -> None:
        super().__init__(content, **kw)
        self.fc_category = category
        self.fc_index = index


class TaskPaneHeader(PaneWidget):
    pane = "task-pane"

    def __init__(self, content: str, **kw: Any) -> None:
        super().__init__(content, **kw)
        self.fc_category = "tasks"


class TaskCard(PaneWidget):
    pane = "task-pane"

    def __init__(self, content: str, index: int, task_name: str = "", **kw: Any) -> None:
        super().__init__(content, **kw)
        self.fc_category = "tasks"
        self.fc_index = index
        self.task_name = task_name


class ResultPaneHeader(PaneWidget):
    pane = "result-pane"


class ResultCard(PaneWidget):
    pane = "result-pane"

    def __init__(self, content: str, task_name: str, **kw: Any) -> None:
        super().__init__(content, **kw)
        self.task_name = task_name
