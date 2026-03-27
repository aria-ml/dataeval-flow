"""Shared helpers for builder modal tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Static

from dataeval_flow._app._model._state import ConfigState
from dataeval_flow._app._screens._base import ComponentModal


@dataclass
class FakeParam:
    """Minimal stand-in for discover.ParamInfo."""

    name: str
    type_hint: str
    required: bool
    default: Any = None
    choices: list[str] = field(default_factory=list)


class _MinimalApp(App[None]):
    """Thin host app that can push arbitrary screens."""

    def compose(self) -> ComposeResult:
        yield Static("host")


class _TestComponentModal(ComponentModal):
    """Concrete subclass so we can test the ABC-like base."""

    def __init__(
        self,
        existing: dict[str, Any] | None = None,
        raw: dict[str, Any] | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(existing, **kw)
        self._raw = raw
        self._original: dict[str, Any] | None = dict(existing) if existing else None

    def compose(self) -> ComposeResult:
        from textual.containers import VerticalScroll

        with VerticalScroll(classes="modal-dialog"):
            yield Input(id="test-input")
            yield from self.compose_buttons()

    def _collect_raw(self) -> dict[str, Any] | None:
        return self._raw

    def _check_dirty(self) -> bool:
        collected = self._collect_raw()
        if not collected:
            return False
        if not self._original:
            return True
        return collected != self._original

    def _collect(self) -> dict[str, Any] | None:
        return self._raw


class _ParamFormApp(App[None]):
    """Host app for testing param form functions."""

    def compose(self) -> ComposeResult:
        yield Vertical(id="container")


def _state_with_datasets() -> ConfigState:
    """Return a ConfigState pre-populated with a dataset for cross-ref tests."""
    state = ConfigState()
    state.add("datasets", {"name": "ds1", "format": "huggingface", "path": "data"})
    return state


async def _wait_for_result(pilot, results: list) -> None:  # type: ignore[type-arg]
    """Pause until the dismiss callback populates *results*.

    Textual's ``dismiss()`` may take several event-loop cycles before the
    callback fires, so a single ``pilot.pause()`` is not always sufficient.
    """
    for _ in range(10):
        await pilot.pause()
        if results:
            return
