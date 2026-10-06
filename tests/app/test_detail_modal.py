"""Tests for _screens._detail: _colorize_marker, ResultDetailModal, ErrorDetailModal."""

from __future__ import annotations

from typing import Any

import pytest
from rich.markup import render
from textual.widgets import Button, Static

from dataeval_flow._app._screens._detail import ErrorDetailModal, ResultDetailModal, _colorize_marker
from dataeval_flow.steps import ChainMetadata, ChainResult, Finding
from tests.workflow_toys import count_result

from .conftest import _MinimalApp, _wait_for_result

pytestmark = pytest.mark.optional

# ---------------------------------------------------------------------------
# _colorize_marker  (lines 35-40)
# ---------------------------------------------------------------------------


class TestColorizeMarker:
    """Pure-function tests for _colorize_marker."""

    def test_ok_marker(self) -> None:
        line = "All checks passed  [ok]"
        result = _colorize_marker(line)
        assert "[green]" in result
        assert result.startswith("All checks passed")

    def test_warning_marker(self) -> None:
        line = "Duplicates found  [!!]"
        result = _colorize_marker(line)
        assert "[bold red]" in result
        assert result.startswith("Duplicates found")

    def test_info_marker(self) -> None:
        line = "Statistics  [..]"
        result = _colorize_marker(line)
        assert "[blue]" in result
        assert result.startswith("Statistics")

    def test_no_marker_returns_unchanged(self) -> None:
        line = "plain text"
        assert _colorize_marker(line) == line

    def test_empty_string(self) -> None:
        assert _colorize_marker("") == ""

    def test_marker_must_be_at_end(self) -> None:
        line = "[ok] at the beginning"
        assert render(_colorize_marker(line)).plain == line

    def test_brackets_in_the_text_show_as_written(self) -> None:
        """A class name is the user's own: a bracket in it is text, never markup."""
        line = "worst: [person], [/x]  [!!]"
        result = _colorize_marker(line)
        assert "[bold red]" in result
        assert render(result).plain == line


# ---------------------------------------------------------------------------
# ErrorDetailModal  (lines 306-333)
# ---------------------------------------------------------------------------


class TestErrorDetailModal:
    def test_init_stores_fields(self) -> None:
        modal = ErrorDetailModal("task_x", "boom")
        assert modal._task_name == "task_x"
        assert modal._error == "boom"

    async def test_a_bracketed_task_name_and_error_show_as_written(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ErrorDetailModal("[/t] task", "KeyError: [/x] [person]")
            app.push_screen(modal)
            await pilot.pause()
            assert str(modal.query_one("#ed-title", Static).render()) == "FAILED: [/t] task"
            assert str(modal.query_one("#ed-error", Static).render()) == "KeyError: [/x] [person]"

    async def test_compose_renders_elements(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ErrorDetailModal("task1", "Something went wrong")
            app.push_screen(modal)
            await pilot.pause()
            # Title, error body, and close button are present
            assert modal.query_one("#ed-title", Static) is not None
            assert modal.query_one("#ed-error", Static) is not None
            assert modal.query_one("#btn-ed-close", Button) is not None

    async def test_close_button_dismisses(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            modal = ErrorDetailModal("task1", "error msg")
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            await pilot.click("#btn-ed-close")
            await _wait_for_result(pilot, results)
            assert results == [None]

    async def test_action_close_dismisses(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            modal = ErrorDetailModal("task1", "error msg")
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            modal.action_close()
            await _wait_for_result(pilot, results)
            assert results == [None]

    async def test_on_button_pressed_non_close_ignored(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            modal = ErrorDetailModal("task1", "error msg")
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            # Fire a button press with unrelated id -- should not dismiss
            event = Button.Pressed(Button("Other", id="btn-other"))
            modal.on_button_pressed(event)
            await pilot.pause()
            assert results == []


# ---------------------------------------------------------------------------
# ResultDetailModal  (lines 124-263)
# ---------------------------------------------------------------------------


class TestResultDetailModal:
    def _result(self) -> ChainResult:
        return count_result(Finding(title="Dup Check", severity="ok"), Finding(title="Coverage", severity="warning"))

    def test_init_stores_fields(self) -> None:
        result = self._result()
        modal = ResultDetailModal("task_a", result)
        assert modal._task_name == "task_a"
        assert modal._result is result

    async def test_bracketed_user_strings_show_as_written(self) -> None:
        """Task names, sources, finding titles and briefs are the user's own: brackets in them are text."""
        app = _MinimalApp()
        finding = Finding(title="[/y] Balance", severity="warning", brief="worst: [person]")
        result = count_result(finding, metadata=ChainMetadata(source_descriptions=["[/s] source"]))
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("[/t] task", result)
            app.push_screen(modal)
            await pilot.pause()
            drawn = [str(widget.render()) for widget in modal.query(Static)]
            assert "Result: [/t] task" in drawn
            assert any("[/s] source" in text for text in drawn)
            assert any("[/y] Balance" in text and "worst: [person]" in text for text in drawn)

    async def test_compose_renders_title_and_close(self) -> None:
        app = _MinimalApp()
        result = self._result()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("task_a", result)
            app.push_screen(modal)
            await pilot.pause()
            # Title and close button are present
            assert modal.query_one("#rd-title", Static) is not None
            assert modal.query_one("#btn-rd-close", Button) is not None

    async def test_close_button_dismisses(self) -> None:
        app = _MinimalApp()
        result = self._result()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            modal = ResultDetailModal("task_a", result)
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            await pilot.click("#btn-rd-close")
            await _wait_for_result(pilot, results)
            assert results == [None]

    async def test_action_close_dismisses(self) -> None:
        app = _MinimalApp()
        result = self._result()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            modal = ResultDetailModal("task_a", result)
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            modal.action_close()
            await _wait_for_result(pilot, results)
            assert results == [None]

    async def test_on_button_pressed_non_close_ignored(self) -> None:
        app = _MinimalApp()
        result = self._result()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            modal = ResultDetailModal("task_a", result)
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            event = Button.Pressed(Button("Other", id="btn-other"))
            modal.on_button_pressed(event)
            await pilot.pause()
            assert results == []


class TestChainDetail:
    async def test_a_chain_shows_its_steps(self, chain_results: dict[str, Any]) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("mixed", chain_results["mixed"])
            app.push_screen(modal)
            await pilot.pause()
            drawn = "\n".join(str(widget.render()) for widget in modal.query(Static))
            assert "clean/at-least" in drawn
            assert "RuntimeError: boom on a" in drawn
