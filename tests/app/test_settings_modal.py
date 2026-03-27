"""Tests for _screens._settings: ExecutionSettings, SettingsModal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.widgets import Button, Input

from dataeval_flow._app._screens._pathpicker import PathPickerScreen
from dataeval_flow._app._screens._settings import ExecutionSettings, SettingsModal

from .conftest import _MinimalApp, _wait_for_result

# ---------------------------------------------------------------------------
# ExecutionSettings dataclass
# ---------------------------------------------------------------------------


class TestExecutionSettings:
    def test_defaults(self) -> None:
        s = ExecutionSettings()
        assert s.data_dir == ""
        assert s.cache_dir == ""
        assert s.output_dir == ""

    def test_custom(self) -> None:
        s = ExecutionSettings(data_dir="/data", cache_dir="/cache", output_dir="/out")
        assert s.data_dir == "/data"
        assert s.cache_dir == "/cache"
        assert s.output_dir == "/out"


# ---------------------------------------------------------------------------
# SettingsModal.to_paths  (static method, lines 177-183)
# ---------------------------------------------------------------------------


class TestToPaths:
    def test_all_populated(self) -> None:
        settings = ExecutionSettings(data_dir="/data", cache_dir="/cache", output_dir="/out")
        d, c, o = SettingsModal.to_paths(settings)
        assert d == Path("/data")
        assert c == Path("/cache")
        assert o == Path("/out")

    def test_all_empty(self) -> None:
        d, c, o = SettingsModal.to_paths(ExecutionSettings())
        assert d is None
        assert c is None
        assert o is None

    def test_mixed(self) -> None:
        settings = ExecutionSettings(data_dir="/data", cache_dir="", output_dir="/out")
        d, c, o = SettingsModal.to_paths(settings)
        assert d == Path("/data")
        assert c is None
        assert o == Path("/out")


# ---------------------------------------------------------------------------
# SettingsModal
# ---------------------------------------------------------------------------


class TestSettingsModal:
    """Tests for SettingsModal compose, collect, and button handlers."""

    async def test_compose_renders_fields(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            settings = ExecutionSettings(data_dir="/data", cache_dir="/cache")
            modal = SettingsModal(settings)
            app.push_screen(modal)
            await pilot.pause()
            data_input = modal.query_one("#input-data-dir", Input)
            assert data_input.value == "/data"
            cache_input = modal.query_one("#input-cache-dir", Input)
            assert cache_input.value == "/cache"
            output_input = modal.query_one("#input-output-dir", Input)
            assert output_input.value == ""
            # Buttons
            assert modal.query_one("#btn-settings-save", Button)
            assert modal.query_one("#btn-settings-cancel", Button)
            assert modal.query_one("#btn-browse-data-dir", Button)
            assert modal.query_one("#btn-browse-cache-dir", Button)
            assert modal.query_one("#btn-browse-output-dir", Button)

    # -- _collect ----------------------------------------------------------

    async def test_collect_returns_settings(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = SettingsModal(ExecutionSettings())
            app.push_screen(modal)
            await pilot.pause()
            modal.query_one("#input-data-dir", Input).value = "/new/data"
            modal.query_one("#input-cache-dir", Input).value = "/new/cache"
            modal.query_one("#input-output-dir", Input).value = "/new/output"
            await pilot.pause()
            collected = modal._collect()
            assert isinstance(collected, ExecutionSettings)
            assert collected.data_dir == "/new/data"
            assert collected.cache_dir == "/new/cache"
            assert collected.output_dir == "/new/output"

    async def test_collect_strips_whitespace(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = SettingsModal(ExecutionSettings())
            app.push_screen(modal)
            await pilot.pause()
            modal.query_one("#input-data-dir", Input).value = "  /spaced  "
            await pilot.pause()
            collected = modal._collect()
            assert collected.data_dir == "/spaced"

    # -- Save button -------------------------------------------------------

    async def test_save_button_dismisses_with_settings(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            settings = ExecutionSettings(data_dir="/d")
            modal = SettingsModal(settings)
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            await pilot.click("#btn-settings-save")
            await _wait_for_result(pilot, results)
            assert len(results) == 1
            assert isinstance(results[0], ExecutionSettings)
            assert results[0].data_dir == "/d"

    # -- Cancel button -----------------------------------------------------

    async def test_cancel_button_dismisses_none(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            modal = SettingsModal(ExecutionSettings())
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            await pilot.click("#btn-settings-cancel")
            await _wait_for_result(pilot, results)
            assert results == [None]

    # -- action_cancel (escape key) ----------------------------------------

    async def test_action_cancel_dismisses_none(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            modal = SettingsModal(ExecutionSettings())
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            modal.action_cancel()
            await _wait_for_result(pilot, results)
            assert results == [None]

    # -- Browse buttons (lines 149-154, 156-165) ---------------------------

    async def test_browse_data_dir_pushes_picker(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = SettingsModal(ExecutionSettings(data_dir="/existing"))
            app.push_screen(modal)
            await pilot.pause()
            await pilot.click("#btn-browse-data-dir")
            await pilot.pause()
            top = app.screen
            assert isinstance(top, PathPickerScreen)
            assert top._start_path == "/existing"

    async def test_browse_cache_dir_pushes_picker(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = SettingsModal(ExecutionSettings())
            app.push_screen(modal)
            await pilot.pause()
            await pilot.click("#btn-browse-cache-dir")
            await pilot.pause()
            top = app.screen
            assert isinstance(top, PathPickerScreen)
            # Empty cache_dir -> default start "."
            assert top._start_path == "."

    async def test_browse_output_dir_pushes_picker(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = SettingsModal(ExecutionSettings(output_dir="/out"))
            app.push_screen(modal)
            await pilot.pause()
            await pilot.click("#btn-browse-output-dir")
            await pilot.pause()
            top = app.screen
            assert isinstance(top, PathPickerScreen)
            assert top._start_path == "/out"

    async def test_browse_callback_sets_input(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = SettingsModal(ExecutionSettings())
            app.push_screen(modal)
            await pilot.pause()
            await pilot.click("#btn-browse-data-dir")
            await pilot.pause()
            pp = app.screen
            assert isinstance(pp, PathPickerScreen)
            pp.dismiss("/picked/path")
            await pilot.pause()
            assert modal.query_one("#input-data-dir", Input).value == "/picked/path"

    async def test_browse_callback_none_leaves_input_unchanged(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = SettingsModal(ExecutionSettings(data_dir="/orig"))
            app.push_screen(modal)
            await pilot.pause()
            await pilot.click("#btn-browse-data-dir")
            await pilot.pause()
            pp = app.screen
            assert isinstance(pp, PathPickerScreen)
            pp.dismiss(None)
            await pilot.pause()
            assert modal.query_one("#input-data-dir", Input).value == "/orig"

    # -- on_button_pressed unrecognised button -----------------------------

    async def test_unrecognised_button_ignored(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            modal = SettingsModal(ExecutionSettings())
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            event = Button.Pressed(Button("Other", id="btn-other"))
            modal.on_button_pressed(event)
            await pilot.pause()
            # Should not dismiss
            assert results == []
