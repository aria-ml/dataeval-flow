"""Tests for _app.modals._base: _select_value, PathPickerScreen, ComponentModal."""

from __future__ import annotations

from pathlib import Path as _Path
from typing import Any

from textual.widgets import Button, Checkbox, DirectoryTree, Input, Select, Static

from dataeval_flow._app._model._item import DELETE_SENTINEL
from dataeval_flow._app._screens import PathPickerScreen, _select_value

from .conftest import _MinimalApp, _TestComponentModal, _wait_for_result

# ---------------------------------------------------------------------------
# _select_value
# ---------------------------------------------------------------------------


class TestSelectValue:
    """Tests for the _select_value helper."""

    async def test_blank_returns_empty(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(80, 24)):
            sel = Select[str]([], id="sv-test")
            await app.mount(sel)
            assert _select_value(sel) == ""

    async def test_cleared_returns_empty(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(80, 24)):
            sel = Select[str]([("a", "a")], id="sv-test2", allow_blank=True)
            await app.mount(sel)
            sel.clear()
            assert _select_value(sel) == ""

    async def test_real_value_returned(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(80, 24)):
            sel = Select[str]([("hello", "hello")], id="sv-test3", value="hello")
            await app.mount(sel)
            assert _select_value(sel) == "hello"

    async def test_none_value_returns_empty(self) -> None:
        """Covers the `v is None` branch."""
        app = _MinimalApp()
        async with app.run_test(size=(80, 24)):
            sel = Select[str]([("a", "a")], id="sv-none", allow_blank=True)
            await app.mount(sel)
            sel.clear()
            assert _select_value(sel) == ""


# ---------------------------------------------------------------------------
# PathPickerScreen
# ---------------------------------------------------------------------------


class TestPathPickerScreen:
    """Tests for PathPickerScreen initialization and compose."""

    def test_init_defaults(self) -> None:
        screen = PathPickerScreen()
        assert screen._start_path == "."
        assert screen._selected_path == "."
        assert screen._mode == "folder"

    def test_init_custom(self) -> None:
        screen = PathPickerScreen(start_path="/test_path", mode="file")
        assert screen._start_path == "/test_path"
        assert screen._selected_path == "/test_path"
        assert screen._mode == "file"

    async def test_compose_folder_mode(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(80, 30)) as pilot:
            screen = PathPickerScreen(start_path="/test_path", mode="folder")
            await app.push_screen(screen)
            await pilot.pause()
            assert screen.query_one("#btn-pp-select", Button)
            assert screen.query_one("#btn-pp-cancel", Button)
            selected_label = screen.query_one("#pp-selected", Static)
            assert "Selected:" in selected_label.content

    async def test_compose_file_mode_title(self) -> None:
        """File mode shows 'Select File' title."""
        app = _MinimalApp()
        async with app.run_test(size=(80, 30)) as pilot:
            screen = PathPickerScreen(start_path=".", mode="file")
            await app.push_screen(screen)
            await pilot.pause()
            assert screen.query_one("#pp-tree")

    async def test_update_selected(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(80, 30)) as pilot:
            screen = PathPickerScreen(start_path=".", mode="folder")
            await app.push_screen(screen)
            await pilot.pause()
            screen._update_selected("/new/path")
            assert screen._selected_path == "/new/path"

    async def test_cancel_button_dismisses_none(self) -> None:
        """Covers line 211->exit (btn-pp-cancel)."""
        app = _MinimalApp()
        results: list[str | None] = []
        async with app.run_test(size=(80, 30)) as pilot:
            screen = PathPickerScreen(start_path=".", mode="folder")
            app.push_screen(screen, callback=lambda r: results.append(r))
            await pilot.pause()
            await pilot.click("#btn-pp-cancel")
            await _wait_for_result(pilot, results)
            assert results == [None]

    async def test_select_button_dismisses_with_path(self) -> None:
        app = _MinimalApp()
        results: list[str | None] = []
        async with app.run_test(size=(80, 30)) as pilot:
            screen = PathPickerScreen(start_path="/mypath", mode="folder")
            app.push_screen(screen, callback=lambda r: results.append(r))
            await pilot.pause()
            await pilot.click("#btn-pp-select")
            await _wait_for_result(pilot, results)
            assert results == ["/mypath"]

    async def test_action_cancel_escape(self) -> None:
        app = _MinimalApp()
        results: list[str | None] = []
        async with app.run_test(size=(80, 30)) as pilot:
            screen = PathPickerScreen(start_path=".", mode="folder")
            app.push_screen(screen, callback=lambda r: results.append(r))
            await pilot.pause()
            screen.action_cancel()
            await _wait_for_result(pilot, results)
            assert results == [None]

    async def test_directory_selected_updates_path(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(80, 30)) as pilot:
            screen = PathPickerScreen(start_path=".", mode="folder")
            await app.push_screen(screen)
            await pilot.pause()
            event = DirectoryTree.DirectorySelected(screen.query_one("#pp-tree", DirectoryTree), _Path("/selected/dir"))
            screen.on_directory_tree_directory_selected(event)
            assert screen._selected_path == "/selected/dir"

    async def test_file_selected_ignored_in_folder_mode(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(80, 30)) as pilot:
            screen = PathPickerScreen(start_path=".", mode="folder")
            await app.push_screen(screen)
            await pilot.pause()
            event = DirectoryTree.FileSelected(screen.query_one("#pp-tree", DirectoryTree), _Path("/a/file.txt"))
            screen.on_directory_tree_file_selected(event)
            assert screen._selected_path == "."

    async def test_file_selected_accepted_in_file_mode(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(80, 30)) as pilot:
            screen = PathPickerScreen(start_path=".", mode="file")
            await app.push_screen(screen)
            await pilot.pause()
            event = DirectoryTree.FileSelected(screen.query_one("#pp-tree", DirectoryTree), _Path("/a/file.txt"))
            screen.on_directory_tree_file_selected(event)
            assert screen._selected_path == "/a/file.txt"

    async def test_unrecognised_button_ignored(self) -> None:
        """Button that is neither select nor cancel is ignored."""
        app = _MinimalApp()
        async with app.run_test(size=(80, 30)) as pilot:
            screen = PathPickerScreen(start_path=".", mode="folder")
            await app.push_screen(screen)
            await pilot.pause()
            event = Button.Pressed(Button("Other", id="btn-other"))
            screen.on_button_pressed(event)
            # Screen should still be active (no dismiss)
            assert screen._selected_path == "."


# ---------------------------------------------------------------------------
# ComponentModal
# ---------------------------------------------------------------------------


class TestComponentModal:
    """Tests for ComponentModal base class behavior."""

    def test_create_mode(self) -> None:
        modal = _TestComponentModal(existing=None)
        assert modal.is_edit_mode is False

    def test_edit_mode(self) -> None:
        modal = _TestComponentModal(existing={"name": "x"})
        assert modal.is_edit_mode is True

    async def test_compose_buttons_create_mode(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(80, 24)) as pilot:
            modal = _TestComponentModal(existing=None)
            await app.push_screen(modal)
            await pilot.pause()
            assert modal.query_one("#btn-modal-ok", Button)
            assert modal.query_one("#btn-modal-cancel", Button)
            assert not modal.query("Button#btn-modal-delete")

    async def test_compose_buttons_edit_mode(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(80, 24)) as pilot:
            modal = _TestComponentModal(existing={"name": "x"})
            await app.push_screen(modal)
            await pilot.pause()
            assert modal.query_one("#btn-modal-ok", Button)
            assert modal.query_one("#btn-modal-cancel", Button)
            assert modal.query_one("#btn-modal-delete", Button)

    async def test_ok_button_starts_disabled(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(80, 24)) as pilot:
            modal = _TestComponentModal(existing=None)
            await app.push_screen(modal)
            await pilot.pause()
            ok_btn = modal.query_one("#btn-modal-ok", Button)
            assert ok_btn.disabled is True

    def test_check_dirty_no_raw(self) -> None:
        modal = _TestComponentModal(existing=None, raw=None)
        assert modal._check_dirty() is False

    def test_check_dirty_new_item(self) -> None:
        modal = _TestComponentModal(existing=None, raw={"name": "new"})
        assert modal._check_dirty() is True

    def test_check_dirty_unchanged(self) -> None:
        modal = _TestComponentModal(existing={"name": "x"}, raw={"name": "x"})
        assert modal._check_dirty() is False

    def test_check_dirty_changed(self) -> None:
        modal = _TestComponentModal(existing={"name": "x"}, raw={"name": "y"})
        assert modal._check_dirty() is True

    async def test_cancel_button_dismisses_none(self) -> None:
        app = _MinimalApp()
        results: list[Any] = []
        async with app.run_test(size=(80, 24)) as pilot:
            modal = _TestComponentModal(existing=None, raw={"a": 1})
            app.push_screen(modal, callback=lambda r: results.append(r))
            await pilot.pause()
            await pilot.click("#btn-modal-cancel")
            await _wait_for_result(pilot, results)
            assert results == [None]

    async def test_delete_button_dismisses_sentinel(self) -> None:
        app = _MinimalApp()
        results: list[Any] = []
        async with app.run_test(size=(80, 24)) as pilot:
            modal = _TestComponentModal(existing={"name": "x"}, raw={"name": "x"})
            app.push_screen(modal, callback=lambda r: results.append(r))
            await pilot.pause()
            await pilot.click("#btn-modal-delete")
            await _wait_for_result(pilot, results)
            assert results == [DELETE_SENTINEL]

    async def test_ok_button_dismisses_with_result(self) -> None:
        app = _MinimalApp()
        results: list[Any] = []
        async with app.run_test(size=(80, 24)) as pilot:
            modal = _TestComponentModal(existing=None, raw={"name": "new"})
            app.push_screen(modal, callback=lambda r: results.append(r))
            await pilot.pause()
            ok = modal.query_one("#btn-modal-ok", Button)
            ok.disabled = False
            await pilot.click("#btn-modal-ok")
            await _wait_for_result(pilot, results)
            assert results == [{"name": "new"}]

    async def test_ok_button_no_dismiss_when_collect_none(self) -> None:
        app = _MinimalApp()
        results: list[Any] = []
        async with app.run_test(size=(80, 24)) as pilot:
            modal = _TestComponentModal(existing=None, raw=None)
            app.push_screen(modal, callback=lambda r: results.append(r))
            await pilot.pause()
            ok = modal.query_one("#btn-modal-ok", Button)
            ok.disabled = False
            await pilot.click("#btn-modal-ok")
            await pilot.pause()
        assert results == []

    async def test_action_cancel(self) -> None:
        app = _MinimalApp()
        results: list[Any] = []
        async with app.run_test(size=(80, 24)) as pilot:
            modal = _TestComponentModal(existing=None)
            app.push_screen(modal, callback=lambda r: results.append(r))
            await pilot.pause()
            modal.action_cancel()
            await _wait_for_result(pilot, results)
            assert results == [None]

    async def test_on_input_changed_updates_ok_state(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(80, 24)) as pilot:
            modal = _TestComponentModal(existing=None, raw={"name": "new"})
            await app.push_screen(modal)
            await pilot.pause()
            inp = modal.query_one("#test-input", Input)
            inp.value = "something"
            await pilot.pause()
            ok = modal.query_one("#btn-modal-ok", Button)
            assert ok.disabled is False

    async def test_on_checkbox_changed_updates_ok_state(self) -> None:
        """Covers line 289 (on_checkbox_changed)."""
        app = _MinimalApp()
        async with app.run_test(size=(80, 24)) as pilot:
            modal = _TestComponentModal(existing=None, raw={"name": "new"})
            await app.push_screen(modal)
            await pilot.pause()
            # Mount a checkbox into the modal
            cb = Checkbox("test", id="cb-test")
            modal.mount(cb)
            await pilot.pause()
            cb.value = True
            await pilot.pause()
            ok = modal.query_one("#btn-modal-ok", Button)
            assert ok.disabled is False

    async def test_update_ok_state_no_button(self) -> None:
        """Covers lines 257-258 (NoMatches on btn-modal-ok)."""
        app = _MinimalApp()
        async with app.run_test(size=(80, 24)) as pilot:
            modal = _TestComponentModal(existing=None, raw={"name": "new"})
            await app.push_screen(modal)
            await pilot.pause()
            # Remove the OK button
            ok = modal.query_one("#btn-modal-ok", Button)
            ok.remove()
            await pilot.pause()
            # _update_ok_state should not raise
            modal._update_ok_state()


# ---------------------------------------------------------------------------
# ComponentModal._browse_for_input  (lines 296-306)
# ---------------------------------------------------------------------------


class TestComponentModalBrowseForInput:
    """Tests for _browse_for_input pushing PathPickerScreen."""

    async def test_browse_pushes_path_picker(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = _TestComponentModal(existing=None, raw={"a": 1})
            await app.push_screen(modal)
            await pilot.pause()
            inp = Input(id="browse-target", value="/some/path")
            modal.mount(inp)
            await pilot.pause()
            modal._browse_for_input("browse-target")
            await pilot.pause()
            top_screen = app.screen
            assert isinstance(top_screen, PathPickerScreen)
            assert top_screen._start_path == "/some/path"

    async def test_browse_default_dot_when_input_empty(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = _TestComponentModal(existing=None, raw={"a": 1})
            await app.push_screen(modal)
            await pilot.pause()
            inp = Input(id="browse-empty", value="")
            modal.mount(inp)
            await pilot.pause()
            modal._browse_for_input("browse-empty")
            await pilot.pause()
            top_screen = app.screen
            assert isinstance(top_screen, PathPickerScreen)
            assert top_screen._start_path == "."

    async def test_browse_default_dot_when_input_missing(self) -> None:
        """Covers lines 296-299 (NoMatches fallback)."""
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = _TestComponentModal(existing=None, raw={"a": 1})
            await app.push_screen(modal)
            await pilot.pause()
            modal._browse_for_input("nonexistent-input-id")
            await pilot.pause()
            top_screen = app.screen
            assert isinstance(top_screen, PathPickerScreen)
            assert top_screen._start_path == "."

    async def test_browse_callback_sets_input_value(self) -> None:
        """Covers lines 301-304 (_on_result callback)."""
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = _TestComponentModal(existing=None, raw={"a": 1})
            await app.push_screen(modal)
            await pilot.pause()
            inp = Input(id="browse-cb", value="")
            modal.mount(inp)
            await pilot.pause()
            modal._browse_for_input("browse-cb")
            await pilot.pause()
            # The top screen is PathPickerScreen, dismiss it with a path
            pp = app.screen
            assert isinstance(pp, PathPickerScreen)
            pp.dismiss("/chosen/path")
            await pilot.pause()
            assert inp.value == "/chosen/path"

    async def test_browse_callback_none_does_not_set(self) -> None:
        """Covers line 302 (result is None branch)."""
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = _TestComponentModal(existing=None, raw={"a": 1})
            await app.push_screen(modal)
            await pilot.pause()
            inp = Input(id="browse-none-cb", value="original")
            modal.mount(inp)
            await pilot.pause()
            modal._browse_for_input("browse-none-cb")
            await pilot.pause()
            pp = app.screen
            assert isinstance(pp, PathPickerScreen)
            pp.dismiss(None)
            await pilot.pause()
            assert inp.value == "original"


# ---------------------------------------------------------------------------
# DELETE_SENTINEL constant
# ---------------------------------------------------------------------------


class TestDeleteSentinel:
    def test_value(self) -> None:
        assert DELETE_SENTINEL == "__DELETE__"
