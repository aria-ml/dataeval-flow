"""Tests for _screens._model: ModelModal."""

from __future__ import annotations

from textual.widgets import Button, Input, Select

from dataeval_flow._app._screens._model import ModelModal
from dataeval_flow._app._screens._pathpicker import PathPickerScreen

from .conftest import _MinimalApp

# ---------------------------------------------------------------------------
# ModelModal
# ---------------------------------------------------------------------------


class TestModelModal:
    """Tests for ModelModal compose, field toggling, collect, and dirty checking."""

    # -- Init & compose ----------------------------------------------------

    def test_init_new(self) -> None:
        modal = ModelModal()
        assert modal.is_edit_mode is False
        assert modal._existing is None

    def test_init_edit(self) -> None:
        existing = {"name": "m1", "type": "onnx", "model_path": "/p"}
        modal = ModelModal(existing=existing)
        assert modal.is_edit_mode is True

    async def test_compose_new_renders_fields(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            assert modal.query_one("#md-model-name", Input)
            assert modal.query_one("#md-model-type", Select)
            assert modal.query_one("#md-model-path", Input)
            assert modal.query_one("#md-model-vocab", Input)
            # Buttons from compose_buttons
            assert modal.query_one("#btn-modal-ok", Button)
            assert modal.query_one("#btn-modal-cancel", Button)
            # No delete button in create mode
            assert not modal.query("Button#btn-modal-delete")

    async def test_compose_edit_has_delete(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            existing = {"name": "m1", "type": "flatten"}
            modal = ModelModal(existing=existing)
            app.push_screen(modal)
            await pilot.pause()
            assert modal.query_one("#btn-modal-delete", Button)

    async def test_compose_edit_populates_fields(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            existing = {"name": "m1", "type": "onnx", "model_path": "/path/to/model.onnx"}
            modal = ModelModal(existing=existing)
            app.push_screen(modal)
            await pilot.pause()
            assert modal.query_one("#md-model-name", Input).value == "m1"
            assert modal.query_one("#md-model-type", Select).value == "onnx"
            assert modal.query_one("#md-model-path", Input).value == "/path/to/model.onnx"

    async def test_compose_edit_populates_vocab(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            existing = {"name": "m2", "type": "bovw", "vocab_size": 2048}
            modal = ModelModal(existing=existing)
            app.push_screen(modal)
            await pilot.pause()
            assert modal.query_one("#md-model-name", Input).value == "m2"
            assert modal.query_one("#md-model-vocab", Input).value == "2048"

    # -- Field toggling ----------------------------------------------------

    async def test_toggle_fields_onnx_shows_path(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            # Select onnx type
            modal.query_one("#md-model-type", Select).value = "onnx"
            await pilot.pause()
            path_group = modal.query_one("#md-model-path-group")
            assert "hidden" not in path_group.classes

    async def test_toggle_fields_flatten_hides_path(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            modal.query_one("#md-model-type", Select).value = "flatten"
            await pilot.pause()
            path_group = modal.query_one("#md-model-path-group")
            assert "hidden" in path_group.classes

    async def test_toggle_fields_bovw_shows_vocab(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            modal.query_one("#md-model-type", Select).value = "bovw"
            await pilot.pause()
            vocab_group = modal.query_one("#md-model-vocab-group")
            assert "hidden" not in vocab_group.classes

    async def test_toggle_fields_torch_shows_path_hides_vocab(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            modal.query_one("#md-model-type", Select).value = "torch"
            await pilot.pause()
            path_group = modal.query_one("#md-model-path-group")
            vocab_group = modal.query_one("#md-model-vocab-group")
            assert "hidden" not in path_group.classes
            assert "hidden" in vocab_group.classes

    # -- _collect_raw & _collect -------------------------------------------

    async def test_collect_valid_flatten(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            modal.query_one("#md-model-name", Input).value = "m1"
            modal.query_one("#md-model-type", Select).value = "flatten"
            await pilot.pause()
            result = modal._collect()
            assert result is not None
            assert result["name"] == "m1"
            assert result["type"] == "flatten"

    async def test_collect_valid_onnx(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            modal.query_one("#md-model-name", Input).value = "resnet"
            modal.query_one("#md-model-type", Select).value = "onnx"
            modal.query_one("#md-model-path", Input).value = "/models/r.onnx"
            await pilot.pause()
            result = modal._collect()
            assert result is not None
            assert result["model_path"] == "/models/r.onnx"

    async def test_collect_invalid_missing_name(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            # Leave name empty, set type
            modal.query_one("#md-model-type", Select).value = "flatten"
            await pilot.pause()
            result = modal._collect()
            assert result is None

    async def test_collect_invalid_missing_type(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            modal.query_one("#md-model-name", Input).value = "m1"
            # Leave type blank
            await pilot.pause()
            result = modal._collect()
            assert result is None

    async def test_collect_invalid_onnx_no_path(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            modal.query_one("#md-model-name", Input).value = "m1"
            modal.query_one("#md-model-type", Select).value = "onnx"
            # Leave path empty
            await pilot.pause()
            result = modal._collect()
            assert result is None

    async def test_collect_raw_returns_none_when_invalid(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            # Both name and type empty
            result = modal._collect_raw()
            assert result is None

    # -- _check_dirty ------------------------------------------------------

    async def test_check_dirty_new_valid(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            modal.query_one("#md-model-name", Input).value = "m1"
            modal.query_one("#md-model-type", Select).value = "flatten"
            await pilot.pause()
            assert modal._check_dirty() is True

    async def test_check_dirty_invalid_not_dirty(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            # All empty -- _collect_raw returns None
            assert modal._check_dirty() is False

    async def test_check_dirty_unchanged_edit(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            existing = {"name": "m1", "type": "flatten"}
            modal = ModelModal(existing=existing)
            app.push_screen(modal)
            await pilot.pause()
            # Fields already populated with same values -- not dirty
            assert modal._check_dirty() is False

    # -- Browse button -----------------------------------------------------

    async def test_browse_button_pushes_path_picker(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            # Make path field visible by selecting onnx
            modal.query_one("#md-model-type", Select).value = "onnx"
            await pilot.pause()
            # Click browse button
            await pilot.click("#btn-browse-md-model-path")
            await pilot.pause()
            top_screen = app.screen
            assert isinstance(top_screen, PathPickerScreen)

    async def test_browse_button_event_routed(self) -> None:
        """on_button_pressed routes btn-browse-md-model-path to _browse_for_input."""
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            modal.query_one("#md-model-type", Select).value = "onnx"
            await pilot.pause()
            event = Button.Pressed(modal.query_one("#btn-browse-md-model-path", Button))
            modal.on_button_pressed(event)
            await pilot.pause()
            assert isinstance(app.screen, PathPickerScreen)

    # -- on_select_changed -------------------------------------------------

    async def test_select_changed_updates_ok_state(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ModelModal()
            app.push_screen(modal)
            await pilot.pause()
            # Fill name first
            modal.query_one("#md-model-name", Input).value = "m1"
            await pilot.pause()
            # Now select type, which triggers on_select_changed
            modal.query_one("#md-model-type", Select).value = "flatten"
            await pilot.pause()
            ok_btn = modal.query_one("#btn-modal-ok", Button)
            assert ok_btn.disabled is False
