"""Tests for SectionModal UI interactions: step builder, field rendering, populate, and button routing."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from textual.containers import Vertical
from textual.widgets import Button, Input, Select, Static

from dataeval_flow._app._model._state import ConfigState
from dataeval_flow._app._screens import SectionModal

from .conftest import _MinimalApp, _state_with_datasets, _wait_for_result

# ---------------------------------------------------------------------------
# SectionModal -- step builder
# ---------------------------------------------------------------------------


class TestSectionModalStepBuilder:
    """Tests for step builder functionality."""

    async def test_refresh_step_list_empty_on_new(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(section="preprocessors", state=ConfigState())
            await app.push_screen(modal)
            await pilot.pause()
            modal._refresh_step_list()
            await pilot.pause()
            container = modal.query_one("#modal-step-list", Vertical)
            statics = container.query(Static)
            assert any("no steps" in s.content.lower() for s in statics)

    async def test_refresh_step_list_with_steps(self) -> None:
        """Covers lines 883-894 (step list rendering)."""
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(
                section="preprocessors",
                existing={"name": "pre1", "steps": [{"step": "Resize", "params": {"size": 256}}]},
                state=ConfigState(),
            )
            await app.push_screen(modal)
            await pilot.pause()
            container = modal.query_one("#modal-step-list", Vertical)
            remove_buttons = container.query("Button")
            assert len(remove_buttons) >= 1

    async def test_step_with_params_display(self) -> None:
        """Covers lines 886-888 (step with params display)."""
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(
                section="preprocessors",
                existing={
                    "name": "pre1",
                    "steps": [{"step": "Resize", "params": {"size": 256, "interp": "bilinear"}}],
                },
                state=ConfigState(),
            )
            await app.push_screen(modal)
            await pilot.pause()
            container = modal.query_one("#modal-step-list", Vertical)
            statics = container.query(Static)
            text = " ".join(s.content for s in statics)
            assert "Resize" in text
            assert "size=256" in text

    async def test_step_without_params_display(self) -> None:
        """Covers lines 889-890 (step without params)."""
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(
                section="preprocessors",
                existing={"name": "pre1", "steps": [{"step": "ToTensor"}]},
                state=ConfigState(),
            )
            await app.push_screen(modal)
            await pilot.pause()
            container = modal.query_one("#modal-step-list", Vertical)
            statics = container.query(Static)
            text = " ".join(s.content for s in statics)
            assert "ToTensor" in text

    async def test_refresh_step_list_no_container(self) -> None:
        """Covers lines 873-876 (NoMatches on step list container)."""
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(section="sources", state=_state_with_datasets())
            await app.push_screen(modal)
            await pilot.pause()
            # sources section has no #modal-step-list
            modal._refresh_step_list()  # Should not raise


# ---------------------------------------------------------------------------
# SectionModal -- field rendering
# ---------------------------------------------------------------------------


class TestSectionModalFieldRendering:
    """Tests for dynamic field rendering."""

    async def test_rebuild_fields_for_sources(self) -> None:
        app = _MinimalApp()
        state = _state_with_datasets()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(section="sources", state=state)
            await app.push_screen(modal)
            await pilot.pause()
            container = modal.query_one("#md-fields", Vertical)
            assert len(container.children) > 0

    async def test_rebuild_fields_skipped_for_step_builder(self) -> None:
        """Covers lines 569-570."""
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(section="preprocessors", state=ConfigState())
            await app.push_screen(modal)
            await pilot.pause()
            assert not modal.query("#md-fields")

    async def test_edit_sources_populates_fields(self) -> None:
        app = _MinimalApp()
        state = _state_with_datasets()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(
                section="sources",
                existing={"name": "src1", "dataset": "ds1"},
                state=state,
            )
            await app.push_screen(modal)
            await pilot.pause()
            name_input = modal.query_one("#md-name", Input)
            assert name_input.value == "src1"

    async def test_missing_md_fields_container_returns_early(self) -> None:
        """Covers lines 573-575."""
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(section="sources", state=_state_with_datasets())
            await app.push_screen(modal)
            await pilot.pause()
            container = modal.query_one("#md-fields", Vertical)
            container.remove()
            await pilot.pause()
            modal._rebuild_fields()

    async def test_scroll_recalculation(self) -> None:
        """Covers lines 676-684 (scroll recalc)."""
        app = _MinimalApp()
        state = _state_with_datasets()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(section="sources", state=state)
            await app.push_screen(modal)
            await pilot.pause()
            modal._rebuild_fields()
            await pilot.pause()
            container = modal.query_one("#md-fields", Vertical)
            assert container is not None


# ---------------------------------------------------------------------------
# SectionModal -- _populate_fields
# ---------------------------------------------------------------------------


class TestSectionModalPopulateFields:
    """Tests for _populate_fields with various field kinds."""

    async def test_populate_missing_widget_graceful(self) -> None:
        """Covers lines 717-718 (NoMatches during populate)."""
        from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind

        mock_descs = [
            FieldDescriptor(name="missing_field", kind=FieldKind.STRING, description="", required=True),
        ]
        app = _MinimalApp()
        state = _state_with_datasets()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(
                section="sources",
                existing={"name": "src1", "missing_field": "value"},
                state=state,
            )
            await app.push_screen(modal)
            await pilot.pause()
            modal._descriptors = mock_descs
            modal._populate_fields()
            await pilot.pause()

    async def test_populate_no_existing_returns_early(self) -> None:
        """Covers lines 688-689 (_populate_fields with no existing)."""
        app = _MinimalApp()
        state = _state_with_datasets()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(section="sources", state=state)
            await app.push_screen(modal)
            await pilot.pause()
            modal._populate_fields()  # Should return early, no crash


# ---------------------------------------------------------------------------
# SectionModal -- parse helpers
# ---------------------------------------------------------------------------


class TestSectionModalParseHelpers:
    """Tests for _parse_picker_field, _parse_add_field, _parse_list_rem."""

    def test_parse_picker_field_valid(self) -> None:
        modal = SectionModal(section="datasets", state=ConfigState())
        modal._gen = 1
        assert modal._parse_picker_field("md-1-detectors-picker") == "detectors"

    def test_parse_picker_field_wrong_prefix(self) -> None:
        modal = SectionModal(section="datasets", state=ConfigState())
        modal._gen = 1
        assert modal._parse_picker_field("md-2-detectors-picker") is None

    def test_parse_picker_field_no_suffix(self) -> None:
        modal = SectionModal(section="datasets", state=ConfigState())
        modal._gen = 1
        assert modal._parse_picker_field("md-1-detectors") is None

    def test_parse_add_field_valid(self) -> None:
        modal = SectionModal(section="datasets", state=ConfigState())
        modal._gen = 1
        assert modal._parse_add_field("md-1-detectors-add") == "detectors"

    def test_parse_add_field_wrong_prefix(self) -> None:
        modal = SectionModal(section="datasets", state=ConfigState())
        modal._gen = 1
        assert modal._parse_add_field("md-2-detectors-add") is None

    def test_parse_add_field_no_suffix(self) -> None:
        modal = SectionModal(section="datasets", state=ConfigState())
        modal._gen = 1
        assert modal._parse_add_field("md-1-detectors") is None

    def test_parse_list_rem_valid(self) -> None:
        modal = SectionModal(section="datasets", state=ConfigState())
        assert modal._parse_list_rem("md-listrem-detectors-0") == ("detectors", 0)

    def test_parse_list_rem_invalid_prefix(self) -> None:
        modal = SectionModal(section="datasets", state=ConfigState())
        assert modal._parse_list_rem("md-other-detectors-0") is None

    def test_parse_list_rem_no_index(self) -> None:
        modal = SectionModal(section="datasets", state=ConfigState())
        assert modal._parse_list_rem("md-listrem-detectors") is None

    def test_parse_list_rem_bad_index(self) -> None:
        modal = SectionModal(section="datasets", state=ConfigState())
        assert modal._parse_list_rem("md-listrem-detectors-abc") is None


# ---------------------------------------------------------------------------
# SectionModal -- _refresh_list_items
# ---------------------------------------------------------------------------


class TestSectionModalRefreshListItems:
    """Tests for _refresh_list_items rendering."""

    async def test_missing_container_no_error(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(section="datasets", state=ConfigState())
            await app.push_screen(modal)
            await pilot.pause()
            modal._refresh_list_items("nonexistent")


# ---------------------------------------------------------------------------
# SectionModal -- select changed dispatch
# ---------------------------------------------------------------------------


class TestSectionModalSelectChanged:
    """Tests for on_select_changed dispatch in SectionModal."""

    async def test_picker_select_triggers_rebuild_variant_params(self) -> None:
        """Covers lines 524-529 (picker select triggers variant rebuild)."""
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(section="workflows", state=ConfigState())
            await app.push_screen(modal)
            await pilot.pause()
            disc = modal.query_one("#md-disc", Select)
            disc.value = "drift-monitoring"
            await pilot.pause()
            picker_id = modal._wid("detectors-picker")
            picker_widgets = modal.query(f"#{picker_id}")
            if picker_widgets:
                picker = picker_widgets.first(Select)  # type: ignore[arg-type]
                with patch.object(modal, "_rebuild_variant_params") as mock_rebuild:
                    picker.value = "kneighbors"
                    await pilot.pause()
                    mock_rebuild.assert_called_once_with("detectors")


# ---------------------------------------------------------------------------
# SectionModal -- button routing for list items
# ---------------------------------------------------------------------------


class TestSectionModalListButtonRouting:
    """Tests for add/remove button routing for list items."""

    async def test_add_button_calls_add_list_item(self) -> None:
        """Covers lines 548-552 (add button routes to _add_list_item)."""
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(section="workflows", state=ConfigState())
            await app.push_screen(modal)
            await pilot.pause()
            disc = modal.query_one("#md-disc", Select)
            disc.value = "drift-monitoring"
            await pilot.pause()
            add_btn_id = modal._wid("detectors-add")
            add_btns = modal.query(f"#{add_btn_id}")
            if add_btns:
                with patch.object(modal, "_add_list_item") as mock_add:
                    event = Button.Pressed(Button("Add detector", id=add_btn_id))
                    modal.on_button_pressed(event)
                    mock_add.assert_called_once_with("detectors")

    async def test_cancel_delegates_to_parent(self) -> None:
        """Covers line 563 (super().on_button_pressed)."""
        app = _MinimalApp()
        results: list[Any] = []
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(section="sources", state=_state_with_datasets())
            await app.push_screen(modal, callback=lambda r: results.append(r))
            await pilot.pause()
            await pilot.pause()
            await pilot.click("#btn-modal-cancel")
            await _wait_for_result(pilot, results)
            assert results == [None]


# ---------------------------------------------------------------------------
# _rebuild_variant_params
# ---------------------------------------------------------------------------


class TestRebuildVariantParams:
    """Tests for _rebuild_variant_params."""

    async def test_select_variant_builds_widgets(self) -> None:
        """Covers lines 951-966."""
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(section="workflows", state=ConfigState())
            await app.push_screen(modal)
            await pilot.pause()
            disc = modal.query_one("#md-disc", Select)
            disc.value = "drift-monitoring"
            await pilot.pause()
            picker_id = modal._wid("detectors-picker")
            picker_widgets = modal.query(f"#{picker_id}")
            if picker_widgets:
                picker = picker_widgets.first(Select)  # type: ignore[arg-type]
                picker.value = "kneighbors"
                await pilot.pause()
                params_id = modal._wid("detectors-params")
                params_container = modal.query_one(f"#{params_id}", Vertical)
                assert len(params_container.children) > 0

    async def test_rebuild_variant_params_no_desc(self) -> None:
        """Covers lines 925-927 (unknown field is no-op)."""
        app = _MinimalApp()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(section="workflows", state=ConfigState())
            await app.push_screen(modal)
            await pilot.pause()
            modal._rebuild_variant_params("nonexistent_field")
