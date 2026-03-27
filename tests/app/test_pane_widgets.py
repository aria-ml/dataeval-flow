"""Tests for pane widget classes and pane mixin coverage."""

from __future__ import annotations

import pytest

from dataeval_flow._app._panes._widgets import (
    CfgItem,
    CfgSectionHeader,
    PaneWidget,
    ResultCard,
    ResultPaneHeader,
    TaskCard,
    TaskPaneHeader,
    uid,
)

# ---------------------------------------------------------------------------
# uid factory
# ---------------------------------------------------------------------------


class TestUid:
    def test_unique(self):
        a = uid("test")
        b = uid("test")
        assert a != b
        assert a.startswith("test-")

    def test_prefix_preserved(self):
        result = uid("my-prefix")
        assert result.startswith("my-prefix-")


# ---------------------------------------------------------------------------
# Widget constructors
# ---------------------------------------------------------------------------


class TestWidgetConstructors:
    def test_cfg_section_header(self):
        w = CfgSectionHeader("Header", category="datasets", id="ch-1")
        assert w.fc_category == "datasets"
        assert w.pane == "config-pane"

    def test_cfg_item(self):
        w = CfgItem("item text", category="datasets", index=3, id="ci-1")
        assert w.fc_category == "datasets"
        assert w.fc_index == 3
        assert w.pane == "config-pane"

    def test_task_pane_header(self):
        w = TaskPaneHeader("Tasks", id="th-1")
        assert w.fc_category == "tasks"
        assert w.pane == "task-pane"

    def test_task_card(self):
        w = TaskCard("card text", index=0, task_name="t1", id="tc-1")
        assert w.fc_category == "tasks"
        assert w.fc_index == 0
        assert w.task_name == "t1"
        assert w.pane == "task-pane"

    def test_result_pane_header(self):
        w = ResultPaneHeader("Results", id="rh-1")
        assert w.pane == "result-pane"

    def test_result_card(self):
        w = ResultCard("result text", task_name="t1", id="rc-1")
        assert w.task_name == "t1"
        assert w.pane == "result-pane"


# ---------------------------------------------------------------------------
# PaneWidget key handlers (require FlowApp for _focus_within_pane etc.)
# ---------------------------------------------------------------------------


class TestPaneWidgetKeys:
    @pytest.fixture
    def builder_app(self):
        from dataeval_flow._app.app import FlowApp

        return FlowApp()

    async def test_key_down(self, builder_app) -> None:
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            widgets = app._get_pane_widgets("config-pane")
            if widgets:
                app.set_focus(widgets[0])
                await pilot.pause()
                widgets[0].key_down()
                await pilot.pause()
                # After key_down, focus should have moved within the pane
                assert app.focused is not None

    async def test_key_up(self, builder_app) -> None:
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            widgets = app._get_pane_widgets("config-pane")
            if len(widgets) > 1:
                app.set_focus(widgets[1])
                await pilot.pause()
                widgets[1].key_up()
                await pilot.pause()
                assert app.focused is not None

    async def test_key_tab(self, builder_app) -> None:
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            widgets = app._get_pane_widgets("config-pane")
            if widgets:
                app.set_focus(widgets[0])
                await pilot.pause()
                widgets[0].key_tab()
                await pilot.pause()
                # Should have cycled to a different pane
                focused = app.focused
                if isinstance(focused, PaneWidget):
                    assert focused.pane != "config-pane" or len(app._get_pane_widgets("task-pane")) == 0

    async def test_key_shift_tab(self, builder_app) -> None:
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            # Focus task pane first
            task_widgets = app._get_pane_widgets("task-pane")
            if task_widgets:
                app.set_focus(task_widgets[0])
                await pilot.pause()
                task_widgets[0].key_shift_tab()
                await pilot.pause()
                focused = app.focused
                assert focused is not None

    async def test_on_key_stops_tab_events(self, builder_app) -> None:
        """Covers _on_key event.stop() and event.prevent_default() for tab/arrow keys."""
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            widgets = app._get_pane_widgets("config-pane")
            if widgets:
                app.set_focus(widgets[0])
                await pilot.pause()
                # Press tab key -- _on_key intercepts it
                await pilot.press("tab")
                await pilot.pause()
                # Press down key
                await pilot.press("down")
                await pilot.pause()
                # Press up key
                await pilot.press("up")
                await pilot.pause()
                # Press shift+tab
                await pilot.press("shift+tab")
                await pilot.pause()


# ---------------------------------------------------------------------------
# ConfigPaneMixin coverage
# ---------------------------------------------------------------------------


class TestConfigPaneCoverage:
    @pytest.fixture
    def builder_app(self):
        from dataeval_flow._app.app import FlowApp

        return FlowApp()

    async def test_rebuild_with_items(self, builder_app) -> None:
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            app._rebuild_config_pane()
            await pilot.pause()
            items = [w for w in app._get_pane_widgets("config-pane") if isinstance(w, CfgItem)]
            assert len(items) >= 1

    async def test_rebuild_section_with_item_focus(self, builder_app) -> None:
        """Focus on a CfgItem, then rebuild the section -- tests save/restore with index."""
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            app._rebuild_config_pane()
            await pilot.pause()
            items = [w for w in app._get_pane_widgets("config-pane") if isinstance(w, CfgItem)]
            if items:
                app.set_focus(items[0])
                await pilot.pause()
                app._rebuild_config_section("datasets")
                await pilot.pause()

    async def test_rebuild_section_focused_header(self, builder_app) -> None:
        """Focus on a CfgSectionHeader, then rebuild that section."""
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_config_pane()
            await pilot.pause()
            headers = [w for w in app._get_pane_widgets("config-pane") if isinstance(w, CfgSectionHeader)]
            if headers:
                app.set_focus(headers[0])
                await pilot.pause()
                app._rebuild_config_section(headers[0].fc_category)
                await pilot.pause()

    async def test_save_focus_no_focus(self, builder_app) -> None:
        """_save_config_focus with None returns the null tuple."""
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            result = app._save_config_focus(None)
            assert result == (None, None, False)

    async def test_restore_focus_none_category(self, builder_app) -> None:
        """_restore_config_focus with None category is a no-op."""
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._restore_config_focus(None, None, False)  # should not raise

    async def test_rebuild_section_focus_different_category(self, builder_app) -> None:
        """Rebuild section when focus is on a different category -- should NOT clear focus."""
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            app._rebuild_config_pane()
            await pilot.pause()
            headers = [w for w in app._get_pane_widgets("config-pane") if isinstance(w, CfgSectionHeader)]
            # Focus on a non-datasets header
            non_ds = [h for h in headers if h.fc_category != "datasets"]
            if non_ds:
                app.set_focus(non_ds[0])
                await pilot.pause()
                # Rebuild datasets section -- should not affect our focus
                app._rebuild_config_section("datasets")
                await pilot.pause()

    async def test_restore_focus_fallback_to_header(self, builder_app) -> None:
        """Restore focus when the target item no longer exists -- should fall back to header."""
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            app._rebuild_config_pane()
            await pilot.pause()
            # Attempt to restore focus to an index that doesn't exist
            app._restore_config_focus("datasets", 99, False)
            await pilot.pause()

    async def test_restore_focus_is_header_flag(self, builder_app) -> None:
        """_restore_config_focus with is_header=True focuses the section header."""
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_config_pane()
            await pilot.pause()
            app._restore_config_focus("datasets", None, True)
            await pilot.pause()


# ---------------------------------------------------------------------------
# TaskPaneMixin coverage
# ---------------------------------------------------------------------------


class TestTaskPaneCoverage:
    @pytest.fixture
    def builder_app(self):
        from dataeval_flow._app.app import FlowApp

        return FlowApp()

    async def test_rebuild_with_tasks(self, builder_app) -> None:
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "s1"})
            app._rebuild_task_pane()
            await pilot.pause()
            cards = [w for w in app._get_pane_widgets("task-pane") if isinstance(w, TaskCard)]
            assert len(cards) >= 1

    async def test_rebuild_with_focus_restore(self, builder_app) -> None:
        """Focus on a TaskCard before rebuild -- should restore focus to same index."""
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "s1"})
            app._rebuild_task_pane()
            await pilot.pause()
            cards = [w for w in app._get_pane_widgets("task-pane") if isinstance(w, TaskCard)]
            if cards:
                app.set_focus(cards[0])
                await pilot.pause()
                app._rebuild_task_pane()
                await pilot.pause()

    async def test_update_task_card_by_name(self, builder_app) -> None:
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "s1"})
            app._rebuild_task_pane()
            await pilot.pause()
            app._update_task_card_by_name("t1")
            await pilot.pause()

    async def test_update_task_card_not_found(self, builder_app) -> None:
        """_update_task_card_by_name with unknown name triggers full rebuild."""
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_task_pane()
            await pilot.pause()
            # No task cards exist, so this should trigger _rebuild_task_pane
            app._update_task_card_by_name("nonexistent")
            await pilot.pause()

    async def test_rebuild_empty_tasks(self, builder_app) -> None:
        """Rebuild with no tasks defined shows placeholder."""
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_task_pane()
            await pilot.pause()
            # Should have a header but no cards
            cards = [w for w in app._get_pane_widgets("task-pane") if isinstance(w, TaskCard)]
            assert len(cards) == 0


# ---------------------------------------------------------------------------
# ResultPaneMixin coverage
# ---------------------------------------------------------------------------


class TestResultPaneCoverage:
    @pytest.fixture
    def builder_app(self):
        from dataeval_flow._app.app import FlowApp

        return FlowApp()

    async def test_rebuild_empty(self, builder_app) -> None:
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_result_pane()
            await pilot.pause()

    async def test_rebuild_with_failed_result(self, builder_app) -> None:
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.mark_task_failed("t1", "some error message")
            app._rebuild_result_pane()
            await pilot.pause()
            cards = [w for w in app._get_pane_widgets("result-pane") if isinstance(w, ResultCard)]
            assert len(cards) >= 1

    async def test_append_or_update_new(self, builder_app) -> None:
        """Append a new result card (task not yet in pane)."""
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_result_pane()
            await pilot.pause()
            app._vm.mark_task_failed("t1", "error")
            app._append_or_update_result("t1")
            await pilot.pause()
            cards = [w for w in app._get_pane_widgets("result-pane") if isinstance(w, ResultCard)]
            assert len(cards) >= 1

    async def test_append_or_update_existing(self, builder_app) -> None:
        """Update an existing result card."""
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.mark_task_failed("t1", "first error")
            app._rebuild_result_pane()
            await pilot.pause()
            # Update existing card
            app._vm.mark_task_failed("t1", "second error")
            app._append_or_update_result("t1")
            await pilot.pause()

    async def test_append_no_execution(self, builder_app) -> None:
        """_append_or_update_result with nonexistent task is a no-op."""
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._append_or_update_result("nonexistent")
            await pilot.pause()

    async def test_update_result_header(self, builder_app) -> None:
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_result_pane()
            await pilot.pause()
            app._update_result_header()
            await pilot.pause()

    async def test_update_result_header_with_results(self, builder_app) -> None:
        """Update header count after adding results."""
        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.mark_task_failed("t1", "err")
            app._vm.mark_task_failed("t2", "err")
            app._rebuild_result_pane()
            await pilot.pause()
            app._update_result_header()
            await pilot.pause()

    async def test_rebuild_with_completed_result(self, builder_app) -> None:
        """Rebuild with a completed (mocked) result for snippet rendering."""
        from unittest.mock import MagicMock

        app = builder_app
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Create a mock result that quacks like WorkflowResult
            mock_result = MagicMock()
            mock_result.metadata.execution_time_s = 1.5
            mock_result.metadata.timestamp = None
            mock_result.metadata.model_id = None
            mock_result.metadata.preprocessor_id = None
            mock_result.metadata.source_descriptions = []
            mock_result.data.report = None
            mock_result.to_dict.return_value = {}
            mock_result.report.return_value = ""
            app._vm.mark_task_completed("t1", mock_result)
            app._rebuild_result_pane()
            await pilot.pause()
            cards = [w for w in app._get_pane_widgets("result-pane") if isinstance(w, ResultCard)]
            assert len(cards) >= 1
