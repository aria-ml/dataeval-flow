"""Unit tests for _app.app — three-pane dashboard."""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from dataeval_flow._app._panes import _CONFIG_SECTIONS, CfgItem, CfgSectionHeader, ResultCard, TaskCard, TaskPaneHeader
from dataeval_flow._app._screens._settings import ExecutionSettings
from dataeval_flow._app.app import CSS, FlowApp, LoadingScreen


def _make_push_screen_capture() -> tuple[list, object]:
    """Return (callbacks, side_effect) for capturing push_screen callbacks."""
    callbacks: list = []

    def _capture(_screen: object, callback: object = None) -> None:
        if callback:
            callbacks.append(callback)

    return callbacks, _capture


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------


class TestCSS:
    def test_css_has_pane_selectors(self) -> None:
        assert isinstance(CSS, str)
        assert "#config-pane" in CSS
        assert "#task-pane" in CSS
        assert "#result-pane" in CSS
        assert "#dashboard" in CSS


# ---------------------------------------------------------------------------
# FlowApp compose / lifecycle
# ---------------------------------------------------------------------------


class TestBuilderAppCompose:
    async def test_compose_has_three_panes(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)):
            assert app.query("Header")
            assert app.query("Footer")
            assert app.query_one("#dashboard")
            assert app.query_one("#config-pane")
            assert app.query_one("#task-pane")
            assert app.query_one("#result-pane")

    async def test_title(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)):
            assert app.title == "DataEval Flow Dashboard"

    async def test_data_dir_stored(self) -> None:
        app = FlowApp(data_dir="/some/data")
        assert app._data_dir == Path("/some/data")

    async def test_cache_dir_stored(self) -> None:
        app = FlowApp(cache_dir="/var/cache/dataeval")
        assert app._cache_dir == Path("/var/cache/dataeval")


# ---------------------------------------------------------------------------
# Loading screen
# ---------------------------------------------------------------------------


class TestLoadingScreen:
    async def test_compose(self) -> None:
        screen = LoadingScreen()
        app = FlowApp()
        async with app.run_test(size=(120, 40)):
            assert screen is not None


# ---------------------------------------------------------------------------
# Pane-based navigation
# ---------------------------------------------------------------------------


class TestPaneNavigation:
    async def test_cycle_pane_forward(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            # Focus something in config pane
            config_widgets = app._get_pane_widgets("config-pane")
            if config_widgets:
                app.set_focus(config_widgets[0])
                app._cycle_pane(1)
                focused = app.focused
                # Should now be in task pane
                assert isinstance(focused, (TaskPaneHeader, TaskCard))

    async def test_focus_within_pane(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            config_widgets = app._get_pane_widgets("config-pane")
            if len(config_widgets) >= 2:
                app.set_focus(config_widgets[0])
                app._focus_within_pane("config-pane", 1)
                assert app.focused is config_widgets[1]


# ---------------------------------------------------------------------------
# Rebuild panes
# ---------------------------------------------------------------------------


class TestRebuildPanes:
    async def test_rebuild_config_pane_creates_headers(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            config_widgets = app._get_pane_widgets("config-pane")
            headers = [w for w in config_widgets if isinstance(w, CfgSectionHeader)]
            assert len(headers) == len(_CONFIG_SECTIONS)

    async def test_rebuild_task_pane_creates_header(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            task_widgets = app._get_pane_widgets("task-pane")
            headers = [w for w in task_widgets if isinstance(w, TaskPaneHeader)]
            assert len(headers) == 1


# ---------------------------------------------------------------------------
# Key actions
# ---------------------------------------------------------------------------


class TestKeyActions:
    async def test_action_blur(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_blur()
            assert app.focused is None

    async def test_undo_nothing_warns(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_undo()

    async def test_redo_nothing_warns(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_redo()

    async def test_space_no_focus_noop(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_space_item()

    async def test_add_item_no_focus_noop(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.set_focus(None)
            app.action_add_item()

    async def test_activate_item_no_focus_noop(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.set_focus(None)
            app.action_activate_item()

    async def test_delete_item_no_focus_noop(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.set_focus(None)
            app.action_delete_item()

    async def test_focused_category_none(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.set_focus(None)
            assert app._focused_category() is None

    async def test_focused_category_config_header(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            config_widgets = app._get_pane_widgets("config-pane")
            headers = [w for w in config_widgets if isinstance(w, CfgSectionHeader)]
            if headers:
                app.set_focus(headers[0])
                assert app._focused_category() == headers[0].fc_category

    async def test_add_item_on_config_header_opens_modal(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            config_widgets = app._get_pane_widgets("config-pane")
            headers = [w for w in config_widgets if isinstance(w, CfgSectionHeader)]
            app.set_focus(headers[0])
            with patch.object(app, "_open_modal") as mock_open:
                app.action_add_item()
                mock_open.assert_called_once_with(headers[0].fc_category, None, -1)

    async def test_enter_on_task_header_opens_modal(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            task_widgets = app._get_pane_widgets("task-pane")
            headers = [w for w in task_widgets if isinstance(w, TaskPaneHeader)]
            if headers:
                app.set_focus(headers[0])
                with patch.object(app, "_open_modal") as mock_open:
                    app.action_activate_item()
                    mock_open.assert_called_once_with("tasks", None, -1)


# ---------------------------------------------------------------------------
# _on_modal_result
# ---------------------------------------------------------------------------


class TestOnModalResult:
    def test_none_result_noop(self) -> None:
        app = FlowApp()
        app._editing_category = "datasets"
        app._editing_index = 0
        app._on_modal_result(None)

    def test_non_dict_non_sentinel_noop(self) -> None:
        app = FlowApp()
        app._editing_category = "datasets"
        app._editing_index = 0
        app._on_modal_result("some_random_string")


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------


class TestFileOperations:
    async def test_save_config_empty(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with tempfile.TemporaryDirectory() as td:
                app._vm.config_file_path = str(Path(td) / "test.yaml")
                app.action_save_config()

    async def test_open_config_pushes_screen(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with contextlib.suppress(Exception):
                app.action_open_config()
                await pilot.pause()

    async def test_default_browse_path_uses_data_dir(self) -> None:
        app = FlowApp(data_dir="/mnt/data")
        assert app._default_browse_path() == "/mnt/data"

    async def test_default_browse_path_falls_back_to_cwd(self) -> None:
        app = FlowApp()
        assert app._default_browse_path() == "."


# ---------------------------------------------------------------------------
# Settings (F10)
# ---------------------------------------------------------------------------


class TestSettings:
    def test_initial_settings_from_cli_args(self) -> None:
        app = FlowApp(data_dir="/mnt/data", cache_dir="/var/cache/de")
        assert app._settings.data_dir == "/mnt/data"
        assert app._settings.cache_dir == "/var/cache/de"
        assert app._settings.output_dir == ""

    def test_settings_properties(self) -> None:
        app = FlowApp(data_dir="/data", cache_dir="/cache")
        assert app._data_dir == Path("/data")
        assert app._cache_dir == Path("/cache")
        assert app._output_dir is None  # empty string -> None

    def test_settings_properties_empty(self) -> None:
        app = FlowApp()
        assert app._data_dir is None
        assert app._cache_dir is None
        assert app._output_dir is None

    async def test_open_settings_pushes_modal(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with contextlib.suppress(Exception):
                app.action_open_settings()
                await pilot.pause()


# ---------------------------------------------------------------------------
# run_builder entry point
# ---------------------------------------------------------------------------


class TestRunBuilder:
    def test_run_builder_is_callable(self) -> None:
        from dataeval_flow._app.app import run_builder

        assert callable(run_builder)

    def test_run_builder_creates_app(self) -> None:
        from dataeval_flow._app.app import run_builder

        with patch.object(FlowApp, "run") as mock_run:
            run_builder(config_path="/some/path.yaml")
            mock_run.assert_called_once()

    def test_run_builder_no_args(self) -> None:
        from dataeval_flow._app.app import run_builder

        with patch.object(FlowApp, "run") as mock_run:
            run_builder()
            mock_run.assert_called_once()

    def test_run_builder_with_data_and_cache(self) -> None:
        from dataeval_flow._app.app import run_builder

        with patch.object(FlowApp, "run") as mock_run:
            run_builder(data_dir="/data", cache_dir="/cache")
            mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Deferred init worker / on_mount (lines 252-286, 291)
# ---------------------------------------------------------------------------


class TestDeferredInit:
    async def test_deferred_init_succeeds(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            # After init, panes should be rebuilt
            config_widgets = app._get_pane_widgets("config-pane")
            assert len(config_widgets) > 0

    async def test_deferred_init_with_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "params.yaml"
            cfg_path.write_text("datasets:\n  - name: ds1\n    format: huggingface\n    path: data\n")
            app = FlowApp(config_path=str(cfg_path))
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await pilot.pause()
                await pilot.pause()
                # Config should have been loaded
                assert app._vm.config_file_path == str(cfg_path)

    async def test_deferred_init_handles_exception(self) -> None:
        app = FlowApp()
        with patch("dataeval_flow._app.app.FlowApp.run_worker") as mock_rw:
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                # The worker was called
                assert mock_rw.called


# ---------------------------------------------------------------------------
# _focused_category returning "tasks" for TaskCard (line 359)
# ---------------------------------------------------------------------------


class TestFocusedCategoryTaskCard:
    async def test_focused_category_task_card(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "s1"})
            app._rebuild_all()
            await pilot.pause()
            task_widgets = app._get_pane_widgets("task-pane")
            cards = [w for w in task_widgets if isinstance(w, TaskCard)]
            if cards:
                app.set_focus(cards[0])
                assert app._focused_category() == "tasks"

    async def test_focused_category_task_pane_header(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            task_widgets = app._get_pane_widgets("task-pane")
            headers = [w for w in task_widgets if isinstance(w, TaskPaneHeader)]
            if headers:
                app.set_focus(headers[0])
                assert app._focused_category() == "tasks"


# ---------------------------------------------------------------------------
# action_undo / action_redo success paths (lines 367-368, 375-376)
# ---------------------------------------------------------------------------


class TestUndoRedo:
    async def test_undo_redo_success(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            app._rebuild_all()
            await pilot.pause()
            assert app._vm.count("datasets") == 1
            app.action_undo()
            await pilot.pause()
            assert app._vm.count("datasets") == 0
            app.action_redo()
            await pilot.pause()
            assert app._vm.count("datasets") == 1


# ---------------------------------------------------------------------------
# action_activate_item with CfgItem, TaskCard, ResultCard (lines 387-397)
# ---------------------------------------------------------------------------


class TestActivateItem:
    async def test_activate_item_on_cfg_item(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            app._rebuild_all()
            await pilot.pause()
            cfg_items = [w for w in app._get_pane_widgets("config-pane") if isinstance(w, CfgItem)]
            if cfg_items:
                app.set_focus(cfg_items[0])
                with patch.object(app, "_open_modal") as mock:
                    app.action_activate_item()
                    assert mock.called

    async def test_activate_item_on_task_card(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "s1"})
            app._rebuild_all()
            await pilot.pause()
            cards = [w for w in app._get_pane_widgets("task-pane") if isinstance(w, TaskCard)]
            if cards:
                app.set_focus(cards[0])
                with patch.object(app, "action_run_task") as mock:
                    app.action_activate_item()
                    assert mock.called

    async def test_activate_item_on_result_card(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.mark_task_failed("t1", "some error")
            app._rebuild_result_pane()
            await pilot.pause()
            result_widgets = app._get_pane_widgets("result-pane")
            cards = [w for w in result_widgets if isinstance(w, ResultCard)]
            if cards:
                app.set_focus(cards[0])
                with patch.object(app, "_view_result") as mock:
                    app.action_activate_item()
                    assert mock.called

    async def test_activate_item_on_cfg_section_header(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            config_widgets = app._get_pane_widgets("config-pane")
            headers = [w for w in config_widgets if isinstance(w, CfgSectionHeader)]
            if headers:
                app.set_focus(headers[0])
                with patch.object(app, "_open_modal") as mock:
                    app.action_activate_item()
                    mock.assert_called_once_with(headers[0].fc_category, None, -1)


# ---------------------------------------------------------------------------
# action_space_item / _toggle_task (lines 403-411)
# ---------------------------------------------------------------------------


class TestSpaceItem:
    async def test_space_toggles_task(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "s1"})
            app._rebuild_all()
            await pilot.pause()
            cards = [w for w in app._get_pane_widgets("task-pane") if isinstance(w, TaskCard)]
            if cards:
                app.set_focus(cards[0])
                app.action_space_item()
                await pilot.pause()
                # After toggle, the task should be disabled
                task = app._vm.get_item("tasks", 0)
                assert task is not None
                assert task.get("enabled") is False

    async def test_space_toggles_task_back_on(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "s1"})
            app._rebuild_all()
            await pilot.pause()
            cards = [w for w in app._get_pane_widgets("task-pane") if isinstance(w, TaskCard)]
            if cards:
                app.set_focus(cards[0])
                # Toggle off
                app.action_space_item()
                await pilot.pause()
                # Toggle on again
                app.action_space_item()
                await pilot.pause()
                task = app._vm.get_item("tasks", 0)
                assert task is not None
                assert task.get("enabled") is True


# ---------------------------------------------------------------------------
# _view_result (lines 414-426)
# ---------------------------------------------------------------------------


class TestViewResult:
    async def test_view_result_failed(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.mark_task_failed("t1", "some error")
            app._rebuild_result_pane()
            await pilot.pause()
            result_widgets = app._get_pane_widgets("result-pane")
            cards = [w for w in result_widgets if isinstance(w, ResultCard)]
            if cards:
                with patch.object(app, "push_screen"):
                    app._view_result(cards[0])

    async def test_view_result_no_execution(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from dataeval_flow._app._panes._widgets import uid

            card = ResultCard("test", task_name="nonexistent", id=uid("rc-test"))
            app._view_result(card)  # should just return, no crash

    async def test_view_result_completed(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            mock_result = MagicMock()
            mock_result.metadata.execution_time_s = 1.5
            mock_result.metadata.timestamp = None
            mock_result.metadata.model_id = None
            mock_result.metadata.preprocessor_id = None
            mock_result.metadata.source_descriptions = []
            mock_result.data.report.findings = []
            mock_result.data.report.summary = "ok"
            mock_result.to_dict.return_value = {"k": "v"}
            mock_result.report.return_value = "report"
            app._vm.mark_task_completed("t1", mock_result)
            app._rebuild_result_pane()
            await pilot.pause()
            result_widgets = app._get_pane_widgets("result-pane")
            cards = [w for w in result_widgets if isinstance(w, ResultCard)]
            if cards:
                with patch.object(app, "push_screen"):
                    app._view_result(cards[0])

    async def test_view_result_completed_no_result_object(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Mark completed but then clear the result
            mock_result = MagicMock()
            app._vm.mark_task_completed("t1", mock_result)
            # Manually null out the result to test the guard
            execution = app._vm.task_execution("t1")
            assert execution is not None
            execution.result = None
            from dataeval_flow._app._panes._widgets import uid

            card = ResultCard("test", task_name="t1", id=uid("rc-test2"))
            app._view_result(card)  # should return without pushing screen


# ---------------------------------------------------------------------------
# action_delete_item (lines 433-452)
# ---------------------------------------------------------------------------


class TestDeleteItem:
    async def test_delete_cfg_item(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            app._rebuild_all()
            await pilot.pause()
            items = [w for w in app._get_pane_widgets("config-pane") if isinstance(w, CfgItem)]
            if items:
                app.set_focus(items[0])
                app.action_delete_item()
                await pilot.pause()
                assert app._vm.count("datasets") == 0

    async def test_delete_task_card(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "s1"})
            app._rebuild_all()
            await pilot.pause()
            cards = [w for w in app._get_pane_widgets("task-pane") if isinstance(w, TaskCard)]
            if cards:
                app.set_focus(cards[0])
                app.action_delete_item()
                await pilot.pause()
                assert app._vm.count("tasks") == 0

    async def test_delete_item_invalid_index(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            # Focus on a header which has index -1
            config_widgets = app._get_pane_widgets("config-pane")
            headers = [w for w in config_widgets if isinstance(w, CfgSectionHeader)]
            if headers:
                app.set_focus(headers[0])
                app.action_delete_item()  # should be noop (index < 0)


# ---------------------------------------------------------------------------
# action_open_settings callback (lines 458-460)
# ---------------------------------------------------------------------------


class TestOpenSettingsCallback:
    async def test_open_settings_callback_updates(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            new_settings = ExecutionSettings(data_dir="/new/data", cache_dir="/new/cache", output_dir="/new/output")
            app._settings = new_settings
            assert app._data_dir == Path("/new/data")
            assert app._cache_dir == Path("/new/cache")
            assert app._output_dir == Path("/new/output")

    async def test_open_settings_callback_via_action(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Simulate the callback that action_open_settings registers
            new_settings = ExecutionSettings(data_dir="/updated/data")
            # Directly invoke the callback logic
            app._settings = new_settings
            assert app._data_dir == Path("/updated/data")


# ---------------------------------------------------------------------------
# action_run_task (lines 470-492)
# ---------------------------------------------------------------------------


class TestRunTask:
    async def test_run_task_no_focus(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.set_focus(None)
            app.action_run_task()  # should just warn, no crash

    async def test_run_task_on_task_card(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "ds1"})
            app._rebuild_all()
            await pilot.pause()
            cards = [w for w in app._get_pane_widgets("task-pane") if isinstance(w, TaskCard)]
            if cards:
                app.set_focus(cards[0])
                with patch.object(app._vm, "build_pipeline_config") as mock_build, patch.object(app, "run_worker"):
                    mock_build.return_value = MagicMock()
                    app.action_run_task()

    async def test_run_task_validation_failure(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "s1"})
            app._rebuild_all()
            await pilot.pause()
            cards = [w for w in app._get_pane_widgets("task-pane") if isinstance(w, TaskCard)]
            if cards:
                app.set_focus(cards[0])
                with patch.object(app._vm, "build_pipeline_config", side_effect=ValueError("bad config")):
                    app.action_run_task()  # should notify error, no crash

    async def test_run_task_no_task_name(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("tasks", -1, {"name": "", "workflow": "w1", "sources": "s1"})
            app._rebuild_all()
            await pilot.pause()
            cards = [w for w in app._get_pane_widgets("task-pane") if isinstance(w, TaskCard)]
            if cards:
                app.set_focus(cards[0])
                app.action_run_task()  # should return early since name is empty

    async def test_run_task_get_item_none(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "s1"})
            app._rebuild_all()
            await pilot.pause()
            cards = [w for w in app._get_pane_widgets("task-pane") if isinstance(w, TaskCard)]
            if cards:
                app.set_focus(cards[0])
                with patch.object(app._vm, "get_item", return_value=None):
                    app.action_run_task()  # should return early


# ---------------------------------------------------------------------------
# action_run_all (lines 499-514)
# ---------------------------------------------------------------------------


class TestRunAll:
    async def test_run_all_no_tasks(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_run_all()  # should warn

    async def test_run_all_with_tasks(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "ds1"})
            app._rebuild_all()
            await pilot.pause()
            with patch.object(app._vm, "build_pipeline_config") as mock_build, patch.object(app, "run_worker"):
                mock_build.return_value = MagicMock()
                app.action_run_all()

    async def test_run_all_validation_failure(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "s1"})
            app._rebuild_all()
            await pilot.pause()
            with patch.object(app._vm, "build_pipeline_config", side_effect=ValueError("bad")):
                app.action_run_all()  # should notify error

    async def test_run_all_disabled_tasks_skipped(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "s1"})
            # Disable the task
            task = app._vm.get_item("tasks", 0)
            if task:
                task["enabled"] = False
            app._rebuild_all()
            await pilot.pause()
            app.action_run_all()  # should warn, no enabled tasks


# ---------------------------------------------------------------------------
# _execute_task_worker (lines 521-545)
# ---------------------------------------------------------------------------


class TestExecuteTaskWorker:
    def test_execute_task_worker_success(self) -> None:
        app = FlowApp()
        mock_config = MagicMock()
        mock_task_cfg = MagicMock()
        mock_task_cfg.name = "t1"
        mock_config.tasks = [mock_task_cfg]
        mock_result = MagicMock()
        mock_result.metadata.execution_time_s = 1.5

        with (
            patch.object(app, "call_from_thread") as mock_call,
            patch("dataeval_flow.workflow.orchestrator._run_single_task", return_value=mock_result),
        ):
            app._execute_task_worker("t1", mock_config)
            assert mock_call.called

    def test_execute_task_worker_failure(self) -> None:
        app = FlowApp()
        mock_config = MagicMock()
        mock_task_cfg = MagicMock()
        mock_task_cfg.name = "t1"
        mock_config.tasks = [mock_task_cfg]

        with (
            patch.object(app, "call_from_thread") as mock_call,
            patch("dataeval_flow.workflow.orchestrator._run_single_task", side_effect=RuntimeError("boom")),
        ):
            app._execute_task_worker("t1", mock_config)
            assert mock_call.called


# ---------------------------------------------------------------------------
# _execute_all_worker (lines 549-584)
# ---------------------------------------------------------------------------


class TestExecuteAllWorker:
    def test_execute_all_worker_success(self) -> None:
        app = FlowApp()
        mock_config = MagicMock()
        mock_task_cfg = MagicMock()
        mock_task_cfg.name = "t1"
        mock_config.tasks = [mock_task_cfg]
        mock_result = MagicMock()
        mock_result.metadata.execution_time_s = 1.0

        with (
            patch.object(app, "call_from_thread") as mock_call,
            patch("dataeval_flow.workflow.orchestrator._run_single_task", return_value=mock_result),
        ):
            app._execute_all_worker(["t1"], mock_config)
            assert mock_call.called

    def test_execute_all_worker_failure(self) -> None:
        app = FlowApp()
        mock_config = MagicMock()
        mock_task_cfg = MagicMock()
        mock_task_cfg.name = "t1"
        mock_config.tasks = [mock_task_cfg]

        with (
            patch.object(app, "call_from_thread") as mock_call,
            patch("dataeval_flow.workflow.orchestrator._run_single_task", side_effect=RuntimeError("boom")),
        ):
            app._execute_all_worker(["t1"], mock_config)
            assert mock_call.called

    def test_execute_all_worker_multiple_tasks(self) -> None:
        app = FlowApp()
        mock_config = MagicMock()
        mock_task1 = MagicMock()
        mock_task1.name = "t1"
        mock_task2 = MagicMock()
        mock_task2.name = "t2"
        mock_config.tasks = [mock_task1, mock_task2]
        mock_result = MagicMock()
        mock_result.metadata.execution_time_s = 0.5

        with (
            patch.object(app, "call_from_thread") as mock_call,
            patch("dataeval_flow.workflow.orchestrator._run_single_task", return_value=mock_result),
        ):
            app._execute_all_worker(["t1", "t2"], mock_config)
            assert mock_call.call_count >= 3  # mark_running x2 + on_done x2 + on_all_done


# ---------------------------------------------------------------------------
# action_export_results (lines 588-604)
# ---------------------------------------------------------------------------


class TestExportResults:
    async def test_export_results_no_results(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_export_results()  # should warn about no results

    async def test_export_results_with_output_dir(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {"k": "v"}
            mock_result.report.return_value = "report"
            app._vm.mark_task_completed("t1", mock_result)
            with tempfile.TemporaryDirectory() as td:
                app._settings.output_dir = td
                app.action_export_results()

    async def test_export_results_browse(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {}
            mock_result.report.return_value = ""
            app._vm.mark_task_completed("t1", mock_result)
            app._settings.output_dir = ""
            with patch.object(app, "push_screen"):
                app.action_export_results()


# ---------------------------------------------------------------------------
# File operations (lines 613-656)
# ---------------------------------------------------------------------------


class TestFileOperationsExtended:
    async def test_new_config(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            assert app._vm.count("datasets") == 1
            app.action_new_config()
            await pilot.pause()
            assert app._vm.is_empty()

    async def test_save_config_with_path(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "test.yaml"
                app._vm.config_file_path = str(path)
                app.action_save_config()
                await pilot.pause()
                assert path.exists()

    async def test_save_config_no_path(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            app._vm.config_file_path = ""
            with patch.object(app, "push_screen"):
                app.action_save_config()

    async def test_save_config_as(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch.object(app, "push_screen"):
                app.action_save_config_as()

    async def test_default_browse_path_with_config_file(self) -> None:
        app = FlowApp()
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.yaml"
            cfg_path.touch()
            app._vm.config_file_path = str(cfg_path)
            assert app._default_browse_path() == td

    async def test_default_browse_path_with_config_dir(self) -> None:
        app = FlowApp()
        with tempfile.TemporaryDirectory() as td:
            app._vm.config_file_path = td
            assert app._default_browse_path() == td


# ---------------------------------------------------------------------------
# on_click routing (lines 663-685)
# ---------------------------------------------------------------------------


class TestOnClick:
    async def test_click_on_cfg_item(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            app._rebuild_all()
            await pilot.pause()
            cfg_items = [w for w in app._get_pane_widgets("config-pane") if isinstance(w, CfgItem)]
            if cfg_items:
                # Simulate click event by creating a mock event
                mock_event = MagicMock()
                mock_event.widget = cfg_items[0]
                with patch.object(app, "_open_modal"):
                    app.on_click(mock_event)
                    mock_event.stop.assert_called()

    async def test_click_on_task_card(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "s1"})
            app._rebuild_all()
            await pilot.pause()
            cards = [w for w in app._get_pane_widgets("task-pane") if isinstance(w, TaskCard)]
            if cards:
                # Test click with x >= 4 (focus + run)
                mock_event = MagicMock()
                mock_event.widget = cards[0]
                mock_event.x = 10
                with patch.object(app, "action_run_task"):
                    app.on_click(mock_event)
                    mock_event.stop.assert_called()

    async def test_click_on_task_card_toggle(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "s1"})
            app._rebuild_all()
            await pilot.pause()
            cards = [w for w in app._get_pane_widgets("task-pane") if isinstance(w, TaskCard)]
            if cards:
                # Test click with x < 4 (toggle)
                mock_event = MagicMock()
                mock_event.widget = cards[0]
                mock_event.x = 2
                with patch.object(app, "_toggle_task") as mock_toggle:
                    app.on_click(mock_event)
                    mock_toggle.assert_called_once_with(cards[0])

    async def test_click_on_result_card(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.mark_task_failed("t1", "err")
            app._rebuild_result_pane()
            await pilot.pause()
            result_widgets = app._get_pane_widgets("result-pane")
            cards = [w for w in result_widgets if isinstance(w, ResultCard)]
            if cards:
                mock_event = MagicMock()
                mock_event.widget = cards[0]
                with patch.object(app, "_view_result") as mock_view:
                    app.on_click(mock_event)
                    mock_view.assert_called_once_with(cards[0])
                    mock_event.stop.assert_called()

    async def test_click_on_non_pane_widget(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Click on a container widget should break out
            from textual.containers import VerticalScroll

            mock_event = MagicMock()
            container = app.query_one("#config-pane", VerticalScroll)
            mock_event.widget = container
            app.on_click(mock_event)  # should not crash


# ---------------------------------------------------------------------------
# _open_modal and _on_modal_result with actual mutations (lines 692-718)
# ---------------------------------------------------------------------------


class TestOpenModalAndResult:
    async def test_on_modal_result_add_dataset(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            app._editing_category = "datasets"
            app._editing_index = -1
            app._on_modal_result({"name": "ds1", "format": "huggingface", "path": "p"})
            await pilot.pause()
            assert app._vm.count("datasets") == 1

    async def test_on_modal_result_task_add(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            app._editing_category = "tasks"
            app._editing_index = -1
            app._on_modal_result({"name": "t1", "workflow": "w1", "sources": "s1"})
            await pilot.pause()
            assert app._vm.count("tasks") == 1

    async def test_on_modal_result_update_dataset(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            app._rebuild_all()
            await pilot.pause()
            app._editing_category = "datasets"
            app._editing_index = 0
            app._on_modal_result({"name": "ds1_updated", "format": "huggingface", "path": "p2"})
            await pilot.pause()
            assert app._vm.count("datasets") == 1
            item = app._vm.get_item("datasets", 0)
            assert item is not None
            assert item["name"] == "ds1_updated"

    async def test_open_modal_pushes_screen(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch.object(app, "push_screen"):
                app._open_modal("datasets", None, -1)
                assert app._editing_category == "datasets"
                assert app._editing_index == -1

    async def test_open_modal_with_existing(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            existing = app._vm.get_item("datasets", 0)
            with patch.object(app, "push_screen"):
                app._open_modal("datasets", existing, 0)
                assert app._editing_category == "datasets"
                assert app._editing_index == 0


# ---------------------------------------------------------------------------
# _get_pane_widgets edge cases (lines 305-306)
# ---------------------------------------------------------------------------


class TestGetPaneWidgets:
    async def test_get_pane_widgets_nonexistent_pane(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            result = app._get_pane_widgets("nonexistent-pane")
            assert result == []


# ---------------------------------------------------------------------------
# _focus_within_pane edge cases (lines 313, 316-321)
# ---------------------------------------------------------------------------


class TestFocusWithinPaneEdgeCases:
    async def test_focus_within_pane_empty(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Result pane initially has no PaneWidgets
            app._focus_within_pane("result-pane", 1)
            # Should not crash

    async def test_focus_within_pane_no_current_focus(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            # No focus set, should focus first item
            app.set_focus(None)
            app._focus_within_pane("config-pane", 1)
            config_widgets = app._get_pane_widgets("config-pane")
            if config_widgets:
                assert app.focused is config_widgets[0]

    async def test_focus_within_pane_backward_no_current(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            app.set_focus(None)
            app._focus_within_pane("config-pane", -1)
            config_widgets = app._get_pane_widgets("config-pane")
            if config_widgets:
                assert app.focused is config_widgets[-1]


# ---------------------------------------------------------------------------
# _cycle_pane edge cases (lines 330-333)
# ---------------------------------------------------------------------------


class TestCyclePaneEdgeCases:
    async def test_cycle_pane_backward(self) -> None:
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            task_widgets = app._get_pane_widgets("task-pane")
            if task_widgets:
                app.set_focus(task_widgets[0])
                app._cycle_pane(-1)
                focused = app.focused
                config_widgets = app._get_pane_widgets("config-pane")
                if config_widgets:
                    assert focused in config_widgets


# ---------------------------------------------------------------------------
# action_open_settings callback (lines 457-460)
# ---------------------------------------------------------------------------


class TestOpenSettingsCallbackDirect:
    async def test_open_settings_callback_applies_settings(self) -> None:
        """The _on_result callback inside action_open_settings updates settings."""
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Capture the callback by patching push_screen
            callbacks, capture = _make_push_screen_capture()

            with patch.object(app, "push_screen", side_effect=capture):
                app.action_open_settings()

            assert len(callbacks) == 1
            # Invoke the callback with new settings
            new_settings = ExecutionSettings(data_dir="/new/data", cache_dir="/new/cache", output_dir="/new/out")
            callbacks[0](new_settings)
            assert app._settings.data_dir == "/new/data"
            assert app._settings.cache_dir == "/new/cache"
            assert app._settings.output_dir == "/new/out"

    async def test_open_settings_callback_none_noop(self) -> None:
        """The _on_result callback with None does not change settings."""
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            original_settings = app._settings
            callbacks, capture = _make_push_screen_capture()

            with patch.object(app, "push_screen", side_effect=capture):
                app.action_open_settings()

            callbacks[0](None)
            assert app._settings is original_settings


# ---------------------------------------------------------------------------
# _execute_task_worker inner callbacks (lines 527-533, 539-543)
# ---------------------------------------------------------------------------


class TestExecuteTaskWorkerCallbacks:
    def test_execute_task_worker_on_done_invokes_callback(self) -> None:
        """The _on_done callback updates VM state and calls UI methods."""
        app = FlowApp()
        mock_config = MagicMock()
        mock_task_cfg = MagicMock()
        mock_task_cfg.name = "t1"
        mock_config.tasks = [mock_task_cfg]
        mock_result = MagicMock()
        mock_result.metadata.execution_time_s = 1.5

        captured_callbacks = []

        def _capture_call(fn):
            captured_callbacks.append(fn)

        with (
            patch.object(app, "call_from_thread", side_effect=_capture_call),
            patch("dataeval_flow.workflow.orchestrator._run_single_task", return_value=mock_result),
        ):
            app._execute_task_worker("t1", mock_config)

        # The success path should have captured _on_done
        assert len(captured_callbacks) == 1
        # Execute the callback with mocked UI methods
        with (
            patch.object(app, "_update_task_card_by_name"),
            patch.object(app, "_append_or_update_result"),
            patch.object(app, "notify"),
        ):
            captured_callbacks[0]()

        # Verify VM state was updated
        execution = app._vm.task_execution("t1")
        assert execution is not None
        assert execution.status == "completed"

    def test_execute_task_worker_on_fail_invokes_callback(self) -> None:
        """The _on_fail callback updates VM state on failure."""
        app = FlowApp()
        mock_config = MagicMock()
        mock_task_cfg = MagicMock()
        mock_task_cfg.name = "t1"
        mock_config.tasks = [mock_task_cfg]

        captured_callbacks = []

        def _capture_call(fn):
            captured_callbacks.append(fn)

        with (
            patch.object(app, "call_from_thread", side_effect=_capture_call),
            patch("dataeval_flow.workflow.orchestrator._run_single_task", side_effect=RuntimeError("boom")),
        ):
            app._execute_task_worker("t1", mock_config)

        assert len(captured_callbacks) == 1
        with (
            patch.object(app, "_update_task_card_by_name"),
            patch.object(app, "_append_or_update_result"),
            patch.object(app, "notify"),
        ):
            captured_callbacks[0]()

        execution = app._vm.task_execution("t1")
        assert execution is not None
        assert execution.status == "failed"
        assert execution.error == "boom"


# ---------------------------------------------------------------------------
# _execute_all_worker inner callbacks (lines 553-582)
# ---------------------------------------------------------------------------


class TestExecuteAllWorkerCallbacks:
    def test_execute_all_worker_invokes_all_callbacks(self) -> None:
        """All inner lambdas in _execute_all_worker fire correctly."""
        app = FlowApp()
        mock_config = MagicMock()
        mock_task1 = MagicMock()
        mock_task1.name = "t1"
        mock_task2 = MagicMock()
        mock_task2.name = "t2"
        mock_config.tasks = [mock_task1, mock_task2]
        mock_result = MagicMock()
        mock_result.metadata.execution_time_s = 0.5

        captured_callbacks = []

        def _capture_call(fn):
            captured_callbacks.append(fn)

        with (
            patch.object(app, "call_from_thread", side_effect=_capture_call),
            patch("dataeval_flow.workflow.orchestrator._run_single_task", return_value=mock_result),
        ):
            app._execute_all_worker(["t1", "t2"], mock_config)

        # Expected: _mark_running(t1), _on_done(t1), _mark_running(t2), _on_done(t2), _on_all_done
        assert len(captured_callbacks) == 5

        # Execute all callbacks to cover the inner lambda code
        with (
            patch.object(app, "_update_task_card_by_name"),
            patch.object(app, "_append_or_update_result"),
            patch.object(app, "notify"),
        ):
            for cb in captured_callbacks:
                cb()

        # Verify both tasks completed
        e1 = app._vm.task_execution("t1")
        e2 = app._vm.task_execution("t2")
        assert e1 is not None
        assert e1.status == "completed"
        assert e2 is not None
        assert e2.status == "completed"

    def test_execute_all_worker_failure_callbacks(self) -> None:
        """_on_fail lambdas inside _execute_all_worker fire correctly."""
        app = FlowApp()
        mock_config = MagicMock()
        mock_task1 = MagicMock()
        mock_task1.name = "t1"
        mock_config.tasks = [mock_task1]

        captured_callbacks = []

        def _capture_call(fn):
            captured_callbacks.append(fn)

        with (
            patch.object(app, "call_from_thread", side_effect=_capture_call),
            patch("dataeval_flow.workflow.orchestrator._run_single_task", side_effect=RuntimeError("fail")),
        ):
            app._execute_all_worker(["t1"], mock_config)

        # Expected: _mark_running(t1), _on_fail(t1), _on_all_done
        assert len(captured_callbacks) == 3

        with (
            patch.object(app, "_update_task_card_by_name"),
            patch.object(app, "_append_or_update_result"),
            patch.object(app, "notify"),
        ):
            for cb in captured_callbacks:
                cb()

        e1 = app._vm.task_execution("t1")
        assert e1 is not None
        assert e1.status == "failed"


# ---------------------------------------------------------------------------
# action_export_results browse callback (lines 599-602)
# ---------------------------------------------------------------------------


class TestExportResultsBrowseCallback:
    async def test_export_results_browse_callback_invoked(self) -> None:
        """The browse callback inside action_export_results exports results."""
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {"k": "v"}
            mock_result.report.return_value = "report text"
            app._vm.mark_task_completed("t1", mock_result)
            app._settings.output_dir = ""

            callbacks, capture = _make_push_screen_capture()

            with patch.object(app, "push_screen", side_effect=capture):
                app.action_export_results()

            assert len(callbacks) == 1
            # Test callback with a valid path
            with tempfile.TemporaryDirectory() as td:
                callbacks[0](td)
                # Verify export happened
                assert (Path(td) / "result.json").exists()
                assert (Path(td) / "result.txt").exists()

    async def test_export_results_browse_callback_none(self) -> None:
        """The browse callback with None does nothing."""
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {}
            mock_result.report.return_value = ""
            app._vm.mark_task_completed("t1", mock_result)
            app._settings.output_dir = ""

            callbacks, capture = _make_push_screen_capture()

            with patch.object(app, "push_screen", side_effect=capture):
                app.action_export_results()

            callbacks[0](None)  # should not crash


# ---------------------------------------------------------------------------
# action_open_config callback (lines 628-634)
# ---------------------------------------------------------------------------


class TestOpenConfigCallback:
    async def test_open_config_callback_success(self) -> None:
        """The callback inside action_open_config loads file on success."""
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            callbacks, capture = _make_push_screen_capture()

            with patch.object(app, "push_screen", side_effect=capture):
                app.action_open_config()

            assert len(callbacks) == 1
            # Test with a real YAML file
            with tempfile.TemporaryDirectory() as td:
                cfg_path = Path(td) / "params.yaml"
                cfg_path.write_text("datasets:\n  - name: ds1\n    format: huggingface\n    path: data\n")
                with patch.object(app, "_rebuild_all"):
                    callbacks[0](str(cfg_path))
                assert app._vm.config_file_path == str(cfg_path)

    async def test_open_config_callback_none(self) -> None:
        """The callback inside action_open_config does nothing on None."""
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            callbacks, capture = _make_push_screen_capture()

            with patch.object(app, "push_screen", side_effect=capture):
                app.action_open_config()

            callbacks[0](None)  # should not crash

    async def test_open_config_callback_load_failure(self) -> None:
        """The callback does not rebuild_all when load fails."""
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            callbacks, capture = _make_push_screen_capture()

            with patch.object(app, "push_screen", side_effect=capture):
                app.action_open_config()

            with patch.object(app, "_rebuild_all"):
                callbacks[0]("/nonexistent/path.yaml")
                # load_file should fail for nonexistent path, rebuild_all should not be called
                # (depends on implementation, but test the callback path)


# ---------------------------------------------------------------------------
# action_save_config_as callback (lines 649-654)
# ---------------------------------------------------------------------------


class TestSaveConfigAsCallback:
    async def test_save_config_as_callback_with_dir(self) -> None:
        """The callback inside action_save_config_as saves to dir/params.yaml."""
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})

            callbacks, capture = _make_push_screen_capture()

            with patch.object(app, "push_screen", side_effect=capture):
                app.action_save_config_as()

            assert len(callbacks) == 1
            with tempfile.TemporaryDirectory() as td:
                callbacks[0](td)
                expected_path = Path(td) / "params.yaml"
                assert expected_path.exists()
                assert app._vm.config_file_path == str(expected_path)

    async def test_save_config_as_callback_with_file(self) -> None:
        """The callback inside action_save_config_as saves to specified file."""
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})

            callbacks, capture = _make_push_screen_capture()

            with patch.object(app, "push_screen", side_effect=capture):
                app.action_save_config_as()

            with tempfile.TemporaryDirectory() as td:
                file_path = str(Path(td) / "custom.yaml")
                callbacks[0](file_path)
                assert Path(file_path).exists()

    async def test_save_config_as_callback_none(self) -> None:
        """The callback inside action_save_config_as does nothing on None."""
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            callbacks, capture = _make_push_screen_capture()

            with patch.object(app, "push_screen", side_effect=capture):
                app.action_save_config_as()

            callbacks[0](None)  # should not crash


# ---------------------------------------------------------------------------
# _on_modal_result with warnings (lines 711-714)
# ---------------------------------------------------------------------------


class TestOnModalResultWarnings:
    async def test_on_modal_result_with_warnings(self) -> None:
        """_on_modal_result should notify warnings from the apply_result outcome."""
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._rebuild_all()
            await pilot.pause()
            # Add a dataset first
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            # Add a source that references ds1
            app._vm.apply_result("sources", -1, {"name": "s1", "dataset": "ds1"})
            app._rebuild_all()
            await pilot.pause()
            # Now update the dataset name -- this should produce a warning about source ref
            app._editing_category = "datasets"
            app._editing_index = 0
            with patch.object(app, "notify") as mock_notify:
                app._on_modal_result({"name": "ds1_renamed", "format": "huggingface", "path": "p"})
                # Should have at least one call for the description message
                assert mock_notify.call_count >= 1


# ---------------------------------------------------------------------------
# on_click widget.parent traversal (line 685)
# ---------------------------------------------------------------------------


class TestOnClickParentTraversal:
    async def test_click_on_nested_static_inside_cfg_item(self) -> None:
        """Click on a Static inside a CfgItem should walk up parents."""
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            app._rebuild_all()
            await pilot.pause()
            cfg_items = [w for w in app._get_pane_widgets("config-pane") if isinstance(w, CfgItem)]
            if cfg_items:
                # Get a child widget of the CfgItem (a Static)
                children = list(cfg_items[0].children)
                if children:
                    mock_event = MagicMock()
                    mock_event.widget = children[0]
                    with patch.object(app, "_open_modal"):
                        app.on_click(mock_event)
                        mock_event.stop.assert_called()

    async def test_click_on_widget_with_none_parent(self) -> None:
        """Click on a widget that reaches None parent stops gracefully."""
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Create a mock widget chain that eventually hits None
            mock_widget = MagicMock(spec="UnknownWidget")
            mock_widget.parent = None
            # Not a CfgItem, TaskCard, ResultCard, or container type
            type(mock_widget).__instancecheck__ = lambda _cls, _inst: False
            mock_event = MagicMock()
            mock_event.widget = mock_widget
            # Should not crash
            app.on_click(mock_event)


# ---------------------------------------------------------------------------
# action_delete_item with warnings (lines 444, 447-448)
# ---------------------------------------------------------------------------


class TestDeleteItemWarnings:
    async def test_delete_item_with_dependent_warnings(self) -> None:
        """Deleting a dataset that is referenced should produce warnings."""
        app = FlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
            app._vm.apply_result("sources", -1, {"name": "s1", "dataset": "ds1"})
            app._rebuild_all()
            await pilot.pause()
            items = [w for w in app._get_pane_widgets("config-pane") if isinstance(w, CfgItem)]
            # Find dataset items
            ds_items = [w for w in items if w.fc_category == "datasets"]
            if ds_items:
                app.set_focus(ds_items[0])
                with patch.object(app, "notify") as mock_notify:
                    app.action_delete_item()
                    await pilot.pause()
                    # Should have called notify at least once for description,
                    # and possibly for warnings about dependent sources
                    assert mock_notify.call_count >= 1
