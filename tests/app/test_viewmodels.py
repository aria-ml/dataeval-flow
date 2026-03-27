from __future__ import annotations

from pathlib import Path

import pytest

from dataeval_flow._app._viewmodel._builder_vm import BuilderViewModel
from dataeval_flow._app._viewmodel._model_vm import ModelViewModel
from dataeval_flow._app._viewmodel._section_vm import SectionViewModel


class TestBuilderViewModel:
    def test_init(self) -> None:
        vm = BuilderViewModel()
        assert vm.is_empty()
        assert vm.config_file_path == ""

    def test_section_data(self) -> None:
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
        data = vm.section_data()
        # Find datasets section in the list of tuples
        ds_section = next(s for s in data if s[0] == "datasets")
        assert len(ds_section[2]) == 1
        assert ds_section[2][0]["name"] == "ds1"

    def test_queries(self) -> None:
        vm = BuilderViewModel()
        item = {"name": "ds1", "format": "huggingface", "path": "p"}
        vm.apply_result("datasets", -1, item)
        assert vm.count("datasets") == 1
        assert vm.names("datasets") == ["ds1"]
        assert vm.items("datasets")[0]["name"] == "ds1"
        dataset = vm.get_item("datasets", 0) or {}
        assert dataset["name"] == "ds1"
        assert vm.sections[0][0] == "datasets"

    def test_undo_redo(self) -> None:
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
        assert vm.count("datasets") == 1

        success, msg = vm.undo()
        assert success
        assert "ds1" in msg
        assert vm.count("datasets") == 0

        success, msg = vm.redo()
        assert success
        assert "ds1" in msg
        assert vm.count("datasets") == 1

    def test_undo_redo_empty(self) -> None:
        vm = BuilderViewModel()
        success, msg = vm.undo()
        assert not success
        assert "Nothing to undo" in msg

        success, msg = vm.redo()
        assert not success
        assert "Nothing to redo" in msg

    def test_toggle_task(self) -> None:
        vm = BuilderViewModel()
        vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": ["s1"]})
        task = vm.get_item("tasks", 0) or {}
        assert task["enabled"] is True

        vm.toggle_task(0)
        task = vm.get_item("tasks", 0) or {}
        assert task["enabled"] is False

        vm.undo()
        task = vm.get_item("tasks", 0) or {}
        assert task["enabled"] is True

        assert vm.toggle_task(99) is None

    def test_delete_item(self) -> None:
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
        vm.delete_item("datasets", 0)
        assert vm.count("datasets") == 0
        vm.undo()
        assert vm.count("datasets") == 1
        assert vm.delete_item("datasets", 99) is None

    def test_apply_result_cancel(self) -> None:
        vm = BuilderViewModel()
        assert vm.apply_result("datasets", -1, None) is None

    def test_file_io(self, tmp_path: Path) -> None:
        vm = BuilderViewModel()
        path = tmp_path / "test.yaml"
        # Save empty
        success, msg = vm.save_file(path)
        assert not success

        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
        success, msg = vm.save_file(path)
        assert success
        assert path.exists()

        vm2 = BuilderViewModel()
        # Load non-existent
        success, msg = vm2.load_file(tmp_path / "none.yaml")
        assert not success

        success, msg = vm2.load_file(path)
        assert success
        assert vm2.count("datasets") == 1
        assert vm2.config_file_path == str(path)

    def test_new_config(self) -> None:
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
        vm.new_config()
        assert vm.is_empty()
        vm.undo()
        assert vm.count("datasets") == 1

    def test_validation_proxies(self) -> None:
        vm = BuilderViewModel()
        # Simple proxy tests
        assert isinstance(vm.validate_all(), list)
        assert isinstance(vm.validate_item("datasets", {}), list)

    def test_snippet_proxies(self) -> None:
        vm = BuilderViewModel()
        item = {"name": "ds1", "format": "huggingface"}
        assert isinstance(vm.item_snippet("datasets", item), str)
        assert isinstance(vm.task_snippet({"name": "t1"}), str)

    def test_create_section_vm(self) -> None:
        vm = BuilderViewModel()
        svm = vm.create_section_vm("datasets")
        assert isinstance(svm, SectionViewModel)
        assert svm.section == "datasets"

    # -- Execution methods -------------------------------------------------

    def test_task_execution_initially_none(self) -> None:
        vm = BuilderViewModel()
        assert vm.task_execution("nonexistent") is None

    def test_mark_task_running(self) -> None:
        vm = BuilderViewModel()
        entry = vm.mark_task_running("t1")
        assert entry.status == "running"
        assert entry.task_name == "t1"
        assert vm.task_execution("t1") is entry

    def test_mark_task_completed(self) -> None:
        from unittest.mock import MagicMock

        vm = BuilderViewModel()
        vm.mark_task_running("t1")
        mock_result = MagicMock()
        entry = vm.mark_task_completed("t1", mock_result)
        assert entry.status == "completed"
        assert entry.result is mock_result

    def test_mark_task_failed(self) -> None:
        vm = BuilderViewModel()
        vm.mark_task_running("t1")
        entry = vm.mark_task_failed("t1", "boom")
        assert entry.status == "failed"
        assert entry.error == "boom"

    def test_all_executions(self) -> None:
        vm = BuilderViewModel()
        vm.mark_task_running("t1")
        vm.mark_task_running("t2")
        assert len(vm.all_executions()) == 2

    def test_clear_task_execution(self) -> None:
        vm = BuilderViewModel()
        vm.mark_task_running("t1")
        vm.clear_task_execution("t1")
        assert vm.task_execution("t1") is None

    def test_clear_all_executions(self) -> None:
        vm = BuilderViewModel()
        vm.mark_task_running("t1")
        vm.mark_task_running("t2")
        vm.clear_task_execution(None)
        assert len(vm.all_executions()) == 0

    def test_completed_results(self) -> None:
        from unittest.mock import MagicMock

        vm = BuilderViewModel()
        mock_result = MagicMock()
        vm.mark_task_completed("t1", mock_result)
        vm.mark_task_failed("t2", "err")
        results = vm.completed_results()
        assert len(results) == 1
        assert results[0] == ("t1", mock_result)

    def test_build_pipeline_config_empty_raises(self) -> None:
        vm = BuilderViewModel()
        # Empty config should raise because PipelineConfig validates tasks etc.
        # but actually empty config is valid (all fields optional)
        config = vm.build_pipeline_config()
        assert config is not None

    def test_export_results_no_results(self) -> None:
        import tempfile

        vm = BuilderViewModel()
        with tempfile.TemporaryDirectory() as td:
            success, msg = vm.export_results(Path(td) / "out")
            assert not success
            assert "No completed" in msg

    def test_export_results_with_results(self) -> None:
        import tempfile
        from unittest.mock import MagicMock

        vm = BuilderViewModel()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"test": "data"}
        mock_result.report.return_value = "Test report"
        vm.mark_task_completed("t1", mock_result)
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            success, msg = vm.export_results(out_dir)
            assert success
            assert (out_dir / "result.json").exists()
            assert (out_dir / "result.txt").exists()

    def test_save_file_disable_tasks(self) -> None:
        import tempfile

        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
        vm.apply_result(
            "tasks",
            -1,
            {
                "name": "t1",
                "workflow": "w1",
                "sources": "s1",
                "enabled": True,
            },
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.yaml"
            success, msg = vm.save_file(path, disable_tasks=True)
            assert success
            # The saved file should have enabled: false
            import yaml

            with open(path) as f:
                saved = yaml.safe_load(f)
            assert saved["tasks"][0]["enabled"] is False
            # But the in-memory state should still be True
            task = vm.get_item("tasks", 0)
            assert task is not None
            assert task["enabled"] is True

    def test_apply_result_delete_sentinel(self) -> None:
        from dataeval_flow._app._model._item import DELETE_SENTINEL

        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
        result = vm.apply_result("datasets", 0, DELETE_SENTINEL)
        assert result is not None
        assert "Delete" in result[0]
        assert vm.count("datasets") == 0

    def test_load_file_state_raises(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        vm = BuilderViewModel()
        path = tmp_path / "exists.yaml"
        path.write_text("datasets: []")
        with patch.object(vm._state, "load_file", side_effect=ValueError("boom")):
            success, msg = vm.load_file(path)
        assert not success
        assert "Failed to load config" in msg
        assert "boom" in msg

    def test_load_file_with_warning(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        vm = BuilderViewModel()
        path = tmp_path / "warn.yaml"
        path.write_text("datasets: []")
        with patch.object(vm._state, "load_file", return_value="Some warning"):
            success, msg = vm.load_file(path)
        assert success
        assert "Some warning" in msg

    def test_new_config_empty(self) -> None:
        vm = BuilderViewModel()
        msg = vm.new_config()
        assert "New config" in msg
        # History should not have a snapshot since state was already empty
        assert not vm.history.can_undo

    def test_save_file_disable_tasks_missing_task(self) -> None:
        import tempfile
        from unittest.mock import patch

        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "p"})
        vm.apply_result("tasks", -1, {"name": "t1", "workflow": "w1", "sources": "s1", "enabled": True})

        original_get = vm._state.get
        call_count = [0]

        def mock_get(section: str, index: int) -> dict | None:
            if section == "tasks" and call_count[0] > 0:
                return None  # simulate missing task during restore
            call_count[0] += 1
            return original_get(section, index)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.yaml"
            with patch.object(vm._state, "get", side_effect=mock_get):
                success, msg = vm.save_file(path, disable_tasks=True)
            assert success


class TestModelViewModel:
    def test_init_empty(self) -> None:
        vm = ModelViewModel()
        assert vm.existing is None
        assert vm.original is None
        assert not vm.is_edit_mode

    def test_init_existing(self) -> None:
        existing = {"name": "m1", "type": "onnx", "model_path": "path/to/model"}
        vm = ModelViewModel(existing)
        assert vm.existing == existing
        assert vm.original == existing
        assert vm.is_edit_mode

    @pytest.mark.parametrize(
        ("model_type", "expected"),
        [
            ("onnx", True),
            ("torch", True),
            ("uncertainty", True),
            ("bovw", False),
            ("flatten", False),
        ],
    )
    def test_needs_path(self, model_type: str, expected: bool) -> None:
        assert ModelViewModel.needs_path(model_type) == expected

    @pytest.mark.parametrize(
        ("model_type", "expected"),
        [
            ("onnx", False),
            ("torch", False),
            ("uncertainty", False),
            ("bovw", True),
            ("flatten", False),
        ],
    )
    def test_needs_vocab(self, model_type: str, expected: bool) -> None:
        assert ModelViewModel.needs_vocab(model_type) == expected

    def test_build_result_invalid_missing_name_or_type(self) -> None:
        vm = ModelViewModel()
        assert vm.build_result("", "onnx", "path", "") is None
        assert vm.build_result("m1", "", "path", "") is None

    def test_build_result_needs_path_missing(self) -> None:
        vm = ModelViewModel()
        assert vm.build_result("m1", "onnx", "", "") is None

    def test_build_result_with_path(self) -> None:
        vm = ModelViewModel()
        result = vm.build_result("m1", "onnx", "path/to/model", "")
        assert result == {"name": "m1", "type": "onnx", "model_path": "path/to/model"}

    def test_build_result_bovw_with_vocab(self) -> None:
        vm = ModelViewModel()
        result = vm.build_result("m1", "bovw", "", "128")
        assert result == {"name": "m1", "type": "bovw", "vocab_size": 128}

    def test_build_result_bovw_invalid_vocab(self) -> None:
        vm = ModelViewModel()
        assert vm.build_result("m1", "bovw", "", "not-an-int") is None

    def test_build_result_flatten(self) -> None:
        vm = ModelViewModel()
        result = vm.build_result("m1", "flatten", "", "")
        assert result == {"name": "m1", "type": "flatten"}

    def test_check_dirty_none_collected(self) -> None:
        vm = ModelViewModel()
        assert not vm.check_dirty(None)

    def test_check_dirty_new_item(self) -> None:
        vm = ModelViewModel()
        assert vm.check_dirty({"name": "m1"})

    def test_check_dirty_edit_mode_not_dirty(self) -> None:
        existing = {"name": "m1", "type": "flatten"}
        vm = ModelViewModel(existing)
        assert not vm.check_dirty(existing)

    def test_check_dirty_edit_mode_is_dirty(self) -> None:
        existing = {"name": "m1", "type": "flatten"}
        vm = ModelViewModel(existing)
        assert vm.check_dirty({"name": "m1", "type": "onnx"})

    def test_validation_message(self) -> None:
        assert "Name and type are required" in ModelViewModel.validation_message()


class TestSectionViewModel:
    def test_init(self) -> None:
        vm = SectionViewModel("datasets")
        assert vm.section == "datasets"
        assert vm.is_step_builder is False
        assert not vm.is_edit_mode

    def test_init_step_builder(self) -> None:
        vm = SectionViewModel("selections")
        assert vm.is_step_builder is True

    def test_init_with_existing(self) -> None:
        existing = {"name": "ds1", "format": "huggingface"}
        vm = SectionViewModel("datasets", existing=existing)
        assert vm.existing == existing
        assert vm.original == existing
        assert vm.is_edit_mode

    def test_init_step_builder_with_existing(self) -> None:
        existing = {"name": "sel1", "steps": [{"type": "Shuffle", "params": {"seed": 42}}]}
        vm = SectionViewModel("selections", existing=existing)
        assert len(vm.steps) == 1
        assert vm.steps[0] == {"type": "Shuffle", "params": {"seed": 42}}

    def test_variant_choices(self) -> None:
        vm = SectionViewModel("datasets")
        choices = vm.variant_choices
        assert choices is not None
        assert "huggingface" in choices

    def test_disc_field(self) -> None:
        vm = SectionViewModel("datasets")
        assert vm.disc_field == "format"

    def test_step_choices(self) -> None:
        vm = SectionViewModel("selections")
        choices = vm.init_step_builder()
        assert len(choices) > 0
        assert "Shuffle" in choices
        assert vm.step_choices == choices

    def test_get_step_params(self) -> None:
        vm = SectionViewModel("selections")
        vm.init_step_builder()
        params = vm.get_step_params("Shuffle")
        assert len(params) > 0

    def test_add_remove_step(self) -> None:
        vm = SectionViewModel("selections")
        msg = vm.add_step("Shuffle", {"seed": 42})
        assert "Added step 'Shuffle'" in msg
        assert len(vm.steps) == 1
        assert vm.step_display_lines() == ["1. Shuffle(seed=42)"]

        assert vm.remove_step(0)
        assert len(vm.steps) == 0
        assert not vm.remove_step(0)

    def test_step_display_lines_no_params(self) -> None:
        vm = SectionViewModel("selections")
        vm.add_step("all", {})
        assert vm.step_display_lines() == ["1. all"]

    def test_add_remove_list_item(self) -> None:
        vm = SectionViewModel("workflows")
        vm.load_fields("drift-monitoring")
        # 'detectors' is a list of discriminated union for drift-monitoring
        msg = vm.add_list_item("detectors", "kneighbors", {"k": 5})
        assert "Added kneighbors" in msg
        assert len(vm.list_items["detectors"]) == 1

        assert vm.remove_list_item("detectors", 0)
        assert len(vm.list_items["detectors"]) == 0
        assert not vm.remove_list_item("detectors", 0)

    def test_get_variant_descriptors(self) -> None:
        vm = SectionViewModel("workflows")
        vm.load_fields("drift-monitoring")
        descriptors = vm.get_variant_descriptors("detectors", "kneighbors")
        assert len(descriptors) > 0
        # Check that 'k' is in the descriptors
        assert any(d.name == "k" for d in descriptors)
        # Discriminator 'method' should be excluded
        assert not any(d.name == "method" for d in descriptors)

    def test_build_result_invalid(self) -> None:
        vm = SectionViewModel("datasets")
        assert vm.build_result("", "huggingface", {}) is None
        assert vm.build_result("ds1", None, {}) is None  # Needs variant

    def test_build_result_step_builder_invalid(self) -> None:
        vm = SectionViewModel("selections")
        assert vm.build_result("sel1", None, {}) is None  # No steps

    def test_build_result_valid(self) -> None:
        vm = SectionViewModel("datasets")
        result = vm.build_result("ds1", "huggingface", {"path": "foo"})
        assert result == {"name": "ds1", "format": "huggingface", "path": "foo"}

    def test_build_result_step_builder_valid(self) -> None:
        vm = SectionViewModel("selections")
        vm.add_step("Shuffle", {"seed": 42})
        result = vm.build_result("sel1", None, {})
        assert result == {"name": "sel1", "steps": [{"type": "Shuffle", "params": {"seed": 42}}]}

    def test_check_dirty(self) -> None:
        vm = SectionViewModel("datasets")
        assert not vm.check_dirty(None)
        assert vm.check_dirty({"name": "ds1"})

        existing = {"name": "ds1", "format": "huggingface"}
        vm = SectionViewModel("datasets", existing=existing)
        assert not vm.check_dirty(existing)
        assert vm.check_dirty({"name": "ds1", "format": "coco"})

    def test_diagnose_failure(self) -> None:
        vm = SectionViewModel("datasets")
        msg = vm.diagnose_failure("ds1")
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_get_step_params_no_init(self) -> None:
        vm = SectionViewModel("selections")
        # No init_step_builder called
        assert vm.get_step_params("Shuffle") == []

    def test_add_list_item_invalid_field(self) -> None:
        vm = SectionViewModel("workflows")
        vm.load_fields("drift-monitoring")
        assert vm.add_list_item("no_such_field", "foo", {}) == ""

    def test_get_variant_descriptors_invalid(self) -> None:
        vm = SectionViewModel("workflows")
        vm.load_fields("drift-monitoring")
        assert vm.get_variant_descriptors("no_such_field", "foo") == []
        assert vm.get_variant_descriptors("detectors", "no_such_variant") == []

    def test_collect_field_select(self) -> None:
        from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind
        from dataeval_flow._app._model._item import SKIP

        vm = SectionViewModel("datasets")
        desc = FieldDescriptor(name="format", kind=FieldKind.SELECT, description="", required=True)
        assert vm.collect_field(desc, "huggingface") == "huggingface"
        assert vm.collect_field(desc, "") == SKIP

    def test_collect_field_bool(self) -> None:
        from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind
        from dataeval_flow._app._model._item import SKIP

        vm = SectionViewModel("datasets")
        desc = FieldDescriptor(name="recursive", kind=FieldKind.BOOL, description="", required=False, default=False)
        assert vm.collect_field(desc, True) is True
        assert vm.collect_field(desc, False) == SKIP

    def test_collect_field_int(self) -> None:
        from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind

        vm = SectionViewModel("datasets")
        desc = FieldDescriptor(name="k", kind=FieldKind.INT, description="", required=True)
        assert vm.collect_field(desc, "10") == 10
        with pytest.raises(ValueError, match="invalid literal for int"):
            vm.collect_field(desc, "not-int")

    def test_collect_field_list_no_union(self) -> None:
        from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind

        vm = SectionViewModel("datasets")
        desc = FieldDescriptor(name="items", kind=FieldKind.LIST, description="", required=True)
        assert vm.collect_field(desc, "[1, 2, 3]") == [1, 2, 3]

    def test_collect_field_unknown(self) -> None:
        from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind
        from dataeval_flow._app._model._item import SKIP

        vm = SectionViewModel("datasets")
        desc = FieldDescriptor(name="unknown", kind=FieldKind.STRING, description="", required=True)
        # To hit fallthrough, we need a kind not handled.
        desc.kind = "non-existent-kind"  # type: ignore
        assert vm.collect_field(desc, "bar") == SKIP

    def test_collect_field_list_union_with_items(self) -> None:
        from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind
        from dataeval_flow._app._model._item import SKIP

        vm = SectionViewModel("workflows")
        vm.load_fields("drift-monitoring")

        # Create a LIST descriptor with union_variants
        desc = FieldDescriptor(
            name="detectors",
            kind=FieldKind.LIST,
            description="",
            required=True,
            union_variants={"kneighbors": object},  # type: ignore
        )

        # With no items - should return SKIP
        assert vm.collect_field(desc, None) == SKIP

        # With items
        vm.list_items["detectors"] = [{"method": "kneighbors", "k": 5}]
        result = vm.collect_field(desc, None)
        assert result == [{"method": "kneighbors", "k": 5}]

    def test_build_result_tasks_preserve_enabled(self) -> None:
        existing = {"name": "t1", "workflow": "w1", "enabled": False}
        vm = SectionViewModel("tasks", existing=existing)
        result = vm.build_result("t1", None, {"workflow": "w1", "sources": ["s1"]})
        assert result is not None
        assert result["enabled"] is False

    def test_build_result_tasks_default_enabled(self) -> None:
        existing = {"name": "t1", "workflow": "w1"}
        vm = SectionViewModel("tasks", existing=existing)
        result = vm.build_result("t1", None, {"workflow": "w1", "sources": ["s1"]})
        assert result is not None
        assert result["enabled"] is True
