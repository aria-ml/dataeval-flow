"""Unit tests for the click-based CLI builder."""

from __future__ import annotations

from dataeval_flow._app._model._introspect import FieldKind
from dataeval_flow._app._viewmodel._builder_vm import BuilderViewModel
from dataeval_flow._app.cli import _show_items


class TestShowItems:
    def test_empty_section(self, capsys):
        vm = BuilderViewModel()
        _show_items("datasets", vm)
        captured = capsys.readouterr()
        assert "(empty)" in captured.out

    def test_with_items(self, capsys):
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "data"})
        _show_items("datasets", vm)
        captured = capsys.readouterr()
        assert "ds1" in captured.out
        assert "format=huggingface" in captured.out

    def test_task_with_enabled(self, capsys):
        vm = BuilderViewModel()
        vm.apply_result("tasks", -1, {"name": "t1", "workflow": "wf1", "sources": "src1", "enabled": True})
        vm.apply_result("tasks", -1, {"name": "t2", "workflow": "wf1", "sources": "src1", "enabled": False})
        _show_items("tasks", vm)
        captured = capsys.readouterr()
        assert "[on]" in captured.out
        assert "[off]" in captured.out

    def test_discriminated_section_shows_type(self, capsys):
        vm = BuilderViewModel()
        vm.apply_result("extractors", -1, {"name": "ext1", "model": "onnx", "model_path": "/m.onnx"})
        _show_items("extractors", vm)
        captured = capsys.readouterr()
        assert "ext1" in captured.out
        assert "model=onnx" in captured.out

    def test_multiple_items(self, capsys):
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "data"})
        vm.apply_result("datasets", -1, {"name": "ds2", "format": "coco", "path": "data2"})
        _show_items("datasets", vm)
        captured = capsys.readouterr()
        assert "1." in captured.out
        assert "2." in captured.out
        assert "ds1" in captured.out
        assert "ds2" in captured.out


class TestPromptFieldDescriptor:
    """Test that FieldDescriptors are correctly generated for CLI prompting."""

    def test_select_field_has_choices(self):
        vm = BuilderViewModel()
        sec_vm = vm.create_section_vm("datasets")
        fields = sec_vm.load_fields("huggingface")
        # All string fields should not have choices (not cross-refs)
        for f in fields:
            if f.name == "path":
                assert f.kind == FieldKind.STRING

    def test_cross_ref_field_has_choices(self):
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "data"})
        sec_vm = vm.create_section_vm("sources")
        fields = sec_vm.load_fields(None)
        dataset_field = next(f for f in fields if f.name == "dataset")
        assert dataset_field.kind == FieldKind.SELECT
        assert "ds1" in dataset_field.choices


class TestCliModuleImports:
    """Test that the CLI module can be imported without textual."""

    def test_import_cli(self):
        from dataeval_flow._app.cli import run_cli_builder

        assert callable(run_cli_builder)

    def test_import_core(self):
        from dataeval_flow._app._model._state import ConfigState

        state = ConfigState()
        assert state.is_empty()
