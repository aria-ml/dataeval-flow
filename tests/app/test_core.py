"""Unit tests for _app._core — ConfigState + helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from dataeval_flow._app._model._coerce import coerce_field_value, coerce_value, validate_value
from dataeval_flow._app._model._introspect import FieldKind
from dataeval_flow._app._model._registry import (
    CROSS_REFS,
    SECTION_KEYS,
    SECTIONS,
    STEP_BUILDER_SECTIONS,
    VARIANT_REGISTRY,
    get_discriminator_field,
    get_fields,
    get_model_for_variant,
    get_variant_choices,
)
from dataeval_flow._app._model._state import ConfigState

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


class TestVariantRegistry:
    def test_get_variant_choices_datasets(self):
        choices = get_variant_choices("datasets")
        assert choices is not None
        assert "huggingface" in choices
        assert "coco" in choices

    def test_get_variant_choices_extractors(self):
        choices = get_variant_choices("extractors")
        assert choices is not None
        assert "onnx" in choices
        assert "flatten" in choices

    def test_get_variant_choices_workflows(self):
        choices = get_variant_choices("workflows")
        assert choices is not None
        assert "data-cleaning" in choices
        assert "drift-monitoring" in choices

    def test_get_variant_choices_non_discriminated(self):
        assert get_variant_choices("sources") is None
        assert get_variant_choices("tasks") is None

    def test_get_discriminator_field(self):
        assert get_discriminator_field("datasets") == "format"
        assert get_discriminator_field("extractors") == "model"
        assert get_discriminator_field("workflows") == "type"
        assert get_discriminator_field("sources") is None

    def test_get_model_for_variant(self):
        from dataeval_flow.config.schemas import HuggingFaceDatasetConfig

        model = get_model_for_variant("datasets", "huggingface")
        assert model is HuggingFaceDatasetConfig

    def test_get_model_for_variant_unknown(self):
        assert get_model_for_variant("datasets", "nonexistent") is None

    def test_get_model_for_non_variant_section(self):
        from dataeval_flow.config._models import SourceConfig

        model = get_model_for_variant("sources", "anything")
        assert model is SourceConfig


# ---------------------------------------------------------------------------
# get_fields
# ---------------------------------------------------------------------------


class TestGetFields:
    def test_dataset_fields_huggingface(self):
        state = ConfigState()
        fields = get_fields("datasets", "huggingface", state)
        names = [f.name for f in fields]
        assert "path" in names
        # name and format should be excluded
        assert "name" not in names
        assert "format" not in names

    def test_dataset_fields_coco(self):
        state = ConfigState()
        fields = get_fields("datasets", "coco", state)
        names = [f.name for f in fields]
        assert "path" in names
        assert "annotations_file" in names
        assert "images_dir" in names

    def test_source_fields_with_cross_refs(self):
        state = ConfigState()
        state.add("datasets", {"name": "ds1", "format": "huggingface", "path": "data"})
        state.add("selections", {"name": "sel1", "steps": [{"type": "Limit", "params": {"size": 100}}]})

        fields = get_fields("sources", None, state)
        names = {f.name: f for f in fields}
        assert "dataset" in names
        assert names["dataset"].kind == FieldKind.SELECT
        assert "ds1" in names["dataset"].choices

        assert "selection" in names
        assert names["selection"].kind == FieldKind.SELECT
        assert "sel1" in names["selection"].choices

    def test_task_fields_multi_ref(self):
        state = ConfigState()
        state.add("sources", {"name": "src1", "dataset": "ds1"})
        state.add(
            "workflows",
            {"name": "wf1", "type": "data-cleaning", "outlier_method": "adaptive", "outlier_flags": ["dimension"]},
        )

        fields = get_fields("tasks", None, state)
        names = {f.name: f for f in fields}
        assert "sources" in names
        assert names["sources"].kind == FieldKind.MULTI_SELECT
        assert "src1" in names["sources"].choices

        assert "workflow" in names
        assert names["workflow"].kind == FieldKind.SELECT

    def test_workflow_fields_skip(self):
        state = ConfigState()
        fields = get_fields("workflows", "data-cleaning", state)
        names = [f.name for f in fields]
        assert "name" not in names
        assert "type" not in names
        assert "mode" not in names
        # Should have outlier_method, outlier_flags, etc.
        assert "outlier_method" in names
        assert "outlier_flags" in names

    def test_step_builder_sections_skip_steps(self):
        state = ConfigState()
        fields = get_fields("preprocessors", None, state)
        names = [f.name for f in fields]
        assert "steps" not in names
        assert "name" not in names

    def test_no_variant_returns_empty(self):
        state = ConfigState()
        fields = get_fields("datasets", None, state)
        assert fields == []


# ---------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------


class TestRegistryCoverage:
    def test_get_literal_value_missing_field(self):
        from pydantic import BaseModel

        from dataeval_flow._app._model._registry import _get_literal_value

        class M(BaseModel):
            name: str = "test"

        assert _get_literal_value(M, "nonexistent") is None

    def test_get_literal_value_plain_default(self):
        from pydantic import BaseModel

        from dataeval_flow._app._model._registry import _get_literal_value

        class M(BaseModel):
            name: str = "hello"

        assert _get_literal_value(M, "name") == "hello"

    def test_get_literal_value_none_default(self):
        from pydantic import BaseModel

        from dataeval_flow._app._model._registry import _get_literal_value

        class M(BaseModel):
            name: str | None = None  # default is literally None

        assert _get_literal_value(M, "name") is None

    def test_extract_discriminated_variants_no_disc(self):
        from dataeval_flow._app._model._registry import _extract_discriminated_variants

        # Pass a non-Union, non-Annotated type
        assert _extract_discriminated_variants(int) is None

    def test_get_fields_unknown_variant_model(self):
        state = ConfigState()
        # "datasets" is discriminated; "nonexistent_variant" returns None model
        fields = get_fields("datasets", "nonexistent_variant", state)
        assert fields == []


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------


class TestCoercion:
    def test_coerce_int(self):
        assert coerce_value("42", "int") == 42

    def test_coerce_float(self):
        assert coerce_value("3.14", "float") == 3.14

    def test_coerce_bool(self):
        assert coerce_value("true", "bool") is True
        assert coerce_value("false", "bool") is False

    def test_coerce_json_list(self):
        assert coerce_value("[1, 2, 3]", "list[int]") == [1, 2, 3]

    def test_coerce_str_passthrough(self):
        assert coerce_value("hello", "str") == "hello"

    def test_coerce_union_str_int(self):
        assert coerce_value("42", "str | int") == 42
        assert coerce_value("hello", "str | int") == "hello"

    def test_coerce_empty(self):
        assert coerce_value("", "int") == ""

    def test_validate_value_int(self):
        assert validate_value("42", "int") is True
        assert validate_value("abc", "int") is False

    def test_validate_value_float(self):
        assert validate_value("3.14", "float") is True

    def test_validate_value_bool(self):
        assert validate_value("true", "bool") is True
        assert validate_value("maybe", "bool") is False

    def test_coerce_field_value_int(self):
        from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind

        desc = FieldDescriptor(name="x", kind=FieldKind.INT, description="", required=True)
        assert coerce_field_value("42", desc) == 42

    def test_coerce_field_value_float(self):
        from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind

        desc = FieldDescriptor(name="x", kind=FieldKind.FLOAT, description="", required=True)
        assert coerce_field_value("3.14", desc) == 3.14

    def test_coerce_field_value_bool(self):
        from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind

        desc = FieldDescriptor(name="x", kind=FieldKind.BOOL, description="", required=True)
        assert coerce_field_value("true", desc) is True
        assert coerce_field_value("false", desc) is False


# ---------------------------------------------------------------------------
# ConfigState
# ---------------------------------------------------------------------------


class TestConfigState:
    def _sample_state(self) -> ConfigState:
        state = ConfigState()
        state.add("datasets", {"name": "ds1", "format": "huggingface", "path": "data/ds1"})
        state.add("datasets", {"name": "ds2", "format": "coco", "path": "data/ds2"})
        state.add("selections", {"name": "sel1", "steps": [{"type": "Limit", "params": {"size": 100}}]})
        state.add("sources", {"name": "src1", "dataset": "ds1", "selection": "sel1"})
        state.add(
            "workflows",
            {"name": "wf1", "type": "data-cleaning", "outlier_method": "adaptive", "outlier_flags": ["dimension"]},
        )
        state.add(
            "tasks", {"name": "t1", "workflow": "wf1", "sources": "src1", "output_format": "json", "enabled": True}
        )
        return state

    def test_empty_state(self):
        state = ConfigState()
        assert state.is_empty()
        for key in SECTION_KEYS:
            assert state.count(key) == 0

    def test_add_and_query(self):
        state = ConfigState()
        state.add("datasets", {"name": "ds1", "format": "huggingface", "path": "data"})
        assert state.count("datasets") == 1
        assert state.names("datasets") == ["ds1"]
        assert state.get("datasets", 0) == {"name": "ds1", "format": "huggingface", "path": "data"}
        assert not state.is_empty()

    def test_update(self):
        state = ConfigState()
        state.add("datasets", {"name": "ds1", "format": "huggingface", "path": "old"})
        state.update("datasets", 0, {"name": "ds1", "format": "huggingface", "path": "new"})
        dataset = state.get("datasets", 0)
        assert dataset is not None
        assert dataset["path"] == "new"

    def test_remove(self):
        state = ConfigState()
        state.add("datasets", {"name": "ds1", "format": "huggingface", "path": "data"})
        name, warnings = state.remove("datasets", 0)
        assert name == "ds1"
        assert state.count("datasets") == 0

    def test_remove_invalid_index(self):
        state = ConfigState()
        name, warnings = state.remove("datasets", 5)
        assert name == ""

    def test_get_invalid_index(self):
        state = ConfigState()
        assert state.get("datasets", 5) is None

    # -- Scrubbing --

    def test_remove_dataset_scrubs_source(self):
        state = self._sample_state()
        name, warnings = state.remove("datasets", 0)  # remove ds1
        assert name == "ds1"
        # Source that depended on ds1 should be removed
        assert state.count("sources") == 0
        # Task that depended on the source should also be removed
        assert state.count("tasks") == 0

    def test_remove_selection_scrubs_source_field(self):
        state = self._sample_state()
        name, warnings = state.remove("selections", 0)  # remove sel1
        assert name == "sel1"
        # Source still exists but selection field is gone
        src = state.get("sources", 0)
        assert src is not None
        assert "selection" not in src

    def test_remove_workflow_scrubs_tasks(self):
        state = self._sample_state()
        name, warnings = state.remove("workflows", 0)  # remove wf1
        assert name == "wf1"
        assert state.count("tasks") == 0

    # -- Load / Save --

    def test_load_dict(self):
        data = {
            "datasets": [{"name": "ds1", "format": "huggingface", "path": "data"}],
            "tasks": [{"name": "t1", "workflow": "wf1", "sources": "src1"}],
        }
        state = ConfigState()
        state.load_dict(data)
        assert state.count("datasets") == 1
        assert state.count("tasks") == 1
        # Tasks should default to enabled
        task = state.get("tasks", 0)
        assert task is not None
        assert task["enabled"] is True

    def test_to_dict_roundtrip(self):
        state = self._sample_state()
        d = state.to_dict()
        assert "datasets" in d
        assert "tasks" in d
        assert len(d["datasets"]) == 2

    def test_save_and_load_yaml(self):
        state = self._sample_state()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            path = Path(f.name)
        state.save_file(path)
        assert path.exists()

        state2 = ConfigState()
        state2.load_file(path)
        assert state2.count("datasets") == 2
        assert state2.count("tasks") == 1
        path.unlink()

    def test_save_and_load_json(self):
        state = self._sample_state()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = Path(f.name)
        state.save_file(path)
        assert path.exists()

        content = json.loads(path.read_text())
        assert "datasets" in content
        path.unlink()

    def test_save_auto_suffix(self):
        state = ConfigState()
        state.add("datasets", {"name": "ds1", "format": "huggingface", "path": "data"})
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            path = Path(f.name)
        yaml_path = path.with_suffix(".yaml")
        state.save_file(path)
        assert yaml_path.exists()
        yaml_path.unlink()
        if path.exists():
            path.unlink()

    # -- Snapshot / Restore --

    def test_snapshot_restore(self):
        state = self._sample_state()
        snap = state.snapshot()
        state.add("datasets", {"name": "ds3", "format": "yolo", "path": "data/ds3"})
        assert state.count("datasets") == 3
        state.restore(snap)
        assert state.count("datasets") == 2

    def test_snapshot_independence(self):
        state = ConfigState()
        state.add("datasets", {"name": "ds1", "format": "huggingface", "path": "data"})
        snap = state.snapshot()
        # Mutating original should not affect snapshot
        state.add("datasets", {"name": "ds2", "format": "coco", "path": "data"})
        assert len(snap["datasets"]) == 1

    # -- Validation --

    def test_validate_item_valid(self):
        state = ConfigState()
        errors = state.validate_item("datasets", {"name": "ds1", "format": "huggingface", "path": "data"})
        assert errors == []

    def test_validate_item_invalid(self):
        state = ConfigState()
        errors = state.validate_item("datasets", {"name": "ds1", "format": "huggingface"})
        assert len(errors) > 0  # missing path

    def test_validate_item_unknown_variant(self):
        state = ConfigState()
        errors = state.validate_item("datasets", {"name": "ds1", "format": "nonexistent", "path": "data"})
        assert len(errors) > 0

    def test_validate_all_empty(self):
        state = ConfigState()
        errors = state.validate_all()
        assert errors == []  # empty config is valid

    def test_items_returns_copy(self):
        state = ConfigState()
        state.add("datasets", {"name": "ds1", "format": "huggingface", "path": "data"})
        items = state.items("datasets")
        items.append({"name": "ds2"})
        assert state.count("datasets") == 1


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_sections_coverage(self):
        assert len(SECTIONS) == 7
        assert set(SECTION_KEYS) == {
            "datasets",
            "preprocessors",
            "selections",
            "sources",
            "extractors",
            "workflows",
            "tasks",
        }

    def test_variant_registry_keys(self):
        assert set(VARIANT_REGISTRY) == {"datasets", "extractors", "workflows"}

    def test_cross_refs(self):
        assert "sources" in CROSS_REFS
        assert "tasks" in CROSS_REFS
        assert "extractors" in CROSS_REFS

    def test_step_builder_sections(self):
        assert set(STEP_BUILDER_SECTIONS) == {"preprocessors", "selections"}


# ---------------------------------------------------------------------------
# UndoStack / UndoEntry
# ---------------------------------------------------------------------------


class TestUndoEntry:
    def test_fields(self):
        from dataeval_flow._app._model._undo import UndoEntry

        state = {"datasets": [{"name": "ds1"}]}
        entry = UndoEntry(state=state, description="test")
        assert entry.state == state
        assert entry.description == "test"


class TestUndoStack:
    @staticmethod
    def _state(n=1):
        return {"datasets": [{"name": f"ds{i}"} for i in range(n)]}

    def test_initial_state(self):
        from dataeval_flow._app._model._undo import UndoStack

        stack = UndoStack()
        assert not stack.can_undo
        assert not stack.can_redo

    def test_push_enables_undo(self):
        from dataeval_flow._app._model._undo import UndoStack

        stack = UndoStack()
        stack.push(self._state(), "add dataset")
        assert stack.can_undo
        assert not stack.can_redo

    def test_push_clears_redo(self):
        from dataeval_flow._app._model._undo import UndoStack

        stack = UndoStack()
        stack.push(self._state(1), "first")
        stack.undo(self._state(2))
        assert stack.can_redo
        stack.push(self._state(3), "new action")
        assert not stack.can_redo

    def test_undo_returns_entry(self):
        from dataeval_flow._app._model._undo import UndoStack

        stack = UndoStack()
        original = self._state(1)
        stack.push(original, "add dataset")
        entry = stack.undo(self._state(2))
        assert entry is not None
        assert entry.description == "add dataset"
        assert entry.state == original

    def test_undo_empty_returns_none(self):
        from dataeval_flow._app._model._undo import UndoStack

        stack = UndoStack()
        assert stack.undo(self._state()) is None

    def test_redo_returns_entry(self):
        from dataeval_flow._app._model._undo import UndoStack

        stack = UndoStack()
        stack.push(self._state(1), "add")
        current = self._state(2)
        stack.undo(current)
        entry = stack.redo(self._state(1))
        assert entry is not None
        assert entry.state == current

    def test_redo_empty_returns_none(self):
        from dataeval_flow._app._model._undo import UndoStack

        stack = UndoStack()
        assert stack.redo(self._state()) is None

    def test_max_depth_enforced(self):
        from dataeval_flow._app._model._undo import UndoStack

        stack = UndoStack(max_depth=3)
        for i in range(5):
            stack.push(self._state(i), f"action {i}")
        count = 0
        while stack.can_undo:
            stack.undo(self._state())
            count += 1
        assert count == 3

    def test_push_deepcopies_state(self):
        from dataeval_flow._app._model._undo import UndoStack

        stack = UndoStack()
        state = self._state(1)
        stack.push(state, "add")
        state["datasets"][0]["name"] = "mutated"
        entry = stack.undo(self._state())
        assert entry is not None
        assert entry.state["datasets"][0]["name"] == "ds0"

    def test_multiple_undo_redo_cycle(self):
        from dataeval_flow._app._model._undo import UndoStack

        stack = UndoStack()
        stack.push(self._state(1), "a")
        stack.push(self._state(2), "b")
        stack.push(self._state(3), "c")
        e3 = stack.undo(self._state(4))
        assert e3 is not None
        assert e3.description == "c"
        e2 = stack.undo(self._state(4))
        assert e2 is not None
        assert e2.description == "b"
        e1 = stack.undo(self._state(4))
        assert e1 is not None
        assert e1.description == "a"
        assert not stack.can_undo
        r1 = stack.redo(self._state(4))
        assert r1 is not None
        assert r1.description == "a"
        r2 = stack.redo(self._state(4))
        assert r2 is not None
        assert r2.description == "b"
        r3 = stack.redo(self._state(4))
        assert r3 is not None
        assert r3.description == "c"
        assert not stack.can_redo


# ---------------------------------------------------------------------------
# apply_modal_result
# ---------------------------------------------------------------------------


class TestApplyModalResult:
    def test_cancel_returns_none(self):
        state = ConfigState()
        assert state.apply_modal_result("datasets", 0, None) is None

    def test_non_dict_non_sentinel_returns_none(self):
        state = ConfigState()
        assert state.apply_modal_result("datasets", 0, "random string") is None

    def test_add_item(self):
        state = ConfigState()
        item = {"name": "ds1", "format": "huggingface", "path": "data"}
        result = state.apply_modal_result("datasets", -1, item)
        assert result is not None
        desc, warnings = result
        assert "Add" in desc
        assert "ds1" in desc
        assert warnings == []
        assert state.count("datasets") == 1

    def test_update_item(self):
        state = ConfigState()
        state.add("datasets", {"name": "ds1", "format": "huggingface", "path": "data"})
        updated = {"name": "ds1", "format": "coco", "path": "new"}
        result = state.apply_modal_result("datasets", 0, updated)
        assert result is not None
        desc, _ = result
        assert "Update" in desc
        item = state.get("datasets", 0)
        assert item is not None
        assert item["format"] == "coco"

    def test_delete_item(self):
        from dataeval_flow._app._model._item import DELETE_SENTINEL

        state = ConfigState()
        state.add("datasets", {"name": "ds1", "format": "huggingface", "path": "data"})
        result = state.apply_modal_result("datasets", 0, DELETE_SENTINEL)
        assert result is not None
        desc, _ = result
        assert "Delete" in desc
        assert "ds1" in desc
        assert state.count("datasets") == 0

    def test_delete_with_cascade_warnings(self):
        from dataeval_flow._app._model._item import DELETE_SENTINEL

        state = ConfigState()
        state.add("datasets", {"name": "ds1", "format": "huggingface", "path": "data"})
        state.add("sources", {"name": "src1", "dataset": "ds1"})
        result = state.apply_modal_result("datasets", 0, DELETE_SENTINEL)
        assert result is not None
        _, warnings = result
        assert len(warnings) > 0
        assert state.count("sources") == 0

    def test_add_task_gets_enabled_default(self):
        state = ConfigState()
        item = {"name": "t1", "workflow": "wf1", "sources": "src1"}
        state.apply_modal_result("tasks", -1, item)
        task = state.get("tasks", 0)
        assert task is not None
        assert task["enabled"] is True


# ---------------------------------------------------------------------------
# finalize_item / build_item_dict
# ---------------------------------------------------------------------------


class TestFinalizeItem:
    def test_task_gets_enabled_default(self):
        from dataeval_flow._app._model._item import finalize_item

        item = {"name": "t1", "workflow": "wf1"}
        result = finalize_item("tasks", item)
        assert result["enabled"] is True

    def test_task_preserves_existing_enabled(self):
        from dataeval_flow._app._model._item import finalize_item

        item = {"name": "t1", "workflow": "wf1", "enabled": False}
        result = finalize_item("tasks", item)
        assert result["enabled"] is False

    def test_non_task_unchanged(self):
        from dataeval_flow._app._model._item import finalize_item

        item = {"name": "ds1", "format": "huggingface"}
        result = finalize_item("datasets", item)
        assert "enabled" not in result


class TestBuildItemDict:
    def test_basic_no_variant(self):
        from dataeval_flow._app._model._item import build_item_dict

        result = build_item_dict("sources", "src1", None, {"dataset": "ds1"})
        assert result == {"name": "src1", "dataset": "ds1"}

    def test_with_variant(self):
        from dataeval_flow._app._model._item import build_item_dict

        result = build_item_dict("datasets", "ds1", "huggingface", {"path": "data"})
        assert result == {"name": "ds1", "format": "huggingface", "path": "data"}

    def test_task_gets_enabled(self):
        from dataeval_flow._app._model._item import build_item_dict

        result = build_item_dict("tasks", "t1", None, {"workflow": "wf1", "sources": "src1"})
        assert result["enabled"] is True

    def test_empty_field_values(self):
        from dataeval_flow._app._model._item import build_item_dict

        result = build_item_dict("sources", "src1", None, {})
        assert result == {"name": "src1"}


# ---------------------------------------------------------------------------
# validate_step_params / coerce_step_params
# ---------------------------------------------------------------------------


class _FakeParam:
    """Minimal ParamInfo stand-in for testing."""

    def __init__(self, name, type_hint, required, default=None, choices=None):
        self.name = name
        self.type_hint = type_hint
        self.required = required
        self.default = default
        self.choices = choices or []


class TestValidateStepParams:
    def test_required_missing(self):
        from dataeval_flow._app._model._item import validate_step_params

        params = [_FakeParam("size", "int", required=True)]
        errors = validate_step_params(params, {"size": None})
        assert any("required" in e for e in errors)

    def test_required_present(self):
        from dataeval_flow._app._model._item import validate_step_params

        params = [_FakeParam("size", "int", required=True)]
        errors = validate_step_params(params, {"size": "42"})
        assert errors == []

    def test_optional_empty_ok(self):
        from dataeval_flow._app._model._item import validate_step_params

        params = [_FakeParam("size", "int", required=False)]
        errors = validate_step_params(params, {"size": None})
        assert errors == []

    def test_invalid_type(self):
        from dataeval_flow._app._model._item import validate_step_params

        params = [_FakeParam("size", "int", required=False)]
        errors = validate_step_params(params, {"size": "abc"})
        assert any("must be" in e for e in errors)

    def test_choice_required_missing(self):
        from dataeval_flow._app._model._item import validate_step_params

        params = [_FakeParam("mode", "select", required=True, choices=["a", "b"])]
        errors = validate_step_params(params, {"mode": ""})
        assert any("required" in e for e in errors)

    def test_choice_present(self):
        from dataeval_flow._app._model._item import validate_step_params

        params = [_FakeParam("mode", "select", required=True, choices=["a", "b"])]
        errors = validate_step_params(params, {"mode": "a"})
        assert errors == []

    def test_bool_always_valid(self):
        from dataeval_flow._app._model._item import validate_step_params

        params = [_FakeParam("flag", "bool", required=True)]
        errors = validate_step_params(params, {"flag": True})
        assert errors == []

    def test_missing_key_skipped(self):
        from dataeval_flow._app._model._item import validate_step_params

        params = [_FakeParam("size", "int", required=True)]
        errors = validate_step_params(params, {})
        assert errors == []


class TestCoerceStepParams:
    def test_int_coercion(self):
        from dataeval_flow._app._model._item import coerce_step_params

        params = [_FakeParam("size", "int", required=True)]
        result = coerce_step_params(params, {"size": "42"})
        assert result == {"size": 42}

    def test_float_coercion(self):
        from dataeval_flow._app._model._item import coerce_step_params

        params = [_FakeParam("rate", "float", required=True)]
        result = coerce_step_params(params, {"rate": "0.5"})
        assert result == {"rate": 0.5}

    def test_choice_passthrough(self):
        from dataeval_flow._app._model._item import coerce_step_params

        params = [_FakeParam("mode", "select", required=True, choices=["a", "b"])]
        result = coerce_step_params(params, {"mode": "a"})
        assert result == {"mode": "a"}

    def test_empty_choice_skipped(self):
        from dataeval_flow._app._model._item import coerce_step_params

        params = [_FakeParam("mode", "select", required=False, choices=["a"])]
        result = coerce_step_params(params, {"mode": None})
        assert result == {}

    def test_bool_changed_from_default(self):
        from dataeval_flow._app._model._item import coerce_step_params

        params = [_FakeParam("flag", "bool", required=False, default=False)]
        result = coerce_step_params(params, {"flag": True})
        assert result == {"flag": True}

    def test_bool_same_as_default_skipped(self):
        from dataeval_flow._app._model._item import coerce_step_params

        params = [_FakeParam("flag", "bool", required=False, default=False)]
        result = coerce_step_params(params, {"flag": False})
        assert result == {}

    def test_empty_string_skipped(self):
        from dataeval_flow._app._model._item import coerce_step_params

        params = [_FakeParam("name", "str", required=False)]
        result = coerce_step_params(params, {"name": None})
        assert result == {}

    def test_missing_key_skipped(self):
        from dataeval_flow._app._model._item import coerce_step_params

        params = [_FakeParam("size", "int", required=True)]
        result = coerce_step_params(params, {})
        assert result == {}

    def test_bool_non_bool_value_skipped(self):
        from dataeval_flow._app._model._item import coerce_step_params

        params = [_FakeParam("flag", "bool", required=False, default=False)]
        result = coerce_step_params(params, {"flag": None})
        assert result == {}

    def test_bool_string_value_skipped(self):
        from dataeval_flow._app._model._item import coerce_step_params

        params = [_FakeParam("flag", "bool", required=False, default=False)]
        result = coerce_step_params(params, {"flag": "true"})
        assert result == {}
