from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataeval_flow._app._model._coerce import (
    _split_type_alternatives,
    coerce_field_value,
    coerce_value,
    validate_value,
)
from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind
from dataeval_flow._app._model._item import (
    DELETE_SENTINEL,
    collect_bool_value,
    collect_field_value,
    collect_json_value,
    collect_multi_select_value,
    diagnose_collect_failure,
)
from dataeval_flow._app._model._state import ConfigState, _strip_empty_params, _to_dict


class TestCoerceLogic:
    @pytest.mark.parametrize(
        ("val", "type_hint", "expected"),
        [
            ("10", "int", 10),
            ("1.5", "float", 1.5),
            ("true", "bool", True),
            ("False", "bool", False),
            ("foo", "str", "foo"),
            ("[1, 2]", "list[int]", [1, 2]),
            ("[1, 2]", "list", [1, 2]),
            ('{"a": 1}', "dict[str, int]", {"a": 1}),
            ('{"a": 1}', "dict", {"a": 1}),
            ("invalid", "list", "invalid"),  # invalid json returns raw val
        ],
    )
    def test_coerce_value(self, val: str, type_hint: str, expected: Any) -> None:
        assert coerce_value(val, type_hint) == expected

    @pytest.mark.parametrize(
        ("val", "type_hint", "expected"),
        [
            ("10", "int", True),
            ("abc", "int", False),
            ("1.5", "float", True),
            ("abc", "float", False),
            ("true", "bool", True),
            ("maybe", "bool", False),
            ("[1, 2]", "list", True),
            ("(1, 2)", "tuple", False),  # json.loads("(1, 2)") fails
            ("[1, 2]", "tuple", True),  # json.loads("[1, 2]") is list, which is accepted
            ("invalid", "list", False),
            ("foo", "str", True),
        ],
    )
    def test_validate_value(self, val: str, type_hint: str, expected: bool) -> None:
        assert validate_value(val, type_hint) == expected

    def test_validate_unknown_type_returns_true(self) -> None:
        assert validate_value("anything", "SomeUnknownType") is True

    def test_coerce_field_value_invalid_json(self) -> None:
        desc_list = FieldDescriptor(name="l", kind=FieldKind.LIST, description="", required=True)
        assert coerce_field_value("invalid", desc_list) == "invalid"

    def test_split_type_alternatives(self) -> None:
        assert _split_type_alternatives("int | float") == ["int", "float"]
        assert _split_type_alternatives("int") == ["int"]

    def test_coerce_field_value_kinds(self) -> None:
        desc_bool = FieldDescriptor(name="b", kind=FieldKind.BOOL, description="", required=True)
        assert coerce_field_value("true", desc_bool) is True
        assert coerce_field_value("false", desc_bool) is False

        desc_str = FieldDescriptor(name="s", kind=FieldKind.STRING, description="", required=True)
        assert coerce_field_value("foo", desc_str) == "foo"

        desc_select = FieldDescriptor(name="sl", kind=FieldKind.SELECT, description="", required=True)
        assert coerce_field_value("choice1", desc_select) == "choice1"

        desc_multi = FieldDescriptor(name="ms", kind=FieldKind.MULTI_SELECT, description="", required=True)
        assert coerce_field_value("v1, v2", desc_multi) == "v1, v2"

        desc_list = FieldDescriptor(name="l", kind=FieldKind.LIST, description="", required=True)
        assert coerce_field_value("[1, 2]", desc_list) == [1, 2]

        desc_nested = FieldDescriptor(name="n", kind=FieldKind.NESTED, description="", required=True)
        assert coerce_field_value('{"a": 1}', desc_nested) == {"a": 1}

    def test_coerce_field_value_empty(self) -> None:
        desc = FieldDescriptor(name="k", kind=FieldKind.INT, description="", required=True)
        assert coerce_field_value("", desc) == ""


class TestItemLogic:
    def test_collect_field_value_empty(self) -> None:
        from dataeval_flow._app._model._item import SKIP

        desc = FieldDescriptor(name="k", kind=FieldKind.INT, description="", required=True)
        assert collect_field_value(desc, "") == SKIP

    def test_collect_json_value(self) -> None:
        from dataeval_flow._app._model._item import SKIP

        assert collect_json_value("") == SKIP
        assert collect_json_value("[1, 2]") == [1, 2]
        assert collect_json_value("invalid") == "invalid"

    def test_collect_multi_select_value(self) -> None:
        from dataeval_flow._app._model._item import SKIP

        assert collect_multi_select_value([], "datasets", "foo") == SKIP
        assert collect_multi_select_value(["a", "b"], "datasets", "foo") == ["a", "b"]
        # Unwrap for task sources
        assert collect_multi_select_value(["s1"], "tasks", "sources") == "s1"
        assert collect_multi_select_value(["s1", "s2"], "tasks", "sources") == ["s1", "s2"]

    def test_collect_bool_value(self) -> None:
        from dataeval_flow._app._model._item import SKIP

        assert collect_bool_value(True, False) is True
        assert collect_bool_value(False, False) == SKIP
        assert collect_bool_value(True, "not-bool") is True  # resolved default False

    def test_diagnose_collect_failure(self) -> None:
        assert diagnose_collect_failure("datasets", "", [], [], None) == "Name is required."
        assert diagnose_collect_failure("selections", "s1", [], [], None) == "Add at least one step."

        desc = FieldDescriptor(name="f1", kind=FieldKind.STRING, description="", required=True)
        # 'sources' does not have a discriminator, so it will check for missing fields
        assert "Required: f1" in diagnose_collect_failure("sources", "d1", [], [desc], {})

        # Test fallback "Invalid input."
        assert diagnose_collect_failure("sources", "d1", [], [], {"f1": "v1"}) == "Invalid input."


class TestConfigState:
    def test_init(self) -> None:
        state = ConfigState()
        assert state.is_empty()
        assert state.count("datasets") == 0

    def test_add_get_items(self) -> None:
        state = ConfigState()
        item = {"name": "ds1", "format": "huggingface"}
        state.add("datasets", item)
        assert state.count("datasets") == 1
        assert state.get("datasets", 0) == item
        assert state.items("datasets") == [item]
        assert state.names("datasets") == ["ds1"]

    def test_update(self) -> None:
        state = ConfigState()
        state.add("datasets", {"name": "ds1"})
        state.update("datasets", 0, {"name": "ds1-updated"})
        assert state.get("datasets", 0) == {"name": "ds1-updated"}

    def test_remove_with_scrubbing(self) -> None:
        state = ConfigState()
        state.add("datasets", {"name": "ds1"})
        state.add("sources", {"name": "s1", "dataset": "ds1"})
        state.add("tasks", {"name": "t1", "sources": ["s1"], "workflow": "w1"})

        name, warnings = state.remove("datasets", 0)
        assert name == "ds1"
        assert "Auto-removed source 's1'" in warnings
        assert "Auto-removed task 't1'" in warnings
        assert state.is_empty()

    def test_load_save_dict(self) -> None:
        state = ConfigState()
        data = {
            "datasets": [{"name": "ds1", "format": "huggingface"}],
            "tasks": [{"name": "t1", "workflow": "w1", "sources": ["s1"]}],
        }
        state.load_dict(data)
        assert state.count("datasets") == 1
        # Task should have enabled=True default
        task = state.get("tasks", 0)
        assert task is not None
        assert task["enabled"] is True

        out = state.to_dict()
        assert out["datasets"] == data["datasets"]

    def test_snapshot_restore(self) -> None:
        state = ConfigState()
        state.add("datasets", {"name": "ds1"})
        snap = state.snapshot()
        state.add("datasets", {"name": "ds2"})
        state.restore(snap)
        assert state.count("datasets") == 1
        assert state.names("datasets") == ["ds1"]

    def test_validate_item(self) -> None:
        state = ConfigState()
        # Valid item
        errs = state.validate_item("datasets", {"name": "ds1", "format": "huggingface", "path": "p"})
        assert len(errs) == 0
        # Invalid item (missing format)
        errs = state.validate_item("datasets", {"name": "ds1"})
        assert len(errs) > 0
        # Unknown section
        errs = state.validate_item("no_such_section", {})
        assert "Unknown" in errs[0]

    def test_validate_all(self) -> None:
        state = ConfigState()
        state.add("datasets", {"name": "ds1", "format": "huggingface", "path": "p"})
        errs = state.validate_all()
        # Should be empty or valid
        assert isinstance(errs, list)

    def test_apply_modal_result_cancel(self) -> None:
        state = ConfigState()
        assert state.apply_modal_result("datasets", -1, None) is None

    def test_apply_modal_result_delete(self) -> None:
        state = ConfigState()
        state.add("datasets", {"name": "ds1"})
        res = state.apply_modal_result("datasets", 0, DELETE_SENTINEL)
        assert res is not None
        assert "Delete dataset 'ds1'" in res[0]
        assert state.count("datasets") == 0

    def test_apply_modal_result_add(self) -> None:
        state = ConfigState()
        item = {"name": "ds1", "format": "huggingface", "path": "p"}
        res = state.apply_modal_result("datasets", -1, item)
        assert res is not None
        assert "Add dataset 'ds1'" in res[0]
        assert state.count("datasets") == 1

    def test_apply_modal_result_update(self) -> None:
        state = ConfigState()
        state.add("datasets", {"name": "ds1"})
        item = {"name": "ds1-mod", "format": "huggingface", "path": "p"}
        res = state.apply_modal_result("datasets", 0, item)
        assert res is not None
        assert "Update dataset 'ds1-mod'" in res[0]
        dataset = state.get("datasets", 0)
        assert dataset is not None
        assert dataset["name"] == "ds1-mod"

    def test_strip_empty_params(self) -> None:
        items = [
            {
                "name": "p1",
                "steps": [{"step": "s1", "params": {}}, {"step": "s2", "params": {"k": 1}}],
            }
        ]
        cleaned = _strip_empty_params(items)
        assert "params" not in cleaned[0]["steps"][0]
        assert "params" in cleaned[0]["steps"][1]

    def test_scrub_references_edge_cases(self) -> None:
        state = ConfigState()
        state.add("selections", {"name": "sel1"})
        state.add("sources", {"name": "s1", "dataset": "ds1", "selection": "sel1"})

        # Scrub optional reference (selections -> sources)
        _, warnings = state.remove("selections", 0)
        item = state.get("sources", 0)
        assert item is not None
        assert item.get("selection") is None
        assert not warnings  # Optional scrub doesn't add warning

        # Scrub extractor -> preprocessor
        state.add("preprocessors", {"name": "p1"})
        state.add("extractors", {"name": "e1", "preprocessor": "p1"})
        state.remove("preprocessors", 0)
        item = state.get("extractors", 0)
        assert item is not None
        assert item.get("preprocessor") is None

    def test_load_save_file(self, tmp_path: Path) -> None:
        state = ConfigState()
        state.add("datasets", {"name": "ds1", "format": "huggingface", "path": "p"})
        yaml_path = tmp_path / "config.yaml"
        state.save_file(yaml_path)
        assert yaml_path.exists()

        state2 = ConfigState()
        state2.load_file(yaml_path)
        assert state2.count("datasets") == 1
        assert state2.names("datasets") == ["ds1"]

        json_path = tmp_path / "config.json"
        state.save_file(json_path)
        assert json_path.exists()

        state3 = ConfigState()
        state3.load_file(json_path)
        assert state3.count("datasets") == 1

    def test_load_file_directory(self, tmp_path: Path) -> None:
        # Mock load_config_folder implicitly by using tmp_path as dir
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "ds.yaml").write_text("datasets:\n  - name: ds1\n    format: huggingface\n    path: p")

        state = ConfigState()
        state.load_file(config_dir)
        assert state.count("datasets") == 1

    def test_load_file_fallback(self, tmp_path: Path) -> None:
        # File that fails pydantic validation but is valid YAML
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("datasets:\n  - name: ds1\n    invalid_field: foo")

        state = ConfigState()
        msg = state.load_file(bad_file)
        assert msg is not None
        assert "Loaded as raw YAML" in msg
        assert state.count("datasets") == 1

    def test_scrub_task_field_list(self) -> None:
        state = ConfigState()
        task = {"name": "t1", "sources": ["s1", "s2"], "workflow": "w1"}
        # Scrub one of many
        assert state._scrub_task_field(task, "sources", "s1") is True
        assert task["sources"] == "s2"  # Unwrapped if 1 left

        # Scrub last one
        assert state._scrub_task_field(task, "sources", "s2") is False

    def test_scrub_task_field_single(self) -> None:
        state = ConfigState()
        task = {"name": "t1", "extractor": "e1", "workflow": "w1"}
        # Scrub optional field
        assert state._scrub_task_field(task, "extractor", "e1") is True
        assert "extractor" not in task

        # Scrub required field
        assert state._scrub_task_field(task, "workflow", "w1") is False

    def test_strip_empty_params_no_steps(self) -> None:
        items = [{"name": "p1", "other_field": "val"}]
        cleaned = _strip_empty_params(items)
        assert cleaned == [{"name": "p1", "other_field": "val"}]

    def test_to_dict_with_plain_dict(self) -> None:
        result = _to_dict({"name": "test", "value": 42})
        assert result == {"name": "test", "value": 42}

    def test_update_out_of_range(self) -> None:
        state = ConfigState()
        state.add("datasets", {"name": "ds1"})
        state.update("datasets", 99, {"name": "ds2"})
        # Should not change anything
        assert state.get("datasets", 0) == {"name": "ds1"}

    def test_save_file_empty_config(self, tmp_path: Path) -> None:
        state = ConfigState()
        path = tmp_path / "empty.yaml"
        state.save_file(path)
        assert not path.exists()

    def test_validate_all_invalid(self) -> None:
        state = ConfigState()
        # Add invalid data that will fail PipelineConfig validation
        state.add("datasets", {"name": "ds1"})  # missing format, path
        errs = state.validate_all()
        assert isinstance(errs, list)

    def test_scrub_task_field_list_no_match(self) -> None:
        state = ConfigState()
        task = {"name": "t1", "sources": ["s1", "s2"], "workflow": "w1"}
        assert state._scrub_task_field(task, "sources", "s3") is True
        assert task["sources"] == ["s1", "s2"]  # unchanged

    def test_scrub_task_field_scalar_no_match(self) -> None:
        state = ConfigState()
        task = {"name": "t1", "workflow": "w1"}
        assert state._scrub_task_field(task, "workflow", "other") is True
        assert task["workflow"] == "w1"  # unchanged

    def test_scrub_references_non_task_section(self) -> None:
        state = ConfigState()
        state.add("preprocessors", {"name": "p1"})
        state.add("extractors", {"name": "e1", "preprocessor": "p1"})
        _, warnings = state.remove("preprocessors", 0)
        ext = state.get("extractors", 0)
        assert ext is not None
        assert "preprocessor" not in ext

    def test_scrub_comp_no_match(self) -> None:
        state = ConfigState()
        state.add("datasets", {"name": "ds1"})
        state.add("sources", {"name": "s1", "dataset": "ds2"})  # different dataset
        _, warnings = state.remove("datasets", 0)
        # source should remain since it references ds2, not ds1
        assert state.count("sources") == 1
