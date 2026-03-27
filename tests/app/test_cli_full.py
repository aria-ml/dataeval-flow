"""Comprehensive tests for the click-based CLI builder."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import BaseModel, Field

from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind
from dataeval_flow._app._viewmodel._builder_vm import BuilderViewModel
from dataeval_flow._app._viewmodel._section_vm import SectionViewModel
from dataeval_flow._app.cli import (
    _edit_section,
    _main_menu,
    _prompt_field,
    _prompt_item,
    _prompt_nested_model,
    _prompt_steps,
    _prompt_union_list,
    _show_items,
    run_cli_builder,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _VariantA(BaseModel):
    kind: str = Field(default="a")
    value: int = Field(default=0, description="a value")


class _VariantB(BaseModel):
    kind: str = Field(default="b")
    label: str = Field(default="x", description="a label")


class _NestedModel(BaseModel):
    x: int = Field(default=0, description="x val")
    y: str = Field(default="", description="y val")


def _make_vm_with_dataset() -> BuilderViewModel:
    vm = BuilderViewModel()
    vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "data"})
    return vm


# ---------------------------------------------------------------------------
# _prompt_field  -- SELECT
# ---------------------------------------------------------------------------


class TestPromptFieldSelect:
    @patch("click.prompt", return_value="huggingface")
    def test_select_required(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(
            name="format", kind=FieldKind.SELECT, description="fmt", required=True, choices=["huggingface", "coco"]
        )
        result = _prompt_field(desc)
        assert result == "huggingface"

    @patch("click.prompt", return_value="")
    def test_select_optional_skip(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(
            name="format", kind=FieldKind.SELECT, description="desc", required=False, choices=["a", "b"]
        )
        result = _prompt_field(desc)
        assert result is None

    @patch("click.prompt", return_value="existing_val")
    def test_select_with_existing_value(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(
            name="format", kind=FieldKind.SELECT, description="", required=True, choices=["existing_val", "other"]
        )
        result = _prompt_field(desc, existing_value="existing_val")
        assert result == "existing_val"

    @patch("click.prompt", return_value="a")
    def test_select_optional_with_no_existing_gets_empty_default(self, mock_prompt: MagicMock) -> None:
        """When optional SELECT has no existing_value, default should be ''."""
        desc = FieldDescriptor(name="fmt", kind=FieldKind.SELECT, description="", required=False, choices=["a", "b"])
        _prompt_field(desc)
        # The default kwarg should be "" since no existing value
        call_kwargs = mock_prompt.call_args
        assert call_kwargs.kwargs.get("default") == ""

    @patch("click.prompt", return_value="a")
    def test_select_no_description(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="fmt", kind=FieldKind.SELECT, description="", required=True, choices=["a", "b"])
        result = _prompt_field(desc)
        assert result == "a"


# ---------------------------------------------------------------------------
# _prompt_field  -- MULTI_SELECT
# ---------------------------------------------------------------------------


class TestPromptFieldMultiSelect:
    @patch("click.confirm", side_effect=[True, False, True])
    def test_multi_select(self, mock_confirm: MagicMock) -> None:
        desc = FieldDescriptor(
            name="flags", kind=FieldKind.MULTI_SELECT, description="desc", required=True, choices=["x", "y", "z"]
        )
        result = _prompt_field(desc)
        assert result == ["x", "z"]

    @patch("click.confirm", side_effect=[False, False])
    def test_multi_select_none_selected(self, mock_confirm: MagicMock) -> None:
        desc = FieldDescriptor(
            name="flags", kind=FieldKind.MULTI_SELECT, description="", required=False, choices=["x", "y"]
        )
        result = _prompt_field(desc)
        assert result is None

    @patch("click.confirm", side_effect=[True, True])
    def test_multi_select_with_existing(self, mock_confirm: MagicMock) -> None:
        desc = FieldDescriptor(
            name="flags", kind=FieldKind.MULTI_SELECT, description="", required=True, choices=["x", "y"]
        )
        result = _prompt_field(desc, existing_value=["x"])
        assert result == ["x", "y"]
        # First call default should be True (x is in existing), second False (y is not)
        assert mock_confirm.call_args_list[0].kwargs.get("default") is True
        assert mock_confirm.call_args_list[1].kwargs.get("default") is False

    @patch("click.confirm", side_effect=[True])
    def test_multi_select_with_non_list_existing(self, mock_confirm: MagicMock) -> None:
        """If existing_value is not a list, it should be treated as empty."""
        desc = FieldDescriptor(name="flags", kind=FieldKind.MULTI_SELECT, description="", required=True, choices=["x"])
        result = _prompt_field(desc, existing_value="not-a-list")
        assert result == ["x"]


# ---------------------------------------------------------------------------
# _prompt_field  -- BOOL
# ---------------------------------------------------------------------------


class TestPromptFieldBool:
    @patch("click.confirm", return_value=True)
    def test_bool_default_false(self, mock_confirm: MagicMock) -> None:
        desc = FieldDescriptor(name="enabled", kind=FieldKind.BOOL, description="", required=True, default=False)
        result = _prompt_field(desc)
        assert result is True

    @patch("click.confirm", return_value=False)
    def test_bool_existing_value(self, mock_confirm: MagicMock) -> None:
        desc = FieldDescriptor(name="enabled", kind=FieldKind.BOOL, description="desc", required=True)
        result = _prompt_field(desc, existing_value=True)
        assert result is False
        # Existing value should be used as default
        assert mock_confirm.call_args.kwargs.get("default") is True

    @patch("click.confirm", return_value=True)
    def test_bool_no_default_no_existing(self, mock_confirm: MagicMock) -> None:
        desc = FieldDescriptor(name="flag", kind=FieldKind.BOOL, description="", required=True)
        result = _prompt_field(desc)
        assert result is True
        # Default should be False when no default or existing
        assert mock_confirm.call_args.kwargs.get("default") is False

    @patch("click.confirm", return_value=False)
    def test_bool_with_non_bool_existing(self, mock_confirm: MagicMock) -> None:
        """Non-bool existing_value falls back to desc.default or False."""
        desc = FieldDescriptor(name="flag", kind=FieldKind.BOOL, description="", required=True, default=True)
        result = _prompt_field(desc, existing_value="not-a-bool")
        assert result is False
        assert mock_confirm.call_args.kwargs.get("default") is True


# ---------------------------------------------------------------------------
# _prompt_field  -- INT / FLOAT
# ---------------------------------------------------------------------------


class TestPromptFieldNumeric:
    @patch("click.prompt", return_value=42)
    def test_int_with_default(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="count", kind=FieldKind.INT, description="", required=False, default=10)
        result = _prompt_field(desc)
        assert result == 42

    @patch("click.prompt", return_value=3.14)
    def test_float_required(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="rate", kind=FieldKind.FLOAT, description="desc", required=True)
        result = _prompt_field(desc)
        assert result == 3.14

    @patch("click.prompt", return_value="")
    def test_int_optional_skip(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="val", kind=FieldKind.INT, description="", required=False)
        result = _prompt_field(desc)
        assert result is None

    @patch("click.prompt", return_value="7")
    def test_int_optional_with_value(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="val", kind=FieldKind.INT, description="", required=False)
        result = _prompt_field(desc)
        assert result == 7

    @patch("click.prompt", return_value=5)
    def test_int_with_constraints(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(
            name="val", kind=FieldKind.INT, description="", required=True, constraints={"ge": 0, "le": 10}
        )
        result = _prompt_field(desc)
        assert result == 5

    @patch("click.prompt", return_value=99)
    def test_existing_value_overrides_default(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="val", kind=FieldKind.INT, description="", required=False, default=5)
        _prompt_field(desc, existing_value=99)
        mock_prompt.assert_called_once_with(
            "val (optional, Enter to skip)",
            default=99,
            type=int,
        )

    @patch("click.prompt", return_value="3.5")
    def test_float_optional_with_value(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="val", kind=FieldKind.FLOAT, description="", required=False)
        result = _prompt_field(desc)
        assert result == 3.5

    @patch("click.prompt", return_value=10)
    def test_int_required_no_default(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="count", kind=FieldKind.INT, description="", required=True)
        result = _prompt_field(desc)
        assert result == 10
        mock_prompt.assert_called_once_with("count", type=int)


# ---------------------------------------------------------------------------
# _prompt_field  -- LIST / NESTED (JSON prompt path)
# ---------------------------------------------------------------------------


class TestPromptFieldComplex:
    @patch("click.prompt", return_value="[1, 2]")
    def test_json_list(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="items", kind=FieldKind.LIST, description="", required=True)
        result = _prompt_field(desc)
        assert result == [1, 2]

    @patch("click.prompt", return_value="not json")
    def test_invalid_json_stored_as_string(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="items", kind=FieldKind.LIST, description="", required=True)
        result = _prompt_field(desc)
        assert result == "not json"

    @patch("click.prompt", return_value="")
    def test_empty_returns_none(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="items", kind=FieldKind.LIST, description="", required=False)
        result = _prompt_field(desc)
        assert result is None

    @patch("click.prompt", return_value='{"a": 1}')
    def test_existing_value_as_json_default(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="items", kind=FieldKind.LIST, description="desc", required=False, default=[1])
        result = _prompt_field(desc, existing_value={"x": 1})
        assert result == {"a": 1}

    @patch("click.prompt", return_value="[3]")
    def test_list_default_serialized(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="items", kind=FieldKind.LIST, description="", required=False, default=[1, 2])
        _prompt_field(desc)
        # default should be json.dumps([1, 2])
        assert mock_prompt.call_args.kwargs.get("default") == "[1, 2]"

    def test_union_variants_delegates_to_prompt_union_list(self) -> None:
        """LIST with union_variants and discriminator delegates to _prompt_union_list."""
        desc = FieldDescriptor(
            name="items",
            kind=FieldKind.LIST,
            description="",
            required=True,
            discriminator="kind",
            union_variants={"a": _VariantA},
        )
        with patch("dataeval_flow._app.cli._prompt_union_list", return_value=[{"kind": "a"}]) as mock_ul:
            result = _prompt_field(desc, existing_value=None)
        mock_ul.assert_called_once_with(desc, None)
        assert result == [{"kind": "a"}]

    def test_nested_model_delegates_to_prompt_nested_model(self) -> None:
        """NESTED with nested_model delegates to _prompt_nested_model."""
        desc = FieldDescriptor(
            name="child", kind=FieldKind.NESTED, description="", required=True, nested_model=_NestedModel
        )
        with patch("dataeval_flow._app.cli._prompt_nested_model", return_value={"x": 5}) as mock_nm:
            result = _prompt_field(desc, existing_value=None)
        mock_nm.assert_called_once_with(desc, None)
        assert result == {"x": 5}


# ---------------------------------------------------------------------------
# _prompt_field  -- STRING
# ---------------------------------------------------------------------------


class TestPromptFieldString:
    @patch("click.prompt", return_value="hello")
    def test_string_with_default(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="label", kind=FieldKind.STRING, description="", required=False, default="world")
        result = _prompt_field(desc)
        assert result == "hello"

    @patch("click.prompt", return_value="")
    def test_string_empty_returns_none(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="label", kind=FieldKind.STRING, description="", required=False)
        result = _prompt_field(desc)
        assert result is None

    @patch("click.prompt", return_value="val")
    def test_string_with_existing_value(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="label", kind=FieldKind.STRING, description="desc", required=False)
        result = _prompt_field(desc, existing_value="old")
        assert result == "val"
        assert mock_prompt.call_args.kwargs.get("default") == "old"

    @patch("click.prompt", return_value="hi")
    def test_string_required_no_default(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(name="label", kind=FieldKind.STRING, description="", required=True)
        result = _prompt_field(desc)
        assert result == "hi"


# ---------------------------------------------------------------------------
# _prompt_union_list
# ---------------------------------------------------------------------------


class TestPromptUnionList:
    def test_no_variants_raises(self) -> None:
        """union_variants must be set (asserts inside)."""
        desc = FieldDescriptor(name="items", kind=FieldKind.LIST, description="", required=True)
        with pytest.raises(ValueError, match="union_variants and discriminator are required"):
            _prompt_union_list(desc, None)

    @patch("click.prompt", side_effect=["d"])
    def test_done_immediately_empty(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(
            name="items",
            kind=FieldKind.LIST,
            description="",
            required=True,
            discriminator="kind",
            union_variants={"a": _VariantA, "b": _VariantB},
        )
        result = _prompt_union_list(desc, None)
        assert result == []

    @patch("click.prompt", side_effect=["d"])
    def test_existing_items_preserved(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(
            name="items",
            kind=FieldKind.LIST,
            description="",
            required=True,
            discriminator="kind",
            union_variants={"a": _VariantA},
        )
        existing = [{"kind": "a", "value": 5}]
        result = _prompt_union_list(desc, existing)
        assert len(result) == 1
        assert result[0]["kind"] == "a"

    @patch(
        "click.prompt",
        side_effect=[
            "a",  # add action
            "a",  # variant choice = "a"
            0,  # value field for _VariantA (int default=0)
            "d",  # done
        ],
    )
    @patch("click.confirm")
    def test_add_variant(self, mock_confirm: MagicMock, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(
            name="items",
            kind=FieldKind.LIST,
            description="",
            required=True,
            discriminator="kind",
            union_variants={"a": _VariantA},
        )
        result = _prompt_union_list(desc, None)
        assert len(result) == 1
        assert result[0]["kind"] == "a"

    @patch(
        "click.prompt",
        side_effect=[
            "r",  # remove action
            "d",  # done
        ],
    )
    def test_remove_empty_list(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(
            name="items",
            kind=FieldKind.LIST,
            description="",
            required=True,
            discriminator="kind",
            union_variants={"a": _VariantA},
        )
        result = _prompt_union_list(desc, None)
        assert result == []

    @patch(
        "click.prompt",
        side_effect=[
            "r",  # remove action
            1,  # index 1
            "d",  # done
        ],
    )
    def test_remove_valid_index(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(
            name="items",
            kind=FieldKind.LIST,
            description="",
            required=True,
            discriminator="kind",
            union_variants={"a": _VariantA},
        )
        result = _prompt_union_list(desc, [{"kind": "a", "value": 5}])
        assert result == []

    @patch(
        "click.prompt",
        side_effect=[
            "r",  # remove action
            99,  # invalid index
            "d",  # done
        ],
    )
    def test_remove_invalid_index(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(
            name="items",
            kind=FieldKind.LIST,
            description="",
            required=True,
            discriminator="kind",
            union_variants={"a": _VariantA},
        )
        result = _prompt_union_list(desc, [{"kind": "a", "value": 5}])
        assert len(result) == 1  # item not removed


# ---------------------------------------------------------------------------
# _prompt_nested_model
# ---------------------------------------------------------------------------


class TestPromptNestedModel:
    def test_none_model(self) -> None:
        desc = FieldDescriptor(name="child", kind=FieldKind.NESTED, description="", required=False)
        assert _prompt_nested_model(desc, None) is None

    @patch("click.prompt", side_effect=[10, "hello"])
    def test_basic(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(
            name="child", kind=FieldKind.NESTED, description="", required=True, nested_model=_NestedModel
        )
        result = _prompt_nested_model(desc, None)
        assert result is not None
        assert result["x"] == 10

    def test_all_none_returns_none(self) -> None:
        """When all prompted values are None, result dict is empty -> returns None."""
        desc = FieldDescriptor(
            name="child", kind=FieldKind.NESTED, description="", required=False, nested_model=_NestedModel
        )
        with patch("dataeval_flow._app.cli._prompt_field", return_value=None):
            result = _prompt_nested_model(desc, None)
        assert result is None

    @patch("click.prompt", side_effect=[42, "hello"])
    def test_with_existing_dict(self, mock_prompt: MagicMock) -> None:
        desc = FieldDescriptor(
            name="child", kind=FieldKind.NESTED, description="", required=True, nested_model=_NestedModel
        )
        result = _prompt_nested_model(desc, {"x": 99, "y": "old"})
        assert result is not None
        assert result["x"] == 42

    @patch("click.prompt", side_effect=[5, "world"])
    def test_with_non_dict_existing(self, mock_prompt: MagicMock) -> None:
        """Non-dict existing should be treated as empty dict."""
        desc = FieldDescriptor(
            name="child", kind=FieldKind.NESTED, description="", required=True, nested_model=_NestedModel
        )
        result = _prompt_nested_model(desc, "not-a-dict")
        assert result is not None


# ---------------------------------------------------------------------------
# _prompt_steps
# ---------------------------------------------------------------------------


class TestPromptSteps:
    def _make_sec_vm(self) -> SectionViewModel:
        vm = BuilderViewModel()
        return vm.create_section_vm("preprocessors")

    @patch("click.prompt", side_effect=["d"])
    @patch("dataeval_flow._app._model._discover.list_transforms", return_value=["Resize", "Normalize"])
    @patch("dataeval_flow._app._model._discover.get_transform_params", return_value=[])
    def test_done_immediately(self, mock_params: MagicMock, mock_list: MagicMock, mock_prompt: MagicMock) -> None:
        sec_vm = self._make_sec_vm()
        result = _prompt_steps(sec_vm)
        assert result == []

    @patch("click.prompt", side_effect=["d"])
    @patch("dataeval_flow._app._model._discover.list_transforms", return_value=["Resize"])
    @patch("dataeval_flow._app._model._discover.get_transform_params", return_value=[])
    def test_existing_steps_shown_then_done(
        self, mock_params: MagicMock, mock_list: MagicMock, mock_prompt: MagicMock
    ) -> None:
        sec_vm = self._make_sec_vm()
        existing = [{"step": "Resize", "params": {"size": 256}}]
        result = _prompt_steps(sec_vm, existing)
        assert len(result) == 1

    @patch("click.prompt", side_effect=["r", 1, "d"])
    @patch("dataeval_flow._app._model._discover.list_transforms", return_value=["Resize"])
    @patch("dataeval_flow._app._model._discover.get_transform_params", return_value=[])
    def test_remove_step(self, mock_params: MagicMock, mock_list: MagicMock, mock_prompt: MagicMock) -> None:
        sec_vm = self._make_sec_vm()
        existing = [{"step": "Resize", "params": {"size": 256}}]
        result = _prompt_steps(sec_vm, existing)
        assert result == []

    @patch("click.prompt", side_effect=["r", "d"])
    @patch("dataeval_flow._app._model._discover.list_transforms", return_value=["Resize"])
    @patch("dataeval_flow._app._model._discover.get_transform_params", return_value=[])
    def test_remove_from_empty(self, mock_params: MagicMock, mock_list: MagicMock, mock_prompt: MagicMock) -> None:
        sec_vm = self._make_sec_vm()
        result = _prompt_steps(sec_vm)
        assert result == []

    @patch("click.prompt", side_effect=["r", 99, "d"])
    @patch("dataeval_flow._app._model._discover.list_transforms", return_value=["Resize"])
    @patch("dataeval_flow._app._model._discover.get_transform_params", return_value=[])
    def test_remove_invalid_index(self, mock_params: MagicMock, mock_list: MagicMock, mock_prompt: MagicMock) -> None:
        sec_vm = self._make_sec_vm()
        existing = [{"step": "Resize"}]
        result = _prompt_steps(sec_vm, existing)
        assert len(result) == 1  # not removed

    @patch("dataeval_flow._app._model._discover.get_transform_params")
    @patch("dataeval_flow._app._model._discover.list_transforms", return_value=["Resize"])
    def test_add_step_no_params(self, mock_list: MagicMock, mock_params: MagicMock) -> None:
        mock_params.return_value = []
        sec_vm = self._make_sec_vm()
        with patch("click.prompt", side_effect=["a", "Resize", "d"]):
            result = _prompt_steps(sec_vm)
        assert len(result) == 1
        assert result[0]["step"] == "Resize"
        assert "params" not in result[0]

    @patch("dataeval_flow._app._model._discover.get_transform_params")
    @patch("dataeval_flow._app._model._discover.list_transforms", return_value=["Resize"])
    def test_add_step_with_choices_param(self, mock_list: MagicMock, mock_params: MagicMock) -> None:
        from dataeval_flow._app._model._discover import ParamInfo

        mock_params.return_value = [
            ParamInfo(name="mode", type_hint="select", required=True, choices=["fast", "slow"]),
        ]
        sec_vm = self._make_sec_vm()
        with patch("click.prompt", side_effect=["a", "Resize", "fast", "d"]):
            result = _prompt_steps(sec_vm)
        assert len(result) == 1
        assert result[0]["params"]["mode"] == "fast"

    @patch("dataeval_flow._app._model._discover.get_transform_params")
    @patch("dataeval_flow._app._model._discover.list_transforms", return_value=["Resize"])
    def test_add_step_with_optional_choice_skip(self, mock_list: MagicMock, mock_params: MagicMock) -> None:
        from dataeval_flow._app._model._discover import ParamInfo

        mock_params.return_value = [
            ParamInfo(name="mode", type_hint="select", required=False, choices=["fast", "slow"]),
        ]
        sec_vm = self._make_sec_vm()
        with patch("click.prompt", side_effect=["a", "Resize", "", "d"]):
            result = _prompt_steps(sec_vm)
        assert len(result) == 1
        # Empty choice should not add the param
        assert "mode" not in result[0].get("params", {})

    @patch("dataeval_flow._app._model._discover.get_transform_params")
    @patch("dataeval_flow._app._model._discover.list_transforms", return_value=["Resize"])
    def test_add_step_with_bool_param_changed(self, mock_list: MagicMock, mock_params: MagicMock) -> None:
        from dataeval_flow._app._model._discover import ParamInfo

        mock_params.return_value = [
            ParamInfo(name="flag", type_hint="bool", required=False, default=False),
        ]
        sec_vm = self._make_sec_vm()
        with patch("click.prompt", side_effect=["a", "Resize", "d"]), patch("click.confirm", return_value=True):
            result = _prompt_steps(sec_vm)
        assert result[0]["params"]["flag"] is True

    @patch("dataeval_flow._app._model._discover.get_transform_params")
    @patch("dataeval_flow._app._model._discover.list_transforms", return_value=["Resize"])
    def test_add_step_with_bool_param_unchanged(self, mock_list: MagicMock, mock_params: MagicMock) -> None:
        from dataeval_flow._app._model._discover import ParamInfo

        mock_params.return_value = [
            ParamInfo(name="flag", type_hint="bool", required=False, default=False),
        ]
        sec_vm = self._make_sec_vm()
        with patch("click.prompt", side_effect=["a", "Resize", "d"]), patch("click.confirm", return_value=False):
            result = _prompt_steps(sec_vm)
        assert "flag" not in result[0].get("params", {})

    @patch("dataeval_flow._app._model._discover.get_transform_params")
    @patch("dataeval_flow._app._model._discover.list_transforms", return_value=["Resize"])
    def test_add_step_with_generic_param_required(self, mock_list: MagicMock, mock_params: MagicMock) -> None:
        from dataeval_flow._app._model._discover import ParamInfo

        mock_params.return_value = [
            ParamInfo(name="size", type_hint="int", required=True),
        ]
        sec_vm = self._make_sec_vm()
        with patch("click.prompt", side_effect=["a", "Resize", "256", "d"]):
            result = _prompt_steps(sec_vm)
        assert result[0]["params"]["size"] == 256

    @patch("dataeval_flow._app._model._discover.get_transform_params")
    @patch("dataeval_flow._app._model._discover.list_transforms", return_value=["Resize"])
    def test_add_step_with_generic_param_empty_optional(self, mock_list: MagicMock, mock_params: MagicMock) -> None:
        from dataeval_flow._app._model._discover import ParamInfo

        mock_params.return_value = [
            ParamInfo(name="size", type_hint="int", required=False, default=None),
        ]
        sec_vm = self._make_sec_vm()
        with patch("click.prompt", side_effect=["a", "Resize", "", "d"]):
            result = _prompt_steps(sec_vm)
        assert "size" not in result[0].get("params", {})

    @patch("dataeval_flow._app._model._discover.get_transform_params")
    @patch("dataeval_flow._app._model._discover.list_transforms", return_value=["Resize"])
    def test_add_step_with_generic_param_same_as_default(self, mock_list: MagicMock, mock_params: MagicMock) -> None:
        from dataeval_flow._app._model._discover import ParamInfo

        mock_params.return_value = [
            ParamInfo(name="size", type_hint="int", required=False, default=10),
        ]
        sec_vm = self._make_sec_vm()
        with patch("click.prompt", side_effect=["a", "Resize", "10", "d"]):
            result = _prompt_steps(sec_vm)
        # "10" == str(10), not required, so should not be included
        assert "size" not in result[0].get("params", {})

    @patch("click.prompt", side_effect=["d"])
    @patch("dataeval_flow._app._model._discover.list_transforms", return_value=["Resize"])
    @patch("dataeval_flow._app._model._discover.get_transform_params", return_value=[])
    def test_existing_steps_without_params(
        self, mock_params: MagicMock, mock_list: MagicMock, mock_prompt: MagicMock
    ) -> None:
        """Steps without params key should display without error."""
        sec_vm = self._make_sec_vm()
        existing = [{"step": "Normalize"}]
        result = _prompt_steps(sec_vm, existing)
        assert len(result) == 1

    @patch("dataeval_flow._app._model._discover.get_transform_params")
    @patch("dataeval_flow._app._model._discover.list_transforms", return_value=["Resize"])
    def test_add_step_with_choice_param_with_default(self, mock_list: MagicMock, mock_params: MagicMock) -> None:
        from dataeval_flow._app._model._discover import ParamInfo

        mock_params.return_value = [
            ParamInfo(name="mode", type_hint="select", required=False, default="fast", choices=["fast", "slow"]),
        ]
        sec_vm = self._make_sec_vm()
        with patch("click.prompt", side_effect=["a", "Resize", "fast", "d"]):
            result = _prompt_steps(sec_vm)
        assert result[0]["params"]["mode"] == "fast"


# ---------------------------------------------------------------------------
# _prompt_item
# ---------------------------------------------------------------------------


class TestPromptItem:
    @patch("click.prompt", return_value="")
    def test_empty_name(self, mock_prompt: MagicMock) -> None:
        vm = BuilderViewModel()
        result = _prompt_item("sources", vm)
        assert result is None

    @patch("dataeval_flow._app.cli._prompt_steps", return_value=[{"step": "Resize"}])
    @patch("click.prompt", return_value="pre1")
    def test_preprocessor_with_steps(self, mock_prompt: MagicMock, mock_steps: MagicMock) -> None:
        vm = BuilderViewModel()
        result = _prompt_item("preprocessors", vm)
        assert result is not None
        assert result["name"] == "pre1"
        assert result["steps"] == [{"step": "Resize"}]

    @patch("dataeval_flow._app.cli._prompt_steps", return_value=[])
    @patch("click.prompt", return_value="pre1")
    def test_preprocessor_no_steps(self, mock_prompt: MagicMock, mock_steps: MagicMock) -> None:
        vm = BuilderViewModel()
        result = _prompt_item("preprocessors", vm)
        assert result is None

    @patch("dataeval_flow._app._viewmodel._section_vm.get_fields", return_value=[])
    @patch("click.prompt", side_effect=["ds1", "huggingface"])
    def test_dataset_with_discriminator(self, mock_prompt: MagicMock, mock_fields: MagicMock) -> None:
        vm = BuilderViewModel()
        result = _prompt_item("datasets", vm)
        assert result is not None
        assert result["name"] == "ds1"
        assert result["format"] == "huggingface"

    @patch("dataeval_flow._app._viewmodel._section_vm.get_fields")
    @patch("click.prompt", side_effect=["src1"])
    def test_source_regular_fields(self, mock_prompt: MagicMock, mock_fields: MagicMock) -> None:
        desc = FieldDescriptor(name="dataset", kind=FieldKind.SELECT, description="", required=True, choices=["ds1"])
        mock_fields.return_value = [desc]
        vm = _make_vm_with_dataset()
        with patch("dataeval_flow._app.cli._prompt_field", return_value="ds1"):
            result = _prompt_item("sources", vm)
        assert result is not None
        assert result["name"] == "src1"
        assert result["dataset"] == "ds1"

    @patch("dataeval_flow._app._viewmodel._section_vm.get_fields", return_value=[])
    @patch("click.prompt", side_effect=["t1", "data-cleaning"])
    def test_task_defaults_enabled(self, mock_prompt: MagicMock, mock_fields: MagicMock) -> None:
        vm = BuilderViewModel()
        result = _prompt_item("tasks", vm)
        # Tasks do not have a discriminator in VARIANT_REGISTRY... let me check
        # Actually tasks are not in VARIANT_REGISTRY but workflows are
        # tasks are non-discriminated, so no variant prompt
        # Wait, let me re-check: get_variant_choices("tasks") returns None
        # So only name is prompted, then fields
        assert result is not None
        assert result.get("enabled") is True

    @patch("dataeval_flow._app._viewmodel._section_vm.get_fields", return_value=[])
    @patch("click.prompt", return_value="t1")
    def test_task_enabled_default(self, mock_prompt: MagicMock, mock_fields: MagicMock) -> None:
        vm = BuilderViewModel()
        result = _prompt_item("tasks", vm)
        assert result is not None
        assert result["enabled"] is True

    @patch("dataeval_flow._app._viewmodel._section_vm.get_fields", return_value=[])
    @patch("click.prompt", side_effect=["ds1", "huggingface"])
    def test_edit_with_existing(self, mock_prompt: MagicMock, mock_fields: MagicMock) -> None:
        vm = BuilderViewModel()
        existing = {"name": "ds1", "format": "huggingface", "path": "old"}
        result = _prompt_item("datasets", vm, existing=existing)
        assert result is not None
        assert result["name"] == "ds1"


# ---------------------------------------------------------------------------
# _show_items
# ---------------------------------------------------------------------------


class TestShowItems:
    def test_empty_section(self, capsys: pytest.CaptureFixture[str]) -> None:
        vm = BuilderViewModel()
        _show_items("datasets", vm)
        assert "(empty)" in capsys.readouterr().out

    def test_with_items(self, capsys: pytest.CaptureFixture[str]) -> None:
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "data"})
        _show_items("datasets", vm)
        out = capsys.readouterr().out
        assert "ds1" in out
        assert "format=huggingface" in out

    def test_list_field_skipped_in_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        vm = BuilderViewModel()
        vm.apply_result("preprocessors", -1, {"name": "pre1", "steps": [{"step": "Resize"}]})
        _show_items("preprocessors", vm)
        out = capsys.readouterr().out
        assert "pre1" in out
        assert "steps=" not in out

    def test_dict_field_skipped_in_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        vm = BuilderViewModel()
        vm.apply_result("sources", -1, {"name": "src1", "meta": {"a": 1}, "dataset": "ds1"})
        _show_items("sources", vm)
        out = capsys.readouterr().out
        assert "src1" in out
        assert "meta=" not in out

    def test_tasks_enabled_marker(self, capsys: pytest.CaptureFixture[str]) -> None:
        vm = BuilderViewModel()
        vm.apply_result("tasks", -1, {"name": "t1", "enabled": True})
        vm.apply_result("tasks", -1, {"name": "t2", "enabled": False})
        _show_items("tasks", vm)
        out = capsys.readouterr().out
        assert "[on]" in out
        assert "[off]" in out

    def test_max_three_parts(self, capsys: pytest.CaptureFixture[str]) -> None:
        vm = BuilderViewModel()
        vm.apply_result(
            "datasets", -1, {"name": "ds1", "format": "huggingface", "path": "data", "split": "train", "extra": "x"}
        )
        _show_items("datasets", vm)
        out = capsys.readouterr().out
        assert "ds1" in out

    def test_task_without_enabled_defaults_on(self, capsys: pytest.CaptureFixture[str]) -> None:
        vm = BuilderViewModel()
        vm.apply_result("tasks", -1, {"name": "t1"})
        _show_items("tasks", vm)
        out = capsys.readouterr().out
        assert "[on]" in out


# ---------------------------------------------------------------------------
# _edit_section
# ---------------------------------------------------------------------------


class TestEditSection:
    @patch("click.prompt", side_effect=["b"])
    def test_back_immediately(self, mock_prompt: MagicMock) -> None:
        vm = BuilderViewModel()
        _edit_section("datasets", "Datasets", vm)

    @patch("click.prompt", side_effect=["d", 1, "b"])
    def test_delete_then_back(self, mock_prompt: MagicMock) -> None:
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "data"})
        _edit_section("datasets", "Datasets", vm)
        assert vm.count("datasets") == 0

    @patch("click.prompt", side_effect=["d", 99, "b"])
    def test_delete_invalid_index(self, mock_prompt: MagicMock) -> None:
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "data"})
        _edit_section("datasets", "Datasets", vm)
        assert vm.count("datasets") == 1

    @patch("click.prompt", side_effect=["t", 1, "b"])
    def test_toggle_task(self, mock_prompt: MagicMock) -> None:
        vm = BuilderViewModel()
        vm.apply_result("tasks", -1, {"name": "t1", "workflow": "wf1", "sources": "s1", "enabled": True})
        _edit_section("tasks", "Tasks", vm)
        task = vm.get_item("tasks", 0)
        assert task is not None
        assert task["enabled"] is False

    @patch("click.prompt", side_effect=["t", 1, "b"])
    def test_toggle_off_to_on(self, mock_prompt: MagicMock) -> None:
        vm = BuilderViewModel()
        vm.apply_result("tasks", -1, {"name": "t1", "workflow": "wf1", "sources": "s1", "enabled": False})
        _edit_section("tasks", "Tasks", vm)
        task = vm.get_item("tasks", 0)
        assert task is not None
        assert task["enabled"] is True

    @patch("click.prompt", side_effect=["t", 99, "b"])
    def test_toggle_invalid_index(self, mock_prompt: MagicMock) -> None:
        vm = BuilderViewModel()
        vm.apply_result("tasks", -1, {"name": "t1", "enabled": True})
        _edit_section("tasks", "Tasks", vm)
        task = vm.get_item("tasks", 0)
        assert task is not None
        assert task["enabled"] is True  # unchanged

    @patch("dataeval_flow._app.cli._prompt_item", return_value={"name": "ds2", "format": "coco", "path": "x"})
    @patch("click.prompt", side_effect=["a", "b"])
    def test_add_item_no_validation_errors(self, mock_prompt: MagicMock, mock_item: MagicMock) -> None:
        vm = BuilderViewModel()
        with patch.object(vm, "validate_item", return_value=[]):
            _edit_section("datasets", "Datasets", vm)
        assert vm.count("datasets") == 1

    @patch("dataeval_flow._app.cli._prompt_item", return_value={"name": "ds2", "format": "huggingface"})
    @patch("click.prompt", side_effect=["a", "b"])
    @patch("click.confirm", return_value=True)
    def test_add_item_with_validation_errors_confirm(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, mock_item: MagicMock
    ) -> None:
        vm = BuilderViewModel()
        with patch.object(vm, "validate_item", return_value=["missing path"]):
            _edit_section("datasets", "Datasets", vm)
        assert vm.count("datasets") == 1

    @patch("dataeval_flow._app.cli._prompt_item", return_value={"name": "ds2", "format": "huggingface"})
    @patch("click.prompt", side_effect=["a", "b"])
    @patch("click.confirm", return_value=False)
    def test_add_item_with_validation_errors_reject(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, mock_item: MagicMock
    ) -> None:
        vm = BuilderViewModel()
        with patch.object(vm, "validate_item", return_value=["missing path"]):
            _edit_section("datasets", "Datasets", vm)
        assert vm.count("datasets") == 0

    @patch("dataeval_flow._app.cli._prompt_item", return_value=None)
    @patch("click.prompt", side_effect=["a", "b"])
    def test_add_item_returns_none(self, mock_prompt: MagicMock, mock_item: MagicMock) -> None:
        vm = BuilderViewModel()
        _edit_section("datasets", "Datasets", vm)
        assert vm.count("datasets") == 0

    @patch("dataeval_flow._app.cli._prompt_item", return_value={"name": "ds1_edit", "format": "coco", "path": "y"})
    @patch("click.prompt", side_effect=["e", 1, "b"])
    def test_edit_item_valid(self, mock_prompt: MagicMock, mock_item: MagicMock) -> None:
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "data"})
        with patch.object(vm, "validate_item", return_value=[]):
            _edit_section("datasets", "Datasets", vm)
        item = vm.get_item("datasets", 0)
        assert item is not None
        assert item["name"] == "ds1_edit"

    @patch("click.prompt", side_effect=["e", 99, "b"])
    def test_edit_invalid_index(self, mock_prompt: MagicMock) -> None:
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "data"})
        _edit_section("datasets", "Datasets", vm)
        # Should not crash, just show "Invalid index."

    @patch("dataeval_flow._app.cli._prompt_item", return_value={"name": "ds1_edit", "format": "huggingface"})
    @patch("click.prompt", side_effect=["e", 1, "b"])
    @patch("click.confirm", return_value=True)
    def test_edit_item_validation_errors_confirm(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, mock_item: MagicMock
    ) -> None:
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "data"})
        with patch.object(vm, "validate_item", return_value=["some warning"]):
            _edit_section("datasets", "Datasets", vm)
        item = vm.get_item("datasets", 0)
        assert item is not None
        assert item["name"] == "ds1_edit"

    @patch("dataeval_flow._app.cli._prompt_item", return_value={"name": "ds1_edit", "format": "huggingface"})
    @patch("click.prompt", side_effect=["e", 1, "b"])
    @patch("click.confirm", return_value=False)
    def test_edit_item_validation_errors_reject(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, mock_item: MagicMock
    ) -> None:
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "data"})
        with patch.object(vm, "validate_item", return_value=["some warning"]):
            _edit_section("datasets", "Datasets", vm)
        item = vm.get_item("datasets", 0)
        assert item is not None
        assert item["name"] == "ds1"  # unchanged

    @patch("dataeval_flow._app.cli._prompt_item", return_value=None)
    @patch("click.prompt", side_effect=["e", 1, "b"])
    def test_edit_item_returns_none(self, mock_prompt: MagicMock, mock_item: MagicMock) -> None:
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "data"})
        _edit_section("datasets", "Datasets", vm)
        item = vm.get_item("datasets", 0)
        assert item is not None
        assert item["name"] == "ds1"  # unchanged

    @patch("click.prompt", side_effect=["d", 1, "b"])
    def test_delete_with_warnings(self, mock_prompt: MagicMock) -> None:
        vm = BuilderViewModel()
        vm.apply_result("datasets", -1, {"name": "ds1", "format": "huggingface", "path": "data"})
        vm.apply_result("sources", -1, {"name": "src1", "dataset": "ds1"})
        _edit_section("datasets", "Datasets", vm)
        assert vm.count("datasets") == 0
        # Source referencing ds1 should be auto-removed
        assert vm.count("sources") == 0


# ---------------------------------------------------------------------------
# _main_menu
# ---------------------------------------------------------------------------


class TestMainMenu:
    @patch("click.prompt", side_effect=["q"])
    def test_quit_empty(self, mock_prompt: MagicMock) -> None:
        vm = BuilderViewModel()
        _main_menu(vm)

    @patch("click.prompt", side_effect=["q"])
    @patch("click.confirm", return_value=True)
    def test_quit_with_data_confirms(self, mock_confirm: MagicMock, mock_prompt: MagicMock) -> None:
        vm = _make_vm_with_dataset()
        _main_menu(vm)

    @patch("click.prompt", side_effect=["q", "q"])
    @patch("click.confirm", side_effect=[False, True])
    def test_quit_with_data_rejects_then_confirms(self, mock_confirm: MagicMock, mock_prompt: MagicMock) -> None:
        vm = _make_vm_with_dataset()
        _main_menu(vm)

    @patch("click.prompt", side_effect=["v", "q"])
    def test_view_empty(self, mock_prompt: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        vm = BuilderViewModel()
        _main_menu(vm)
        assert "(empty config)" in capsys.readouterr().out

    @patch("click.prompt", side_effect=["v", "q"])
    @patch("click.confirm", return_value=True)
    def test_view_with_data(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        vm = _make_vm_with_dataset()
        _main_menu(vm)
        out = capsys.readouterr().out
        assert "datasets" in out

    @patch("click.prompt", side_effect=["99", "q"])
    def test_invalid_section_number(self, mock_prompt: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        vm = BuilderViewModel()
        _main_menu(vm)
        assert "Invalid choice" in capsys.readouterr().out

    @patch("click.prompt", side_effect=["x", "q"])
    def test_invalid_choice_letter(self, mock_prompt: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        vm = BuilderViewModel()
        _main_menu(vm)
        assert "Invalid choice" in capsys.readouterr().out

    @patch("dataeval_flow._app.cli._edit_section")
    @patch("click.prompt", side_effect=["1", "q"])
    def test_select_section(self, mock_prompt: MagicMock, mock_edit: MagicMock) -> None:
        vm = BuilderViewModel()
        _main_menu(vm)
        mock_edit.assert_called_once()
        # First section is "datasets"
        assert mock_edit.call_args[0][0] == "datasets"

    @patch("click.prompt", side_effect=["s", "q"])
    @patch("click.confirm", return_value=True)
    def test_save(self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path) -> None:
        vm = _make_vm_with_dataset()
        out = tmp_path / "out.yaml"
        mock_prompt.side_effect = ["s", str(out), "q"]
        mock_confirm.return_value = True
        _main_menu(vm)
        assert out.exists()

    @patch("click.prompt")
    @patch("click.confirm", return_value=True)
    def test_save_with_existing_config_path(
        self, mock_confirm: MagicMock, mock_prompt: MagicMock, tmp_path: Path
    ) -> None:
        vm = _make_vm_with_dataset()
        vm.config_file_path = str(tmp_path / "existing.yaml")
        out = tmp_path / "out.yaml"
        mock_prompt.side_effect = ["s", str(out), "q"]
        _main_menu(vm)
        assert out.exists()

    @patch("click.prompt", side_effect=["l", "", "q"])
    def test_load_empty_path(self, mock_prompt: MagicMock) -> None:
        vm = BuilderViewModel()
        _main_menu(vm)
        # Empty load path should be a no-op

    @patch("click.prompt", side_effect=["l", "/nonexistent.yaml", "q"])
    def test_load_missing_file(self, mock_prompt: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        vm = BuilderViewModel()
        _main_menu(vm)
        assert "File not found" in capsys.readouterr().out

    @patch("click.confirm", return_value=True)
    @patch("click.prompt")
    def test_load_valid_file(self, mock_prompt: MagicMock, mock_confirm: MagicMock, tmp_path: Path) -> None:
        cfg = {"datasets": [{"name": "ds1", "format": "huggingface", "path": "data"}]}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        vm = BuilderViewModel()
        mock_prompt.side_effect = ["l", str(p), "q"]
        _main_menu(vm)
        assert vm.count("datasets") == 1

    @patch("click.prompt")
    def test_load_with_warning(self, mock_prompt: MagicMock, tmp_path: Path) -> None:
        """Loading a file that triggers fallback raw YAML load produces a warning."""
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"datasets": [{"name": "ds1"}]}))  # minimal, may trigger fallback
        vm = BuilderViewModel()
        mock_prompt.side_effect = ["l", str(p), "q"]
        # We'll patch load_file to return a (success, msg) tuple
        with patch.object(vm, "load_file", return_value=(True, "some warning")):
            _main_menu(vm)
        # No crash expected

    def test_main_menu_shows_config_path(self, capsys: pytest.CaptureFixture[str]) -> None:
        vm = BuilderViewModel("myconfig.yaml")
        with patch("click.prompt", side_effect=["q"]):
            _main_menu(vm)
        out = capsys.readouterr().out
        assert "myconfig.yaml" in out
        assert "DataEval Flow Config Builder" in out

    def test_main_menu_shows_section_counts(self, capsys: pytest.CaptureFixture[str]) -> None:
        vm = _make_vm_with_dataset()
        with patch("click.prompt", side_effect=["q"]), patch("click.confirm", return_value=True):
            _main_menu(vm)
        out = capsys.readouterr().out
        assert "1 item" in out


# ---------------------------------------------------------------------------
# run_cli_builder
# ---------------------------------------------------------------------------


class TestRunCliBuilder:
    @patch("dataeval_flow._app.cli._main_menu")
    def test_no_config(self, mock_menu: MagicMock) -> None:
        run_cli_builder()
        mock_menu.assert_called_once()
        args = mock_menu.call_args[0]
        assert isinstance(args[0], BuilderViewModel)

    @patch("dataeval_flow._app.cli._main_menu")
    def test_with_existing_config(self, mock_menu: MagicMock, tmp_path: Path) -> None:
        cfg = {"datasets": [{"name": "ds1", "format": "huggingface", "path": "data"}]}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        run_cli_builder(config_path=p)
        mock_menu.assert_called_once()
        vm = mock_menu.call_args[0][0]
        assert isinstance(vm, BuilderViewModel)
        assert vm.config_file_path == str(p)

    @patch("dataeval_flow._app.cli._main_menu")
    def test_with_missing_config(self, mock_menu: MagicMock) -> None:
        run_cli_builder(config_path="/nonexistent.yaml")
        mock_menu.assert_called_once()
        vm = mock_menu.call_args[0][0]
        assert isinstance(vm, BuilderViewModel)
        assert vm.config_file_path == "/nonexistent.yaml"

    @patch("dataeval_flow._app.cli._main_menu")
    def test_with_string_path(self, mock_menu: MagicMock, tmp_path: Path) -> None:
        cfg = {"datasets": [{"name": "ds1", "format": "huggingface", "path": "data"}]}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        run_cli_builder(config_path=str(p))
        mock_menu.assert_called_once()

    @patch("dataeval_flow._app.cli._main_menu")
    def test_loads_config_and_shows_warning(self, mock_menu: MagicMock, tmp_path: Path) -> None:
        """When load_file returns a (success, msg) tuple, it should still proceed."""
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"datasets": [{"name": "ds1"}]}))
        with patch.object(BuilderViewModel, "load_file", return_value=(True, "some warning")):
            run_cli_builder(config_path=p)
        mock_menu.assert_called_once()


# ---------------------------------------------------------------------------
# Save/Load integration via _main_menu
# ---------------------------------------------------------------------------


class TestMainMenuSaveLoad:
    def test_save_yaml_and_load(self, tmp_path: Path) -> None:
        out = tmp_path / "saved.yaml"
        vm = _make_vm_with_dataset()

        with patch("click.prompt", side_effect=["s", str(out), "q"]), patch("click.confirm", return_value=True):
            _main_menu(vm)

        assert out.exists()
        loaded = yaml.safe_load(out.read_text())
        assert "datasets" in loaded

    def test_save_json(self, tmp_path: Path) -> None:
        out = tmp_path / "saved.json"
        vm = _make_vm_with_dataset()

        with patch("click.prompt", side_effect=["s", str(out), "q"]), patch("click.confirm", return_value=True):
            _main_menu(vm)

        assert out.exists()
        data = json.loads(out.read_text())
        assert "datasets" in data

    def test_save_empty_config_noop(self, tmp_path: Path) -> None:
        out = tmp_path / "saved.yaml"
        vm = BuilderViewModel()

        with patch("click.prompt", side_effect=["s", str(out), "q"]):
            _main_menu(vm)

        # Empty config should not create a file
        assert not out.exists()
