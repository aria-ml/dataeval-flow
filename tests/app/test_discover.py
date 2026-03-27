"""Tests for the builder discovery module."""

import inspect
import typing
from typing import Literal

from dataeval_flow._app._model._discover import (
    ParamInfo,
    _introspect_params,
    _simplify_type,
    get_selection_params,
    get_transform_params,
    list_selection_classes,
    list_transforms,
)

# ── _simplify_type ────────────────────────────────────────────────────────


class TestSimplifyType:
    def test_empty_annotation(self):
        assert _simplify_type(inspect.Parameter.empty) == ("any", [])

    def test_any(self):
        from typing import Any

        assert _simplify_type(Any) == ("any", [])

    def test_int(self):
        assert _simplify_type(int) == ("int", [])

    def test_float(self):
        assert _simplify_type(float) == ("float", [])

    def test_bool(self):
        assert _simplify_type(bool) == ("bool", [])

    def test_str(self):
        assert _simplify_type(str) == ("str", [])

    def test_literal(self):
        hint, choices = _simplify_type(Literal["a", "b", "c"])
        assert hint == "select"
        assert choices == ["a", "b", "c"]

    def test_optional_int(self):
        hint, choices = _simplify_type(int | None)
        assert hint == "int"
        assert choices == []

    def test_union(self):
        hint, choices = _simplify_type(int | float)
        assert "int" in hint
        assert "float" in hint
        assert choices == []

    def test_list_int(self):
        assert _simplify_type(list[int]) == ("list[int]", [])

    def test_list_bare(self):
        assert _simplify_type(list) == ("list", [])

    def test_list_literal_returns_choices(self):
        hint, choices = _simplify_type(list[Literal["x", "y"]])
        assert hint == "select"
        assert choices == ["x", "y"]

    def test_fallback_class(self):
        class MyClass:
            pass

        hint, choices = _simplify_type(MyClass)
        assert hint == "MyClass"
        assert choices == []

    def test_sequence_type(self):
        hint, _ = _simplify_type(typing.Sequence[int])
        assert hint == "list[int]"


# ── _introspect_params ────────────────────────────────────────────────────


class TestIntrospectParams:
    def test_basic_params(self):
        class Dummy:
            def __init__(self, a: int, b: str = "hello"):
                pass

        params = _introspect_params(Dummy)
        assert len(params) == 2
        by_name = {p.name: p for p in params}
        assert by_name["a"].required is True
        assert by_name["a"].type_hint == "int"
        assert by_name["b"].required is False
        assert by_name["b"].default == "hello"

    def test_skips_self_args_kwargs(self):
        class Dummy:
            def __init__(self, x: int, *args, **kwargs):
                pass

        params = _introspect_params(Dummy)
        names = [p.name for p in params]
        assert names == ["x"]

    def test_literal_choices(self):
        class Dummy:
            def __init__(self, mode: Literal["fast", "slow"] = "fast"):
                pass

        params = _introspect_params(Dummy)
        assert params[0].choices == ["fast", "slow"]
        assert params[0].type_hint == "select"

    def test_broken_signature(self):
        # builtins like `object` may not have inspectable __init__
        params = _introspect_params(object)
        assert isinstance(params, list)

    def test_param_info_defaults(self):
        p = ParamInfo(name="x", type_hint="int", required=True)
        assert p.default is None
        assert p.choices == []


# ── list_transforms / list_selection_classes ───────────────────────────────


class TestListFunctions:
    def test_list_transforms_sorted_nonempty(self):
        names = list_transforms()
        assert len(names) > 0
        assert names == sorted(names)

    def test_list_transforms_excludes_skips(self):
        names = list_transforms()
        for skip in ("Transform", "Compose", "Identity", "Lambda"):
            assert skip not in names

    def test_list_transforms_includes_resize(self):
        names = list_transforms()
        assert "Resize" in names

    def test_list_selection_classes_sorted_nonempty(self):
        names = list_selection_classes()
        assert len(names) > 0
        assert names == sorted(names)

    def test_list_selection_classes_excludes_skips(self):
        names = list_selection_classes()
        for skip in ("Selection", "Subselection"):
            assert skip not in names


# ── get_transform_params / get_selection_params ───────────────────────────


class TestGetParams:
    def test_transform_params_valid(self):
        params = get_transform_params("Resize")
        assert len(params) > 0
        names = [p.name for p in params]
        assert "size" in names

    def test_transform_params_invalid(self):
        assert get_transform_params("NonExistentTransform") == []

    def test_selection_params_invalid(self):
        assert get_selection_params("NonExistentClass") == []


# ── Coverage: lru_cache clearing ───────────────────────────────────────────


class TestListFunctionsCoverage:
    """Tests that clear lru_cache to ensure coverage tracking."""

    def test_list_transforms_cache_cleared(self):
        list_transforms.cache_clear()
        names = list_transforms()
        assert len(names) > 0
        assert names == sorted(names)

    def test_list_selection_classes_cache_cleared(self):
        list_selection_classes.cache_clear()
        names = list_selection_classes()
        assert len(names) > 0
        assert names == sorted(names)


class TestGetParamsCoverage:
    """Tests that clear lru_cache to ensure coverage tracking."""

    def test_transform_params_cache_cleared(self):
        get_transform_params.cache_clear()
        params = get_transform_params("Resize")
        assert len(params) > 0

    def test_selection_params_cache_cleared(self):
        get_selection_params.cache_clear()
        # Get a valid selection class name
        names = list_selection_classes()
        if names:
            params = get_selection_params(names[0])
            assert isinstance(params, list)


# ── Coverage: _simplify_type bare list/tuple ────────────────────────────────


class TestSimplifyTypeCoverage:
    def test_typing_list_bare(self):
        # typing.List (unsubscripted) has get_origin() -> list but get_args() -> ()
        hint, choices = _simplify_type(list)  # type: ignore[attr-defined]
        assert hint == "list"
        assert choices == []


# ── Coverage: _introspect_params exception handling ─────────────────────────


class TestIntrospectParamsCoverage:
    def test_signature_raises_value_error(self):
        from unittest.mock import patch

        class Dummy:
            def __init__(self, x: int):
                pass

        with patch("inspect.signature", side_effect=ValueError("test")):
            params = _introspect_params(Dummy)
            assert params == []
