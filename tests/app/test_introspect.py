"""Tests for the builder introspection module."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from dataeval_flow._app._model._introspect import FieldKind, introspect_model


class SimpleModel(BaseModel):
    name: str = Field(description="A name")
    count: int = Field(default=10, ge=0, le=100, description="A count")
    rate: float = Field(default=0.5, description="A rate")
    enabled: bool = Field(default=True, description="Toggle")


class SelectModel(BaseModel):
    method: Literal["a", "b", "c"] = Field(default="a", description="Choose one")
    flags: list[Literal["x", "y", "z"]] = Field(description="Choose multiple")


class NestedChild(BaseModel):
    value: int = Field(default=0, description="Child value")


class ParentModel(BaseModel):
    child: NestedChild = Field(default_factory=NestedChild, description="Nested child")
    label: str = Field(default="", description="A label")


class OptionalFieldsModel(BaseModel):
    required_field: str = Field(description="This is required")
    optional_field: str | None = Field(default=None, description="This is optional")


def test_primitive_fields():
    descriptors = introspect_model(SimpleModel)
    by_name = {d.name: d for d in descriptors}

    assert by_name["name"].kind == FieldKind.STRING
    assert by_name["name"].required is True

    assert by_name["count"].kind == FieldKind.INT
    assert by_name["count"].default == 10
    assert by_name["count"].constraints["ge"] == 0
    assert by_name["count"].constraints["le"] == 100

    assert by_name["rate"].kind == FieldKind.FLOAT
    assert by_name["enabled"].kind == FieldKind.BOOL


def test_select_fields():
    descriptors = introspect_model(SelectModel)
    by_name = {d.name: d for d in descriptors}

    assert by_name["method"].kind == FieldKind.SELECT
    assert by_name["method"].choices == ["a", "b", "c"]

    assert by_name["flags"].kind == FieldKind.MULTI_SELECT
    assert by_name["flags"].choices == ["x", "y", "z"]


def test_nested_model():
    descriptors = introspect_model(ParentModel)
    by_name = {d.name: d for d in descriptors}

    assert by_name["child"].kind == FieldKind.NESTED
    assert by_name["child"].nested_model is NestedChild


def test_optional_fields():
    descriptors = introspect_model(OptionalFieldsModel)
    by_name = {d.name: d for d in descriptors}

    assert by_name["required_field"].required is True
    assert by_name["optional_field"].required is False


def test_real_cleaning_params():
    from dataeval_flow.workflows.cleaning.params import DataCleaningParameters

    descriptors = introspect_model(DataCleaningParameters)
    by_name = {d.name: d for d in descriptors}

    assert by_name["outlier_method"].kind == FieldKind.SELECT
    assert "adaptive" in by_name["outlier_method"].choices

    assert by_name["outlier_flags"].kind == FieldKind.MULTI_SELECT
    assert "dimension" in by_name["outlier_flags"].choices

    assert by_name["health_thresholds"].kind == FieldKind.NESTED


def test_real_drift_params():
    from dataeval_flow.workflows.drift.params import DriftMonitoringParameters

    descriptors = introspect_model(DriftMonitoringParameters)
    by_name = {d.name: d for d in descriptors}

    assert by_name["detectors"].kind == FieldKind.LIST
    assert by_name["health_thresholds"].kind == FieldKind.NESTED


# ---------------------------------------------------------------------------
# Coverage: _resolve_discriminated_union with non-string discriminator
# ---------------------------------------------------------------------------


def test_resolve_discriminated_union_non_string_disc():
    from pydantic.fields import FieldInfo as PydanticFieldInfo

    from dataeval_flow._app._model._introspect import _resolve_discriminated_union

    # Create a FieldInfo with a non-string discriminator
    fi = PydanticFieldInfo(discriminator=42)  # type: ignore
    d, v = _resolve_discriminated_union(int, fi)
    assert d is None
    assert v is None


# ---------------------------------------------------------------------------
# Coverage: _introspect_list_field returns None for plain inner type
# ---------------------------------------------------------------------------


def test_list_plain_inner_type():
    """List of plain type (not literal, not BaseModel, not union) returns None."""
    from dataeval_flow._app._model._introspect import _introspect_list_field

    # list[int] - inner is int, not Literal, not BaseModel
    result = _introspect_list_field(list[int], "items", "desc", True, None, {})
    assert result is None


# ---------------------------------------------------------------------------
# Coverage: discriminated union field (not in list)
# ---------------------------------------------------------------------------


def test_discriminated_union_field():
    class TypeA(BaseModel):
        kind: Literal["a"] = "a"
        val: int = 0

    class TypeB(BaseModel):
        kind: Literal["b"] = "b"
        msg: str = ""

    class Container(BaseModel):
        item: Annotated[TypeA | TypeB, Field(discriminator="kind")]

    descriptors = introspect_model(Container)
    by_name = {d.name: d for d in descriptors}
    assert "item" in by_name
    assert by_name["item"].kind == FieldKind.NESTED
    assert by_name["item"].discriminator == "kind"
    assert by_name["item"].union_variants is not None


# ---------------------------------------------------------------------------
# Coverage: plain list field (falls through _introspect_list_field)
# ---------------------------------------------------------------------------


def test_plain_list_field():
    class M(BaseModel):
        tags: list[str] = Field(default_factory=list, description="Tags")

    descriptors = introspect_model(M)
    by_name = {d.name: d for d in descriptors}
    assert by_name["tags"].kind == FieldKind.LIST
