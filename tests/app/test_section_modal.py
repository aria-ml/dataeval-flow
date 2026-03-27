"""Unit tests for the generic SectionModal."""

from __future__ import annotations

import pytest

from dataeval_flow._app._model._registry import get_fields, get_variant_choices
from dataeval_flow._app._model._state import ConfigState
from dataeval_flow._app._screens import SectionModal


class TestSectionModalInit:
    """Test that SectionModal initializes correctly for each section type."""

    @pytest.mark.parametrize("section", ["datasets", "extractors", "workflows"])
    def test_discriminated_section(self, section: str):
        """Discriminated sections should have variant choices."""
        SectionModal(section=section, state=ConfigState())
        assert get_variant_choices(section) is not None

    @pytest.mark.parametrize("section", ["sources", "tasks"])
    def test_non_discriminated_section(self, section: str):
        """Non-discriminated sections should NOT have variant choices."""
        SectionModal(section=section, state=ConfigState())
        assert get_variant_choices(section) is None

    def test_edit_mode_with_existing(self):
        existing = {"name": "ds1", "format": "huggingface", "path": "data"}
        modal = SectionModal(section="datasets", existing=existing, state=ConfigState())
        assert modal.is_edit_mode is True
        assert modal._existing == existing

    def test_create_mode_without_existing(self):
        modal = SectionModal(section="datasets", state=ConfigState())
        assert modal.is_edit_mode is False


class TestSectionModalFieldGeneration:
    """Test that get_fields produces correct descriptors for each section."""

    def test_huggingface_dataset_fields(self):
        state = ConfigState()
        fields = get_fields("datasets", "huggingface", state)
        names = [f.name for f in fields]
        assert "path" in names
        assert "split" in names
        assert "name" not in names
        assert "format" not in names

    def test_coco_dataset_fields(self):
        state = ConfigState()
        fields = get_fields("datasets", "coco", state)
        names = [f.name for f in fields]
        assert "path" in names
        assert "annotations_file" in names
        assert "images_dir" in names
        assert "classes_file" in names
        # HuggingFace-specific field should NOT be here
        assert "split" not in names

    def test_onnx_extractor_fields(self):
        state = ConfigState()
        fields = get_fields("extractors", "onnx", state)
        names = [f.name for f in fields]
        assert "model_path" in names
        assert "output_name" in names
        assert "flatten" in names

    def test_flatten_extractor_minimal_fields(self):
        state = ConfigState()
        fields = get_fields("extractors", "flatten", state)
        names = [f.name for f in fields]
        # flatten extractor has minimal fields (just base: preprocessor, batch_size)
        assert "model_path" not in names
        assert "preprocessor" in names

    def test_source_cross_refs_populated(self):
        state = ConfigState()
        state.add("datasets", {"name": "ds1", "format": "huggingface", "path": "data"})
        state.add("datasets", {"name": "ds2", "format": "coco", "path": "data"})

        fields = get_fields("sources", None, state)
        dataset_field = next(f for f in fields if f.name == "dataset")
        assert "ds1" in dataset_field.choices
        assert "ds2" in dataset_field.choices

    def test_task_multi_ref_sources(self):
        state = ConfigState()
        state.add("sources", {"name": "src1", "dataset": "ds1"})
        state.add("sources", {"name": "src2", "dataset": "ds2"})

        fields = get_fields("tasks", None, state)
        sources_field = next(f for f in fields if f.name == "sources")
        from dataeval_flow._app._model._introspect import FieldKind

        assert sources_field.kind == FieldKind.MULTI_SELECT
        assert "src1" in sources_field.choices
        assert "src2" in sources_field.choices
