"""TC-33-1 (NFR-3) — type-safety infrastructure."""

from __future__ import annotations

from pathlib import Path

import pytest

from verification.fixtures import write_image_folder

pytestmark = pytest.mark.required


class TestTypeSafety:
    def test_py_typed_marker_present(self) -> None:
        import dataeval_flow

        assert (Path(dataeval_flow.__file__).parent / "py.typed").exists()

    def test_top_level_all_defined(self) -> None:
        import dataeval_flow

        assert hasattr(dataeval_flow, "__all__")
        assert isinstance(dataeval_flow.__all__, list)
        assert len(dataeval_flow.__all__) > 0

    def test_maite_dataset_protocol_is_runtime_checkable(self, tmp_path: Path) -> None:
        """Datasets are accepted by structure, so the protocol that describes them can be tested at run time."""
        from dataeval_flow import load_dataset
        from dataeval_flow._dataset import MaiteDataset

        write_image_folder(tmp_path, n_per_class=2, n_classes=2, size=16)

        assert isinstance(load_dataset(tmp_path, dataset_format="image_folder", infer_labels=True), MaiteDataset)
        assert not isinstance(object(), MaiteDataset)

    def test_run_returns_the_result_class_of_the_config_it_was_given(self, tmp_path: Path) -> None:
        """``run`` is typed to its result: a config class determines the result class a caller gets back."""
        from dataeval_flow import Result, load_dataset, run
        from dataeval_flow.evaluators.quality import (
            DuplicatesConfig,
            DuplicatesResult,
            LabelHealthConfig,
            LabelHealthResult,
        )

        write_image_folder(tmp_path, n_per_class=4, n_classes=2, size=32)
        dataset = load_dataset(tmp_path, dataset_format="image_folder", infer_labels=True)

        duplicates = run(DuplicatesConfig(), dataset)
        label_health = run(LabelHealthConfig(), dataset)

        assert type(duplicates) is DuplicatesResult
        assert type(label_health) is LabelHealthResult
        assert isinstance(duplicates, Result)
