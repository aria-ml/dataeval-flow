"""TC-9-3 — drift in a model's uncertainty, read through the `uncertainty` extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow.config.extractors import UncertaintyExtractorConfig
from verification.functional.workflows._synthetic import Images, pipeline, run_preset, write_onnx_classifier

pytestmark = pytest.mark.required

SHAPE = (3, 16, 16)
UNCERTAINTY = UncertaintyExtractorConfig(
    name="unc", model_path="model.onnx", metadata_path="model.json", preds_type="logits", batch_size=8
)


def sources() -> dict[str, Any]:
    """A reference, a source like it, and one of bright images, which the model reads with a different certainty."""
    return {
        "reference": Images(60, seed=0, shape=SHAPE),
        "same": Images(60, seed=1, shape=SHAPE),
        "bright": Images(60, seed=2, shape=SHAPE, value_range=(150, 255)),
    }


@pytest.fixture
def model_root(tmp_path: Path) -> Path:
    """The data root, holding a small ONNX classifier and DataEval's metadata for it."""
    pytest.importorskip("onnx", reason="the onnx extra is not installed")
    pytest.importorskip("onnxruntime", reason="the onnx extra is not installed")
    write_onnx_classifier(tmp_path, size=SHAPE[1])
    return tmp_path


def run_shift(model_root: Path, entry: dict[str, Any], **task: Any) -> Any:
    return run_preset(
        {"type": "shift", **entry},
        sources(),
        task={"extractor": "unc", **task},
        data_dir=model_root,
        extra={"extractors": [UNCERTAINTY]},
    )


class TestUncertaintyExtractor:
    def test_a_source_the_model_is_less_sure_of_drifts_in_its_uncertainty(self, model_root: Path) -> None:
        result = run_shift(model_root, {"detectors": [{"name": "ks", "type": "drift-univariate"}]})

        assert result.success, result.errors
        assert {f.step: f.brief for f in result.findings} == {
            "ks-check[same]": "no drift",
            "ks-check[bright]": "drift",
        }
        assert not result.steps["ks"].elements["same"].output.drifted
        assert result.steps["ks"].elements["bright"].output.drifted

    def test_a_detector_may_name_its_own_extractor_beside_the_tasks(self, model_root: Path) -> None:
        detectors = [
            {"name": "ks", "type": "drift-univariate", "extractor": "unc"},
            {"name": "mmd", "type": "drift-mmd"},
        ]

        result = run_preset(
            {"type": "shift", "detectors": detectors},
            sources(),
            extractor=True,  # the task's extractor is `flat`
            data_dir=model_root,
            extra={"extractors": [UNCERTAINTY]},
        )

        assert result.success, result.errors
        found = {f.step: f.brief for f in result.findings}
        assert found["ks-check[bright]"] == "drift"  # in the model's uncertainty
        assert found["mmd-check[bright]"] == "drift"  # in the flattened pixels
        assert found["mmd-check[same]"] == "no drift"

    def test_by_predicted_runs_a_drift_detector_once_per_class_the_model_predicts(self, model_root: Path) -> None:
        entry = {"detectors": [{"name": "ks", "type": "drift-univariate"}], "classwise": {"ks": "predicted"}}

        result = run_shift(model_root, entry)

        assert result.success, result.errors
        by_class = result.steps["ks-by-class"].elements["bright"].output
        assert list(by_class.outputs) == ["cat"]  # the stub model predicts class 0 for every image
        found = {f.step: f.brief for f in result.findings}
        assert found["ks-by-class-check[bright]"] == "1/1 predicted classes warn"
        assert found["ks-by-class-check[same]"] == "0/1 predicted classes warn"

    def test_an_evaluator_that_needs_one_embedding_per_item_refuses_the_extractor(self, model_root: Path) -> None:
        with pytest.raises(ValidationError, match="only drift and OOD evaluators read it"):
            pipeline(
                evaluators=[{"name": "cov", "type": "coverage"}],
                tasks=[{"name": "t", "evaluator": "cov", "sources": ["images"], "extractor": "unc"}],
                datasets={"images": Images(30, shape=SHAPE)},
                extra={"extractors": [UNCERTAINTY]},
            )

    def test_the_model_and_its_metadata_are_read_from_the_data_root(self, model_root: Path, tmp_path: Path) -> None:
        empty = tmp_path / "elsewhere"
        empty.mkdir()
        result = run_preset(
            {"type": "shift", "detectors": [{"name": "ks", "type": "drift-univariate"}]},
            sources(),
            task={"extractor": "unc"},
            data_dir=empty,  # no model under this root
            extra={"extractors": [UNCERTAINTY]},
        )

        assert not result.success
        assert any("model.onnx" in error or "model.json" in error for error in result.errors), result.errors
