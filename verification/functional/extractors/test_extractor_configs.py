"""TC-4-1 — extractor configs."""

from __future__ import annotations

import pytest

from dataeval_flow import (
    BoVWExtractorConfig,
    FlattenExtractorConfig,
    OnnxExtractorConfig,
    TorchExtractorConfig,
    UncertaintyExtractorConfig,
)


@pytest.mark.test_case("4-1")
class TestExtractorConfigs:
    def test_flatten_extractor_config_constructs(self) -> None:
        cfg = FlattenExtractorConfig(name="flat")
        assert cfg.model == "flatten"

    def test_torch_extractor_config_constructs(self) -> None:
        cfg = TorchExtractorConfig(name="torch_ex", model_path="model.pt")
        assert cfg.model_path == "model.pt"

    def test_onnx_extractor_config_constructs(self) -> None:
        cfg = OnnxExtractorConfig(name="onnx_ex", model_path="model.onnx")
        assert cfg.model_path.endswith(".onnx")

    def test_bovw_extractor_config_constructs(self) -> None:
        cfg = BoVWExtractorConfig(name="bovw_ex")
        assert cfg.model == "bovw"

    def test_uncertainty_extractor_config_constructs(self) -> None:
        cfg = UncertaintyExtractorConfig(name="unc_ex", model_path="classifier.pt")
        assert cfg.model == "uncertainty"

    def test_all_extractor_configs_exported(self) -> None:
        import dataeval_flow

        for name in (
            "FlattenExtractorConfig",
            "TorchExtractorConfig",
            "OnnxExtractorConfig",
            "BoVWExtractorConfig",
            "UncertaintyExtractorConfig",
        ):
            assert name in dataeval_flow.__all__
