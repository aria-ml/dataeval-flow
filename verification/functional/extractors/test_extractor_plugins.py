"""TC-4-2 — extractors from installed plug-in packages."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from dataeval_flow import load_config, run
from dataeval_flow.config.extractors import Extractor, ExtractorConfig, get_extractor, list_extractors
from dataeval_flow.evaluators.scope import CompletenessConfig
from verification.functional.orchestration.support import MODULE, InMemoryImages, MeanConfig, Means

pytestmark = [pytest.mark.required, pytest.mark.usefixtures("fresh_caches")]

BUILT_IN = ["bovw", "flatten", "onnx", "torch", "uncertainty"]


def _images(n: int = 6, shape: tuple[int, int, int] = (3, 8, 8), seed: int = 0) -> list[Any]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, shape, dtype=np.uint8) for _ in range(n)]


class TestPluginExtractors:
    def test_an_installed_extractor_is_listed_and_validates_from_yaml(
        self,
        example_plugin: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        names = [cls.name for cls in list_extractors()]
        assert names == sorted(names)
        assert "example.mean" in names

        path = tmp_path / "params.yaml"
        path.write_text(yaml.safe_dump({"extractors": [{"model": "example.mean", "batch_size": 4}]}))
        config = load_config(path)
        assert config.extractors is not None
        assert isinstance(config.extractors[0], MeanConfig)
        assert config.extractors[0].name == "example.mean"

    def test_an_installed_extractor_builds_embeddings_like_a_built_in(
        self,
        example_plugin: dict[str, Any],
    ) -> None:
        extractor = get_extractor("example.mean")().build(MeanConfig(), None)
        images = _images(4)
        embeddings = np.asarray(extractor(images))
        assert embeddings.shape == (4, 3)
        assert np.allclose(embeddings[1], images[1].mean(axis=(1, 2)))

    def test_a_task_runs_with_an_installed_extractor(self, example_plugin: dict[str, Any]) -> None:
        Means.batches.clear()
        result = run(CompletenessConfig(), InMemoryImages(20), extractor=MeanConfig(batch_size=20))
        assert result.success, result.errors
        assert result.metadata.model_id == "example.mean (example.mean)"
        assert sum(Means.batches) == 20

    def test_a_broken_extractor_plugin_is_left_out_and_says_why(
        self,
        plugins: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        plugins["dataeval_flow.extractors"] = [("example.gone", "verification.no_such_module:Nope")]
        with caplog.at_level(logging.WARNING):
            names = [cls.name for cls in list_extractors()]
        assert names == BUILT_IN
        assert "example.gone" in caplog.text
        with pytest.raises(ValueError, match="no_such_module"):
            get_extractor("example.gone")

    def test_a_plugin_cannot_take_a_built_in_extractors_name(
        self,
        plugins: dict[str, Any],
    ) -> None:

        plugins["dataeval_flow.extractors"] = [("flatten", f"{MODULE}:MeanExtractor")]
        assert get_extractor("flatten").__name__ == "_FlattenExtractor"

    def test_a_plugin_whose_config_configures_another_model_is_refused(
        self,
        plugins: dict[str, Any],
    ) -> None:

        plugins["dataeval_flow.extractors"] = [("example.other", f"{MODULE}:MeanExtractor")]
        with pytest.raises(ValueError, match="example.mean"):
            get_extractor("example.other")

    def test_the_extractor_base_is_not_a_config(self) -> None:
        assert not issubclass(Extractor, ExtractorConfig)
