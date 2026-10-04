"""`library_versions` on every result: the installed version of each library its numbers depend on (audit spec §7.3)."""

from importlib.metadata import version
from types import SimpleNamespace

import pytest

from dataeval_flow._versions import library_versions
from dataeval_flow.config.extractors import (
    BoVWExtractorConfig,
    FlattenExtractorConfig,
    OnnxExtractorConfig,
    TorchExtractorConfig,
    UncertaintyExtractorConfig,
)
from tests.evaluator_toys import toy_run

_ALWAYS = ("dataeval", "numpy", "pillow", "datamaite")


def test_every_run_depends_on_dataeval_numpy_pillow_and_datamaite() -> None:
    assert library_versions() == {name: version(name) for name in _ALWAYS}


def test_the_extractors_runtime_is_recorded_where_installed() -> None:
    pytest.importorskip("torch")
    assert library_versions(TorchExtractorConfig(model_path="model.pt"))["torch"] == version("torch")


def test_a_runtime_that_is_not_installed_is_left_out() -> None:
    absent = SimpleNamespace(runtime_distributions=("not-a-real-distribution",))
    assert "not-a-real-distribution" not in library_versions(absent)


def test_each_built_in_extractor_names_its_runtime() -> None:
    assert OnnxExtractorConfig.runtime_distributions == ("onnxruntime", "onnxruntime-gpu")
    assert UncertaintyExtractorConfig.runtime_distributions == ("onnxruntime", "onnxruntime-gpu")
    assert TorchExtractorConfig.runtime_distributions == ("torch",)
    assert BoVWExtractorConfig.runtime_distributions == ("opencv-python-headless", "opencv-python")
    assert FlattenExtractorConfig.runtime_distributions == ()


def test_a_result_records_them_in_its_envelope() -> None:
    result = toy_run("label-health")
    assert result.metadata.library_versions == library_versions()
    assert result.to_dict()["metadata"]["library_versions"]["numpy"] == version("numpy")


def test_a_flatten_extractor_adds_no_runtime() -> None:
    assert set(toy_run("coverage").metadata.library_versions) == set(_ALWAYS)
