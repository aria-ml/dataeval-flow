"""The `divergence` evaluator and the `embedding-divergence` check (audit spec §10.1, §10.2)."""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import run
from dataeval_flow.evaluators.shift import DivergenceConfig, DivergenceOutput
from dataeval_flow.steps import CheckContext
from dataeval_flow.steps.checks import EmbeddingDivergenceCheck, EmbeddingDivergenceConfig
from tests.evaluator_toys import FLAT, ToyImages, shifted_sources


def test_it_measures_how_far_two_sources_sit_apart() -> None:
    result = run(DivergenceConfig(), shifted_sources(40), extractor=FLAT)
    assert result.success, result.errors
    data = result.output.data()
    assert 0.0 <= data["divergence"] <= 1.0
    assert data["method"] == "mst"
    assert isinstance(data["errors"], int)


def test_fnn_runs_too() -> None:
    result = run(DivergenceConfig(method="fnn"), shifted_sources(40), extractor=FLAT)
    assert result.success, result.errors
    assert result.output.data()["method"] == "fnn"


def test_an_empty_source_fails_with_a_reason() -> None:
    result = run(DivergenceConfig(), {"a": ToyImages(count=12), "b": ToyImages(count=0)}, extractor=FLAT)
    assert not result.success
    assert "needs embeddings from both sources" in result.errors[0]


def _judge(value: float, **limits: float | None) -> Any:
    output = DivergenceOutput({"divergence": value, "errors": 3, "method": "mst"}, None)
    node = SimpleNamespace(value=output, computed_on=(), address="shift")
    (finding,) = EmbeddingDivergenceCheck().run(
        EmbeddingDivergenceConfig(input="shift", **limits), {"input": node}, CheckContext("t", "s")
    )
    return finding


@pytest.mark.parametrize(
    ("value", "severity", "level"), [(0.9, "warning", "High"), (0.45, "info", "Moderate"), (0.1, "ok", "Low")]
)
def test_the_finding_bands_divergence_as_legacy_did(value: float, severity: str, level: str) -> None:
    finding = _judge(value)
    assert finding.severity == severity
    assert finding.title == "Embedding Divergence"
    assert finding.brief == f"{level.lower()} divergence: {value:.4f} (mst)"
    assert finding.description == f"{level} divergence: {value:.4f} (mst)."


def test_the_info_band_follows_warning_unless_set() -> None:
    assert EmbeddingDivergenceConfig(input="s").info == 0.2
    assert EmbeddingDivergenceConfig(input="s", warning=0.2).info == 0.08
    assert EmbeddingDivergenceConfig(input="s", info=None).info is None
    assert "info" not in EmbeddingDivergenceConfig(input="s").model_fields_set
    with pytest.raises(ValidationError, match="must not exceed"):
        EmbeddingDivergenceConfig(input="s", info=0.9)


def test_with_no_limits_it_judges_nothing() -> None:
    finding = _judge(0.9, warning=None, info=None)
    assert finding.severity == "info"
    assert finding.brief == "divergence 0.9000 (mst)"
