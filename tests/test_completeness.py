"""`completeness` and `completeness-score`: dimensional completeness of embeddings rescaled to the unit interval, and
legacy data-coverage's Dimensional Completeness finding (coverage spec §6.1, §6.2)."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from dataeval_flow import run
from dataeval_flow.evaluators.scope import CompletenessConfig, CompletenessOutput
from dataeval_flow.steps.checks import CompletenessScoreCheck, CompletenessScoreConfig
from tests.evaluator_toys import FLAT, ToyImages


def test_it_scores_the_embeddings_rescaled_to_the_unit_interval() -> None:
    from dataeval.core import completeness

    from dataeval_flow.workflows._common import normalize_unit_interval
    from tests.evaluator_toys import output_json

    images = ToyImages(count=20)
    result = run(CompletenessConfig(), images, extractor=FLAT)
    assert result.success, result.errors
    assert isinstance(result.output, CompletenessOutput)
    flat = np.stack([np.asarray(images[index][0], dtype=np.float32).ravel() for index in range(len(images))])
    expected = completeness(normalize_unit_interval(flat))
    assert result.output.data()["completeness"] == pytest.approx(float(expected["completeness"]))
    pairs = output_json(result)["data"]["nearest_neighbor_pairs"]
    assert all(isinstance(value, int) for pair in pairs for value in pair)


def _judge(score: float, **settings: Any) -> Any:
    data = {"completeness": score, "nearest_neighbor_pairs": [[0, 1]]}
    node = SimpleNamespace(value=SimpleNamespace(data=lambda: data))
    config = CompletenessScoreConfig(input="c", **settings)
    (finding,) = CompletenessScoreCheck().run(config, {"input": node}, None)  # type: ignore[arg-type]
    return finding


@pytest.mark.parametrize(("score", "severity"), [(0.3, "warning"), (0.6, "info"), (0.9, "ok")])
def test_it_bands_the_score(score: float, severity: str) -> None:
    finding = _judge(score)
    assert (finding.severity, finding.title, finding.brief) == (
        severity,
        "Dimensional Completeness",
        f"Completeness: {score}",
    )
    assert finding.description == f"Dimensional completeness score is {score} (threshold: 0.5)."


def test_it_judges_the_score_rounded_to_three_places() -> None:
    assert _judge(0.49996).severity == "info"  # rounds to 0.5, which is not under 0.5


def test_null_thresholds_judge_nothing() -> None:
    assert _judge(0.1, warning=None, info=None).severity == "info"


def test_warning_above_info_is_refused() -> None:
    with pytest.raises(ValidationError, match="warning"):
        CompletenessScoreConfig(input="c", warning=0.9, info=0.8)


def test_a_score_of_exactly_the_info_band_is_ok() -> None:
    assert _judge(0.8).severity == "ok"  # the bands are strictly below


def test_no_warning_threshold_leaves_the_clause_out() -> None:
    assert _judge(0.6, warning=None).description == "Dimensional completeness score is 0.6."


def test_fewer_than_two_embeddings_fail_the_run() -> None:
    result = run(CompletenessConfig(), ToyImages(count=1), extractor=FLAT)
    assert not result.success
    assert any("needs at least two embeddings; the source has 1" in error for error in result.errors), result.errors
