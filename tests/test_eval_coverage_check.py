"""The `eval-coverage` check: the share of an evaluation split lying beyond train (audit spec §10.2, §5.2)."""

from types import SimpleNamespace
from typing import Any

import numpy as np

from dataeval_flow import run
from dataeval_flow.evaluators.shift import OODKNeighborsConfig
from dataeval_flow.steps import CheckContext
from dataeval_flow.steps.checks import EvalCoverageCheck, EvalCoverageConfig
from dataeval_flow.steps.checks._ood import ood_severity
from tests.evaluator_toys import FLAT, ToyImages

_EUCLID: dict[str, Any] = {"distance_metric": "euclidean"}  # flatten's cosine ignores a brightness shift


class _White(ToyImages):
    """Images all at full brightness: far from `ToyImages`' noise, which flattening embeds as distant points."""

    def __init__(self, count: int = 40) -> None:
        super().__init__(count=count)
        self._images = [np.full((3, 16, 16), 255, dtype=np.uint8) for _ in range(count)]


def _shifted() -> dict[str, Any]:
    return {"train": ToyImages(count=40), "test": _White()}


def _judge(sources: dict[str, Any], config: OODKNeighborsConfig, **limits: Any) -> Any:
    output = run(config, sources, extractor=FLAT).output
    on = (SimpleNamespace(address="train"), SimpleNamespace(address="evals[test]"))
    node = SimpleNamespace(value=output, computed_on=on, address="coverage[test]", config=config)
    (finding,) = EvalCoverageCheck().run(
        EvalCoverageConfig(input="coverage", **limits), {"input": node}, CheckContext("t", "s")
    )
    return finding


def test_a_split_drawn_like_train_reads_ok() -> None:
    assert EvalCoverageConfig(input="o").info == 2.0
    assert EvalCoverageConfig(input="o").warning == 10.0
    assert ood_severity(1.0, EvalCoverageConfig(input="o")) == "ok"
    sources = {"train": ToyImages(count=200, seed=0), "test": ToyImages(count=200, seed=1)}
    finding = _judge(sources, OODKNeighborsConfig(threshold_perc=99, **_EUCLID))
    assert finding.severity == "ok"
    assert finding.title == "Evaluation Coverage"
    assert "than 99% of it" in finding.brief


def test_a_shifted_split_warns_and_names_the_percentile() -> None:
    finding = _judge(_shifted(), OODKNeighborsConfig(threshold_perc=99, **_EUCLID))
    assert finding.severity == "warning"
    assert "farther from `train` than 99% of `train`" in finding.description


def test_an_unset_percentile_reads_as_dataeval_s_default() -> None:
    finding = _judge(_shifted(), OODKNeighborsConfig(**_EUCLID))
    assert "than 95% of it" in finding.brief
