"""The metadata factors behind what OOD detectors flagged: `factor-predictors`, `factor-deviation`, and legacy's
rules for which factors count (ood-detection spec §5.4)."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest
from dataeval.core import factor_deviation, factor_predictors
from dataeval.flags import ImageStats
from pydantic import ValidationError

from dataeval_flow import run_task
from dataeval_flow._cache import get_or_compute_metadata, get_or_compute_stats
from dataeval_flow._stats import ResolvedStatsPolicy
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators.shift import OODKNeighborsConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.steps.combines import (
    FactorDeviationOutput,
    FactorPredictorsCombine,
    FactorPredictorsConfig,
    FactorPredictorsOutput,
    OODUnionOutput,
)
from dataeval_flow.steps.combines._factors import collect_factors
from tests.chain_toys import chain_pipeline
from tests.ood_toys import FactorImages

_REF, _TEST = SimpleNamespace(address="reference"), SimpleNamespace(address="tests[cam1]")
_SHIFTED = range(0, 40, 5)


class _Metadata:
    """Stands in for DataEval's Metadata: item-level rows, the factor names, and the class labels."""

    item_level = "unit"

    def __init__(self, rows: dict[str, Any], labels: Any = None) -> None:
        self._rows = pl.DataFrame(rows)
        self.factor_names = list(rows)
        self.class_labels = labels

    def rows_at(self, level: str) -> pl.DataFrame:
        assert level == "unit"
        return self._rows


def _context(metadata: dict[str, Any], stats: dict[str, Any]) -> Any:
    """A combine context reading `metadata` and `stats` by node address; a value that is an exception raises it."""

    def reader(table: dict[str, Any]) -> Any:
        def derive(node: Any) -> Any:
            value = table[node.address]
            if isinstance(value, Exception):
                raise value
            return value

        return derive

    return SimpleNamespace(derive_metadata=reader(metadata), derive_stats=reader(stats))


def _stats(count: int, **arrays: Any) -> dict[str, Any]:
    return {"stats": arrays, "image_count": count}


def test_factors_follow_legacy_s_rules() -> None:
    reference = _Metadata(
        {
            "id": [0, 1, 2, 3],
            "altitude": [1.0, 2.0, 3.0, 4.0],
            "site": ["a", "b", "a", "b"],
            "flat": [1.0, 1.0, 1.0, 1.0],
            "gap": [1.0, np.nan, 2.0, 3.0],
        },
        labels=[0, 1, 0, 1],
    )
    test = _Metadata(
        {
            "id": [0, 1, 2],
            "altitude": [5.0, 6.0, 7.0],
            "site": ["a", "a", "b"],
            "flat": [1.0, 2.0, 3.0],
            "gap": [1.0, 2.0, 3.0],
        },
        labels=[1, 1, 0],
    )
    stats = {
        "reference": _stats(
            4, brightness=np.arange(4.0), histogram=np.ones((4, 8)), short=np.arange(2.0), still=np.ones(4)
        ),
        "tests[cam1]": _stats(
            3, brightness=np.arange(3.0), histogram=np.ones((3, 8)), short=np.arange(2.0), still=np.ones(3)
        ),
    }
    collected = collect_factors(_context({"reference": reference, "tests[cam1]": test}, stats), _REF, _TEST)
    # `id` goes; `site` is not numeric; `gap` is not finite in the reference; `f_histogram` is two-dimensional;
    # `f_short` is not one per image; `f_still` is constant in the test. `flat`, constant only in the reference, stays.
    assert sorted(collected.test) == ["altitude", "class_label", "f_brightness", "flat"]
    assert collected.unavailable == []


def test_class_labels_count_only_where_one_per_item() -> None:
    boxes = {"reference": _Metadata({"altitude": [1.0, 2.0, 3.0]}, labels=[0, 1, 1, 0, 1])}
    boxes["tests[cam1]"] = _Metadata({"altitude": [2.0, 4.0, 6.0]}, labels=[0, 1, 1, 0])
    stats = {"reference": _stats(3), "tests[cam1]": _stats(3)}
    assert sorted(collect_factors(_context(boxes, stats), _REF, _TEST).test) == ["altitude"]


def test_unreadable_metadata_falls_back_to_statistics_and_back() -> None:
    stats = {"reference": _stats(3, brightness=np.arange(3.0)), "tests[cam1]": _stats(3, brightness=np.arange(3.0))}
    failing = {"reference": ValueError("no metadata"), "tests[cam1]": ValueError("no metadata")}
    collected = collect_factors(_context(failing, stats), _REF, _TEST)
    assert sorted(collected.test) == ["f_brightness"]
    assert collected.unavailable == [
        "the metadata of `reference`: no metadata",
        "the metadata of `tests[cam1]`: no metadata",
    ]
    metadata = {
        "reference": _Metadata({"altitude": [1.0, 2.0, 3.0]}),
        "tests[cam1]": _Metadata({"altitude": [3.0, 4.0, 6.0]}),
    }
    broken = {"reference": RuntimeError("no stats"), "tests[cam1]": RuntimeError("no stats")}
    assert sorted(collect_factors(_context(metadata, broken), _REF, _TEST).test) == ["altitude"]


def test_on_detection_rows_predictors_read_assessed_images_only(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def record(factors: dict[str, Any], indices: Any) -> dict[str, float]:
        seen.update(factors=factors, indices=list(indices))
        return dict.fromkeys(factors, 0.5)

    monkeypatch.setattr("dataeval.core.factor_predictors", record)
    union = OODUnionOutput(
        source="tests[cam1]",
        detectors=["u"],
        left_out=[],
        images=5,
        assessed=4,
        union=[1, 4],
        mutual=[1, 4],
        partial=[],
        unique={"u": []},
        scores=[1.0, 2.0, None, 0.5, 3.0],
        thresholds={"u": 1.0},
        flagged_detections=[0, 1, 0, 0, 2],
    )
    metadata = {
        "reference": _Metadata({"altitude": [1.0, 2.0, 3.0, 4.0]}),
        "tests[cam1]": _Metadata({"altitude": [1.0, 5.0, 9.0, 2.0, 7.0]}),
    }
    context = _context(metadata, {"reference": RuntimeError("x"), "tests[cam1]": RuntimeError("x")})
    config = FactorPredictorsConfig(ood="u", reference="reference", input="tests")
    inputs = {"ood": SimpleNamespace(value=union), "reference": _REF, "input": _TEST}
    output = FactorPredictorsCombine().run(config, inputs, context)["output"]
    assert seen["factors"]["altitude"].tolist() == [1.0, 5.0, 2.0, 7.0]
    assert seen["indices"] == [1, 3]
    assert output.flagged == 2


def _chain(datasets: dict[str, Any], *, ood_input: str = "tests", **deviation: Any) -> ChainResult:
    on = "tests" if ood_input == "tests" else "a"
    steps = [
        {"name": "knn", "evaluator": "knn", "input": ["reference", on]},
        {"name": "agreement", "combine": "ood-union", "input": ["knn"]},
        {
            "name": "predictors",
            "combine": "factor-predictors",
            "ood": "agreement",
            "reference": "reference",
            "input": ood_input,
        },
        {
            "name": "deviation",
            "combine": "factor-deviation",
            "ood": "agreement",
            "reference": "reference",
            "input": ood_input,
            **deviation,
        },
    ]
    inputs = ["reference", {"name": "tests", "list": True}] if on == "tests" else list(datasets)
    workflow = {"name": "w", "inputs": inputs, "steps": steps}
    knn = OODKNeighborsConfig(name="knn", k=5, distance_metric="euclidean")
    # Seeded: DataEval's `factor_predictors` draws sklearn's tie-breaking noise from its global seed, which the run sets
    # and the test's own recomputation then reads.
    config = chain_pipeline(
        workflows=[workflow], evaluators=[knn], datasets=datasets, extractor=True, extra={"seed": 0}
    )
    result = run_task(config, TaskConfig(name="t", workflow="w", sources=list(datasets), extractor="flat"))
    assert isinstance(result, ChainResult)
    return result


def _output(result: ChainResult, step: str) -> Any:
    return (result.steps[step].elements or {})["cam1"].output


def test_the_factors_behind_flagged_images_are_dataeval_s_on_the_factors_collected() -> None:
    reference, test = FactorImages(40), FactorImages(40, seed=1, shifted=_SHIFTED)
    result = _chain({"reference": reference, "cam1": test}, max_items=3)
    union, predictors, deviation = (_output(result, step) for step in ("agreement", "predictors", "deviation"))
    assert isinstance(predictors, FactorPredictorsOutput)
    assert isinstance(deviation, FactorDeviationOutput)
    assert {"altitude", "hour", "class_label", "f_brightness"} <= set(predictors.factors)
    assert "id" not in predictors.factors
    assert list(predictors.factors.values()) == sorted(predictors.factors.values(), reverse=True)
    data = {"reference": reference, "tests[cam1]": test}
    every = ResolvedStatsPolicy.of_flags(ImageStats.ALL)
    context: Any = SimpleNamespace(
        derive_metadata=lambda node: get_or_compute_metadata(data[node.address], None),
        derive_stats=lambda node: get_or_compute_stats(every, data[node.address], per_image=True, per_target=False),
    )
    collected = collect_factors(context, _REF, _TEST)
    expected = factor_predictors(collected.test, union.union)
    assert predictors.factors == pytest.approx({name: round(float(value), 4) for name, value in expected.items()})
    chosen = sorted(union.mutual, key=lambda index: (-(union.scores[index] or 0.0), index))[:3]
    assert [item.index for item in deviation.items] == chosen
    for item, wanted in zip(
        deviation.items, factor_deviation(collected.reference, collected.test, chosen), strict=True
    ):
        assert item.deviations == pytest.approx(dict(wanted))
    assert deviation.source == "tests[cam1]"
    report = result.report()
    assert "MI (normalized)" in report
    assert "Top factors" in report


def test_metadata_with_only_id_falls_back_to_class_labels_and_statistics() -> None:  # Review Focus 4
    reference, test = FactorImages(40, factors=False), FactorImages(40, seed=1, shifted=_SHIFTED, factors=False)
    predictors = _output(_chain({"reference": reference, "cam1": test}), "predictors")
    assert "class_label" in predictors.factors
    assert any(name.startswith("f_") for name in predictors.factors)
    assert predictors.unavailable == []


def test_with_nothing_flagged_each_says_why() -> None:
    result = _chain({"reference": FactorImages(40), "cam1": FactorImages(40)})
    assert _output(result, "predictors").reason == "No image was flagged, so no factor was compared."
    assert _output(result, "deviation").reason == "No image was flagged by every detector, so none is explained."
    assert "No image was flagged" in result.report()


def test_a_factor_step_reading_another_comparison_s_flags_is_refused() -> None:
    wanted = r"Step 'predictors' reads `agreement`, which was computed on `reference`, `a`, not on `reference`, `b`"
    datasets = {"reference": FactorImages(20), "a": FactorImages(20, seed=1), "b": FactorImages(20, seed=2)}
    with pytest.raises(ValidationError, match=wanted):
        _chain(datasets, ood_input="b")
