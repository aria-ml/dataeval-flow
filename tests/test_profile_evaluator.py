"""`profile`: each measured statistic and supplied metadata field summarized and binned, every row's value kept."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from dataeval_flow import run_task
from dataeval_flow._cache import DatasetCache
from dataeval_flow._runner import run
from dataeval_flow.evaluators.quality._profile import summarize
from tests.chain_toys import ToyDetections, chain_pipeline
from tests.evaluator_toys import ToyImages, output_json


@pytest.fixture(autouse=True)
def _fresh_caches():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


class _Occluded(ToyDetections):
    """`ToyDetections` whose boxes each carry an `occluded` flag, a target-level metadata field."""

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        image, target, meta = super().__getitem__(index)
        return image, target, {**meta, "occluded": [bool(box % 2) for box in range(len(target.labels))]}


_DETECTIONS = _Occluded([[0], [0, 1], [1], [0], [1, 1], [0]], {0: "a", 1: "b"})


def _config(dataset: Any, **settings: Any) -> Any:
    return chain_pipeline(
        evaluators=[{"name": "profile", "type": "profile", **settings}],
        tasks=[{"name": "p", "evaluator": "profile", "sources": "src"}],
        datasets={"src": dataset},
    )


def _profile(dataset: Any, **settings: Any) -> Any:
    config = _config(dataset, **settings)
    result = run_task(config, config.tasks[0])
    assert result.success, result.errors
    return result


def _field(data: dict[str, Any], name: str, scope: str) -> dict[str, Any]:
    (found,) = [field for field in data["fields"] if field["name"] == name and field["scope"] == scope]
    return found


def test_a_numeric_field_bins_as_numpy_does() -> None:
    values = pl.Series([0.0, 1.0, 2.5, 4.0, 10.0])
    field = summarize(values, bins=4, categories=20)
    counts, edges = np.histogram(values.to_numpy(), bins=4)
    assert field["type"] == "numeric"
    assert field["histogram"] == {"edges": edges.tolist(), "counts": counts.tolist()}
    assert (field["min"], field["max"], field["median"]) == (0.0, 10.0, 2.5)
    assert field["std"] == pytest.approx(float(np.std(values.to_numpy())))


def test_nulls_are_missing_and_nan_and_infinity_are_non_finite() -> None:
    field = summarize(pl.Series([1.0, None, float("nan"), float("inf"), 3.0]), bins=2, categories=20)
    assert (field["rows"], field["missing"], field["non_finite"], field["finite"]) == (5, 1, 2, 2)
    assert field["histogram"]["counts"] == [1, 1]


def test_a_constant_field_is_one_closed_bin() -> None:
    field = summarize(pl.Series([7.0, 7.0, 7.0]), bins=10, categories=20)
    assert field["histogram"] == {"edges": [7.0, 7.0], "counts": [3]}


def test_a_field_with_no_finite_value_has_no_summary() -> None:
    field = summarize(pl.Series([None, float("nan")], dtype=pl.Float64), bins=10, categories=20)
    assert field["finite"] == 0
    assert (field["mean"], field["histogram"]) == (None, None)


def test_categories_beyond_the_limit_fall_in_other() -> None:
    field = summarize(pl.Series(["a", "a", "a", "b", "b", "c", None]), bins=10, categories=2)
    assert field["type"] == "categorical"
    assert field["values"] == [{"value": "a", "count": 3}, {"value": "b", "count": 2}]
    assert field["other"] == {"values": 1, "count": 1}
    assert field["missing"] == 1


def test_a_detection_profile_has_an_image_and_a_target_scope() -> None:
    data = output_json(_profile(_DETECTIONS))["data"]
    assert data["scopes"] == {"image": {"rows": 6}, "target": {"rows": 8}}
    brightness = _field(data, "brightness", "target")
    assert (brightness["origin"], brightness["family"], brightness["group"]) == ("computed", "visual", None)
    assert _field(data, "brightness", "image")["rows"] == 6


def test_supplied_fields_are_profiled_at_their_own_level() -> None:
    data = output_json(_profile(_DETECTIONS))["data"]
    site = _field(data, "site", "image")
    assert (site["origin"], site["type"]) == ("supplied", "categorical")
    assert {value["value"]: value["count"] for value in site["values"]} == {"north": 2, "south": 2, "east": 2}
    assert _field(data, "occluded", "target")["rows"] == 8


def test_vector_valued_statistics_are_unsupported() -> None:
    data = output_json(_profile(_DETECTIONS))["data"]
    assert _field(data, "histogram", "image") == {
        **_field(data, "histogram", "image"),
        "type": "unsupported",
        "reason": "vector-valued",
    }


def test_a_classification_profile_has_only_an_image_scope() -> None:
    data = output_json(_profile(ToyImages()))["data"]
    assert data["scopes"] == {"image": {"rows": 12}}
    assert data["binning"] == {"method": "equal_width", "bins": 10, "intervals": "left-closed, last bin closed"}


def test_each_scope_s_rows_are_kept_for_selections() -> None:
    result = _profile(_DETECTIONS, bins=4)
    frames = result.output.frames()
    target = frames["target"]
    assert target.select("item", "target").rows()[:3] == [(0, 0), (1, 0), (1, 1)]
    brightness = _field(output_json(result)["data"], "brightness", "target")
    assert brightness["column"] == "computed:brightness"
    assert target["computed:brightness"].is_not_null().sum() == brightness["rows"] - brightness["missing"]
    assert set(frames["image"]["supplied:site"].to_list()) == {'"north"', '"south"', '"east"'}


def test_the_run_writes_each_scope_s_rows_beside_the_results(tmp_path: Path) -> None:
    with patch("dataeval_flow._runner._resolve_config", return_value=_config(_DETECTIONS)):
        run("pipeline.yaml", tmp_path / "out", data_dir=tmp_path)
    root = tmp_path / "out" / "results" / "profiles" / "p"
    assert pl.read_parquet(root / "image.parquet").height == 6
    assert pl.read_parquet(root / "target.parquet").height == 8


class _Named(ToyDetections):
    """`ToyDetections` whose metadata names a field `brightness`, as a statistic is named, and one `item`."""

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        image, target, meta = super().__getitem__(index)
        return image, target, {**meta, "brightness": ("dark", "light")[index % 2], "item": index * 10}


def test_a_metadata_field_named_like_a_statistic_is_kept_apart_from_it() -> None:
    result = _profile(_Named([[0], [0, 1], [1], [0]], {0: "a", 1: "b"}))
    data = output_json(result)["data"]
    image = {(field["name"], field["origin"]): field for field in data["fields"] if field["scope"] == "image"}
    supplied, computed = image["brightness", "supplied"], image["brightness", "computed"]
    assert (supplied["type"], computed["type"]) == ("categorical", "numeric")
    assert {value["value"] for value in supplied["values"]} == {"dark", "light"}
    assert image["item", "supplied"]["max"] == 30.0
    frame = result.output.frames()["image"]
    assert frame["item"].to_list() == [0, 1, 2, 3]
    assert {"computed:brightness", "supplied:brightness", "supplied:item"} <= set(frame.columns)
