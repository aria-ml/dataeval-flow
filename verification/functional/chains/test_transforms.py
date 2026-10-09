"""TC-19-3 — transforms: the steps that make a Dataset from a Dataset (view, merge, split, kfold, select, wrap,
conform, remove, collect, export)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow.steps import ChainResult
from verification.functional.chains._toys import Detections, Images, yaml_pipeline

pytestmark = pytest.mark.required


def _run(text: str, datasets: dict, *, extractor: bool = False, output_dir: Path | None = None) -> ChainResult:
    config = yaml_pipeline(text, datasets, extractor=extractor, extra={"seed": 0})
    result = run_tasks(config, output_dir=output_dir)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


def _sizes(result: ChainResult) -> dict[str, int]:
    return {record.name: record.items for record in result.metadata.lineage}


_TWO = {"a": Images(20, planted=False), "b": Images(20, seed=1, planted=False)}
_A = Images(20, planted=False)


class TestDatasetTransforms:
    def test_view_applies_dataeval_view_operations(self) -> None:
        text = """
workflows:
  - name: w
    inputs: [a]
    steps:
      - {name: first5, transform: view, input: a, operations: [{type: Limit, params: {size: 5}}]}
tasks:
  - {name: t, workflow: w, sources: [a]}
"""
        result = _run(text, {"a": Images(20)})
        assert len(result.steps["first5"].output) == 5
        assert result.steps["first5"].details == {"indices": [0, 1, 2, 3, 4]}

    def test_merge_concatenates_datasets_in_the_order_named(self) -> None:
        text = """
workflows:
  - name: w
    inputs: [a, b]
    steps:
      - {name: both, transform: merge, input: [a, b]}
tasks:
  - {name: t, workflow: w, sources: [a, b]}
"""
        result = _run(text, _TWO)
        merged = result.steps["both"].output
        assert len(merged) == 40
        assert (merged[0][0] == _TWO["a"][0][0]).all()
        assert (merged[20][0] == _TWO["b"][0][0]).all()

    def test_split_makes_train_val_and_test_that_do_not_overlap(self) -> None:
        text = """
workflows:
  - name: w
    inputs: [a]
    steps:
      - {name: sp, transform: split, input: a, test_frac: 0.25, val_frac: 0.25}
tasks:
  - {name: t, workflow: w, sources: [a]}
"""
        result = _run(text, {"a": Images(40, planted=False)})
        sizes = _sizes(result)
        assert sizes["sp.test"] == 10
        assert sizes["sp.val"] > 0
        indices = result.steps["sp"].details["indices"]
        assert sorted(i for part in indices.values() for i in part) == list(range(40))

    def test_split_is_repeatable_under_a_seed(self) -> None:
        text = """
workflows:
  - name: w
    inputs: [a]
    steps:
      - {name: sp, transform: split, input: a, test_frac: 0.25}
tasks:
  - {name: t, workflow: w, sources: [a]}
"""
        first = _run(text, {"a": Images(40, planted=False)}).steps["sp"].details
        second = _run(text, {"a": Images(40, planted=False)}).steps["sp"].details
        assert first == second

    def test_kfold_makes_one_train_and_one_val_per_fold_and_a_single_test(self) -> None:
        text = """
evaluators:
  - {name: lh, type: label-health}
workflows:
  - name: w
    inputs: [a]
    steps:
      - {name: kf, transform: kfold, input: a, folds: 3}
      - {name: per_fold, evaluator: lh, input: kf.train}
      - {name: first_fold, evaluator: lh, input: "kf.train[0]"}
tasks:
  - {name: t, workflow: w, sources: [a]}
"""
        result = _run(text, {"a": Images(40, planted=False)})
        sizes = _sizes(result)
        assert {key for key in sizes if key.startswith("kf.")} == {
            "kf.train[0]", "kf.train[1]", "kf.train[2]", "kf.val[0]", "kf.val[1]", "kf.val[2]", "kf.test",
        }  # fmt: skip
        assert all(sizes[f"kf.train[{k}]"] + sizes[f"kf.val[{k}]"] + sizes["kf.test"] == 40 for k in range(3))
        # A step reading the whole list runs once per fold; one reading an element runs once.
        assert list(result.steps["per_fold"].elements) == ["0", "1", "2"]
        assert result.steps["first_fold"].output.data()["item_count"] == sizes["kf.train[0]"]

    def test_select_keeps_the_top_of_a_prioritization_ranking(self) -> None:
        text = """
evaluators:
  - {name: prio, type: prioritization}
workflows:
  - name: w
    inputs: [a]
    steps:
      - {name: rank, evaluator: prio, input: a}
      - {name: top, transform: select, input: a, ranking: rank, n: 4}
tasks:
  - {name: t, workflow: w, sources: [a], extractor: flat}
"""
        result = _run(text, {"a": _A}, extractor=True)
        ranking = [int(i) for i in result.steps["rank"].output.data()]
        top, source = result.steps["top"].output, _A
        assert len(top) == 4
        assert [(top[k][0] == source[ranking[k]][0]).all() for k in range(4)] == [True] * 4

    def test_remove_drops_what_a_duplicates_plan_names_and_reports_what_it_dropped(self) -> None:
        text = """
evaluators:
  - {name: dupes, type: duplicates}
workflows:
  - name: w
    inputs: [a]
    steps:
      - {name: dupes, evaluator: dupes, input: a}
      - {name: clean, transform: remove, input: a, plans: {dupes: {keep: first}}}
tasks:
  - {name: t, workflow: w, sources: [a]}
"""
        result = _run(text, {"a": Images(12)})
        assert _sizes(result) == {"a": 12, "clean": 11}
        assert result.steps["clean"].details == {
            "removed": {"items": 1, "detections": 0, "tracks": 0, "frames": 0},
            "by_plan": {"dupes": {"items": 1}},
        }
        assert "Kept 11 of 12 images. Removed 1 image: 1 named by `dupes`." in result.report()

    def test_remove_drops_the_outliers_an_outliers_plan_names(self) -> None:
        text = """
evaluators:
  - {name: out, type: outliers, flags: [pixel], outlier_threshold: zscore}
workflows:
  - name: w
    inputs: [a]
    steps:
      - {name: out, evaluator: out, input: a}
      - {name: clean, transform: remove, input: a, plans: {out: {}}}
tasks:
  - {name: t, workflow: w, sources: [a]}
"""
        result = _run(text, {"a": Images(12)})
        assert _sizes(result)["clean"] == 11
        assert result.steps["clean"].details["by_plan"] == {"out": {"items": 1}}

    def test_remove_refuses_a_plan_computed_on_another_dataset_when_the_config_loads(self) -> None:
        text = """
evaluators:
  - {name: dupes, type: duplicates}
workflows:
  - name: w
    inputs: [a, b]
    steps:
      - {name: on_a, evaluator: dupes, input: a}
      - {name: clean_b, transform: remove, input: b, plans: {on_a: {}}}
tasks:
  - {name: t, workflow: w, sources: [a, b]}
"""
        with pytest.raises(ValidationError, match="which was computed on `a`, not on `b`"):
            yaml_pipeline(text, _TWO)

    def test_collect_gathers_datasets_into_a_list_keyed_by_name(self) -> None:
        text = """
evaluators:
  - {name: lh, type: label-health}
workflows:
  - name: w
    inputs: [a, b]
    steps:
      - {name: both, transform: collect, input: [a, b]}
      - {name: each, evaluator: lh, input: both}
tasks:
  - {name: t, workflow: w, sources: [a, b]}
"""
        result = _run(text, _TWO)
        assert result.steps["both"].details == {"elements": {"a": "a", "b": "b"}}
        assert list(result.steps["each"].elements) == ["a", "b"]


_DETECTIONS = Detections(
    [[0] if i % 2 == 0 else [1, 0] for i in range(12)], {0: "car", 1: "person"}, dataset_id="det-a", duplicate_of={5: 0}
)
_DETECTIONS_B = Detections(
    [[0, 2] if i % 2 == 0 else [1] for i in range(12)], {0: "car", 1: "truck", 2: "pedestrian"}, dataset_id="det-b"
)
_VEHICLES = """
ontologies:
  - name: vehicles
    concepts:
      - {id: Vehicle, label: Vehicle, synonyms: [car, truck]}
      - {id: Person, label: Person, synonyms: [person, pedestrian]}
evaluators:
  - {name: align, type: label-alignment, ontology: vehicles}
  - {name: lh, type: label-health}
  - {name: dupes, type: duplicates}
"""


class TestDetectionTransforms:
    def test_wrap_turns_each_detection_into_an_item_of_its_own(self) -> None:
        text = """
workflows:
  - name: w
    inputs: [a]
    steps:
      - {name: crops, transform: wrap, input: a, wrapper: DetectionCrops, params: {min_size: 1}}
tasks:
  - {name: t, workflow: w, sources: [a]}
"""
        result = _run(text, {"a": _DETECTIONS})
        assert len(result.steps["crops"].output) == 18  # one crop per box: 6 items x 1 box, 6 x 2
        assert result.steps["crops"].details["kind"] == "object_detection"

    def test_conform_refuses_a_lossy_alignment_until_allow_says_to_accept_it(self) -> None:
        refused = (
            _VEHICLES
            + """
workflows:
  - name: w
    inputs: [b]
    steps:
      - {name: align_b, evaluator: align, input: b}
      - {name: b_c, transform: conform, input: b, alignment: align_b}
      - {name: counts, evaluator: lh, input: b_c}
tasks:
  - {name: t, workflow: w, sources: [b]}
"""
        )
        result = run_tasks(yaml_pipeline(refused, {"b": _DETECTIONS_B}))["t"]
        assert not result.success
        assert "car, truck collapse onto Vehicle" in result.steps["b_c"].errors[0]
        assert result.steps["counts"].status == "skipped"

        accepted = _run(
            refused.replace("alignment: align_b}", "alignment: align_b, allow: lossy}"), {"b": _DETECTIONS_B}
        )
        assert accepted.steps["b_c"].details["collapses"] == {"Vehicle": ["car", "truck"]}
        # Six even items hold a car and a pedestrian, six odd ones a truck.
        assert accepted.steps["counts"].output.data()["label_counts_per_class"] == {"Vehicle": 12, "Person": 6}

    def test_conform_merge_remove_and_export_make_one_cleaned_dataset_on_disk(self, tmp_path: Path) -> None:
        text = (
            _VEHICLES
            + """
workflows:
  - name: w
    inputs: [street, drone]
    steps:
      - {name: align_street, evaluator: align, input: street}
      - {name: align_drone, evaluator: align, input: drone}
      - {name: street_c, transform: conform, input: street, alignment: align_street}
      - {name: drone_c, transform: conform, input: drone, alignment: align_drone, allow: lossy}
      - {name: merged, transform: merge, input: [street_c, drone_c]}
      - {name: dupes, evaluator: dupes, input: merged}
      - {name: clean, transform: remove, input: merged, plans: {dupes: {keep: first}}}
      - {name: dataset, transform: export, input: clean, format: coco}
tasks:
  - {name: t, workflow: w, sources: [street, drone]}
"""
        )
        result = _run(text, {"street": _DETECTIONS, "drone": _DETECTIONS_B}, output_dir=tmp_path)
        assert _sizes(result)["merged"] == 24
        assert _sizes(result)["clean"] == 23
        exported = result.steps["dataset"].output
        assert (exported.format, exported.items) == ("coco", 23)
        written = tmp_path / "datasets" / "t.dataset"
        assert Path(exported.path) == written
        (provenance,) = json.loads((written / "provenance.json").read_text())["runs"]
        assert provenance["step"] == "dataset"
        assert {operand["source"] for operand in provenance["operands"]} == {"street", "drone"}
        assert [record["source"] for record in provenance["label_space"]] == ["street_c", "drone_c"]
        assert provenance["digest"]["items"] == 23

    def test_export_with_no_output_directory_is_skipped_not_failed(self) -> None:
        text = """
workflows:
  - name: w
    inputs: [a]
    steps:
      - {name: dataset, transform: export, input: a, format: coco}
tasks:
  - {name: t, workflow: w, sources: [a]}
"""
        result = _run(text, {"a": _DETECTIONS})
        assert result.steps["dataset"].status == "skipped"
        assert result.steps["dataset"].reason == "the run has no output directory"
