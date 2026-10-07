"""The chains 2a exists for, written as a user would write them (spec §1, §11.2)."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dataeval_flow import load_dataset, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._ci_reports import junit_report, markdown_summary
from dataeval_flow.evaluators.quality import OutliersEvaluator
from dataeval_flow.steps import ChainResult
from tests.chain_toys import ToyDetections, yaml_pipeline
from tests.evaluator_toys import ToyImages

_ONTOLOGY = """
ontologies:
  - name: vehicles
    concepts:
      - {id: Vehicle, label: Vehicle, synonyms: [car, truck]}
      - {id: Person, label: Person, synonyms: [person, pedestrian]}
extractors:
  - {name: flat, model: flatten, batch_size: 8}
"""


def _datasets() -> dict[str, ToyDetections]:
    """Two datasets of 24 detection images: `a` names car and person, `b` car, truck and pedestrian (lossy)."""
    a = ToyDetections(
        [[i % 2, (i + 1) % 2] for i in range(24)], {0: "car", 1: "person"}, duplicate_of={9: 3}, dataset_id="a"
    )
    b = ToyDetections([[i % 3] for i in range(24)], {0: "car", 1: "truck", 2: "pedestrian"}, dataset_id="b")
    return {"a": a, "b": b}


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _run(text: str, datasets: dict[str, Any], output_dir: Path, task: str) -> ChainResult:
    result = run_tasks(yaml_pipeline(text, datasets), output_dir=output_dir)[task]
    assert isinstance(result, ChainResult)
    return result


def _coco(root: Path) -> dict:
    return json.loads((root / "annotations" / "instances.json").read_text())


def test_the_parent_specs_example_chain_runs_from_yaml(tmp_path: Path) -> None:
    text = (
        _ONTOLOGY
        + """
evaluators:
  - {name: align, type: label-alignment, ontology: vehicles}
  - {name: dupes, type: duplicates}
  - {name: coverage, type: coverage}
  - {name: balance, type: balance}
workflows:
  - name: dataset
    inputs: [a, b]
    steps:
      - {name: align_a, evaluator: align, input: a}
      - {name: align_b, evaluator: align, input: b}
      - {name: a2, transform: conform, input: a, alignment: align_a}
      - {name: b2, transform: conform, input: b, alignment: align_b, allow: lossy}
      - {name: merged, transform: merge, input: [a2, b2]}
      - {name: dupes, evaluator: dupes, input: merged}
      - name: clean
        transform: remove
        input: merged
        plans: {dupes: {keep: first}}
      - {name: crops, transform: wrap, input: clean, wrapper: DetectionCrops}
      - {name: coverage, evaluator: coverage, input: crops}
      - {name: balance, evaluator: balance, input: clean}
      - {name: split, transform: split, input: clean, test_frac: 0.25, val_frac: 0, stratify: false}
      - {name: train_balance, evaluator: balance, input: split.train}
tasks:
  - {name: dataset, workflow: dataset, sources: [a, b], extractor: flat}
"""
    )
    # Coverage reads the crops' metadata, where DataEval drops the `source_id` DetectionCrops adds to every crop.
    with pytest.warns(UserWarning, match="`source_id` was dropped"):
        result = _run(text, _datasets(), tmp_path, "dataset")
    assert result.success, result.errors
    assert {record.status for record in result.steps.values()} == {"ok"}
    sizes = {record.name: record.items for record in result.metadata.lineage}
    assert (sizes["merged"], sizes["clean"]) == (48, 47)  # item 9 of `a` copies item 3
    assert (sizes["split.train"], sizes["split.val"], sizes["split.test"]) == (35, 0, 12)
    assert {record.source for record in result.metadata.label_space} == {"a2", "b2"}


def test_clean_remove_export_writes_a_dataset_without_the_duplicate(tmp_path: Path) -> None:
    text = (
        _ONTOLOGY
        + """
evaluators:
  - {name: dupes, type: duplicates}
workflows:
  - name: clean_export
    inputs: [data]
    steps:
      - {name: dupes, evaluator: dupes, input: data}
      - {name: clean, transform: remove, input: data, plans: {dupes: {keep: first}}}
      - {name: dataset, transform: export, input: clean, format: coco}
tasks:
  - {name: prep, workflow: clean_export, sources: [a]}
"""
    )
    result = _run(text, {"a": _datasets()["a"]}, tmp_path, "prep")
    assert result.success, result.errors
    written = _coco(tmp_path / "datasets" / "prep.dataset")
    assert len(written["images"]) == 23
    assert result.steps["dataset"].output.items == 23
    reloaded = load_dataset(tmp_path / "datasets" / "prep.dataset", dataset_format="coco")
    assert len(reloaded) == 23
    assert [reloaded[i][2]["source_id"] for i in range(23)] == [str(i) for i in range(24) if i != 9]


_CONFORM = (
    _ONTOLOGY
    + """
evaluators:
  - {name: align, type: label-alignment, ontology: vehicles}
workflows:
  - name: conform_and_merge
    inputs: [a, b]
    steps:
      - {name: align_a, evaluator: align, input: a}
      - {name: align_b, evaluator: align, input: b}
      - {name: a2, transform: conform, input: a, alignment: align_a}
      - {name: b2, transform: conform, input: b, alignment: align_b%s}
      - {name: merged, transform: merge, input: [a2, b2]}
      - {name: dataset, transform: export, input: merged, format: coco, to: conformed}
tasks:
  - {name: build, workflow: conform_and_merge, sources: [a, b]}
"""
)


def test_align_conform_merge_export_refuses_a_collapse_until_allowed(tmp_path: Path) -> None:
    refused = _run(_CONFORM % "", _datasets(), tmp_path / "refused", "build")
    assert not refused.success
    assert refused.steps["b2"].status == "failed"
    assert "is lossy, beyond `allow: lossless`" in refused.steps["b2"].errors[0]
    assert (refused.steps["merged"].status, refused.steps["dataset"].status) == ("skipped", "skipped")
    assert not (tmp_path / "refused" / "datasets" / "conformed").exists()

    allowed = _run(_CONFORM % ", allow: lossy", _datasets(), tmp_path / "allowed", "build")
    assert allowed.success, allowed.errors
    dataset = tmp_path / "allowed" / "datasets" / "conformed"
    written = _coco(dataset)
    assert len(written["images"]) == 48
    assert sorted(category["name"] for category in written["categories"]) == ["Person", "Vehicle"]
    a2_remap = {"car": "Vehicle", "person": "Person"}
    b2_remap = {"car": "Vehicle", "truck": "Vehicle", "pedestrian": "Person"}
    spaces = [
        (record.source, record.ontology, record.ontology_digest, dict(record.class_remap), list(record.target))
        for record in allowed.metadata.label_space
    ]
    assert spaces == [
        ("a2", "vehicles", "7322513e6772", a2_remap, ["Vehicle", "Person"]),
        ("b2", "vehicles", "7322513e6772", b2_remap, ["Vehicle", "Person"]),
    ]
    provenance = json.loads((dataset / "provenance.json").read_text())["runs"][-1]
    assert provenance["operands"] == [
        {"source": "a", "dataset": "a_data", "view": None, "class_remap": {}, "provenance": {}},
        {"source": "b", "dataset": "b_data", "view": None, "class_remap": {}, "provenance": {}},
    ]
    assert [record["name"] for record in provenance["lineage"]] == ["merged", "a2", "b2", "a", "b"]
    # The roots `a` and `b` read no view, so the label space holds each conform on the way, in chain order.
    assert "conforms" not in provenance
    records = [
        (entry["source"], entry["ontology"], entry["ontology_digest"], entry["class_remap"])
        for entry in provenance["label_space"]
    ]
    assert records == [("a2", "vehicles", "7322513e6772", a2_remap), ("b2", "vehicles", "7322513e6772", b2_remap)]


_CLEANING_REPORT = """
evaluators:
  - {name: outliers, type: outliers, flags: [pixel, visual], outlier_threshold: zscore, per_target: true}
  - {name: dupes, type: duplicates, merge_near_duplicates: true}
  - {name: labels, type: label-health}

workflows:
  - name: cleaning_report
    inputs: [data]
    steps:
      - {name: outliers, evaluator: outliers, input: data}
      - {name: labels, evaluator: labels, input: data}
      - {name: by-class, combine: outliers-by-class, input: data, outliers: outliers}
      - {name: dupes, evaluator: dupes, input: data}
      - {name: image-outliers, check: image-outliers, input: outliers}
      - {name: target-outliers, check: target-outliers, input: outliers, labels: labels}
      - {name: classwise, check: class-outliers, input: by-class}
      - {name: duplicates, check: image-duplicates, input: dupes}
      - {name: imbalance, check: class-imbalance, input: labels}

tasks:
  - {name: report, workflow: cleaning_report, sources: [src]}
"""


def test_data_cleaning_s_report_runs_as_a_chain_and_every_output_shows_its_verdict(tmp_path: Path) -> None:
    result = _run(_CLEANING_REPORT, {"src": ToyImages(count=24)}, tmp_path, "report")
    assert result.success, result.errors
    assert result.health == {"status": "warning", "warnings": 3, "findings": 4, "failed_steps": []}
    payload = result.to_dict()
    assert [finding["step"] for finding in payload["findings"]] == [  # type: ignore[union-attr]
        "image-outliers",
        "classwise",
        "duplicates",
        "imbalance",
    ]
    assert "Health: 3 warning(s)" in result.report()
    assert '<span class="badge warning">3 warnings</span>' in result.to_html()
    junit = junit_report({"report": result})
    assert 'tests="4" failures="3" errors="0"' in junit
    assert "**Health:** 3 warnings" in markdown_summary({"report": result})


def test_a_check_whose_evaluator_failed_is_not_assessed_in_a_run_from_yaml(tmp_path: Path) -> None:
    text = _CLEANING_REPORT.replace(
        "{name: outliers, evaluator: outliers, input: data}",
        "{name: outliers, evaluator: outliers, input: data, optional: true}",
    )
    with patch.object(OutliersEvaluator, "run", side_effect=RuntimeError("no stats")):
        result = _run(text, {"src": ToyImages(count=24)}, tmp_path, "report")
    image = next(finding for finding in result.findings if finding.step == "image-outliers")
    assert (image.severity, image.brief) == ("info", "not assessed")
    assert image.description == "Not assessed: `outliers` was skipped: failed: RuntimeError: no stats."
    assert result.failed_steps == []
    assert result.health["status"] == "warning"  # the duplicates are still judged, and still warn


def test_a_not_assessed_description_ends_in_one_full_stop(tmp_path: Path) -> None:
    text = _CLEANING_REPORT.replace(
        "{name: outliers, evaluator: outliers, input: data}",
        "{name: outliers, evaluator: outliers, input: data, optional: true}",
    )
    with patch.object(OutliersEvaluator, "run", side_effect=RuntimeError("no stats.")):
        result = _run(text, {"src": ToyImages(count=24)}, tmp_path, "report")
    image = next(finding for finding in result.findings if finding.step == "image-outliers")
    assert image.description == "Not assessed: `outliers` was skipped: failed: RuntimeError: no stats."


def test_a_data_cleaning_step_hands_its_cleaned_dataset_to_an_export(tmp_path: Path) -> None:
    text = """
workflows:
  - {name: basic_clean, type: data-cleaning, outliers: {flags: [pixel, visual], outlier_threshold: zscore}}
  - name: clean_export
    inputs: [data]
    steps:
      - {name: cleaning, workflow: basic_clean, input: data}
      - {name: dataset, transform: export, input: cleaning.clean, format: coco}
tasks:
  - {name: prep, workflow: clean_export, sources: [a]}
"""
    result = _run(text, {"a": _datasets()["a"]}, tmp_path, "prep")
    assert result.success, result.errors
    assert [(f.severity, f.title, f.step) for f in result.findings] == [
        ("ok", "Image Outliers", "cleaning/image-outliers"),
        ("ok", "Class Outliers", "cleaning/class-outliers"),
        ("warning", "Image Duplicates", "cleaning/image-duplicates"),
    ]
    assert result.health == {"status": "warning", "warnings": 1, "findings": 3, "failed_steps": []}
    assert result.steps["cleaning/clean"].details == {
        "removed": {"items": 1, "detections": 0, "tracks": 0, "frames": 0},
        "by_plan": {"duplicates": {"items": 1}, "outliers": {}},
    }
    written = _coco(tmp_path / "datasets" / "prep.dataset")
    assert (len(written["images"]), len(written["annotations"])) == (23, 46)
    reloaded = load_dataset(tmp_path / "datasets" / "prep.dataset", dataset_format="coco")
    assert [reloaded[i][2]["source_id"] for i in range(23)] == [str(i) for i in range(24) if i != 9]
