"""The chains 2a exists for, written as a user would write them (spec §1, §11.2)."""

import json
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import load_dataset, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.steps import ChainResult
from tests.chain_toys import ToyDetections, yaml_pipeline

_ONTOLOGY = """
ontologies:
  - name: vehicles
    concepts:
      - {id: Vehicle, label: Vehicle, synonyms: [car, truck]}
      - {id: Person, label: Person, synonyms: [person, pedestrian]}
extractors:
  - {name: flat, model: flatten, batch_size: 8}
"""


def _corpora() -> dict[str, ToyDetections]:
    """Two corpora of 24 detection images: `a` names car and person, `b` car, truck and pedestrian (lossy)."""
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
  - {name: align, type: scope.label-alignment, ontology: vehicles}
  - {name: dupes, type: quality.duplicates}
  - {name: coverage, type: scope.coverage}
  - {name: balance, type: bias.balance}
workflows:
  - name: corpus
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
      - {name: split, transform: split, input: clean, test_frac: 0.25}
      - {name: train_balance, evaluator: balance, input: split.train}
tasks:
  - {name: corpus, workflow: corpus, sources: [a, b], extractor: flat}
"""
    )
    # Coverage reads the crops' metadata, where DataEval drops the `source_id` DetectionCrops adds to every crop.
    with pytest.warns(UserWarning, match="`source_id` was dropped"):
        result = _run(text, _corpora(), tmp_path, "corpus")
    assert result.success, result.errors
    assert {record.status for record in result.steps.values()} == {"ok"}
    sizes = {record.name: record.items for record in result.metadata.lineage}
    assert (sizes["merged"], sizes["clean"]) == (48, 47)  # item 9 of `a` copies item 3
    assert (sizes["split.train"], sizes["split.val"], sizes["split.test"]) == (35, 0, 12)
    assert {record.source for record in result.metadata.label_space} == {"a2", "b2"}


def test_clean_remove_export_writes_a_corpus_without_the_duplicate(tmp_path: Path) -> None:
    text = (
        _ONTOLOGY
        + """
evaluators:
  - {name: dupes, type: quality.duplicates}
workflows:
  - name: clean_export
    inputs: [data]
    steps:
      - {name: dupes, evaluator: dupes, input: data}
      - {name: clean, transform: remove, input: data, plans: {dupes: {keep: first}}}
      - {name: corpus, transform: export, input: clean, format: coco}
tasks:
  - {name: prep, workflow: clean_export, sources: [a]}
"""
    )
    result = _run(text, {"a": _corpora()["a"]}, tmp_path, "prep")
    assert result.success, result.errors
    written = _coco(tmp_path / "datasets" / "prep.corpus")
    assert len(written["images"]) == 23
    assert result.steps["corpus"].output.items == 23
    reloaded = load_dataset(tmp_path / "datasets" / "prep.corpus", dataset_format="coco")
    assert len(reloaded) == 23
    assert [reloaded[i][2]["source_id"] for i in range(23)] == [str(i) for i in range(24) if i != 9]


_CONFORM = (
    _ONTOLOGY
    + """
evaluators:
  - {name: align, type: scope.label-alignment, ontology: vehicles}
workflows:
  - name: conform_and_merge
    inputs: [a, b]
    steps:
      - {name: align_a, evaluator: align, input: a}
      - {name: align_b, evaluator: align, input: b}
      - {name: a2, transform: conform, input: a, alignment: align_a}
      - {name: b2, transform: conform, input: b, alignment: align_b%s}
      - {name: merged, transform: merge, input: [a2, b2]}
      - {name: corpus, transform: export, input: merged, format: coco, to: conformed}
tasks:
  - {name: build, workflow: conform_and_merge, sources: [a, b]}
"""
)


def test_align_conform_merge_export_refuses_a_collapse_until_allowed(tmp_path: Path) -> None:
    refused = _run(_CONFORM % "", _corpora(), tmp_path / "refused", "build")
    assert not refused.success
    assert refused.steps["b2"].status == "failed"
    assert "is lossy, beyond `allow: lossless`" in refused.steps["b2"].errors[0]
    assert (refused.steps["merged"].status, refused.steps["corpus"].status) == ("skipped", "skipped")
    assert not (tmp_path / "refused" / "datasets" / "conformed").exists()

    allowed = _run(_CONFORM % ", allow: lossy", _corpora(), tmp_path / "allowed", "build")
    assert allowed.success, allowed.errors
    corpus = tmp_path / "allowed" / "datasets" / "conformed"
    written = _coco(corpus)
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
    provenance = json.loads((corpus / "provenance.json").read_text())["runs"][-1]
    assert [record["name"] for record in provenance["lineage"]] == ["merged", "a2", "b2", "a", "b"]
    conforms = [
        (entry["source"], entry["ontology"], entry["ontology_digest"], entry["class_remap"])
        for entry in provenance["conforms"]
    ]
    assert conforms == [("a2", "vehicles", "7322513e6772", a2_remap), ("b2", "vehicles", "7322513e6772", b2_remap)]
