"""Drift over rows that are detections: chunks of whole images, chunks without detections left out, and what was
compared (uncertainty-drift spec §4.3, §4.4, §7)."""

from typing import Any

import numpy as np

from dataeval_flow._blocks import Fields, Paragraph, Table
from dataeval_flow.evaluators.shift import ChunkedDriftConfig, DriftUnivariateConfig, DriftWassersteinConfig
from dataeval_flow.evaluators.shift._report import drift_section
from dataeval_flow.evaluators.shift._rows import DriftRowsOutput, image_chunks
from tests.onnx_toys import DETECTOR, Frames, element, install, model_files, run_uncertainty

_INPUTS = ["reference", {"name": "tests", "list": True}]


def _workflow(*steps: dict[str, Any], reads: str = "tests") -> dict[str, Any]:
    ks = {"name": "ks", "evaluator": "ks", "input": ["reference", reads]}
    return {"name": "w", "inputs": _INPUTS, "steps": [*steps, ks]}


def _run(tmp_path, monkeypatch, reference, test, *, chunking=None, workflow=None, extra=None):
    install(monkeypatch, DETECTOR)
    model_files(tmp_path, "IMAGE_OBJECT_DETECTION")
    ks = DriftUnivariateConfig(name="ks", chunking=chunking)
    datasets = {"reference": reference, "cam1": test}
    return run_uncertainty(tmp_path, workflow or _workflow(), [ks], datasets, detector=True, extra=extra)


def _frames(*spans: tuple[int, float], seed: int = 0) -> Frames:
    """Frames in runs of (count, brightness), each jittered by up to 0.05 so entropies spread; a brightness of 0.05
    holds no detections at confidence 0.3, and one of 0.5 holds two."""
    rng = np.random.default_rng(seed + 100)
    brightness = [max(0.0, level + rng.uniform(-0.05, 0.05)) for count, level in spans for _ in range(count)]
    return Frames(brightness, seed=seed)


def test_image_chunks_cut_images_as_dataeval_cuts_rows():
    from dataeval.shift._drift._chunk import CountChunker, SizeChunker

    for incomplete in ("keep", "drop", "append"):
        ours = image_chunks(2, ChunkedDriftConfig(chunk_size=3, incomplete=incomplete))
        assert [c.tolist() for c in ours] == [c.tolist() for c in SizeChunker(3, incomplete).split(2)]
    for n in (7, 10, 12):
        for size in (3, 5):
            for incomplete in ("keep", "drop", "append"):
                ours = image_chunks(n, ChunkedDriftConfig(chunk_size=size, incomplete=incomplete))
                assert [c.tolist() for c in ours] == [c.tolist() for c in SizeChunker(size, incomplete).split(n)]
        for count in (2, 3):
            ours = image_chunks(n, ChunkedDriftConfig(chunk_count=count))
            assert [c.tolist() for c in ours] == [c.tolist() for c in CountChunker(count).split(n)]


def test_an_unchunked_run_records_what_it_compared(tmp_path, monkeypatch):
    output = element(_run(tmp_path, monkeypatch, _frames((20, 0.5)), _frames((15, 0.5), seed=1)), "ks").output
    assert isinstance(output, DriftRowsOutput)
    assert output.rows == {
        "unit": "detections",
        "compared": {"reference": 40, "tests[cam1]": 30},
        "images": {"reference": 20, "tests[cam1]": 15},
        "confidence": 0.3,
    }


def test_chunks_hold_whole_images_cut_at_the_references_first_chunk(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, _frames((30, 0.5)), _frames((25, 0.5), seed=1), chunking={"chunk_count": 3})
    output = element(result, "ks").output
    assert output.rows["chunk_images"] == [[0, 9], [10, 24]]
    assert output.rows["unassessed"] == []
    assert output.details.height == 2


def test_a_test_chunk_without_detections_is_listed_and_left_out_and_the_check_skips_it(tmp_path, monkeypatch):
    check = {"name": "c", "check": "drift", "input": "ks", "chunk_percent": 50.0}
    workflow = {"name": "w", "inputs": _INPUTS, "steps": [*_workflow()["steps"], check]}
    test = _frames((10, 0.5), (10, 0.05), (10, 0.5), seed=1)
    result = _run(tmp_path, monkeypatch, _frames((30, 0.5)), test, chunking={"chunk_count": 3}, workflow=workflow)
    output = element(result, "ks").output
    assert output.rows["chunk_images"] == [[0, 9], [20, 29]]
    assert output.rows["unassessed"] == [{"source": "tests[cam1]", "images": [10, 19], "reason": "no detections"}]
    assert output.details.height == 2
    (finding,) = element(result, "c").output
    assert finding.brief.endswith("/2 chunks drifted")


def test_an_empty_reference_chunk_is_dropped_and_fewer_than_three_left_is_refused_in_images(tmp_path, monkeypatch):
    reference = _frames((10, 0.05), (20, 0.5))
    result = _run(tmp_path, monkeypatch, reference, _frames((30, 0.5), seed=1), chunking={"chunk_count": 3})
    ks = element(result, "ks")
    assert ks.status == "failed"
    assert "30 images into 3 chunks, 2 with detections" in ks.errors[0]


def test_a_source_without_detections_fails_its_step_naming_it(tmp_path, monkeypatch):
    ks = element(_run(tmp_path, monkeypatch, _frames((20, 0.5)), _frames((20, 0.05), seed=1)), "ks")
    assert ks.status == "failed"
    assert "No detections at `confidence` ≥ 0.3 in `tests[cam1]`" in ks.errors[0]


def test_rows_count_the_source_after_its_view(tmp_path, monkeypatch):  # Review Focus 1
    views = [{"name": "first10", "operations": [{"type": "Limit", "params": {"size": 10}}]}]
    workflow = _workflow({"name": "few", "transform": "view", "input": "tests", "view": "first10"}, reads="few")
    result = _run(
        tmp_path, monkeypatch, _frames((20, 0.5)), _frames((30, 0.5), seed=1), workflow=workflow, extra={"views": views}
    )
    output = element(result, "ks").output
    assert sorted(output.rows["images"].values()) == [10, 20]
    assert sorted(output.rows["compared"].values()) == [20, 40]


def test_wasserstein_with_a_validation_source_needs_detections_in_each(tmp_path, monkeypatch):  # Review Focus 2
    install(monkeypatch, DETECTOR)
    model_files(tmp_path, "IMAGE_OBJECT_DETECTION")
    workflow = {
        "name": "w",
        "inputs": ["reference", "validation", {"name": "tests", "list": True}],
        "steps": [{"name": "ws", "evaluator": "ws", "input": ["reference", "validation", "tests"]}],
    }
    datasets = {
        "reference": _frames((20, 0.5)),
        "validation": _frames((20, 0.5), seed=2),
        "cam1": _frames((20, 0.9), seed=1),
    }
    result = run_uncertainty(tmp_path, workflow, [DriftWassersteinConfig(name="ws")], datasets, detector=True)
    assert element(result, "ws").output.rows["compared"] == {"reference": 40, "validation": 40, "tests[cam1]": 40}
    datasets["validation"] = _frames((20, 0.05), seed=2)
    result = run_uncertainty(tmp_path, workflow, [DriftWassersteinConfig(name="ws")], datasets, detector=True)
    assert "in `validation`" in element(result, "ws").errors[0]


def test_the_section_says_what_was_compared_and_labels_chunks_by_image_range():
    rows = {
        "unit": "detections",
        "compared": {"reference": 4812, "cam1": 3977},
        "images": {"reference": 300, "cam1": 280},
        "confidence": 0.25,
        "chunk_images": [[0, 99], [100, 279]],
        "unassessed": [{"source": "cam1", "images": [180, 199], "reason": "no detections"}],
    }
    chunk = {"key": "[0:10]", "value": 0.1, "drifted": False, "lower_threshold": None, "upper_threshold": 0.3}
    details = {"shape": "table", "rows": [chunk, chunk | {"key": "[10:20]", "drifted": True}]}
    data = {"drifted": True, "distance": 0.1, "threshold": 0.3, "metric_name": "ks", "details": details, "rows": rows}
    compared, table, unassessed = drift_section({"data": data})
    assert compared == Paragraph(
        text="Compared detections at confidence ≥ 0.25: `reference` 4,812 in 300 images; `cam1` 3,977 in 280 images."
    )
    assert isinstance(table, Table)
    assert [row["chunk"] for row in table.rows] == ["images 0–99", "images 100–279"]
    assert unassessed == Paragraph(text="Not assessed: images 180–199 in `cam1` (no detections).")


def test_a_section_on_embeddings_is_unchanged():
    data = {"drifted": False, "distance": 0.1, "threshold": 0.05, "metric_name": "mmd2", "details": {"p_val": 0.4}}
    (fields,) = drift_section({"data": data})
    assert isinstance(fields, Fields)


def test_a_chunk_size_over_the_reference_makes_no_chunk_and_fails_in_images(tmp_path, monkeypatch):
    chunking = {"chunk_size": 50, "incomplete": "drop"}
    ks = element(_run(tmp_path, monkeypatch, _frames((30, 0.5)), _frames((30, 0.5), seed=1), chunking=chunking), "ks")
    assert ks.status == "failed"
    assert "`chunking` cuts `reference`'s 30 images into no chunk" in ks.errors[0]


def test_a_test_source_under_one_chunk_fails_in_images(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, _frames((30, 0.5)), _frames((5, 0.5), seed=1), chunking={"chunk_count": 3})
    ks = element(result, "ks")
    assert ks.status == "failed"
    assert "`tests[cam1]` holds 5 images, fewer than one chunk of 10" in ks.errors[0]
