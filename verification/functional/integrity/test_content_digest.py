"""TC-22-2 — Content-digest evaluator: a run pins its dataset in its result and in a manifest it writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataeval_flow import DatasetManifest, dataset_digest, dataset_manifest, load_dataset, run
from dataeval_flow.evaluators.quality import ContentDigestConfig, ContentDigestResult
from verification.functional.integrity._data import Items, digest_pipeline, make_items, write_coco, write_pipeline
from verification.functional.integrity.conftest import RecordedRun
from verification.helpers import run_cli

pytestmark = pytest.mark.required


def _digest_values(digest) -> dict[str, object]:
    return {"content": digest.content, "metadata": digest.metadata, "items": digest.items, "scheme": digest.scheme}


class TestContentDigestEvaluator:
    def test_evaluator_gives_the_digests_dataset_digest_gives(self) -> None:
        data = Items(make_items())
        result = run(ContentDigestConfig(), data)
        assert isinstance(result, ContentDigestResult)
        assert result.success, result.errors
        assert result.output.data() == _digest_values(dataset_digest(data))

    def test_a_pipeline_task_records_the_digests_of_its_source_in_its_result(self, recorded: RecordedRun) -> None:
        results = json.loads((recorded.output / "results" / "result.json").read_text())
        data = results["digest"]["output"]["data"]
        expected = dataset_digest(load_dataset(recorded.data / "fixture", dataset_format="coco"))
        assert data == _digest_values(expected)
        assert data["items"] == 6

    def test_the_text_report_shows_the_item_count_and_digests(self, recorded: RecordedRun) -> None:
        text = (recorded.output / "results" / "result.txt").read_text()
        digest = dataset_digest(load_dataset(recorded.data / "fixture", dataset_format="coco"))
        assert "Items:" in text
        assert digest.content in text
        assert digest.metadata in text


class TestManifestsWrittenByARun:
    def test_the_task_writes_a_manifest_named_for_it(self, recorded: RecordedRun) -> None:
        assert recorded.manifest.is_file()
        manifest = DatasetManifest.load(recorded.manifest)
        expected = dataset_manifest(load_dataset(recorded.data / "fixture", dataset_format="coco"))
        assert manifest.digest == expected.digest
        assert manifest.classes == {0: "boat", 1: "swimmer"}

    def test_the_manifest_names_its_source_and_each_item_s_id_and_hashes(self, recorded: RecordedRun) -> None:
        manifest = DatasetManifest.load(recorded.manifest)
        assert manifest.source == "data"
        assert [(entry.index, entry.id, entry.root) for entry in manifest.entries] == [(i, i, i) for i in range(6)]
        assert all(
            len(entry.content) == 64 and entry.metadata and len(entry.metadata) == 64 for entry in manifest.entries
        )

    def test_the_manifest_digest_equals_the_digest_in_the_result(self, recorded: RecordedRun) -> None:
        results = json.loads((recorded.output / "results" / "result.json").read_text())
        recorded_digest = results["digest"]["output"]["data"]
        digest = DatasetManifest.load(recorded.manifest).digest
        assert (digest.content, digest.metadata) == (recorded_digest["content"], recorded_digest["metadata"])

    def test_each_content_digest_task_has_its_own_manifest(self, tmp_path: Path) -> None:
        write_coco(tmp_path)
        write_coco(tmp_path, "other", count=4)
        pipeline = digest_pipeline()
        pipeline["datasets"].append({"name": "ds2", "format": "coco", "path": "other"})
        pipeline["sources"].append({"name": "data2", "dataset": "ds2"})
        pipeline["tasks"].append({"name": "digest2", "evaluator": "digest", "sources": "data2"})
        config = write_pipeline(tmp_path / "pipeline.yaml", pipeline)
        proc = run_cli("-c", str(config), "-d", str(tmp_path), "-o", str(tmp_path / "out"))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        root = tmp_path / "out" / "results" / "manifests"
        assert DatasetManifest.load(root / "digest" / "content-digest.json").digest.items == 6
        assert DatasetManifest.load(root / "digest2" / "content-digest.json").digest.items == 4

    def test_a_run_without_an_output_directory_prints_the_digest_and_writes_no_manifest(self, tmp_path: Path) -> None:
        write_coco(tmp_path)
        config = write_pipeline(tmp_path / "pipeline.yaml", digest_pipeline())
        proc = run_cli("-c", str(config), "-d", str(tmp_path), cwd=tmp_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        digest = dataset_digest(load_dataset(tmp_path / "fixture", dataset_format="coco"))
        assert digest.content in proc.stdout
        assert sorted(path.name for path in tmp_path.iterdir()) == ["fixture", "pipeline.yaml"]
