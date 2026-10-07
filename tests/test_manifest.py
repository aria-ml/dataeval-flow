"""`dataset_manifest`, its file and `compare`; the manifests a run writes; and `verify` (follow-ons spec §5)."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dataeval_flow import DatasetManifest, ManifestDiff, dataset_digest, dataset_manifest
from dataeval_flow._cache import DatasetCache
from dataeval_flow._runner import run
from dataeval_flow._verify_cli import verify
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages
from tests.test_audit_preset import _OUTLIERS


@pytest.fixture(autouse=True)
def _fresh_caches():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


class _Edited(ToyImages):
    """`ToyImages` with item 3's image inverted."""

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        image, target, meta = super().__getitem__(index)
        return (255 - image if index == 3 else image), target, meta


class _Unnamed(ToyImages):
    """`ToyImages` whose items carry `id` 0 each, or none: no id tells two items apart."""

    def __init__(self, *, edited: bool = False, ids: bool = False) -> None:
        super().__init__()
        self._edited, self._ids = edited, ids

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        image, target, _ = super().__getitem__(index)
        image = 255 - image if self._edited and index == 3 else image
        return image, target, ({"id": 0} if self._ids else {})


def test_the_manifest_holds_the_digest_dataset_digest_gives() -> None:
    manifest = dataset_manifest(ToyImages())
    assert manifest.digest == dataset_digest(ToyImages())
    assert [entry.id for entry in manifest.entries] == list(range(12))
    assert manifest.classes == {0: "a", 1: "b"}


def test_a_manifest_reads_back_as_written(tmp_path: Path) -> None:
    manifest = dataset_manifest(ToyImages())
    manifest.save(tmp_path / "manifests" / "train.json")
    assert DatasetManifest.load(tmp_path / "manifests" / "train.json") == manifest


def test_a_manifest_of_another_scheme_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    dataset_manifest(ToyImages()).save(path)
    path.write_text(json.dumps({**json.loads(path.read_text()), "scheme": 99}))
    with pytest.raises(ValueError, match="scheme 99 manifest"):
        DatasetManifest.load(path)


def test_the_same_items_compare_equal() -> None:
    assert not dataset_manifest(ToyImages()).compare(dataset_manifest(ToyImages()))


def test_an_edited_item_is_named_by_its_id() -> None:
    assert dataset_manifest(ToyImages()).compare(dataset_manifest(_Edited())) == ManifestDiff(changed=(3,))


def test_a_dropped_item_is_missing() -> None:
    diff = dataset_manifest(ToyImages(12)).compare(dataset_manifest(ToyImages(11)))
    assert diff.missing == (11,)
    assert diff.added == ()


@pytest.mark.parametrize("ids", [False, True], ids=["no-ids", "repeated-ids"])
def test_items_without_unique_ids_compare_by_index(ids: bool) -> None:
    diff = dataset_manifest(_Unnamed(ids=ids)).compare(dataset_manifest(_Unnamed(ids=ids, edited=True)))
    assert diff == ManifestDiff(missing=(3,), added=(3,))


def test_renamed_classes_are_reported() -> None:
    renamed = ToyImages()
    renamed.metadata["index2label"] = {0: "cat", 1: "dog"}  # type: ignore[typeddict-item]
    assert dataset_manifest(ToyImages()).compare(dataset_manifest(renamed)) == ManifestDiff(classes=True)


def test_the_cli_writes_each_split_s_manifest_beside_the_results(tmp_path: Path) -> None:
    config = chain_pipeline(
        workflows=[{"name": "w", "type": "audit", **_OUTLIERS}],
        tasks=[{"name": "t", "workflow": "w", "sources": ["train", "test"]}],
        datasets={"train": ToyImages(), "test": ToyImages(seed=1)},
    )
    with patch("dataeval_flow._runner._resolve_config", return_value=config):
        run("pipeline.yaml", tmp_path / "out", data_dir=tmp_path)
    root = tmp_path / "out" / "results" / "manifests" / "t"
    assert DatasetManifest.load(root / "content-digest-train.json").digest == dataset_digest(ToyImages())
    assert DatasetManifest.load(root / "content-digest-evals" / "test.json").digest == dataset_digest(ToyImages(seed=1))


def _verify(dataset: Any, tmp_path: Path) -> int:
    dataset_manifest(ToyImages()).save(tmp_path / "train.json")
    with patch("dataeval_flow.config._loader.load_config", return_value=chain_pipeline(datasets={"train": dataset})):
        return verify(tmp_path / "train.json", tmp_path / "pipeline.yaml", "train")


def test_verify_passes_a_source_that_holds_the_recorded_items(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert _verify(ToyImages(), tmp_path) == 0
    assert "holds the 12 items" in capsys.readouterr().out


def test_verify_names_the_item_that_changed(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert _verify(_Edited(), tmp_path) == 1
    assert "Changed: 1 (3)" in capsys.readouterr().out


def test_verify_reports_a_manifest_it_cannot_read(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert verify(tmp_path / "missing.json", tmp_path / "pipeline.yaml", "train") == 1
    assert "ERROR" in capsys.readouterr().err


def test_the_cli_parses_verify() -> None:
    from dataeval_flow.__main__ import _build_parser

    args = _build_parser().parse_args(["verify", "m.json", "--config", "p.yaml", "--source", "train"])
    parsed = (args.command, args.manifest, args.config, args.source)
    assert parsed == ("verify", Path("m.json"), Path("p.yaml"), "train")


@pytest.mark.parametrize(
    "argv",
    [
        ["--data", "/x", "verify", "m.json", "--config", "p.yaml", "--source", "train"],
        ["verify", "m.json", "--config", "p.yaml", "--source", "train", "--data", "/x"],
    ],
    ids=["before-verify", "after-verify"],
)
def test_verify_reads_data_given_before_or_after_it(argv: list[str]) -> None:
    from dataeval_flow.__main__ import _build_parser

    assert _build_parser().parse_args(argv).data == Path("/x")


def test_a_source_name_that_is_not_a_directory_skips_its_manifest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = chain_pipeline(
        workflows=[{"name": "w", "type": "audit", **_OUTLIERS}],
        tasks=[{"name": "t", "workflow": "w", "sources": ["train", "a/b"]}],
        datasets={"train": ToyImages(), "a/b": ToyImages(seed=1)},
    )
    with patch("dataeval_flow._runner._resolve_config", return_value=config), caplog.at_level("WARNING"):
        run("pipeline.yaml", tmp_path / "out", data_dir=tmp_path)
    root = tmp_path / "out" / "results" / "manifests" / "t"
    assert (root / "content-digest-train.json").exists()
    assert sorted(p.name for p in (tmp_path / "out").rglob("b.json")) == []
    assert not (root / "content-digest-evals" / "a").exists()
    assert "source name 'a/b' must be one directory segment" in caplog.text


def test_a_manifest_records_each_item_s_index_in_the_dataset_under_its_views() -> None:
    from dataeval.data import Indices, View

    manifest = dataset_manifest(View(ToyImages(), operations=[Indices([2, 0])]))
    assert [(entry.index, entry.root) for entry in manifest.entries] == [(0, 2), (1, 0)]


def test_a_manifest_records_each_item_s_hashes_as_item_hashes_gives_them() -> None:
    from dataeval_flow._digest import item_hashes

    toys = ToyImages()
    entries = dataset_manifest(toys).entries
    assert [(entry.content, entry.metadata) for entry in entries] == [item_hashes(toys[i]) for i in range(len(toys))]


def test_recording_roots_and_metadata_hashes_leaves_the_digests_as_they_were() -> None:
    digest = dataset_digest(ToyImages())
    assert (digest.content[:12], digest.metadata[:12]) == ("771c534e9e79", "97aec4f1e57c")


def test_a_manifest_reads_back_with_its_source(tmp_path: Path) -> None:
    import dataclasses

    manifest = dataclasses.replace(dataset_manifest(ToyImages()), source="train")
    manifest.save(tmp_path / "train.json")
    assert DatasetManifest.load(tmp_path / "train.json") == manifest


def test_a_manifest_written_before_roots_reads_back_without_them(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    dataset_manifest(ToyImages()).save(path)
    data = json.loads(path.read_text())
    data["entries"] = [{key: entry[key] for key in ("index", "id", "content")} for entry in data["entries"]]
    path.write_text(json.dumps(data))
    loaded = DatasetManifest.load(path)
    assert loaded.source is None
    assert {(entry.root, entry.metadata) for entry in loaded.entries} == {(None, None)}


def test_a_content_digest_task_s_manifest_names_its_source_and_lands_as_the_task_finishes(tmp_path: Path) -> None:
    from dataeval_flow import _orchestrator

    config = chain_pipeline(
        evaluators=[{"name": "digest", "type": "content-digest"}],
        tasks=[{"name": "d", "evaluator": "digest", "sources": "train"}],
        datasets={"train": ToyImages()},
    )
    path = tmp_path / "out" / "results" / "manifests" / "d" / "content-digest.json"
    landed: list[bool] = []
    real = _orchestrator.run_tasks

    def spy(*args: Any, on_result: Any, **kwargs: Any) -> Any:
        def finished(name: str, result: Any) -> None:
            on_result(name, result)
            landed.append(path.exists())

        return real(*args, on_result=finished, **kwargs)

    with (
        patch("dataeval_flow._runner._resolve_config", return_value=config),
        patch.object(_orchestrator, "run_tasks", spy),
    ):
        run("pipeline.yaml", tmp_path / "out", data_dir=tmp_path)
    assert landed == [True]
    manifest = DatasetManifest.load(path)
    assert manifest.source == "train"
    assert [entry.root for entry in manifest.entries] == list(range(12))


def test_a_task_s_manifest_lands_before_its_result_names_it(tmp_path: Path) -> None:
    from dataeval_flow import _runner

    config = chain_pipeline(
        evaluators=[{"name": "digest", "type": "content-digest"}],
        tasks=[{"name": "d", "evaluator": "digest", "sources": "train"}],
        datasets={"train": ToyImages()},
    )
    path = tmp_path / "out" / "results" / "manifests" / "d" / "content-digest.json"
    landed: list[bool] = []
    real = _runner._write_results

    def spy(*args: Any, **kwargs: Any) -> Any:
        landed.append(path.exists())
        return real(*args, **kwargs)

    with (
        patch("dataeval_flow._runner._resolve_config", return_value=config),
        patch.object(_runner, "_write_results", spy),
    ):
        run("pipeline.yaml", tmp_path / "out", data_dir=tmp_path)
    assert landed == [True]


def test_a_manifest_is_written_whole(tmp_path: Path) -> None:
    path = tmp_path / "manifests" / "train.json"
    dataset_manifest(ToyImages()).save(path)
    assert sorted(p.name for p in path.parent.iterdir()) == ["train.json"]
    with patch("pathlib.Path.replace", side_effect=OSError("disk full")), pytest.raises(OSError, match="disk full"):
        dataset_manifest(ToyImages(seed=1)).save(path)
    assert DatasetManifest.load(path) == dataset_manifest(ToyImages())
