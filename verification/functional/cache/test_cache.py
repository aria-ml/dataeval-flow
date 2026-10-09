"""TC-14-1 — the disk cache a pipeline run fills and reuses, and the in-memory cache it falls back to."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml

from dataeval_flow import load_config, run_tasks
from dataeval_flow.config import PipelineConfig
from verification.functional.reporting._project import Invocation, pipeline, task, write_images, write_project

pytestmark = pytest.mark.required

IMAGES = 12  # two classes of six
FLATTENED = 3 * 32 * 32  # a flattened 32-pixel RGB image


def _files(cache: Path) -> dict[str, int]:
    """Every file under the cache root, with the time it was last written."""
    return {str(path.relative_to(cache)): path.stat().st_mtime_ns for path in cache.rglob("*") if path.is_file()}


def _entries(cache: Path, pattern: str) -> list[Path]:
    return sorted(cache.glob(f"v1/*/sel_*/{pattern}"))


@pytest.fixture
def project(tmp_path: Path) -> tuple[PipelineConfig, Path]:
    """Two image folders (the first holds two exact duplicates) and tasks that read them in different ways."""
    write_images(tmp_path, per_class=6, duplicates=2, name="imgs")
    write_images(tmp_path, per_class=6, seed=3, name="imgs2")
    config = pipeline(
        tasks=[
            {"name": "dup_main", "evaluator": "dupes", "sources": "main", "extractor": "flat"},
            {"name": "dup_main_batch4", "evaluator": "dupes", "sources": "main", "extractor": "flat4"},
            {"name": "dup_view", "evaluator": "dupes", "sources": "first4", "extractor": "flat"},
            {"name": "dup_other", "evaluator": "dupes", "sources": "other", "extractor": "flat"},
            task("quality_pixel", "q_pixel"),
            task("quality_dimension", "q_dimension"),
            task("quality_policy", "q_policy"),
        ],
        workflows=[
            {"name": "q_pixel", "type": "quality", "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}},
            {
                "name": "q_policy",
                "type": "quality",
                "metadata": "no_filename",
                "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"},
            },
            {
                "name": "q_dimension",
                "type": "quality",
                "outliers": {"flags": ["pixel", "dimension"], "outlier_threshold": "zscore"},
            },
        ],
        extra={
            "metadata": [{"name": "no_filename", "exclude": ["filename"]}],
            "evaluators": [{"name": "dupes", "type": "duplicates", "cluster_sensitivity": 1.0}],
            "views": [{"name": "limit4", "operations": [{"type": "Limit", "params": {"size": 4}}]}],
            "extractors": [
                {"name": "flat", "model": "flatten", "batch_size": 8},
                {"name": "flat4", "model": "flatten", "batch_size": 4},
            ],
        },
    )
    config["datasets"].append({"name": "ds2", "format": "image_folder", "path": "imgs2", "infer_labels": True})
    config["sources"] = [
        {"name": "main", "dataset": "ds"},
        {"name": "first4", "dataset": "ds", "view": "limit4"},
        {"name": "other", "dataset": "ds2"},
    ]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_config(path), tmp_path


def _run(project: tuple[PipelineConfig, Path], name: str, cache: Path | None) -> Any:
    config, root = project
    return run_tasks(config, tasks=name, data_dir=root, cache_dir=cache)[name]


class TestDiskCache:
    def test_the_directory_is_created_on_first_use_and_holds_versioned_artifacts(
        self, project: tuple[PipelineConfig, Path], tmp_path: Path
    ) -> None:
        cache = tmp_path / "does" / "not" / "exist" / "cache"
        result = _run(project, "dup_main", cache)

        assert result.success, result.errors
        assert cache.is_dir()
        assert [path.name for path in cache.iterdir()] == ["v1"]
        (dataset_dir,) = (cache / "v1").iterdir()
        (selection_dir,) = dataset_dir.iterdir()
        assert dataset_dir.name.startswith("ds_")
        assert selection_dir.name.startswith("sel_")
        kinds = sorted(path.suffix for path in selection_dir.iterdir())
        assert kinds == [".json", ".npy", ".npz", ".parquet"]  # stats (two files), embeddings, clusters
        (embeddings,) = selection_dir.glob("embeddings_*.npy")
        assert np.load(embeddings).shape == (IMAGES + 2, FLATTENED)
        assert not list(cache.rglob("*.tmp"))  # every write is replaced whole

    def test_quality_leaves_metadata_and_statistics(self, project: tuple[PipelineConfig, Path], tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        assert _run(project, "quality_pixel", cache).success
        (selection_dir,) = (path for path in cache.glob("v1/*/sel_*"))
        names = sorted(path.suffix for path in selection_dir.iterdir())
        assert names == [".dem", ".json", ".json", ".parquet"]  # metadata archive and sidecar, stats and sidecar

    def test_metadata_is_cached_per_encoding_policy(self, project: tuple[PipelineConfig, Path], tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        _run(project, "quality_pixel", cache)
        (first,) = _entries(cache, "metadata_*.dem")

        policy = _run(project, "quality_policy", cache)

        assert policy.success, policy.errors
        archives = _entries(cache, "metadata_*.dem")
        assert len(archives) == 2  # one archive for each way of encoding the factors
        assert first in archives
        assert len(_entries(cache, "stats_*.parquet")) == 1  # the statistics do not depend on the policy
        assert len(_entries(cache, "metadata_*.json")) == 2  # each with the sidecar that names its encoding

    def test_an_unchanged_rerun_reads_the_cache_and_writes_nothing(
        self, project: tuple[PipelineConfig, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        cache = tmp_path / "cache"
        first = _run(project, "dup_main", cache)
        written = _files(cache)

        with caplog.at_level(logging.INFO, logger="dataeval_flow._cache"):
            second = _run(project, "dup_main", cache)

        assert _files(cache) == written  # same files, none rewritten
        assert "Cache hit: embeddings" in caplog.text
        assert "Cache hit: cluster result" in caplog.text
        assert "Cache hit: stats" in caplog.text
        assert "Computing embeddings" not in caplog.text
        assert second.to_dict()["output"] == first.to_dict()["output"]
        assert second.to_dict()["output"]["rows"], "the planted duplicate pair is found from the cache too"

    def test_a_different_extractor_setting_is_a_different_entry(
        self, project: tuple[PipelineConfig, Path], tmp_path: Path
    ) -> None:
        cache = tmp_path / "cache"
        _run(project, "dup_main", cache)
        before = _files(cache)
        (first,) = _entries(cache, "embeddings_*.npy")

        _run(project, "dup_main_batch4", cache)

        after = _files(cache)
        assert len(_entries(cache, "embeddings_*.npy")) == 2
        assert {
            name: stamp for name, stamp in after.items() if name in before
        } == before  # the first entry is untouched
        assert first.exists()

    def test_changed_dataset_content_is_a_new_entry_and_the_old_one_is_kept(
        self, project: tuple[PipelineConfig, Path], tmp_path: Path
    ) -> None:
        from PIL import Image

        cache = tmp_path / "cache"
        _run(project, "dup_main", cache)
        (before,) = _entries(cache, "embeddings_*.npy")
        old = np.load(before)

        # Replace one image in place: same path and file name, different pixels.
        Image.fromarray(np.full((32, 32, 3), 7, dtype=np.uint8)).save(project[1] / "imgs" / "class_1" / "img_0.png")
        changed = _run(project, "dup_main", cache)

        assert changed.success, changed.errors
        embeddings = _entries(cache, "embeddings_*.npy")
        assert len(embeddings) == 2
        assert before in embeddings
        np.testing.assert_array_equal(np.load(before), old)  # the earlier entry is as it was
        (fresh,) = (path for path in embeddings if path != before)
        assert np.load(fresh).shape == old.shape
        assert not np.array_equal(np.load(fresh), old)

    def test_the_same_dataset_under_another_view_is_kept_apart(
        self, project: tuple[PipelineConfig, Path], tmp_path: Path
    ) -> None:
        cache = tmp_path / "cache"
        _run(project, "dup_main", cache)
        view = _run(project, "dup_view", cache)

        assert view.success, view.errors
        embeddings = _entries(cache, "embeddings_*.npy")
        assert len(embeddings) == 2
        assert sorted(np.load(path).shape[0] for path in embeddings) == [4, IMAGES + 2]
        assert len({path.parent.parent for path in embeddings}) == 2  # a dataset directory per view

    def test_entries_are_isolated_per_source_and_never_served_across(
        self, project: tuple[PipelineConfig, Path], tmp_path: Path
    ) -> None:
        cache = tmp_path / "cache"
        main = _run(project, "dup_main", cache)
        other = _run(project, "dup_other", cache)  # a different dataset, with no duplicates

        assert len({path.parent.parent for path in _entries(cache, "embeddings_*.npy")}) == 2
        assert main.to_dict()["output"]["rows"], "the first dataset has a duplicate pair"
        assert other.to_dict()["output"]["rows"] == [], "and the second dataset must not inherit it from the cache"
        # The second dataset's cached embeddings are its own pixels, not the first dataset's.
        shapes = sorted(np.load(path).shape for path in _entries(cache, "embeddings_*.npy"))
        assert shapes == [(IMAGES, FLATTENED), (IMAGES + 2, FLATTENED)]

    def test_statistics_accumulate_in_one_entry_across_workflows(
        self, project: tuple[PipelineConfig, Path], tmp_path: Path
    ) -> None:
        cache = tmp_path / "cache"
        _run(project, "quality_pixel", cache)
        (stats,) = _entries(cache, "stats_*.parquet")
        narrow = pl.read_parquet(stats)

        _run(project, "quality_dimension", cache)

        (same,) = _entries(cache, "stats_*.parquet")
        wide = pl.read_parquet(same)
        assert same == stats
        assert {"mean", "std", "xxhash"} <= set(narrow.columns)
        assert "aspect_ratio" not in narrow.columns
        assert {"aspect_ratio", "width", "height"} <= set(wide.columns)
        assert set(narrow.columns) < set(wide.columns)
        assert wide.select(narrow.columns).equals(narrow)  # what was cached is kept as it was

    def test_a_corrupt_entry_is_recomputed_with_a_warning_and_does_not_fail_the_run(
        self, project: tuple[PipelineConfig, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        cache = tmp_path / "cache"
        first = _run(project, "dup_main", cache)
        (embeddings,) = _entries(cache, "embeddings_*.npy")
        embeddings.write_bytes(b"not a numpy file")

        with caplog.at_level(logging.WARNING, logger="dataeval_flow._cache"):
            second = _run(project, "dup_main", cache)

        assert second.success, second.errors
        assert "Failed to load embeddings from cache" in caplog.text
        assert second.to_dict()["output"] == first.to_dict()["output"]
        assert np.load(embeddings).shape == (IMAGES + 2, FLATTENED)  # replaced with a good entry


class TestMemoryOnlyCache:
    def test_without_a_directory_nothing_is_written(self, project: tuple[PipelineConfig, Path], tmp_path: Path) -> None:
        before = {path.name for path in tmp_path.iterdir()}
        result = _run(project, "dup_main", None)
        assert result.success
        assert {path.name for path in tmp_path.iterdir()} == before

    def test_a_second_run_in_the_same_process_reuses_what_the_first_computed(
        self, project: tuple[PipelineConfig, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        first = _run(project, "dup_main", None)
        with caplog.at_level(logging.DEBUG, logger="dataeval_flow._cache"):
            second = _run(project, "dup_main", None)
        assert "Memory hit: embeddings" in caplog.text
        assert "Computing embeddings" not in caplog.text
        assert second.to_dict()["output"] == first.to_dict()["output"]


class TestCacheOnTheCommandLine:
    def test_the_flag_fills_the_directory_and_a_second_run_reads_it(
        self, tmp_path: Path, cli: Callable[..., Invocation]
    ) -> None:
        config = write_project(tmp_path, duplicates=2)
        cache = tmp_path / "cache"

        assert cli("-c", config, "-d", tmp_path, "-o", tmp_path / "out1", "-k", cache).code == 0
        assert _entries(cache, "stats_*.parquet")
        written = _files(cache)

        assert cli("-c", config, "-d", tmp_path, "-o", tmp_path / "out2", "--cache", cache).code == 0
        assert _files(cache) == written
        assert "Cache hit: stats" in (tmp_path / "out2" / "result.log").read_text()

    def test_the_environment_variable_names_the_directory_and_the_flag_overrides_it(
        self, tmp_path: Path, cli: Callable[..., Invocation]
    ) -> None:
        config = write_project(tmp_path)
        from_env, from_flag = tmp_path / "env_cache", tmp_path / "flag_cache"

        assert cli("-c", config, "-d", tmp_path, env={"DATAEVAL_CACHE": str(from_env)}).code == 0
        assert _entries(from_env, "stats_*.parquet")

        assert (
            cli("-c", config, "-d", tmp_path, "-k", from_flag, env={"DATAEVAL_CACHE": str(tmp_path / "ignored")}).code
            == 0
        )
        assert _entries(from_flag, "stats_*.parquet")
        assert not (tmp_path / "ignored").exists()
