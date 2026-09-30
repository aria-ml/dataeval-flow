"""`run` on in-memory data matches the same run through a pipeline."""

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, assert_type

import numpy as np
import pytest
from dataeval.config import use_batch_size
from dataeval.extractors import FlattenExtractor
from pydantic import ValidationError

from dataeval_flow import PipelineConfig, load_config, run, run_tasks
from dataeval_flow.config import SourceConfig, StatsMeasureConfig, StatsPolicyConfig, TaskConfig, ViewConfig
from dataeval_flow.config.extractors import FlattenExtractorConfig, list_extractors
from dataeval_flow.evaluators.quality import DuplicatesConfig, DuplicatesResult, OutliersConfig, OutliersResult
from dataeval_flow.workflows import DatasetContext, WorkflowConfig, WorkflowContext
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig
from dataeval_flow.workflows.drift_monitoring import DriftDetectorMMD, DriftMonitoringConfig, DriftMonitoringResult
from dataeval_flow.workflows.parameter_sweep import ParameterSweepConfig
from tests.evaluator_toys import ToyImages, toy_pipeline


def _flatten(images: Any) -> np.ndarray:
    return np.stack([np.asarray(image, dtype=np.float32).ravel() for image in images])


def test_run_matches_a_pipeline() -> None:
    direct = run(DuplicatesConfig(), ToyImages())
    config = toy_pipeline(
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[TaskConfig(name="t", workflow="dupes", kind="evaluator", sources="src")],
    )
    piped = run_tasks(config)["t"]
    assert isinstance(direct, DuplicatesResult)
    assert direct.to_dict()["output"] == piped.to_dict()["output"]


def test_run_is_typed_to_the_configs_result() -> None:
    """Checked by pyright: the config's type parameter is the type `run` returns."""
    assert_type(run(DuplicatesConfig(), ToyImages()), DuplicatesResult)
    drift = DriftMonitoringConfig(detectors=[DriftDetectorMMD(method="mmd")])
    data = {"reference": ToyImages(seed=0), "test": ToyImages(seed=1)}
    assert_type(run(drift, data, extractor=FlattenExtractorConfig(batch_size=8)), DriftMonitoringResult)


def test_several_sources_run_in_the_order_given() -> None:
    """Out of alphabetical order, so a run that sorted its sources would read `incoming` as the reference."""
    drift = DriftMonitoringConfig(detectors=[DriftDetectorMMD(method="mmd")])
    result = run(
        drift,
        {"reference": ToyImages(seed=0), "incoming": ToyImages(seed=1)},
        extractor=FlattenExtractorConfig(batch_size=8),
    )
    assert isinstance(result, DriftMonitoringResult)
    assert result.success
    assert result.sources is not None
    assert list(result.sources) == ["reference", "incoming"]
    assert result.metadata.resolved_config["sources"][0]["name"] == "reference"


def test_an_extractor_config_needs_no_name() -> None:
    """Named after its `model`, as a workflow or evaluator config is after its `type`."""
    assert FlattenExtractorConfig().name == "flatten"
    assert FlattenExtractorConfig(name="flat").name == "flat"
    config = PipelineConfig.model_validate(
        {
            "extractors": [{"model": "flatten"}],
            "evaluators": [{"type": "duplicates", "cluster_sensitivity": 1.0}],
            "tasks": [{"name": "t", "evaluator": "duplicates", "sources": "src", "extractor": "flatten"}],
        }
    )
    assert config.extractors is not None
    assert config.extractors[0].name == "flatten"
    result = run(DuplicatesConfig(cluster_sensitivity=1.0), ToyImages(), extractor=FlattenExtractorConfig(batch_size=8))
    assert result.success
    assert result.metadata.model_id == "flatten (flatten)"


def test_a_protocol_extractor_runs_and_is_not_cached_to_disk(tmp_path: Path) -> None:
    drift = DriftMonitoringConfig(detectors=[DriftDetectorMMD(method="mmd")])
    with use_batch_size(8):
        result = run(
            drift, {"reference": ToyImages(seed=0), "test": ToyImages(seed=1)}, extractor=_flatten, cache_dir=tmp_path
        )
    assert result.success
    assert not list(tmp_path.rglob("embeddings_*"))


def test_an_extractor_config_is_cached_to_disk(tmp_path: Path) -> None:
    """The counterpart of the test above, so its empty cache means what it says."""
    drift = DriftMonitoringConfig(detectors=[DriftDetectorMMD(method="mmd")])
    extractor = FlattenExtractorConfig(batch_size=8)
    result = run(
        drift, {"reference": ToyImages(seed=0), "test": ToyImages(seed=1)}, extractor=extractor, cache_dir=tmp_path
    )
    assert result.success
    assert list(tmp_path.rglob("embeddings_*"))


def test_a_protocol_extractor_matches_the_same_extractor_as_config() -> None:
    """DataEval's own `FlattenExtractor`, passed as an object, finds what the `flatten` config finds."""
    dupes = DuplicatesConfig(cluster_sensitivity=1.0)
    data = ToyImages(near_duplicate=True)
    by_config = run(dupes, data, extractor=FlattenExtractorConfig(batch_size=8))
    with use_batch_size(8):
        by_object = run(dupes, data, extractor=FlattenExtractor())
    assert by_config.success
    assert by_object.success
    assert by_object.to_dict()["output"] == by_config.to_dict()["output"]


def test_two_protocol_extractors_never_share_embeddings_or_clusters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both serialize alike, so any keyed cache would serve one's embeddings and clusters to the other."""
    import dataeval_flow._cache as cache_module
    from dataeval_flow._cache import DatasetCache
    from dataeval_flow.config.extractors._base import _InstanceExtractorConfig

    clustered: list[np.ndarray] = []
    original = cache_module._do_compute_clusters
    monkeypatch.setattr(
        cache_module, "_do_compute_clusters", lambda e, *a, **k: clustered.append(e) or original(e, *a, **k)
    )
    toy = ToyImages()
    cache = DatasetCache.get_or_create(cache_dir=None, name="toy", cache_key=f"toy-run-{uuid.uuid4()}")

    def context(extractor: Any) -> WorkflowContext:
        config = _InstanceExtractorConfig(name="extractor", extractor=extractor)
        dc = DatasetContext(name="src", dataset=toy, cache=cache, extractor=config, batch_size=8)
        return WorkflowContext(dataset_contexts={"src": dc})

    pixels = context(_flatten)
    means = context(lambda images: _flatten(images).mean(axis=1, keepdims=True))
    assert pixels.embeddings("src").shape == (12, 3 * 16 * 16)
    assert means.embeddings("src").shape == (12, 1)
    pixels.clusters("src")
    means.clusters("src")
    assert [embeddings.shape for embeddings in clustered] == [(12, 3 * 16 * 16), (12, 1)]


_CLUSTERING_CLEANERS = [
    DataCleaningConfig(
        outlier_method="zscore",
        outlier_flags=["dimension"],
        outlier_cluster_threshold=2.0,
        outlier_cluster_algorithm="kmeans",
        outlier_n_clusters=2,
    ),
    ParameterSweepConfig(
        outlier_method=["zscore"],
        outlier_flags=["dimension"],
        outlier_cluster_threshold=[2.0],
        outlier_cluster_algorithm=["hdbscan"],
    ),
]


@pytest.mark.parametrize("clean", _CLUSTERING_CLEANERS, ids=lambda config: config.type)
def test_cleaning_never_serves_one_protocol_extractors_clusters_to_another(
    monkeypatch: pytest.MonkeyPatch, clean: WorkflowConfig[Any]
) -> None:
    """Two `run()` calls clustering in cleaning, one extractor object each, cluster their own embeddings."""
    import dataeval_flow._cache as cache_module
    from dataeval_flow._cache import DatasetCache

    clustered: list[tuple[int, ...]] = []
    original = cache_module._do_compute_clusters
    monkeypatch.setattr(
        cache_module, "_do_compute_clusters", lambda e, *a, **k: clustered.append(e.shape) or original(e, *a, **k)
    )
    DatasetCache.clear_instances()
    toy = ToyImages()
    with use_batch_size(8):
        pixels = run(clean, toy, extractor=_flatten)
        means = run(clean, toy, extractor=lambda images: _flatten(images).mean(axis=1, keepdims=True))
    assert pixels.success, pixels.errors
    assert means.success, means.errors
    assert clustered == [(12, 3 * 16 * 16), (12, 1)]


def test_the_instance_extractor_is_never_registered_nor_stateful() -> None:
    from dataeval_flow._embeddings import build_extractor, is_stateful_extractor
    from dataeval_flow.config.extractors._base import _InstanceExtractorConfig

    config = _InstanceExtractorConfig(name="extractor", extractor=_flatten)
    assert build_extractor(config) is _flatten
    assert is_stateful_extractor(config) is False
    assert "instance" not in {cls.name for cls in list_extractors()}
    type_ids = set(_consts(PipelineConfig.model_json_schema()))
    assert "flatten" in type_ids
    assert "instance" not in type_ids
    assert "extractor" not in config.model_dump()


def _consts(schema: object) -> Iterator[object]:
    """Every ``const`` in a JSON schema: the type id each registered branch states."""
    if isinstance(schema, dict):
        if "const" in schema:
            yield schema["const"]
        for value in schema.values():
            yield from _consts(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _consts(value)


def test_definitions_resolve_the_names_a_config_refers_to() -> None:
    outliers = OutliersConfig(flags=["visual"], stats="visual_only")
    policy = StatsPolicyConfig(name="visual_only", measure=[StatsMeasureConfig(families=["visual"])])
    result = run(outliers, ToyImages(), definitions=[policy])
    assert isinstance(result, OutliersResult)
    assert result.success
    with pytest.raises(ValueError, match="visual_only"):
        run(outliers, ToyImages())


def test_an_unregistered_plugin_says_how_it_is_found(plugins: dict[str, list[tuple[str, str]]]) -> None:
    """Its class exists in this process, but Flow finds a workflow by its type id, through an entry point."""
    from tests.example_plugin import CountConfig

    assert plugins == {}
    with pytest.raises(ValueError, match=r"Unknown workflow: 'example.count'.*'dataeval_flow.workflows' group"):
        run(CountConfig(), ToyImages())


def test_no_datasets_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one dataset"):
        run(DuplicatesConfig(), {})


def test_a_definition_of_another_type_is_refused() -> None:
    with pytest.raises(TypeError, match="SourceConfig"):
        run(DuplicatesConfig(), ToyImages(), definitions=[SourceConfig(name="s", dataset="d")])  # type: ignore[list-item]


def test_inputs_are_checked_before_anything_runs() -> None:
    drift = DriftMonitoringConfig(detectors=[DriftDetectorMMD(method="mmd")])
    with pytest.raises(ValidationError, match="two or more sources"):
        run(drift, ToyImages(), extractor=FlattenExtractorConfig())


def test_run_tasks_keys_results_by_task() -> None:
    config = toy_pipeline(
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[TaskConfig(name="t", workflow="dupes", kind="evaluator", sources="src")],
    )
    assert list(run_tasks(config)) == ["t"]


def test_load_config_reads_a_folder(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("evaluators:\n  - type: duplicates\n")
    (tmp_path / "b.yaml").write_text("seed: 3\n")
    config = load_config(tmp_path)
    assert config.seed == 3
    assert config.evaluators is not None


def test_load_config_reads_a_file_named_as_a_string(tmp_path: Path) -> None:
    (tmp_path / "params.yaml").write_text("seed: 5\n")
    assert load_config(str(tmp_path / "params.yaml")).seed == 5


def test_a_cleaning_run_carries_a_thumbnail_of_each_item_its_report_names() -> None:
    """The white image its outliers flag (7), and both members of its exact (0, 5) and near (3, 9) duplicate groups,
    each captured once. Each is named by ``data``, the preset's name for the Dataset ``run`` hands it."""
    config = DataCleaningConfig(outlier_method="zscore", outlier_flags=["pixel", "visual"])
    result = run(config, ToyImages(count=40, near_duplicate=True))
    assert sorted(asset.item.index for asset in result.assets) == [0, 3, 5, 7, 9]
    assert {(asset.item.source, asset.media_type, asset.width, asset.height) for asset in result.assets} == {
        ("data", "image/webp", 16, 16)
    }
    page = result.to_html()
    assert page.count('<details class="thumb">') == 5
    assert 'alt="data 7"' in page


def test_an_ood_run_reads_each_sample_s_thumbnail_from_its_own_test_source() -> None:
    """Test sources are scored joined end to end; night's black images, all out of distribution, are night's own."""
    import base64
    import io

    from PIL import Image

    from dataeval_flow.workflows.ood_detection import OODDetectionConfig, OODDetectorKNeighbors

    class Dark(ToyImages):
        def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
            image, target, datum = super().__getitem__(index)
            return np.zeros_like(image), target, datum

    config = OODDetectionConfig(
        detectors=[OODDetectorKNeighbors(k=3), OODDetectorKNeighbors(k=5)], metadata_insights=False
    )
    data = {"reference": ToyImages(seed=0, count=20), "day": ToyImages(seed=1, count=12), "night": Dark(count=4)}
    result = run(config, data, extractor=FlattenExtractorConfig(batch_size=8))
    night = [asset for asset in result.assets if asset.item.source == "night"]
    assert sorted(asset.item.index for asset in night) == [0, 1, 2, 3]
    for asset in night:
        assert np.asarray(Image.open(io.BytesIO(base64.b64decode(asset.data))).convert("L")).max() < 8
    assert {asset.item.source for asset in result.assets} <= {"day", "night"}


def test_with_images_off_no_item_is_read_for_a_thumbnail(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataeval_flow._capture as capture_module

    def refuse(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise AssertionError("captured with images off")

    monkeypatch.setattr(capture_module, "capture", refuse)
    config = DataCleaningConfig(outlier_method="zscore", outlier_flags=["pixel", "visual"])
    result = run(config, ToyImages(count=40, near_duplicate=True), report_images=False)
    assert result.assets == []
    assert "assets" not in result.to_dict()
    assert '<span class="item">7</span>' in result.to_html()


def test_an_ood_thumbnail_is_the_sample_scored_though_its_view_shuffles_unseeded() -> None:
    """The detectors score one draw of a random view; each thumbnail is read from that draw, not a fresh one."""
    import base64
    import io

    from PIL import Image

    from dataeval_flow._blocks._items import refs_in
    from dataeval_flow.config import DatasetProtocolConfig, ViewOperation
    from dataeval_flow.workflows.ood_detection import OODDetectionConfig, OODDetectionResult, OODDetectorKNeighbors
    from tests.finding_blocks import tables

    class Mixed(ToyImages):
        """Items 0 to 3 black, which the reference has never seen; the rest noise like the reference's."""

        def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
            image, target, datum = super().__getitem__(index)
            return (np.zeros_like(image) if index < 4 else image), target, datum

    config = PipelineConfig(
        datasets=[
            DatasetProtocolConfig(name="ref", format="maite", dataset=ToyImages(seed=0, count=20)),
            DatasetProtocolConfig(name="mixed", format="maite", dataset=Mixed(seed=1, count=16)),
        ],
        views=[ViewConfig(name="shuffled", operations=[ViewOperation(type="Shuffle", params={})])],
        sources=[
            SourceConfig(name="reference", dataset="ref"),
            SourceConfig(name="test", dataset="mixed", view="shuffled"),
        ],
        extractors=[FlattenExtractorConfig(name="flat", batch_size=8)],
        workflows=[
            OODDetectionConfig(
                name="ood", detectors=[OODDetectorKNeighbors(k=3), OODDetectorKNeighbors(k=5)], metadata_insights=False
            )
        ],
        tasks=[TaskConfig(name="t", workflow="ood", sources=["reference", "test"], extractor="flat")],
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, OODDetectionResult)
    (aggregate,) = [finding for finding in result.findings if finding.title.startswith("Aggregate")]
    agreed = {ref for row in tables(aggregate)[0].rows for ref in refs_in(row.get("image"))}
    shades = {
        asset.item: int(np.asarray(Image.open(io.BytesIO(base64.b64decode(asset.data))).convert("L")).max())
        for asset in result.assets
    }
    assert len(agreed) == 4
    assert {item: shades[item] < 8 for item in agreed} == dict.fromkeys(agreed, True)


def test_the_result_block_limits_a_run_s_tables() -> None:
    """``max_rows`` and ``preview_rows`` reach the tables a real preset's steps build, for that run alone."""
    from dataeval_flow._blocks import Paragraph, Section, Table
    from dataeval_flow._tables import TableLimits, table_limits
    from dataeval_flow.config import ResultConfig
    from dataeval_flow.steps import ChainResult
    from tests.finding_blocks import walk

    config = toy_pipeline(
        workflows=[DataCleaningConfig(name="clean", outlier_method="zscore", outlier_flags=["pixel", "visual"])],
        tasks=[TaskConfig(name="t", workflow="clean", sources="src")],
        dataset=ToyImages(count=40, near_duplicate=True),
    )
    config.result = ResultConfig(max_rows=1, preview_rows=-1, max_images=0)
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.steps["dupes"].type == "duplicates"
    report = result._document(detailed=True).blocks
    (dupes,) = [block for block in report if isinstance(block, Section) and block.title == "dupes (duplicates)"]
    blocks = list(walk(dupes.blocks))
    (groups,) = [block for block in blocks if isinstance(block, Table)]
    assert (len(groups.rows), groups.preview) == (1, None)
    assert "2 groups of images; the 1 largest are listed, and every one is in `output.rows`." in [
        block.text for block in blocks if isinstance(block, Paragraph)
    ]
    assert table_limits() == TableLimits()
