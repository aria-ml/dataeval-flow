"""A stateful extractor is fitted once per task, on the source that first asks, whatever the cache holds.

The extractor here fits on the first batch it is given, as BoVW does, and describes every image as its mean
brightness less the fitted batch's. An embedding therefore carries the data it was fitted on, so a test can
tell a vocabulary fitted on the reference from one fitted anywhere else.
"""

import logging
import uuid
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

import numpy as np
import pytest
from dataeval.config import use_batch_size, use_seed

from dataeval_flow import PipelineConfig, run_task, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._embeddings import shared_extractor_scope
from dataeval_flow._orchestrator import _run_single_task
from dataeval_flow.config import DatasetProtocolConfig, SourceConfig, TaskConfig
from dataeval_flow.config.extractors import Extractor, ExtractorConfig, FlattenExtractorConfig
from dataeval_flow.workflows import DatasetContext, WorkflowContext
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig
from dataeval_flow.workflows.drift_monitoring import DriftDetectorMMD, DriftMonitoringConfig
from tests.evaluator_toys import ToyImages

BATCH = 4


class RecordingConfig(ExtractorConfig):
    """Settings for `test.recording`."""

    model: str = "test.recording"


class _Recording:
    """Fits on the first batch it is given, then describes each image as its mean less that batch's."""

    builds: ClassVar[list["_Recording"]] = []

    def __init__(self) -> None:
        self.fitted_on: np.ndarray | None = None
        _Recording.builds.append(self)

    def __call__(self, data: Any, /) -> np.ndarray:
        means = _means(data)
        if self.fitted_on is None:
            self.fitted_on = np.stack([np.asarray(image) for image in data])
        return means - self.center

    @property
    def center(self) -> float:
        assert self.fitted_on is not None
        return float(self.fitted_on.astype(np.float32).mean())


class RecordingExtractor(Extractor[RecordingConfig]):
    """A stateful test extractor that records the data it was fitted on."""

    name: ClassVar[str] = "test.recording"
    description: ClassVar[str] = "Mean brightness less the first batch's."
    stateful: ClassVar[bool] = True

    def build(self, config: RecordingConfig, transforms: Any) -> Any:
        return _Recording()


def _means(images: Any) -> np.ndarray:
    return np.array([[float(np.asarray(image, dtype=np.float32).mean())] for image in images], dtype=np.float32)


def _first_batch(dataset: ToyImages) -> np.ndarray:
    return np.stack([dataset[i][0] for i in range(BATCH)])


def _fitted_on(dataset: ToyImages, described: ToyImages) -> np.ndarray:
    """What `described` embeds to under an extractor fitted on `dataset`."""
    center = float(_first_batch(dataset).astype(np.float32).mean())
    return _means([described[i][0] for i in range(len(described))]) - center


@pytest.fixture
def recording(plugins: dict[str, list[tuple[str, str]]]) -> Iterator[list[_Recording]]:
    """Register the recording extractor, and start from empty in-memory caches so every run computes."""
    plugins["dataeval_flow.extractors"] = [("test.recording", f"{__name__}:RecordingExtractor")]
    _Recording.builds.clear()
    DatasetCache.clear_instances()
    yield _Recording.builds
    _Recording.builds.clear()
    DatasetCache.clear_instances()


REFERENCE, INCOMING = ToyImages(seed=0), ToyImages(seed=1)


def _pipeline() -> tuple[TaskConfig, PipelineConfig]:
    task = TaskConfig(name="drift", workflow="drift", sources=["reference", "incoming"], extractor="rec")
    config = PipelineConfig(
        datasets=[
            DatasetProtocolConfig(name="reference", dataset=REFERENCE),
            DatasetProtocolConfig(name="incoming", dataset=INCOMING),
        ],
        sources=[
            SourceConfig(name="reference", dataset="reference"),
            SourceConfig(name="incoming", dataset="incoming"),
        ],
        extractors=[RecordingConfig(name="rec", batch_size=BATCH)],
        workflows=[DriftMonitoringConfig(name="drift", detectors=[DriftDetectorMMD()])],
        tasks=[task],
    )
    return task, config


def _via_run_task(task: TaskConfig, config: PipelineConfig) -> Any:
    return run_task(task, config)


def _via_run_tasks(task: TaskConfig, config: PipelineConfig) -> Any:
    return run_tasks(config)[task.name]


def _via_the_tui(task: TaskConfig, config: PipelineConfig) -> Any:
    """What `_app/app.py` calls for one task and for "run all"."""
    return _run_single_task(task, config, data_dir=None, cache_dir=None)


@pytest.mark.parametrize("entry_point", [_via_run_task, _via_run_tasks, _via_the_tui])
def test_every_entry_point_builds_and_fits_once_per_task(
    recording: list[_Recording], entry_point: Callable[[TaskConfig, PipelineConfig], Any]
) -> None:
    task, config = _pipeline()
    result = entry_point(task, config)
    assert result.success, result.errors
    assert len(recording) == 1
    assert recording[0].fitted_on is not None
    np.testing.assert_array_equal(recording[0].fitted_on, _first_batch(REFERENCE))


def _contexts(*names: str, cache_key: str, batch_size: int | None = BATCH) -> WorkflowContext:
    """A context over the reference and incoming toys, each with its own cache, as the orchestrator builds them."""
    data = {"reference": REFERENCE, "incoming": INCOMING}
    return WorkflowContext(
        dataset_contexts={
            name: DatasetContext(
                name=name,
                dataset=data[name],
                extractor=RecordingConfig(name="rec", batch_size=batch_size),
                batch_size=batch_size,
                cache=DatasetCache.get_or_create(cache_dir=None, name=name, cache_key=cache_key),
            )
            for name in names
        }
    )


@pytest.fixture
def computed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The selections whose embeddings are computed rather than served, in order."""
    import dataeval_flow._cache as cache_module

    selections: list[str] = []
    compute = cache_module._do_compute_embeddings

    def spy(*args: Any, **kwargs: Any) -> Any:
        selections.append(kwargs.get("selection") or "unnamed")
        return compute(*args, **kwargs)

    monkeypatch.setattr(cache_module, "_do_compute_embeddings", spy)
    return selections


@pytest.mark.parametrize("seed", [0, None])
def test_embeddings_each_selection_fitted_on_itself_never_serve_a_task(
    recording: list[_Recording], seed: int | None
) -> None:
    """Outside a scope a selection fits itself, and its entry names it, so a task's other sources miss it."""
    key = f"poison-{uuid.uuid4()}"
    with use_seed(seed):
        unscoped = _contexts("reference", "incoming", cache_key=key)
        unscoped.embeddings("reference")
        np.testing.assert_allclose(unscoped.embeddings("incoming"), _fitted_on(INCOMING, INCOMING))
        assert len(recording) == 2

        with shared_extractor_scope():
            task = _contexts("reference", "incoming", cache_key=key)
            reference, incoming = task.embeddings("reference"), task.embeddings("incoming")

    np.testing.assert_allclose(reference, _fitted_on(REFERENCE, REFERENCE))
    np.testing.assert_allclose(incoming, _fitted_on(REFERENCE, INCOMING))
    np.testing.assert_array_equal(recording[-1].fitted_on, _first_batch(REFERENCE))


def _sources(selections: list[str]) -> list[str]:
    """The source each computed selection belongs to: its dataset identity is `{source}_{hash}/{selection}`."""
    return [selection.split("/")[0].rsplit("_", 1)[0] for selection in selections]


def _partly_warm(key: str) -> np.ndarray:
    """Task 1 embeds the reference alone; task 2 reads it again and embeds the incoming source. Returns the latter."""
    with shared_extractor_scope():
        _contexts("reference", cache_key=key).embeddings("reference")
    with shared_extractor_scope():
        task = _contexts("reference", "incoming", cache_key=key)
        task.embeddings("reference")
        return task.embeddings("incoming")


def test_a_seeded_partly_warm_cache_serves_the_fitter_and_refits_on_it(
    recording: list[_Recording], computed: list[str]
) -> None:
    """Under a seed the refit is the fit, so the reference is served and the extractor is fitted on it again."""
    with use_seed(0):
        incoming = _partly_warm(f"warm-{uuid.uuid4()}")

    assert _sources(computed) == ["reference", "incoming"]
    assert len(recording) == 2
    np.testing.assert_array_equal(recording[1].fitted_on, _first_batch(REFERENCE))
    np.testing.assert_allclose(incoming, _fitted_on(REFERENCE, INCOMING))


def test_an_unseeded_partly_warm_cache_serves_nothing_across_tasks(
    recording: list[_Recording], computed: list[str]
) -> None:
    """Without a seed a refit learns another vocabulary, so the second task computes its reference again."""
    with use_seed(None):
        incoming = _partly_warm(f"warm-{uuid.uuid4()}")

    assert _sources(computed) == ["reference", "reference", "incoming"]
    assert len(recording) == 2
    np.testing.assert_array_equal(recording[1].fitted_on, _first_batch(REFERENCE))
    np.testing.assert_allclose(incoming, _fitted_on(REFERENCE, INCOMING))


def test_unseeded_embeddings_are_reused_within_their_task(recording: list[_Recording], computed: list[str]) -> None:
    with use_seed(None), shared_extractor_scope():
        task = _contexts("reference", cache_key=f"within-{uuid.uuid4()}")
        first, again = task.embeddings("reference"), task.embeddings("reference")
    assert len(computed) == 1
    assert len(recording) == 1
    np.testing.assert_array_equal(first, again)


def test_unseeded_embeddings_say_once_why_they_are_not_cached(
    recording: list[_Recording],  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import dataeval_flow._embeddings as embeddings_module

    monkeypatch.setattr(embeddings_module, "_unseeded_noted", set())
    with use_seed(None), caplog.at_level(logging.WARNING, logger="dataeval_flow._embeddings"):
        _partly_warm(f"note-{uuid.uuid4()}")
    notes = [record.getMessage() for record in caplog.records if "`seed:`" in record.getMessage()]
    assert len(notes) == 1
    assert "'test.recording'" in notes[0]


def _spy_keys(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Record the config key every embeddings and clusters lookup in `DatasetCache` is given."""
    keys: dict[str, list[str]] = {"embeddings": [], "clusters": []}
    load_embeddings, load_clusters = (
        DatasetCache.load_or_compute_embeddings,
        DatasetCache.load_or_compute_cluster_result,
    )

    def embeddings(self: DatasetCache, sel_key: str, config_json: str, *args: Any, **kwargs: Any) -> Any:
        keys["embeddings"].append(config_json)
        return load_embeddings(self, sel_key, config_json, *args, **kwargs)

    def clusters(self: DatasetCache, sel_key: str, config_json: str, *args: Any, **kwargs: Any) -> Any:
        keys["clusters"].append(config_json)
        return load_clusters(self, sel_key, config_json, *args, **kwargs)

    monkeypatch.setattr(DatasetCache, "load_or_compute_embeddings", embeddings)
    monkeypatch.setattr(DatasetCache, "load_or_compute_cluster_result", clusters)
    return keys


def _embed_and_cluster() -> None:
    with shared_extractor_scope():
        task = _contexts("reference", "incoming", cache_key=f"clusters-{uuid.uuid4()}")
        task.embeddings("reference")
        task.clusters("incoming", algorithm="kmeans", n_clusters=2)


def test_clusters_are_keyed_by_the_fit_their_embeddings_are(
    recording: list[_Recording],  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = _spy_keys(monkeypatch)
    with use_seed(0):
        _embed_and_cluster()

    reference_key, incoming_key = keys["embeddings"]
    assert "|fitted_on=" in reference_key
    assert reference_key.endswith("|seed=0")
    assert incoming_key == reference_key
    assert keys["clusters"] == [incoming_key]


def test_unseeded_stateful_results_never_reach_the_cache(
    recording: list[_Recording],  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = _spy_keys(monkeypatch)
    with use_seed(None):
        _embed_and_cluster()
    assert keys == {"embeddings": [], "clusters": []}


def test_a_fit_under_another_global_batch_size_is_another_fit(
    recording: list[_Recording],
    computed: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no batch size of its own, the extractor fits on a first batch of DataEval's global size."""
    keys = _spy_keys(monkeypatch)
    key = f"batch-{uuid.uuid4()}"
    with use_seed(0):
        with use_batch_size(8), shared_extractor_scope():
            _contexts("reference", cache_key=key, batch_size=None).embeddings("reference")
        with use_batch_size(4), shared_extractor_scope():
            _contexts("reference", cache_key=key, batch_size=None).embeddings("reference")

    first, second = keys["embeddings"]
    assert first.endswith("|batch=8|seed=0")
    assert second.endswith("|batch=4|seed=0")
    assert _sources(computed) == ["reference", "reference"]
    assert [len(extractor.fitted_on) for extractor in recording if extractor.fitted_on is not None] == [8, 4]


def test_unscoped_clusters_are_keyed_at_the_batch_size_their_embeddings_were_made_at(
    recording: list[_Recording],  # noqa: ARG001
    computed: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside a task a context may embed at another batch size than its extractor config names; its clusters follow."""
    import dataeval_flow._cache as cache_module

    clustered: list[int] = []
    compute = cache_module._do_compute_clusters

    def spy(embeddings: Any, algorithm: Any, n_clusters: Any) -> Any:
        clustered.append(len(embeddings))
        return compute(embeddings, algorithm, n_clusters)

    monkeypatch.setattr(cache_module, "_do_compute_clusters", spy)
    key = f"unscoped-{uuid.uuid4()}"

    def context(batch_size: int) -> WorkflowContext:
        return WorkflowContext(
            dataset_contexts={
                "reference": DatasetContext(
                    name="reference",
                    dataset=REFERENCE,
                    extractor=RecordingConfig(name="rec", batch_size=12),
                    batch_size=batch_size,
                    cache=DatasetCache.get_or_create(cache_dir=None, name="reference", cache_key=key),
                )
            }
        )

    with use_seed(0):
        context(8).clusters("reference", algorithm="kmeans", n_clusters=2)
        context(4).clusters("reference", algorithm="kmeans", n_clusters=2)
    assert len(computed) == 2
    assert len(clustered) == 2


def _cleaning(seed: int | None) -> PipelineConfig:
    """Two cleaning tasks on one source, clustering it twice each: once for outliers, once for duplicates."""
    clean = DataCleaningConfig(
        name="clean",
        outlier_method="zscore",
        outlier_flags=["dimension"],
        outlier_cluster_threshold=2.0,
        outlier_cluster_algorithm="kmeans",
        outlier_n_clusters=2,
        duplicate_cluster_sensitivity=1.0,
        duplicate_cluster_algorithm="kmeans",
        duplicate_n_clusters=3,
    )
    return PipelineConfig(
        seed=seed,
        datasets=[DatasetProtocolConfig(name="reference", dataset=REFERENCE)],
        sources=[SourceConfig(name="reference", dataset="reference")],
        extractors=[RecordingConfig(name="rec", batch_size=BATCH)],
        workflows=[clean],
        tasks=[TaskConfig(name=f"clean_{i}", workflow="clean", sources="reference", extractor="rec") for i in (1, 2)],
    )


@pytest.mark.parametrize(("seed", "clusterings"), [(None, 4), (0, 2)])
def test_cleaning_clusters_follow_the_fit_of_a_stateful_extractor(
    recording: list[_Recording],  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
    seed: int | None,
    clusterings: int,
) -> None:
    """Unseeded, each task clusters its own fit; seeded, the second task is served the first's clusters."""
    import dataeval_flow._cache as cache_module

    computed: list[tuple[str, int | None]] = []
    compute = cache_module._do_compute_clusters

    def spy(embeddings: Any, algorithm: Any, n_clusters: Any) -> Any:
        computed.append((algorithm, n_clusters))
        return compute(embeddings, algorithm, n_clusters)

    monkeypatch.setattr(cache_module, "_do_compute_clusters", spy)
    with use_seed(None):
        results = run_tasks(_cleaning(seed))
    assert all(result.success for result in results.values()), [result.errors for result in results.values()]
    assert len(computed) == clusterings
    assert set(computed) == {("kmeans", 2), ("kmeans", 3)}


@pytest.mark.parametrize("seed", [0, None])
def test_a_stateless_extractor_keys_as_it_always_has(monkeypatch: pytest.MonkeyPatch, seed: int | None) -> None:
    seen: list[str] = []
    load = DatasetCache.load_or_compute_embeddings

    def spy(self: DatasetCache, sel_key: str, config_json: str, *args: Any, **kwargs: Any) -> Any:
        seen.append(config_json)
        return load(self, sel_key, config_json, *args, **kwargs)

    monkeypatch.setattr(DatasetCache, "load_or_compute_embeddings", spy)
    context = WorkflowContext(
        dataset_contexts={
            "src": DatasetContext(
                name="src",
                dataset=REFERENCE,
                extractor=FlattenExtractorConfig(name="flat", batch_size=8),
                batch_size=8,
                cache=DatasetCache.get_or_create(cache_dir=None, name="src", cache_key=f"flat-{uuid.uuid4()}"),
            )
        }
    )
    with use_seed(seed), shared_extractor_scope():
        context.embeddings("src")
    assert seen == ['{"name":"flat","preprocessor":null,"batch_size":8,"model":"flatten"}']
