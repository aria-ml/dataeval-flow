"""A stateful extractor must describe every source in one vocabulary.

BoVW fits its visual vocabulary from the data it first sees. Building one extractor per
source therefore gives each source its own codebook, and histograms drawn from different
codebooks are not comparable -- the numbers that come back are wrong. That is the failure
this pins.
"""

import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from dataeval.config import use_seed

from dataeval_flow import PipelineConfig, run_task, run_tasks
from dataeval_flow.config import DatasetProtocolConfig, SourceConfig, TaskConfig
from dataeval_flow.config.extractors import BoVWExtractorConfig
from dataeval_flow.workflows.drift_monitoring import DriftDetectorMMD, DriftMonitoringConfig

if TYPE_CHECKING:
    from dataeval.protocols import DatasetMetadata


class _Images:
    """A tiny MAITE-shaped image classification dataset with structure SIFT can key on."""

    def __init__(self, seed: int, count: int = 64) -> None:
        rng = np.random.default_rng(seed)
        self._images = []
        for _ in range(count):
            base = rng.integers(0, 255, (96, 96, 1), dtype=np.uint8).repeat(3, axis=2)
            # Hard edges give SIFT keypoints to find; pure noise yields few. A grid of
            # them supplies enough descriptors to fill a 256-word vocabulary.
            for y in range(8, 96, 24):
                for x in range(8, 96, 24):
                    base[y : y + 12, x : x + 12] = 255 - base[y : y + 12, x : x + 12]
            self._images.append(np.transpose(base, (2, 0, 1)))
        self.metadata: DatasetMetadata = {"id": f"images_{seed}", "index2label": {0: "a", 1: "b"}}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, i: int):
        target = np.zeros(2, dtype=np.float32)
        target[i % 2] = 1.0
        return self._images[i], target, {"id": str(i)}


def _drift_between(reference, incoming, tmp_path):
    task = TaskConfig(name="t", workflow="w", sources=["ref", "inc"], extractor="bovw")
    config = PipelineConfig(
        datasets=[
            DatasetProtocolConfig(name="reference", format="maite", dataset=reference),
            DatasetProtocolConfig(name="incoming", format="maite", dataset=incoming),
        ],
        sources=[
            SourceConfig(name="ref", dataset="reference"),
            SourceConfig(name="inc", dataset="incoming"),
        ],
        extractors=[BoVWExtractorConfig(name="bovw", vocab_size=256, batch_size=16)],
        workflows=[DriftMonitoringConfig(name="w", detectors=[DriftDetectorMMD()])],
        tasks=[task],
    )
    result = run_task(task, config, cache_dir=tmp_path)
    assert result.success, result.errors
    return result


def test_identical_sources_do_not_drift(tmp_path):
    """The same images on both sides cannot have drifted from themselves.

    They only appear to when each side is described in its own vocabulary.
    """
    images = _Images(seed=0)
    result = _drift_between(images, images, tmp_path)

    drifted = {name: detector for name, detector in result.output.raw.detectors.items() if detector.get("drifted")}
    assert not drifted, f"identical data reported drift: {drifted}"


def _task_histograms(cache_key: str, *, sources: tuple[str, ...], identical: bool = False) -> list[np.ndarray]:
    """One task's histograms, one array per source in the order given.

    ``identical`` gives both sources the same images, which cannot differ unless they are
    described in two vocabularies.
    """
    from dataeval_flow._cache import DatasetCache
    from dataeval_flow._embeddings import shared_extractor_scope
    from dataeval_flow.workflows import DatasetContext, WorkflowContext

    data = {"reference": _Images(seed=0, count=32), "incoming": _Images(seed=0 if identical else 1, count=32)}
    context = WorkflowContext(
        dataset_contexts={
            name: DatasetContext(
                name=name,
                dataset=data[name],
                extractor=BoVWExtractorConfig(vocab_size=256, batch_size=16),
                batch_size=16,
                cache=DatasetCache.get_or_create(cache_dir=None, name=name, cache_key=cache_key),
            )
            for name in sources
        }
    )
    with shared_extractor_scope():
        return [context.embeddings(name) for name in sources]


@pytest.fixture
def computed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The selections whose embeddings are computed rather than served, in order."""
    import dataeval_flow._cache as cache_module

    selections: list[str] = []
    compute = cache_module._do_compute_embeddings

    def spy(*args: Any, **kwargs: Any) -> Any:
        selections.append(str(kwargs.get("selection")).split("/")[0].rsplit("_", 1)[0])
        return compute(*args, **kwargs)

    monkeypatch.setattr(cache_module, "_do_compute_embeddings", spy)
    return selections


def test_a_seeded_partly_warm_cache_describes_the_incoming_source_as_a_cold_task_does(computed: list[str]):
    """Under a seed the reference's histograms are served, and its vocabulary is fitted again for the incoming source.

    Refitted on the reference's data under the same seed, it is the vocabulary a task computing both sources fits.
    """
    with use_seed(0):
        _, cold = _task_histograms(f"cold-{uuid.uuid4()}", sources=("reference", "incoming"))
        warm_key = f"warm-{uuid.uuid4()}"
        _task_histograms(warm_key, sources=("reference",))
        computed.clear()
        _, warm = _task_histograms(warm_key, sources=("reference", "incoming"))
    assert computed == ["incoming"]
    np.testing.assert_array_equal(warm, cold)


def test_an_unseeded_partly_warm_cache_never_splits_the_vocabulary(computed: list[str]):
    """The default, unseeded path: identical sources stay identical when a previous task embedded the reference.

    A refit without a seed learns another vocabulary, so the reference is computed again beside the incoming source.
    """
    with use_seed(None):
        warm_key = f"warm-{uuid.uuid4()}"
        _task_histograms(warm_key, sources=("reference",), identical=True)
        computed.clear()
        reference, incoming = _task_histograms(warm_key, sources=("reference", "incoming"), identical=True)
    assert computed == ["reference", "incoming"]
    np.testing.assert_array_equal(incoming, reference)


def test_unseeded_drift_tasks_sharing_a_reference_see_no_drift_in_identical_data():
    """The CLI's path, seedless: two drift tasks monitor two batches against one reference, all three the same images.

    The second task's reference came from the first task's cache, refitted into another vocabulary.
    """
    names = ["reference", "incoming_a", "incoming_b"]
    data = {name: _Images(seed=0, count=32) for name in names}
    config = PipelineConfig(
        datasets=[DatasetProtocolConfig(name=name, dataset=data[name]) for name in names],
        sources=[SourceConfig(name=name, dataset=name) for name in names],
        extractors=[BoVWExtractorConfig(name="bovw", vocab_size=256, batch_size=16)],
        workflows=[DriftMonitoringConfig(name="drift", detectors=[DriftDetectorMMD()])],
        tasks=[
            TaskConfig(
                name=f"drift_{batch}", workflow="drift", sources=["reference", f"incoming_{batch}"], extractor="bovw"
            )
            for batch in "ab"
        ],
    )
    with use_seed(None):
        results = run_tasks(config)
    for name, result in results.items():
        assert result.success, result.errors
        drifted = {detector: raw.get("drifted") for detector, raw in result.output.raw.detectors.items()}
        assert not any(drifted.values()), f"{name} reported drift between identical images: {drifted}"


class TestTheScopeSharesOnlyWhatItMust:
    """The sharing is narrow on purpose: one instance per identity, stateful only."""

    def _bovw(self):
        return BoVWExtractorConfig(name="bovw", vocab_size=256, batch_size=16)

    def test_one_instance_serves_a_scope(self):
        from dataeval_flow._embeddings import build_extractor, shared_extractor_scope

        config = self._bovw()
        with shared_extractor_scope():
            assert build_extractor(config) is build_extractor(config)

    def test_instances_do_not_leak_between_scopes(self):
        """A later task must not inherit the vocabulary an earlier one fitted."""
        from dataeval_flow._embeddings import build_extractor, shared_extractor_scope

        config = self._bovw()
        with shared_extractor_scope():
            first = build_extractor(config)
        with shared_extractor_scope():
            assert build_extractor(config) is not first

    def test_nothing_is_shared_without_a_scope(self):
        from dataeval_flow._embeddings import build_extractor

        config = self._bovw()
        assert build_extractor(config) is not build_extractor(config)

    def test_a_stateless_extractor_is_never_shared(self):
        """Flatten carries no state, so reusing one instance gains nothing."""
        from dataeval_flow._embeddings import build_extractor, shared_extractor_scope
        from dataeval_flow.config.extractors import FlattenExtractorConfig

        config = FlattenExtractorConfig(name="flatten", batch_size=8)
        with shared_extractor_scope():
            assert build_extractor(config) is not build_extractor(config)

    def test_the_first_selection_to_ask_is_claimed_as_the_fitter(self):
        """The cache key needs to name the vocabulary's origin, not just the config.

        Claimed when embeddings are first asked for, before anything is built.
        """
        from dataeval_flow._embeddings import claim_fitter, fit_identity, shared_extractor_scope

        config = self._bovw()
        with use_seed(0), shared_extractor_scope():
            assert claim_fitter(config, None, "reference", MagicMock(), 16) == "reference"
            # The incoming source is described in the reference's vocabulary, so it
            # shares the reference's cache identity rather than claiming its own.
            assert claim_fitter(config, None, "incoming", MagicMock(), 16) == "reference"
            assert fit_identity(config, None, "incoming", 16) == "fitted_on=reference|batch=16|seed=0"

    def test_outside_a_scope_a_selection_fits_itself(self):
        """Nothing is shared, so an entry can only serve the selection it was fitted on."""
        from dataeval_flow._embeddings import claim_fitter, fit_identity

        config = self._bovw()
        with use_seed(0):
            assert claim_fitter(config, None, "reference", MagicMock(), 16) == "reference"
            assert claim_fitter(config, None, "incoming", MagicMock(), 16) == "incoming"
            assert fit_identity(config, None, "incoming", 16) == "fitted_on=incoming|batch=16|seed=0"

    def test_no_fit_is_named_without_a_seed(self):
        """An unseeded refit learns another vocabulary, so no key could name the fit."""
        from dataeval_flow._embeddings import claim_fitter, fit_identity, shared_extractor_scope

        config = self._bovw()
        with use_seed(None), shared_extractor_scope():
            claim_fitter(config, None, "reference", MagicMock(), 16)
            assert fit_identity(config, None, "reference", 16) is None

    def test_the_global_batch_size_names_the_fit_when_the_config_sets_none(self):
        """The batch size picks the images a stateful extractor fits on, so it is part of the fit."""
        from dataeval.config import use_batch_size

        from dataeval_flow._embeddings import fit_identity

        config = BoVWExtractorConfig(vocab_size=256)
        with use_seed(0):
            with use_batch_size(8):
                small = fit_identity(config, None, "reference", None)
            with use_batch_size(16):
                large = fit_identity(config, None, "reference", None)
        assert small == "fitted_on=reference|batch=8|seed=0"
        assert large == "fitted_on=reference|batch=16|seed=0"

    def test_a_stateless_extractor_names_no_other_fitter(self):
        from dataeval_flow._embeddings import claim_fitter, shared_extractor_scope
        from dataeval_flow.config.extractors import FlattenExtractorConfig

        config = FlattenExtractorConfig(batch_size=8)
        with shared_extractor_scope():
            claim_fitter(config, None, "reference", MagicMock(), 8)
            assert claim_fitter(config, None, "incoming", MagicMock(), 8) == "incoming"
