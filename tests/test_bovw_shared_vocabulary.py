"""A stateful extractor must describe every source in one vocabulary.

BoVW fits its visual vocabulary from the data it first sees. Building one extractor per
source therefore gives each source its own codebook, and histograms drawn from different
codebooks are not comparable -- the numbers that come back are confident and wrong rather
than an error, which is the failure this pins.
"""

import numpy as np

from dataeval_flow.config import (
    BoVWExtractorConfig,
    DatasetProtocolConfig,
    DriftMonitoringTaskConfig,
    DriftMonitoringWorkflowConfig,
    PipelineConfig,
    SourceConfig,
)
from dataeval_flow.workflow import run_task
from dataeval_flow.workflows.drift.params import DriftDetectorMMD


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
        self.metadata = {"id": f"images_{seed}", "index2label": {0: "a", 1: "b"}}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, i: int):
        target = np.zeros(2, dtype=np.float32)
        target[i % 2] = 1.0
        return self._images[i], target, {"id": str(i)}


def _drift_between(reference, incoming, tmp_path):
    task = DriftMonitoringTaskConfig(name="t", workflow="w", sources=["ref", "inc"], extractor="bovw")
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
        workflows=[DriftMonitoringWorkflowConfig(name="w", detectors=[DriftDetectorMMD()])],
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

    drifted = {name: detector for name, detector in result.data.raw.detectors.items() if detector.get("drifted")}
    assert not drifted, f"identical data reported drift: {drifted}"


class TestTheScopeSharesOnlyWhatItMust:
    """The sharing is narrow on purpose: one instance per identity, stateful only."""

    def _bovw(self):
        return BoVWExtractorConfig(name="bovw", vocab_size=256, batch_size=16)

    def test_one_instance_serves_a_scope(self):
        from dataeval_flow.embeddings import build_extractor, shared_extractor_scope

        config = self._bovw()
        with shared_extractor_scope():
            assert build_extractor(config) is build_extractor(config)

    def test_instances_do_not_leak_between_scopes(self):
        """A later task must not inherit the vocabulary an earlier one fitted."""
        from dataeval_flow.embeddings import build_extractor, shared_extractor_scope

        config = self._bovw()
        with shared_extractor_scope():
            first = build_extractor(config)
        with shared_extractor_scope():
            assert build_extractor(config) is not first

    def test_nothing_is_shared_without_a_scope(self):
        from dataeval_flow.embeddings import build_extractor

        config = self._bovw()
        assert build_extractor(config) is not build_extractor(config)

    def test_a_stateless_extractor_is_never_shared(self):
        """Flatten carries no state, so reusing one buys nothing and hides a lifetime."""
        from dataeval_flow.config import FlattenExtractorConfig
        from dataeval_flow.embeddings import build_extractor, shared_extractor_scope

        config = FlattenExtractorConfig(name="flatten", batch_size=8)
        with shared_extractor_scope():
            assert build_extractor(config) is not build_extractor(config)

    def test_the_first_selection_to_ask_is_recorded_as_the_fitter(self):
        """The cache key needs to name the vocabulary's origin, not just the config."""
        from dataeval_flow.embeddings import build_extractor, fitting_source, shared_extractor_scope

        config = self._bovw()
        key = f"{config.model_dump_json()}|None"
        with shared_extractor_scope():
            build_extractor(config)
            assert fitting_source(key, "reference") == "reference"
            # The incoming source is described in the reference's vocabulary, so it
            # shares the reference's cache identity rather than claiming its own.
            assert fitting_source(key, "incoming") == "reference"

    def test_no_fitter_is_claimed_outside_a_scope(self):
        from dataeval_flow.embeddings import fitting_source

        assert fitting_source("anything", "reference") is None
