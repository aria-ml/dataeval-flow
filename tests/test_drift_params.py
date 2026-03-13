"""Tests for drift monitoring workflow parameters."""

import pytest
from pydantic import ValidationError

from dataeval_app.workflows.drift.params import (
    ChunkingConfig,
    DriftDetectorDomainClassifier,
    DriftDetectorKNeighbors,
    DriftDetectorMMD,
    DriftDetectorUnivariate,
    DriftHealthThresholds,
    DriftMonitoringParameters,
    UpdateStrategyConfig,
)

# ---------------------------------------------------------------------------
# DriftDetectorUnivariate
# ---------------------------------------------------------------------------


class TestDriftDetectorUnivariate:
    def test_defaults(self):
        cfg = DriftDetectorUnivariate()
        assert cfg.method == "univariate"
        assert cfg.test == "ks"
        assert cfg.p_val == 0.05
        assert cfg.correction == "bonferroni"
        assert cfg.alternative == "two-sided"
        assert cfg.n_features is None

    @pytest.mark.parametrize("test", ["ks", "cvm", "mwu", "anderson", "bws"])
    def test_valid_test_methods(self, test: str):
        cfg = DriftDetectorUnivariate(test=test)  # type: ignore
        assert cfg.test == test

    def test_invalid_test_method(self):
        with pytest.raises(ValidationError, match="test"):
            DriftDetectorUnivariate(test="invalid")  # type: ignore

    def test_p_val_bounds(self):
        with pytest.raises(ValidationError):
            DriftDetectorUnivariate(p_val=0.0)
        with pytest.raises(ValidationError):
            DriftDetectorUnivariate(p_val=1.0)

    def test_n_features_positive(self):
        cfg = DriftDetectorUnivariate(n_features=10)
        assert cfg.n_features == 10
        with pytest.raises(ValidationError):
            DriftDetectorUnivariate(n_features=0)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs"):
            DriftDetectorUnivariate(k=10)  # type: ignore

    def test_extra_fields_from_other_detector_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs"):
            DriftDetectorUnivariate(n_permutations=100)  # type: ignore


# ---------------------------------------------------------------------------
# DriftDetectorMMD
# ---------------------------------------------------------------------------


class TestDriftDetectorMMD:
    def test_defaults(self):
        cfg = DriftDetectorMMD()
        assert cfg.method == "mmd"
        assert cfg.p_val == 0.05
        assert cfg.n_permutations == 100
        assert cfg.device is None

    def test_custom_values(self):
        cfg = DriftDetectorMMD(p_val=0.01, n_permutations=200, device="cuda:0")
        assert cfg.p_val == 0.01
        assert cfg.n_permutations == 200
        assert cfg.device == "cuda:0"

    def test_n_permutations_positive(self):
        with pytest.raises(ValidationError):
            DriftDetectorMMD(n_permutations=0)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs"):
            DriftDetectorMMD(test="ks")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# DriftDetectorDomainClassifier
# ---------------------------------------------------------------------------


class TestDriftDetectorDomainClassifier:
    def test_defaults(self):
        cfg = DriftDetectorDomainClassifier()
        assert cfg.method == "domain_classifier"
        assert cfg.n_folds == 5
        assert cfg.threshold == 0.55

    def test_n_folds_minimum(self):
        cfg = DriftDetectorDomainClassifier(n_folds=2)
        assert cfg.n_folds == 2
        with pytest.raises(ValidationError):
            DriftDetectorDomainClassifier(n_folds=1)

    def test_threshold_bounds(self):
        with pytest.raises(ValidationError):
            DriftDetectorDomainClassifier(threshold=0.5)  # must be > 0.5
        with pytest.raises(ValidationError):
            DriftDetectorDomainClassifier(threshold=1.1)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs"):
            DriftDetectorDomainClassifier(k=10)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# DriftDetectorKNeighbors
# ---------------------------------------------------------------------------


class TestDriftDetectorKNeighbors:
    def test_defaults(self):
        cfg = DriftDetectorKNeighbors()
        assert cfg.method == "kneighbors"
        assert cfg.k == 10
        assert cfg.distance_metric == "euclidean"
        assert cfg.p_val == 0.05

    def test_custom_values(self):
        cfg = DriftDetectorKNeighbors(k=5, distance_metric="cosine", p_val=0.01)
        assert cfg.k == 5
        assert cfg.distance_metric == "cosine"
        assert cfg.p_val == 0.01

    def test_k_positive(self):
        with pytest.raises(ValidationError):
            DriftDetectorKNeighbors(k=0)

    def test_invalid_metric(self):
        with pytest.raises(ValidationError, match="distance_metric"):
            DriftDetectorKNeighbors(distance_metric="manhattan")  # type: ignore[arg-type]

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs"):
            DriftDetectorKNeighbors(n_folds=5)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Discriminated union dispatch
# ---------------------------------------------------------------------------


class TestDriftDetectorConfigUnion:
    def test_univariate_dispatched(self):
        params = DriftMonitoringParameters.model_validate({"detectors": [{"method": "univariate", "test": "cvm"}]})
        assert isinstance(params.detectors[0], DriftDetectorUnivariate)
        assert params.detectors[0].test == "cvm"

    def test_mmd_dispatched(self):
        params = DriftMonitoringParameters.model_validate({"detectors": [{"method": "mmd", "n_permutations": 50}]})
        assert isinstance(params.detectors[0], DriftDetectorMMD)
        assert params.detectors[0].n_permutations == 50

    def test_domain_classifier_dispatched(self):
        params = DriftMonitoringParameters.model_validate(
            {"detectors": [{"method": "domain_classifier", "threshold": 0.6}]}
        )
        assert isinstance(params.detectors[0], DriftDetectorDomainClassifier)
        assert params.detectors[0].threshold == 0.6

    def test_kneighbors_dispatched(self):
        params = DriftMonitoringParameters.model_validate({"detectors": [{"method": "kneighbors", "k": 20}]})
        assert isinstance(params.detectors[0], DriftDetectorKNeighbors)
        assert params.detectors[0].k == 20

    def test_multiple_detectors(self):
        params = DriftMonitoringParameters.model_validate(
            {
                "detectors": [
                    {"method": "univariate"},
                    {"method": "mmd"},
                    {"method": "domain_classifier"},
                    {"method": "kneighbors"},
                ]
            }
        )
        assert len(params.detectors) == 4
        types = [type(d) for d in params.detectors]
        assert types == [
            DriftDetectorUnivariate,
            DriftDetectorMMD,
            DriftDetectorDomainClassifier,
            DriftDetectorKNeighbors,
        ]

    def test_invalid_method_rejected(self):
        with pytest.raises(ValidationError, match="method"):
            DriftMonitoringParameters.model_validate({"detectors": [{"method": "invalid_method"}]})

    def test_wrong_params_for_method_rejected(self):
        with pytest.raises(ValidationError):
            DriftMonitoringParameters.model_validate({"detectors": [{"method": "univariate", "k": 10}]})

    def test_empty_detectors_rejected(self):
        with pytest.raises(ValidationError, match="detectors"):
            DriftMonitoringParameters.model_validate({"detectors": []})


# ---------------------------------------------------------------------------
# ChunkingConfig
# ---------------------------------------------------------------------------


class TestChunkingConfig:
    def test_enabled_with_chunk_size(self):
        cfg = ChunkingConfig(chunk_size=50)
        assert cfg.chunk_size == 50

    def test_enabled_with_chunk_count(self):
        cfg = ChunkingConfig(chunk_count=10)
        assert cfg.chunk_count == 10

    def test_enabled_without_size_or_count_fails(self):
        with pytest.raises(ValidationError, match="chunk_size or chunk_count must be set"):
            ChunkingConfig()

    def test_both_size_and_count_fails(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            ChunkingConfig(chunk_size=50, chunk_count=10)

    def test_requires_size_or_count(self):
        with pytest.raises(ValidationError, match="chunk_size or chunk_count must be set"):
            ChunkingConfig()

    def test_chunk_size_positive(self):
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=0)

    def test_chunk_count_positive(self):
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_count=0)

    @pytest.mark.parametrize("incomplete", ["keep", "drop", "append"])
    def test_valid_incomplete_values(self, incomplete: str):
        cfg = ChunkingConfig(chunk_size=10, incomplete=incomplete)  # type: ignore[arg-type]
        assert cfg.incomplete == incomplete


# ---------------------------------------------------------------------------
# UpdateStrategyConfig
# ---------------------------------------------------------------------------


class TestUpdateStrategyConfig:
    def test_last_seen(self):
        cfg = UpdateStrategyConfig(type="last_seen", n=100)
        assert cfg.type == "last_seen"
        assert cfg.n == 100

    def test_reservoir_sampling(self):
        cfg = UpdateStrategyConfig(type="reservoir_sampling", n=500)
        assert cfg.type == "reservoir_sampling"

    def test_n_positive(self):
        with pytest.raises(ValidationError):
            UpdateStrategyConfig(type="last_seen", n=0)

    def test_invalid_type(self):
        with pytest.raises(ValidationError, match="type"):
            UpdateStrategyConfig(type="invalid", n=100)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DriftHealthThresholds
# ---------------------------------------------------------------------------


class TestDriftHealthThresholds:
    def test_defaults(self):
        t = DriftHealthThresholds()
        assert t.any_drift_is_warning is True
        assert t.chunk_drift_pct_warning == 10.0
        assert t.consecutive_chunks_warning == 3
        assert t.classwise_any_drift_is_warning is True

    def test_custom_values(self):
        t = DriftHealthThresholds(
            any_drift_is_warning=False,
            chunk_drift_pct_warning=25.0,
            consecutive_chunks_warning=5,
            classwise_any_drift_is_warning=False,
        )
        assert t.any_drift_is_warning is False
        assert t.chunk_drift_pct_warning == 25.0

    def test_chunk_pct_bounds(self):
        with pytest.raises(ValidationError):
            DriftHealthThresholds(chunk_drift_pct_warning=-1.0)
        with pytest.raises(ValidationError):
            DriftHealthThresholds(chunk_drift_pct_warning=101.0)

    def test_consecutive_minimum(self):
        with pytest.raises(ValidationError):
            DriftHealthThresholds(consecutive_chunks_warning=0)


# ---------------------------------------------------------------------------
# DriftMonitoringParameters (top-level)
# ---------------------------------------------------------------------------


class TestDriftMonitoringParameters:
    def test_minimal_valid(self):
        params = DriftMonitoringParameters.model_validate({"detectors": [{"method": "mmd"}]})
        assert len(params.detectors) == 1
        assert params.detectors[0].chunking is None
        assert params.classwise is False
        assert params.update_strategy is None
        assert params.mode == "advisory"

    def test_full_config(self):
        params = DriftMonitoringParameters.model_validate(
            {
                "mode": "preparatory",
                "detectors": [
                    {"method": "univariate", "test": "ks", "chunking": {"enabled": True, "chunk_size": 100}},
                    {"method": "mmd", "n_permutations": 200},
                ],
                "classwise": True,
                "update_strategy": {"type": "last_seen", "n": 500},
                "health_thresholds": {
                    "any_drift_is_warning": False,
                    "chunk_drift_pct_warning": 20.0,
                },
            }
        )
        assert params.mode == "preparatory"
        assert len(params.detectors) == 2
        assert params.detectors[0].chunking is not None
        assert params.detectors[0].chunking.chunk_size == 100
        assert params.detectors[1].chunking is None
        assert params.classwise is True
        assert params.update_strategy is not None
        assert params.update_strategy.type == "last_seen"
        assert params.health_thresholds.any_drift_is_warning is False

    def test_no_detectors_fails(self):
        with pytest.raises(ValidationError):
            DriftMonitoringParameters.model_validate({"detectors": []})

    def test_defaults_for_optional_fields(self):
        params = DriftMonitoringParameters.model_validate({"detectors": [{"method": "kneighbors"}]})
        assert isinstance(params.health_thresholds, DriftHealthThresholds)
