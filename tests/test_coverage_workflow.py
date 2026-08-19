"""Tests for the data coverage workflow."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from dataeval.protocols import DatasetMetadata, DatumMetadata
from pydantic import BaseModel, ValidationError

from dataeval_flow.workflow import DatasetContext, WorkflowContext
from dataeval_flow.workflow._text_report import _render_detail_section
from dataeval_flow.workflow.base import MetadataConfigMixin, WorkflowParametersBase
from dataeval_flow.workflows._common import serialize_coverage as _serialize_coverage
from dataeval_flow.workflows.coverage.outputs import (
    ClassCoverageRow,
    ClassMetadataGap,
    CompletenessAssessment,
    CoverageAssessment,
    DarkBranch,
    DataCoverageMetadata,
    DataCoverageOutputs,
    DataCoverageRawOutputs,
    DataCoverageReport,
    LabelConformance,
    LabelDistributionResult,
    LabelSpaceCoverage,
    MetadataDistributionResult,
    MetadataGapResult,
    OntologyAssessment,
    OntologyStructure,
    RepresentationRow,
    RepresentationViolation,
    is_coverage_result,
)
from dataeval_flow.workflows.coverage.params import DataCoverageHealthThresholds, DataCoverageParameters
from dataeval_flow.workflows.coverage.report import build_findings
from dataeval_flow.workflows.coverage.workflow import (
    DataCoverageWorkflow,
    _crop_view,
    _is_object_detection,
    _mi_from_balance,
    _run_completeness,
    _run_coverage,
    _run_gap_analysis,
    _to_serializable,
)

pytestmark = pytest.mark.required

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(**overrides: Any) -> DataCoverageParameters:
    defaults: dict[str, Any] = {}
    defaults.update(overrides)
    return DataCoverageParameters(**defaults)


def _make_dataset(n: int = 100) -> MagicMock:
    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=n)
    ds.metadata = {"index2label": {0: "cat", 1: "dog", 2: "bird"}}
    return ds


def _make_context(dataset: MagicMock | None = None, n: int = 100) -> WorkflowContext:
    """Build a single-dataset ``WorkflowContext`` with no extractor or cache."""
    return WorkflowContext(dataset_contexts={"ds": DatasetContext(name="ds", dataset=dataset or _make_dataset(n))})


def _make_metadata(n: int = 100, num_classes: int = 3) -> MagicMock:
    meta = MagicMock()
    meta.class_labels = np.array([i % num_classes for i in range(n)], dtype=np.intp)
    meta.item_indices = np.arange(n)
    meta.factor_names = ["brightness", "contrast"]
    meta.index2label = {i: name for i, name in enumerate(["cat", "dog", "bird"][:num_classes])}
    meta.factor_data = np.column_stack(
        [
            np.random.default_rng(42).integers(0, 5, size=n),
            np.random.default_rng(43).integers(0, 3, size=n),
        ]
    )
    meta.is_discrete = np.array([True, True])
    meta.multi_target = False
    # Set explicitly: dataeval's evaluators refuse a filtered metadata paired with
    # embeddings, and an auto-created MagicMock attribute is truthy, so leaving it
    # unset makes every Coverage call look like it was filtered by where()/having().
    meta.is_filtered = False

    df = pl.DataFrame(
        {
            "brightness": np.random.default_rng(42).integers(0, 5, size=n).tolist(),
            "contrast": np.random.default_rng(43).integers(0, 3, size=n).tolist(),
        }
    )
    meta.dataframe = df
    meta.label_level = "instance"
    meta.view = "instance"
    meta.rows_at.return_value = df

    # Mock factor_info
    factor_info = {}
    for name in ["brightness", "contrast"]:
        info = MagicMock()
        info.factor_type = "discrete"
        info.level = "unit"
        info.is_binned = False
        factor_info[name] = info
    meta.factor_info = factor_info
    meta.dropped_factors = {}

    # Delete metadata attribute so it does not match AnnotatedDataset protocol
    del meta.metadata

    return meta


def _make_label_stats(num_classes: int = 3, n: int = 100) -> dict[str, Any]:
    counts_per_class: dict[int, int] = {}
    for i in range(n):
        cls = i % num_classes
        counts_per_class[cls] = counts_per_class.get(cls, 0) + 1

    return {
        "class_count": num_classes,
        "label_count": n,
        "image_count": n,
        "label_counts_per_class": counts_per_class,
        "image_counts_per_class": counts_per_class,
        "image_indices_per_class": {i: list(range(i, n, num_classes)) for i in range(num_classes)},
        "classes_per_image": [[i % num_classes] for i in range(n)],
        "empty_image_indices": np.array([], dtype=int),
        "empty_image_count": 0,
        "label_counts_per_image": [1] * n,
        "index2label": {0: "cat", 1: "dog", 2: "bird"},
    }


class _Target:
    """Minimal object-detection target (boxes / labels / scores)."""

    def __init__(self, boxes: list[list[float]], labels: list[int]) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float64)
        self.labels = np.asarray(labels, dtype=np.intp)
        self.scores = np.ones((len(self.labels), 2), dtype=np.float32)


class _ODDatumMetadata(DatumMetadata):
    """Per-datum metadata carrying one factor for the metadata analyses."""

    weather: str


class _ODDataset:
    """A real (unmocked) tiny object-detection dataset — two boxes per image."""

    metadata: DatasetMetadata = DatasetMetadata({"id": "od", "index2label": {0: "cat", 1: "dog"}})

    def __init__(self, n_images: int = 6, degenerate: bool = False) -> None:
        rng = np.random.default_rng(0)
        self._items: list[tuple[Any, Any, DatumMetadata]] = []
        for i in range(n_images):
            image = rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)
            boxes: list[list[float]] = [[2, 2, 20, 20], [8, 8, 30, 30]]
            labels = [0, 1]
            if degenerate and i == 0:
                # Zero-area box — DetectionCrops drops it, so crops no longer align
                # 1:1 with the source dataset's labels.
                boxes.append([5.0, 5.0, 5.0, 5.0])
                labels.append(0)
            datum_metadata: _ODDatumMetadata = {"id": i, "weather": "sun" if i % 2 else "rain"}
            self._items.append((image, _Target(boxes, labels), datum_metadata))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[Any, Any, DatumMetadata]:
        return self._items[index]


# ---------------------------------------------------------------------------
# TestDataCoverageParameters
# ---------------------------------------------------------------------------


class TestDataCoverageParameters:
    def test_defaults(self) -> None:
        p = DataCoverageParameters()
        assert p.coverage_method == "adaptive"
        assert p.coverage_percent == 0.01
        assert p.num_observations == 50
        assert p.min_class_samples == 20
        assert p.isotropy_min_samples is None
        assert p.near_duplicate_factor == 0.5
        assert p.run_completeness is True
        assert p.balance is True
        assert p.diversity_method == "simpson"
        assert p.run_gap_analysis is True
        assert p.gap_mi_threshold == 0.1
        assert p.gap_min_representation == 5

    def test_inherits_workflow_params_base(self) -> None:
        assert issubclass(DataCoverageParameters, WorkflowParametersBase)

    def test_inherits_metadata_config_mixin(self) -> None:
        assert issubclass(DataCoverageParameters, MetadataConfigMixin)

    def test_custom_values(self) -> None:
        p = _make_params(
            coverage_method="naive",
            coverage_percent=0.05,
            num_observations=100,
            run_completeness=False,
            balance=False,
            diversity_method="shannon",
            run_gap_analysis=False,
        )
        assert p.coverage_method == "naive"
        assert p.coverage_percent == 0.05
        assert p.num_observations == 100
        assert p.run_completeness is False
        assert p.balance is False
        assert p.diversity_method == "shannon"
        assert p.run_gap_analysis is False

    def test_ontology_expected_rejects_out_of_range_share(self) -> None:
        """ontology_expected values are documented as fractions in [0, 1] — a value
        like 5.0 or -1 would silently produce a nonsense target and, prior to this
        fix, was the most likely trigger for an unguarded crash in analysis."""
        import pytest

        with pytest.raises(ValidationError):
            _make_params(ontology_expected={"cat": 5.0})
        with pytest.raises(ValidationError):
            _make_params(ontology_expected={"cat": -1.0})

    def test_ontology_expected_accepts_valid_share(self) -> None:
        p = _make_params(ontology_expected={"cat": 0.25})
        assert p.ontology_expected == {"cat": 0.25}


class TestDataCoverageHealthThresholds:
    def test_defaults(self) -> None:
        t = DataCoverageHealthThresholds()
        assert t.uncovered_rate == 10.0
        assert t.completeness_score == 0.5
        assert t.class_imbalance_ratio == 5.0
        assert t.gap_count == 3

    def test_custom(self) -> None:
        t = DataCoverageHealthThresholds(
            uncovered_rate=5.0,
            completeness_score=0.7,
            class_imbalance_ratio=3.0,
            gap_count=5,
        )
        assert t.uncovered_rate == 5.0
        assert t.completeness_score == 0.7


# ---------------------------------------------------------------------------
# TestSerializers
# ---------------------------------------------------------------------------


class TestToSerializable:
    def test_numpy_types(self) -> None:
        assert _to_serializable(np.int64(42)) == 42
        assert _to_serializable(np.float64(3.14)) == 3.14
        assert _to_serializable(np.bool_(True)) is True

    def test_ndarray(self) -> None:
        result = _to_serializable(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_nested(self) -> None:
        result = _to_serializable({"a": np.int64(1), "b": [np.float64(2.0)]})
        assert result == {"a": 1, "b": [2.0]}


class TestSerializeCoverage:
    def test_dict_result(self) -> None:
        result = {
            "uncovered_indices": np.array([0, 5, 10]),
            "critical_value_radii": np.array([0.1, 0.2]),
            "coverage_radius": 0.5,
        }
        serialized = _serialize_coverage(result)
        assert serialized["uncovered_indices"] == [0, 5, 10]
        assert serialized["coverage_radius"] == 0.5


# ---------------------------------------------------------------------------
# TestRunCoverage
# ---------------------------------------------------------------------------


class TestRunCoverage:
    def test_per_class_columns_present(self) -> None:
        """scope.Coverage reports dispersion/isotropy/near-duplicates per class."""
        rng = np.random.default_rng(42)
        # Two classes of 60: class 0 broadly spread, class 1 collapsed into a pocket.
        emb = np.vstack([rng.random((60, 8)), rng.random((60, 8)) * 0.01 + 0.5])
        meta = _make_metadata(n=120, num_classes=2)
        meta.class_labels = np.array([0] * 60 + [1] * 60, dtype=np.intp)
        meta.index2label = {0: "wide", 1: "tight"}

        result = _run_coverage(meta, emb, _make_params(coverage_method="adaptive", num_observations=10))

        assert isinstance(result, CoverageAssessment)
        assert result.method == "adaptive"
        assert {row.class_name for row in result.per_class} == {"wide", "tight"}
        by_name = {row.class_name: row for row in result.per_class}
        assert by_name["tight"].dispersion is not None
        assert by_name["wide"].dispersion is not None
        # The collapsed class is the lower-dispersion one, and rows are sorted by it.
        assert by_name["tight"].dispersion < by_name["wide"].dispersion
        assert result.per_class[0].class_name == "tight"

    def test_thin_class_is_unassessable(self) -> None:
        """A class below min_class_samples reports null signals, not an error."""
        rng = np.random.default_rng(7)
        emb = np.vstack([rng.random((60, 8)), rng.random((5, 8))])
        meta = _make_metadata(n=65, num_classes=2)
        meta.class_labels = np.array([0] * 60 + [1] * 5, dtype=np.intp)
        meta.index2label = {0: "plenty", 1: "thin"}

        result = _run_coverage(meta, emb, _make_params(num_observations=10, min_class_samples=20))

        thin = next(row for row in result.per_class if row.class_name == "thin")
        assert thin.assessable is False
        assert thin.dispersion is None
        assert thin.isotropy is None
        assert thin.near_duplicate_fraction is None

    def test_naive_overflow_falls_back_to_adaptive(self) -> None:
        """The naive radius includes gamma(d/2 + 1), which overflows past ~340 dims.

        The real coverage code runs here, so the overflow is genuine.
        """
        rng = np.random.default_rng(42)
        emb = rng.random((400, 512))
        meta = _make_metadata(n=400, num_classes=2)
        meta.class_labels = np.array([i % 2 for i in range(400)], dtype=np.intp)
        meta.index2label = {0: "a", 1: "b"}

        result = _run_coverage(meta, emb, _make_params(coverage_method="naive", num_observations=50))

        assert result.method == "adaptive (naive overflowed)"
        assert result.uncovered_count >= 0

    def test_both_is_rejected(self) -> None:
        """coverage_method='both' no longer exists — scope.Coverage runs one method."""
        import pytest

        with pytest.raises(ValidationError):
            _make_params(coverage_method="both")


# ---------------------------------------------------------------------------
# TestRunCompleteness
# ---------------------------------------------------------------------------


class TestCropView:
    """Object-detection datasets are embedded as detection crops, not whole images."""

    def test_image_classification_passes_through(self) -> None:
        ds = _make_dataset(10)
        meta = _make_metadata(10)
        out_ds, out_meta, key, unit = _crop_view(ds, meta, "sel:all")
        assert out_ds is ds
        assert out_meta is meta
        assert key == "sel:all"
        assert unit == "image"
        assert _is_object_detection(ds) is False

    def test_object_detection_is_wrapped_in_crops(self) -> None:
        from dataeval import Metadata
        from dataeval.data import DetectionCrops

        ds = _ODDataset(n_images=6)
        meta = Metadata(ds)
        crops, crop_meta, key, unit = _crop_view(ds, meta, "sel:all")

        assert _is_object_detection(ds) is True
        assert isinstance(crops, DetectionCrops)
        assert len(crops) == 12  # 6 images x 2 boxes
        assert unit == "detection crop"
        # Crop embeddings must not be served from — or overwrite — the whole-image
        # embeddings cached under the dataset's own selection key.
        assert key != "sel:all"
        # No boxes dropped, so the source labels already align 1:1 with the crops.
        assert crop_meta is meta
        assert len(crop_meta.class_labels) == len(crops)

    def test_dropped_boxes_force_metadata_over_the_crops(self) -> None:
        from dataeval import Metadata
        from dataeval.data import DetectionCrops

        ds = _ODDataset(n_images=6, degenerate=True)
        meta = Metadata(ds)
        crops, crop_meta, _, _ = _crop_view(ds, meta, "sel:all")

        assert isinstance(crops, DetectionCrops)
        assert crops.n_dropped == 1
        assert len(crops) == len(meta.class_labels) - 1
        # The source labels no longer line up, so a Metadata over the view is built.
        assert crop_meta is not meta
        assert len(crop_meta.class_labels) == len(crops)


# ---------------------------------------------------------------------------
# TestRunCompleteness
# ---------------------------------------------------------------------------


class TestRunCompleteness:
    @patch("dataeval.core.completeness")
    def test_basic(self, mock_comp: MagicMock) -> None:
        mock_comp.return_value = {
            "completeness": 0.85,
            "nearest_neighbor_pairs": [(0, 1), (2, 3)],
        }
        emb = np.random.default_rng(42).random((50, 10))
        result = _run_completeness(emb)
        assert isinstance(result, CompletenessAssessment)
        assert result.completeness_score == 0.85
        assert len(result.nearest_neighbor_pairs) == 2


# ---------------------------------------------------------------------------
# TestRunGapAnalysis
# ---------------------------------------------------------------------------


class TestRunGapAnalysis:
    @patch("dataeval_flow.workflows.coverage.workflow._balance_class_to_factor")
    def test_identifies_gaps(self, mock_mi: MagicMock) -> None:
        """Test that gap analysis identifies under-represented class-factor combinations."""
        mock_mi.return_value = {"time_of_day": 0.5, "weather": 0.02}

        n = 90
        # time_of_day=1 appears in 40/90 samples overall, but never for class 2 ("bird"),
        # whose 10 samples therefore expect ~4.4 and have 0.
        time_of_day = [0] * 30 + [1] * 30 + [0] * 10 + [1] * 10 + [0] * 10
        meta = MagicMock()
        meta.class_labels = np.array([0] * 60 + [1] * 20 + [2] * 10, dtype=np.intp)
        meta.factor_names = ["time_of_day", "weather"]
        meta.factor_data = np.column_stack([np.array(time_of_day), np.zeros(n, dtype=int)])
        meta.is_discrete = np.array([True, True])
        meta.multi_target = False

        df = pl.DataFrame({"time_of_day": time_of_day, "weather": [0] * n})
        meta.dataframe = df
        meta.label_level = "instance"
        meta.rows_at.return_value = df

        index2label = {0: "cat", 1: "dog", 2: "bird"}
        params = _make_params(gap_mi_threshold=0.1, gap_min_representation=3)

        result = _run_gap_analysis(meta, index2label, params)

        assert isinstance(result, MetadataGapResult)
        assert "time_of_day" in result.mutual_info_class_to_factor
        # "weather" has MI 0.02 < gap_mi_threshold, so only time_of_day is cross-tabulated.
        assert [(g.class_name, g.factor_name, g.factor_value) for g in result.gaps] == [("bird", "time_of_day", "1")]
        gap = result.gaps[0]
        assert gap.class_count == 0
        assert gap.expected_count == 4.4
        assert gap.deficit == 1.0

    @patch("dataeval_flow.workflows.coverage.workflow._balance_class_to_factor")
    def test_object_detection_uses_target_rows(self, mock_mi: MagicMock) -> None:
        """Gap analysis reads target-level rows, which always align with class_labels.

        For OD datasets ``Metadata.dataframe`` interleaves one image-level row per
        image with one target-level row per detection, so it is longer than
        ``class_labels`` and cannot be masked by class.
        """
        mock_mi.return_value = {"weather": 0.5}

        # 12 images, 30 detections: class 0 gets 20, class 1 gets 10.
        n_images, n_targets = 12, 30
        # weather=1 covers 10/30 detections overall but none of class 1 ("dog").
        weather = [0] * 10 + [1] * 10 + [0] * 10
        meta = MagicMock()
        meta.class_labels = np.array([0] * 20 + [1] * 10, dtype=np.intp)
        meta.factor_names = ["weather"]
        meta.factor_data = np.array(weather).reshape(-1, 1)
        meta.is_discrete = np.array([True])
        meta.multi_target = True

        meta.label_level = "instance"
        meta.rows_at.return_value = pl.DataFrame({"weather": weather})
        # The misaligned source: image rows prepended to target rows.
        meta.dataframe = pl.DataFrame({"weather": [0] * n_images + weather})
        assert len(meta.dataframe) != n_targets

        params = _make_params(gap_mi_threshold=0.1, gap_min_representation=3)
        result = _run_gap_analysis(meta, {0: "cat", 1: "dog"}, params)

        assert [(g.class_name, g.factor_name, g.factor_value) for g in result.gaps] == [("dog", "weather", "1")]
        assert result.gaps[0].class_count == 0
        assert result.gaps[0].expected_count == 3.3

    @patch("dataeval_flow.workflows.coverage.workflow._balance_class_to_factor")
    def test_no_factors(self, mock_mi: MagicMock) -> None:
        meta = MagicMock()
        meta.factor_names = []
        params = _make_params()
        result = _run_gap_analysis(meta, None, params)
        assert result.gaps == []
        mock_mi.assert_not_called()

    @patch("dataeval_flow.workflows.coverage.workflow._balance_class_to_factor")
    def test_precomputed_mi_skips_second_pass(self, mock_mi: MagicMock) -> None:
        """Balance already computed this MI — do not pay for it twice."""
        weather = [0] * 10 + [1] * 10 + [0] * 10
        meta = MagicMock()
        meta.class_labels = np.array([0] * 20 + [1] * 10, dtype=np.intp)
        meta.factor_names = ["weather"]
        meta.is_discrete = np.array([True])
        meta.multi_target = False
        meta.label_level = "instance"
        meta.rows_at.return_value = pl.DataFrame({"weather": weather})

        params = _make_params(gap_mi_threshold=0.1, gap_min_representation=3)
        result = _run_gap_analysis(meta, {0: "cat", 1: "dog"}, params, precomputed_mi={"weather": 0.5})

        mock_mi.assert_not_called()
        assert result.mutual_info_class_to_factor == {"weather": 0.5}
        assert [(g.class_name, g.factor_value) for g in result.gaps] == [("dog", "1")]


class TestMiFromBalance:
    """Balance's `balance` frame carries the class-to-factor MI row gap analysis needs."""

    def test_extracts_all_factors(self) -> None:
        summary = {
            "balance": [
                {"factor_name": "class_label", "mi_value": 1.0},
                {"factor_name": "weather", "mi_value": 0.25},
                {"factor_name": "zone", "mi_value": 0.75},
            ]
        }
        assert _mi_from_balance(summary, ["weather", "zone"]) == {"weather": 0.25, "zone": 0.75}

    def test_partial_coverage_falls_back(self) -> None:
        summary = {"balance": [{"factor_name": "weather", "mi_value": 0.25}]}
        assert _mi_from_balance(summary, ["weather", "zone"]) is None

    def test_missing_or_empty_summary_falls_back(self) -> None:
        assert _mi_from_balance(None, ["weather"]) is None
        assert _mi_from_balance({}, ["weather"]) is None
        assert _mi_from_balance({"balance": []}, ["weather"]) is None

    def test_unexpected_shape_falls_back(self) -> None:
        """A renamed column must not silently produce empty MI."""
        summary = {"balance": [{"factor": "weather", "mi": 0.25}]}
        assert _mi_from_balance(summary, ["weather"]) is None


# ---------------------------------------------------------------------------
# TestBuildFindings
# ---------------------------------------------------------------------------


class TestBuildFindings:
    def test_without_extractor(self) -> None:
        raw = DataCoverageRawOutputs(
            dataset_size=100,
            coverage=None,
            completeness=None,
            metadata_distribution=MetadataDistributionResult(
                metadata_factors=["f1"],
                metadata_summary={"f1": {"type": "discrete", "unique_values": 5}},
            ),
            metadata_gaps=None,
            label_distribution=LabelDistributionResult(
                num_classes=3,
                class_distribution={"cat": 40, "dog": 35, "bird": 25},
            ),
        )
        thresholds = DataCoverageHealthThresholds()
        findings = build_findings(raw, thresholds)

        # Should have label distribution + metadata distribution (no coverage/completeness/gaps)
        assert len(findings) == 2
        titles = [f.title for f in findings]
        assert "Label Distribution" in titles
        assert "Metadata Distribution" in titles

    def test_with_extractor(self) -> None:
        raw = DataCoverageRawOutputs(
            dataset_size=100,
            coverage=CoverageAssessment(
                method="adaptive",
                uncovered_count=5,
                uncovered_rate=0.05,
                coverage_radius=0.3,
            ),
            completeness=CompletenessAssessment(
                completeness_score=0.8,
                nearest_neighbor_pairs=[[0, 1]],
            ),
            metadata_distribution=MetadataDistributionResult(
                metadata_factors=["f1"],
                metadata_summary={"f1": {"type": "discrete"}},
            ),
            metadata_gaps=MetadataGapResult(
                mutual_info_class_to_factor={"f1": 0.5},
                gaps=[
                    ClassMetadataGap(
                        class_name="cat",
                        factor_name="f1",
                        factor_value="night",
                        class_count=0,
                        expected_count=15.0,
                        deficit=1.0,
                    )
                ],
            ),
            label_distribution=LabelDistributionResult(
                num_classes=3,
                class_distribution={"cat": 40, "dog": 35, "bird": 25},
            ),
        )
        thresholds = DataCoverageHealthThresholds()
        findings = build_findings(raw, thresholds)

        titles = [f.title for f in findings]
        assert "Embedding Coverage" in titles
        assert "Dimensional Completeness" in titles
        assert "Label Distribution" in titles
        assert "Metadata Distribution" in titles
        assert "Metadata Coverage Gaps" in titles
        assert len(findings) == 5

    def test_coverage_warning_severity(self) -> None:
        raw = DataCoverageRawOutputs(
            dataset_size=100,
            coverage=CoverageAssessment(
                method="naive",
                uncovered_count=20,
                uncovered_rate=0.20,
                coverage_radius=0.3,
            ),
            metadata_distribution=MetadataDistributionResult(metadata_factors=[], metadata_summary={}),
            label_distribution=LabelDistributionResult(
                num_classes=2,
                class_distribution={"a": 50, "b": 50},
            ),
        )
        thresholds = DataCoverageHealthThresholds(uncovered_rate=10.0)
        findings = build_findings(raw, thresholds)
        cov_finding = next(f for f in findings if f.title == "Embedding Coverage")
        assert cov_finding.severity == "warning"

    def test_adaptive_rate_is_not_health_checked(self) -> None:
        """An adaptive rate over the threshold must not warn — it is set by config."""
        raw = DataCoverageRawOutputs(
            dataset_size=100,
            coverage=CoverageAssessment(
                method="adaptive",
                uncovered_count=20,
                uncovered_rate=0.20,
                coverage_radius=0.3,
            ),
            metadata_distribution=MetadataDistributionResult(metadata_factors=[], metadata_summary={}),
            label_distribution=LabelDistributionResult(num_classes=2, class_distribution={"a": 50, "b": 50}),
        )
        findings = build_findings(raw, DataCoverageHealthThresholds(uncovered_rate=10.0))
        cov_finding = next(f for f in findings if f.title == "Embedding Coverage")
        assert cov_finding.severity == "info"
        assert "not health-checked" in (cov_finding.description or "")

    def test_completeness_warning_severity(self) -> None:
        raw = DataCoverageRawOutputs(
            dataset_size=100,
            completeness=CompletenessAssessment(
                completeness_score=0.3,
                nearest_neighbor_pairs=[],
            ),
            metadata_distribution=MetadataDistributionResult(metadata_factors=[], metadata_summary={}),
            label_distribution=LabelDistributionResult(
                num_classes=2,
                class_distribution={"a": 50, "b": 50},
            ),
        )
        thresholds = DataCoverageHealthThresholds(completeness_score=0.5)
        findings = build_findings(raw, thresholds)
        comp_finding = next(f for f in findings if f.title == "Dimensional Completeness")
        assert comp_finding.severity == "warning"

    def test_gap_warning_severity(self) -> None:
        gaps = [
            ClassMetadataGap(
                class_name=f"cls_{i}",
                factor_name="f1",
                factor_value="v",
                class_count=0,
                expected_count=10.0,
                deficit=1.0,
            )
            for i in range(5)
        ]
        raw = DataCoverageRawOutputs(
            dataset_size=100,
            metadata_distribution=MetadataDistributionResult(metadata_factors=[], metadata_summary={}),
            metadata_gaps=MetadataGapResult(
                mutual_info_class_to_factor={"f1": 0.5},
                gaps=gaps,
            ),
            label_distribution=LabelDistributionResult(
                num_classes=2,
                class_distribution={"a": 50, "b": 50},
            ),
        )
        thresholds = DataCoverageHealthThresholds(gap_count=3)
        findings = build_findings(raw, thresholds)
        gap_finding = next(f for f in findings if f.title == "Metadata Coverage Gaps")
        assert gap_finding.severity == "warning"

    def test_missing_class_warns_despite_balanced_present_classes(self) -> None:
        """A declared class with zero samples must warn even when the rest are balanced."""
        raw = DataCoverageRawOutputs(
            dataset_size=100,
            metadata_distribution=MetadataDistributionResult(metadata_factors=[], metadata_summary={}),
            label_distribution=LabelDistributionResult(
                num_classes=2,
                class_distribution={"a": 50, "b": 50, "c": 0},
                missing_classes=["c"],
            ),
        )
        findings = build_findings(raw, DataCoverageHealthThresholds(class_imbalance_ratio=5.0))
        label_finding = next(f for f in findings if f.title == "Label Distribution")

        assert label_finding.severity == "warning"
        assert isinstance(label_finding.data, dict)
        # Ratio is computed over present classes only, so the zero does not zero it out.
        assert "ratio 1.0:1" in label_finding.data["brief"]
        assert "1 with no samples" in label_finding.data["brief"]
        assert "zero samples: c" in (label_finding.description or "")

    def test_no_missing_classes_stays_ok(self) -> None:
        raw = DataCoverageRawOutputs(
            dataset_size=100,
            metadata_distribution=MetadataDistributionResult(metadata_factors=[], metadata_summary={}),
            label_distribution=LabelDistributionResult(
                num_classes=2,
                class_distribution={"a": 50, "b": 50},
            ),
        )
        findings = build_findings(raw, DataCoverageHealthThresholds())
        label_finding = next(f for f in findings if f.title == "Label Distribution")
        assert label_finding.severity == "ok"
        assert isinstance(label_finding.data, dict)
        assert "no samples" not in label_finding.data["brief"]

    def test_label_percentages_are_shares_of_the_label_pool(self) -> None:
        """A multi-label / OD dataset has more labels than images.

        Dividing label counts by the image count made the column sum well past 100%.
        """
        raw = DataCoverageRawOutputs(
            dataset_size=50,  # 50 images carrying 100 labels
            metadata_distribution=MetadataDistributionResult(metadata_factors=[], metadata_summary={}),
            label_distribution=LabelDistributionResult(
                num_classes=2,
                class_distribution={"cat": 75, "dog": 25},
            ),
        )
        findings = build_findings(raw, DataCoverageHealthThresholds())
        label_finding = next(f for f in findings if f.title == "Label Distribution")
        assert isinstance(label_finding.data, dict)
        pcts = {row["Class"]: row["pct"] for row in label_finding.data["table_data"]}
        assert pcts == {"cat": 75.0, "dog": 25.0}
        assert sum(pcts.values()) == 100.0
        assert "shares of 100 labels across 50 images" in (label_finding.description or "")

    def test_all_null_continuous_factor_renders(self) -> None:
        """A continuous factor whose column is all-null summarizes to a null mean.

        Rounding that None raised TypeError, failing report building — and with it
        the whole run.
        """
        raw = DataCoverageRawOutputs(
            dataset_size=100,
            metadata_distribution=MetadataDistributionResult(
                metadata_factors=["altitude"],
                metadata_summary={
                    "altitude": {"type": "continuous", "null_count": 100, "mean": None, "std": None},
                },
            ),
            label_distribution=LabelDistributionResult(num_classes=1, class_distribution={"a": 100}),
        )
        findings = build_findings(raw, DataCoverageHealthThresholds())
        md_finding = next(f for f in findings if f.title == "Metadata Distribution")
        assert isinstance(md_finding.data, dict)
        assert md_finding.data["table_data"][0]["Unique"] == "-"
        assert md_finding.data["table_data"][0]["Nulls"] == 100

    def test_ontology_skip_reason_is_reported(self) -> None:
        """A failed ontology must explain itself, not silently drop its sections."""
        raw = DataCoverageRawOutputs(
            dataset_size=100,
            metadata_distribution=MetadataDistributionResult(metadata_factors=[], metadata_summary={}),
            label_distribution=LabelDistributionResult(num_classes=1, class_distribution={"a": 100}),
            ontology=None,
            ontology_skipped_reason="ontology file not found: /data/missing.ttl",
        )
        findings = build_findings(raw, DataCoverageHealthThresholds())
        onto_finding = next(f for f in findings if f.title == "Ontology Analysis")
        assert onto_finding.severity == "info"
        assert "missing.ttl" in (onto_finding.description or "")

    def test_no_ontology_skip_finding_when_analysis_ran(self) -> None:
        raw = _populated_raw()
        raw.ontology = OntologyAssessment(
            source="index2label",
            synthesized=True,
            representation=LabelSpaceCoverage(leaf_coverage=1.0, total_deficit=0),
        )
        titles = [f.title for f in build_findings(raw, DataCoverageHealthThresholds())]
        assert "Ontology Analysis" not in titles

    def test_clustered_class_warns(self) -> None:
        raw = _populated_raw()
        findings = build_findings(raw, DataCoverageHealthThresholds())
        cov = next(f for f in findings if f.title == "Embedding Coverage")
        # _populated_raw's "cat" has dispersion 0.21, below the 0.5 default.
        assert cov.severity == "warning"
        assert isinstance(cov.data, dict)
        assert "clustered" in cov.data["brief"]
        assert cov.description
        assert "cat" in cov.description

    def test_healthy_classes_do_not_warn(self) -> None:
        raw = _populated_raw()
        # _populated_raw's "cat" is unhealthy on dispersion, isotropy, and the overall
        # rate (12% > the 10% default) — neutralize all three to isolate "no warning
        # when everything is healthy" from the other warning paths.
        assert raw.coverage
        raw.coverage.uncovered_rate = 0.05
        raw.coverage.per_class[0].dispersion = 1.0
        raw.coverage.per_class[0].isotropy = 1.0
        findings = build_findings(raw, DataCoverageHealthThresholds())
        cov = next(f for f in findings if f.title == "Embedding Coverage")
        assert cov.severity != "warning"

    def _raw_with_ontology(self, *, synthesized: bool) -> DataCoverageRawOutputs:
        raw = _populated_raw()
        raw.ontology = OntologyAssessment(
            source="index2label" if synthesized else "inline",
            synthesized=synthesized,
            representation=LabelSpaceCoverage(
                leaf_coverage=0.5,
                total_deficit=50,
                worklist=[
                    RepresentationRow(
                        concept="frog",
                        label="frog",
                        parent="amphibian",
                        action="acquire",
                        count=0,
                        target=40,
                        deficit=40,
                    ),
                    RepresentationRow(
                        concept="dog",
                        label="dog",
                        parent="mammal",
                        action="augment",
                        count=30,
                        target=40,
                        deficit=10,
                    ),
                ],
                dark_branches=[] if synthesized else [DarkBranch(concept="amphibian", label="amphibian", leaves=1)],
            ),
            conformance=None
            if synthesized
            else LabelConformance(conforms=False, matched={"cat": "cat"}, unmatched=["kitteh"], ambiguous={}),
            structure=None
            if synthesized
            else OntologyStructure(concept_count=9, leaf_count=5, max_depth=3, roots=["subject"]),
        )
        return raw

    def test_configured_ontology_emits_three_findings(self) -> None:
        findings = build_findings(self._raw_with_ontology(synthesized=False), DataCoverageHealthThresholds())
        titles = [f.title for f in findings]
        assert "Label Space Coverage" in titles
        assert "Label Conformance" in titles
        assert "Ontology Structure" in titles
        assert "Class Balance Worklist" not in titles

    def test_synthesized_emits_only_the_balance_worklist(self) -> None:
        findings = build_findings(self._raw_with_ontology(synthesized=True), DataCoverageHealthThresholds())
        titles = [f.title for f in findings]
        assert "Class Balance Worklist" in titles
        assert "Label Space Coverage" not in titles
        assert "Label Conformance" not in titles
        assert "Ontology Structure" not in titles

    def test_leaf_coverage_threshold_warns_only_when_configured(self) -> None:
        thresholds = DataCoverageHealthThresholds()  # leaf_coverage default 0.9
        configured = build_findings(self._raw_with_ontology(synthesized=False), thresholds)
        assert next(f for f in configured if f.title == "Label Space Coverage").severity == "warning"

        synthesized = build_findings(self._raw_with_ontology(synthesized=True), thresholds)
        # Same 0.5 leaf_coverage in the data, but not health-checked when synthesized.
        assert next(f for f in synthesized if f.title == "Class Balance Worklist").severity != "warning"

    def test_unmatched_class_warns(self) -> None:
        findings = build_findings(self._raw_with_ontology(synthesized=False), DataCoverageHealthThresholds())
        conf = next(f for f in findings if f.title == "Label Conformance")
        assert conf.severity == "warning"
        assert conf.description
        assert "kitteh" in conf.description

    def test_label_collisions_warn(self) -> None:
        raw = self._raw_with_ontology(synthesized=False)
        assert raw.ontology is not None
        assert raw.ontology.structure is not None
        raw.ontology.structure.label_collisions = {"bat": ["mammal_bat", "sports_bat"]}
        findings = build_findings(raw, DataCoverageHealthThresholds())
        assert next(f for f in findings if f.title == "Ontology Structure").severity == "warning"

    def test_violations_warn_in_both_modes(self) -> None:
        for synthesized in (True, False):
            raw = self._raw_with_ontology(synthesized=synthesized)
            assert raw.ontology is not None
            raw.ontology.representation.violations = [
                RepresentationViolation(concept="dog", label="dog", floor=0.25, actual=0.05, shortfall=40)
            ]
            findings = build_findings(raw, DataCoverageHealthThresholds())
            title = "Class Balance Worklist" if synthesized else "Label Space Coverage"
            assert next(f for f in findings if f.title == title).severity == "warning"

    def test_no_ontology_emits_nothing(self) -> None:
        findings = build_findings(_populated_raw(), DataCoverageHealthThresholds())
        titles = [f.title for f in findings]
        assert "Label Space Coverage" not in titles
        assert "Class Balance Worklist" not in titles


# ---------------------------------------------------------------------------
# TestRenderFindings
# ---------------------------------------------------------------------------


def _populated_raw() -> DataCoverageRawOutputs:
    """Raw outputs with every optional section present, for render coverage."""
    return DataCoverageRawOutputs(
        dataset_size=100,
        coverage=CoverageAssessment(
            method="naive",
            uncovered_count=12,
            uncovered_rate=0.12,
            coverage_radius=0.3456,
            per_class=[
                ClassCoverageRow(
                    class_name="cat",
                    count=60,
                    uncovered=8,
                    uncovered_fraction=0.133,
                    dispersion=0.21,
                    isotropy=0.44,
                    near_duplicate_fraction=0.02,
                    assessable=True,
                ),
                ClassCoverageRow(
                    class_name="dog",
                    count=30,
                    uncovered=4,
                    uncovered_fraction=0.133,
                    dispersion=1.05,
                    isotropy=0.91,
                    near_duplicate_fraction=0.01,
                    assessable=True,
                ),
            ],
        ),
        completeness=CompletenessAssessment(completeness_score=0.42, nearest_neighbor_pairs=[[0, 1], [2, 3]]),
        metadata_distribution=MetadataDistributionResult(
            metadata_factors=["brightness", "weather"],
            metadata_summary={
                "brightness": {"type": "continuous", "null_count": 0, "mean": 0.51},
                "weather": {"type": "discrete", "null_count": 2, "unique_values": 3},
            },
            balance_summary={"balance": []},
            diversity_summary={"factors": []},
        ),
        metadata_gaps=MetadataGapResult(
            mutual_info_class_to_factor={"weather": 0.4},
            gaps=[
                ClassMetadataGap(
                    class_name="cat",
                    factor_name="weather",
                    factor_value="snow",
                    class_count=0,
                    expected_count=8.0,
                    deficit=1.0,
                ),
            ],
        ),
        label_distribution=LabelDistributionResult(
            num_classes=3,
            class_distribution={"cat": 60, "dog": 30, "bird": 10},
            empty_images=[5],
        ),
    )


class TestRenderFindings:
    """Every finding must survive the text renderer.

    ``_render_detail_section`` dispatches on ``report_type`` and each renderer
    expects a specific ``data`` shape, so a mismatched pairing either raises or
    silently drops the table. Asserting on finding metadata alone cannot see that.
    """

    def test_all_findings_render_without_error(self) -> None:
        findings = build_findings(_populated_raw(), DataCoverageHealthThresholds())
        assert len(findings) == 5
        for finding in findings:
            lines = _render_detail_section(finding)
            assert lines, f"{finding.title} rendered nothing"

    def test_coverage_values_appear(self) -> None:
        findings = build_findings(_populated_raw(), DataCoverageHealthThresholds())
        text = "\n".join(_render_detail_section(next(f for f in findings if f.title == "Embedding Coverage")))
        assert "naive" in text
        assert "12" in text
        assert "0.3456" in text

    def test_label_distribution_counts_appear(self) -> None:
        """The pivot renderer aliases the "Count"/"%" headers to lowercase row keys."""
        findings = build_findings(_populated_raw(), DataCoverageHealthThresholds())
        lines = _render_detail_section(next(f for f in findings if f.title == "Label Distribution"))
        cat_row = next(line for line in lines if line.strip().startswith("cat"))
        assert "60" in cat_row
        assert "60.0%" in cat_row

    def test_metadata_distribution_rows_appear(self) -> None:
        findings = build_findings(_populated_raw(), DataCoverageHealthThresholds())
        lines = _render_detail_section(next(f for f in findings if f.title == "Metadata Distribution"))
        assert any("brightness" in line and "continuous" in line for line in lines)
        assert any("weather" in line and "discrete" in line for line in lines)

    def test_gap_rows_appear(self) -> None:
        """Gap rows are the workflow's most actionable output — they must reach the page."""
        findings = build_findings(_populated_raw(), DataCoverageHealthThresholds())
        lines = _render_detail_section(next(f for f in findings if f.title == "Metadata Coverage Gaps"))
        gap_row = next(line for line in lines if line.strip().startswith("cat"))
        assert "weather" in gap_row
        assert "snow" in gap_row
        assert "8.0" in gap_row
        assert "100.0%" in gap_row


# ---------------------------------------------------------------------------
# TestDataCoverageWorkflow
# ---------------------------------------------------------------------------


class TestDataCoverageWorkflow:
    def test_properties(self) -> None:
        wf = DataCoverageWorkflow()
        assert wf.name == "data-coverage"
        assert "coverage" in wf.description.lower()
        assert wf.params_schema is DataCoverageParameters
        assert wf.output_schema is DataCoverageOutputs

    def test_rejects_non_context(self) -> None:
        wf = DataCoverageWorkflow()
        result = wf.execute("not a context")  # type: ignore[arg-type]
        assert result.success is False
        assert any("WorkflowContext" in e for e in result.errors)

    def test_rejects_no_params(self) -> None:
        wf = DataCoverageWorkflow()
        ctx = WorkflowContext()
        result = wf.execute(ctx, params=None)
        assert result.success is False
        assert any("required" in e.lower() for e in result.errors)

    def test_rejects_wrong_params(self) -> None:
        class OtherParams(BaseModel):
            x: int = 1

        wf = DataCoverageWorkflow()
        ctx = WorkflowContext()
        result = wf.execute(ctx, params=OtherParams())
        assert result.success is False
        assert any("DataCoverageParameters" in e for e in result.errors)

    def test_empty_context_returns_failed(self) -> None:
        wf = DataCoverageWorkflow()
        ctx = WorkflowContext()
        result = wf.execute(ctx, _make_params())
        assert result.success is False

    @patch("dataeval_flow.workflows.coverage.workflow.get_or_compute_metadata")
    @patch("dataeval_flow.workflows.coverage.workflow.label_stats")
    @patch("dataeval.bias.Balance")
    @patch("dataeval.bias.Diversity")
    def test_without_extractor(
        self,
        mock_diversity: MagicMock,
        mock_balance: MagicMock,
        mock_label_stats: MagicMock,
        mock_get_metadata: MagicMock,
    ) -> None:
        dataset = _make_dataset(100)
        metadata = _make_metadata(100)
        mock_get_metadata.return_value = metadata
        mock_label_stats.return_value = _make_label_stats()

        mock_balance_inst = MagicMock()
        mock_balance_inst.evaluate.return_value = MagicMock(
            balance=pl.DataFrame({"factor": ["brightness"], "mi": [0.1]}),
            factors=pl.DataFrame({"factor": ["brightness"], "score": [0.5]}),
            classwise=pl.DataFrame({"class": ["cat"], "factor": ["brightness"], "score": [0.3]}),
        )
        mock_balance.return_value = mock_balance_inst

        mock_diversity_inst = MagicMock()
        mock_diversity_inst.evaluate.return_value = MagicMock(
            factors=pl.DataFrame({"factor_name": ["brightness"], "diversity_value": [0.9]}),
            classwise=pl.DataFrame({"class": ["cat"], "factor": ["brightness"], "score": [0.8]}),
        )
        mock_diversity.return_value = mock_diversity_inst

        ctx = _make_context(dataset)

        wf = DataCoverageWorkflow()
        result = wf.execute(ctx, _make_params(run_gap_analysis=False))

        assert result.success is True
        assert result.data.raw.coverage is None
        assert result.data.raw.completeness is None
        assert result.data.raw.metadata_distribution.metadata_factors == ["brightness", "contrast"]
        assert result.data.raw.label_distribution.num_classes == 3
        assert result.data.raw.label_distribution.missing_classes == []
        assert result.metadata.has_extractor is False

    @patch("dataeval_flow.workflows.coverage.workflow.get_or_compute_metadata")
    @patch("dataeval_flow.workflows.coverage.workflow.label_stats")
    @patch("dataeval.bias.Balance")
    @patch("dataeval.bias.Diversity")
    def test_declared_class_with_no_samples_is_reported(
        self,
        mock_diversity: MagicMock,
        mock_balance: MagicMock,
        mock_label_stats: MagicMock,
        mock_get_metadata: MagicMock,
    ) -> None:
        """A class in index2label that never appears must surface as a missing class.

        ``label_stats`` only counts observed labels, so the absence is invisible
        unless the workflow diffs against the dataset's declared index2label.
        """
        dataset = _make_dataset(100)
        dataset.metadata = {"index2label": {0: "cat", 1: "dog", 2: "bird", 3: "fish"}}
        mock_get_metadata.return_value = _make_metadata(100)
        mock_label_stats.return_value = _make_label_stats()  # only classes 0-2 observed

        ctx = _make_context(dataset)
        result = DataCoverageWorkflow().execute(ctx, _make_params(run_gap_analysis=False, balance=False))

        assert result.success is True
        ld = result.data.raw.label_distribution
        assert ld.missing_classes == ["fish"]
        assert ld.class_distribution["fish"] == 0
        assert ld.num_classes == 3  # observed classes only

        label_finding = next(f for f in result.data.report.findings if f.title == "Label Distribution")
        assert label_finding.severity == "warning"

    @patch("dataeval_flow.embeddings.build_embeddings")
    @patch("dataeval_flow.workflows.coverage.workflow.get_or_compute_metadata")
    @patch("dataeval_flow.workflows.coverage.workflow.label_stats")
    @patch("dataeval.bias.Balance")
    @patch("dataeval.bias.Diversity")
    @patch("dataeval.core.completeness")
    def test_with_extractor(
        self,
        mock_completeness: MagicMock,
        mock_diversity: MagicMock,
        mock_balance: MagicMock,
        mock_label_stats: MagicMock,
        mock_get_metadata: MagicMock,
        mock_build_emb: MagicMock,
    ) -> None:
        """Coverage now runs for real (scope.Coverage has no module-level function to mock)."""
        dataset = _make_dataset(50)
        metadata = _make_metadata(50)
        mock_get_metadata.return_value = metadata
        mock_label_stats.return_value = _make_label_stats(n=50)

        mock_build_emb.return_value = np.random.default_rng(42).random((50, 10))

        mock_completeness.return_value = {
            "completeness": 0.85,
            "nearest_neighbor_pairs": [(0, 1), (2, 3)],
        }

        mock_balance_inst = MagicMock()
        mock_balance_inst.evaluate.return_value = MagicMock(
            balance=pl.DataFrame({"factor": ["brightness"], "mi": [0.1]}),
            factors=pl.DataFrame({"factor": ["brightness"], "score": [0.5]}),
            classwise=pl.DataFrame({"class": ["cat"], "factor": ["brightness"], "score": [0.3]}),
        )
        mock_balance.return_value = mock_balance_inst

        mock_diversity_inst = MagicMock()
        mock_diversity_inst.evaluate.return_value = MagicMock(
            factors=pl.DataFrame({"factor_name": ["brightness"], "diversity_value": [0.9]}),
            classwise=pl.DataFrame({"class": ["cat"], "factor": ["brightness"], "score": [0.8]}),
        )
        mock_diversity.return_value = mock_diversity_inst

        extractor_config = MagicMock()
        ctx = WorkflowContext(
            dataset_contexts={"ds": DatasetContext(name="ds", dataset=dataset, extractor=extractor_config)},
        )

        wf = DataCoverageWorkflow()
        # num_observations must be strictly less than the sample count, otherwise
        # dataeval's coverage functions reject the input (see the test below).
        result = wf.execute(ctx, _make_params(run_gap_analysis=False, coverage_method="adaptive", num_observations=20))

        assert result.success is True
        assert result.data.raw.coverage is not None
        # Real coverage math now runs (no module-level function left to mock) — assert
        # structure rather than an exact count, which TestRunCoverage already covers.
        assert result.data.raw.coverage.uncovered_count >= 0
        assert len(result.data.raw.coverage.per_class) == 3
        assert result.data.raw.coverage_skipped_reason is None
        assert result.data.raw.completeness is not None
        assert result.data.raw.completeness.completeness_score == 0.85
        assert result.metadata.has_extractor is True

    @patch("dataeval_flow.workflows.coverage.workflow.get_or_compute_embeddings")
    def test_object_detection_embeds_detection_crops(self, mock_emb: MagicMock) -> None:
        """An OD dataset with an extractor must not abort the whole workflow.

        Coverage assumes one embedding per label; whole-image embeddings against
        per-detection labels raise ShapeMismatchError, which used to take label,
        metadata and gap analysis down with it. Real ``Metadata``, ``label_stats``
        and ``scope.Coverage`` run here — only extraction is stubbed.
        """
        from dataeval.data import DetectionCrops

        n_images = 40
        n_crops = n_images * 2
        dataset = _ODDataset(n_images=n_images)
        mock_emb.return_value = np.random.default_rng(42).random((n_crops, 8))

        ctx = WorkflowContext(
            dataset_contexts={"ds": DatasetContext(name="ds", dataset=dataset, extractor=MagicMock())},
        )
        params = _make_params(
            num_observations=10,
            min_class_samples=10,
            run_completeness=False,
            run_gap_analysis=False,
            balance=False,
            diversity_method=None,
        )
        result = DataCoverageWorkflow().execute(ctx, params)

        assert result.errors == []
        assert result.success is True

        # Extraction ran over the crop view, not the source images.
        embedded = mock_emb.call_args[0][0]
        assert isinstance(embedded, DetectionCrops)
        assert len(embedded) == n_crops

        cov = result.data.raw.coverage
        assert cov is not None
        assert result.data.raw.coverage_skipped_reason is None
        assert cov.observation_count == n_crops
        assert cov.observation_unit == "detection crop"
        assert sum(row.count for row in cov.per_class) == n_crops
        assert {row.class_name for row in cov.per_class} == {"cat", "dog"}

        cov_finding = next(f for f in result.data.report.findings if f.title == "Embedding Coverage")
        assert f"of {n_crops} detection crops" in (cov_finding.description or "")

    @patch("dataeval_flow.workflows.coverage.workflow.get_or_compute_embeddings")
    @patch("dataeval_flow.workflows.coverage.workflow.get_or_compute_metadata")
    @patch("dataeval_flow.workflows.coverage.workflow.label_stats")
    @patch("dataeval.core.completeness")
    def test_embeddings_go_through_the_cache(
        self,
        mock_completeness: MagicMock,
        mock_label_stats: MagicMock,
        mock_get_metadata: MagicMock,
        mock_get_or_compute_emb: MagicMock,
    ) -> None:
        """Embedding extraction must be cacheable like every other extractor workflow.

        Calling ``build_embeddings`` directly re-runs extraction on every invocation
        even when ``cache_dir`` is set.
        """
        dataset = _make_dataset(100)
        mock_get_metadata.return_value = _make_metadata(100)
        mock_label_stats.return_value = _make_label_stats()
        mock_get_or_compute_emb.return_value = np.random.default_rng(42).random((100, 10))
        mock_completeness.return_value = {"completeness": 0.7, "nearest_neighbor_pairs": []}

        cache = MagicMock()
        extractor = MagicMock()
        transforms = MagicMock()
        ctx = WorkflowContext(
            dataset_contexts={
                "ds": DatasetContext(
                    name="ds",
                    dataset=dataset,
                    extractor=extractor,
                    transforms=transforms,
                    batch_size=16,
                    cache=cache,
                )
            },
        )
        params = _make_params(run_gap_analysis=False, balance=False, diversity_method=None, num_observations=20)
        result = DataCoverageWorkflow().execute(ctx, params)

        assert result.success is True
        mock_get_or_compute_emb.assert_called_once_with(dataset, extractor, transforms, 16)

    @patch("dataeval_flow.workflows.coverage.workflow._balance_class_to_factor")
    @patch("dataeval_flow.workflows.coverage.workflow.get_or_compute_metadata")
    @patch("dataeval_flow.workflows.coverage.workflow.label_stats")
    @patch("dataeval.bias.Balance")
    @patch("dataeval.bias.Diversity")
    def test_gap_analysis_reuses_balance_mi(
        self,
        mock_diversity: MagicMock,
        mock_balance: MagicMock,
        mock_label_stats: MagicMock,
        mock_get_metadata: MagicMock,
        mock_mi: MagicMock,
    ) -> None:
        """With both balance and gap analysis on (the defaults), MI is computed once."""
        mock_get_metadata.return_value = _make_metadata(100)
        mock_label_stats.return_value = _make_label_stats()

        mock_balance_inst = MagicMock()
        mock_balance_inst.evaluate.return_value = MagicMock(
            # Real column names from dataeval's BalanceOutput.balance
            balance=pl.DataFrame(
                {
                    "factor_name": ["class_label", "brightness", "contrast"],
                    "mi_value": [1.0, 0.4, 0.05],
                }
            ),
            factors=pl.DataFrame({"factor1": ["brightness"], "factor2": ["contrast"], "mi_value": [0.1]}),
            classwise=pl.DataFrame({"class_name": ["cat"], "factor_name": ["brightness"], "mi_value": [0.3]}),
        )
        mock_balance.return_value = mock_balance_inst
        mock_diversity.return_value = MagicMock(
            evaluate=MagicMock(
                return_value=MagicMock(
                    factors=pl.DataFrame({"factor_name": ["brightness"], "diversity_value": [0.9]}),
                    classwise=pl.DataFrame({"class_name": ["cat"], "diversity_value": [0.8]}),
                )
            )
        )

        ctx = _make_context()
        result = DataCoverageWorkflow().execute(ctx, _make_params(balance=True, run_gap_analysis=True))

        assert result.success is True
        # Balance's internal mutual_info is inside the mocked evaluate; the workflow's
        # own direct call must not happen at all.
        mock_mi.assert_not_called()
        gaps = result.data.raw.metadata_gaps
        assert gaps is not None
        assert gaps.mutual_info_class_to_factor == {"brightness": 0.4, "contrast": 0.05}

    @patch("dataeval_flow.embeddings.build_embeddings")
    @patch("dataeval_flow.workflows.coverage.workflow.get_or_compute_metadata")
    @patch("dataeval_flow.workflows.coverage.workflow.label_stats")
    def test_dataset_smaller_than_num_observations_skips_only_coverage(
        self,
        mock_label_stats: MagicMock,
        mock_get_metadata: MagicMock,
        mock_build_emb: MagicMock,
    ) -> None:
        """A too-small dataset must not cost us the metadata and label analysis.

        ``coverage_naive``/``coverage_adaptive`` raise ValueError when
        ``len(embeddings) <= num_observations``. The real functions run here — no
        coverage mock — so the precondition is genuinely exercised.
        """
        dataset = _make_dataset(40)
        mock_get_metadata.return_value = _make_metadata(40)
        mock_label_stats.return_value = _make_label_stats(n=40)
        mock_build_emb.return_value = np.random.default_rng(42).random((40, 10))

        ctx = WorkflowContext(
            dataset_contexts={"ds": DatasetContext(name="ds", dataset=dataset, extractor=MagicMock())},
        )
        params = _make_params(run_gap_analysis=False, balance=False, diversity_method=None, num_observations=50)
        result = DataCoverageWorkflow().execute(ctx, params)

        assert result.success is True
        assert result.errors == []
        assert result.data.raw.coverage is None
        assert result.data.raw.coverage_skipped_reason is not None
        assert "num_observations=50" in result.data.raw.coverage_skipped_reason
        # The rest of the analysis survived.
        assert result.data.raw.label_distribution.num_classes == 3
        assert result.data.raw.metadata_distribution.metadata_factors == ["brightness", "contrast"]

        cov_finding = next(f for f in result.data.report.findings if f.title == "Embedding Coverage")
        assert isinstance(cov_finding.data, dict)
        assert cov_finding.data["brief"] == "skipped"


# ---------------------------------------------------------------------------
# TestOutputModels
# ---------------------------------------------------------------------------


class TestOutputModels:
    def test_is_coverage_result_guard(self) -> None:
        from dataeval_flow.workflow import WorkflowResult

        result = WorkflowResult(
            name="data-coverage",
            success=True,
            data=DataCoverageOutputs(
                raw=DataCoverageRawOutputs(
                    dataset_size=0,
                    metadata_distribution=MetadataDistributionResult(metadata_factors=[], metadata_summary={}),
                    label_distribution=LabelDistributionResult(num_classes=0, class_distribution={}),
                ),
                report=DataCoverageReport(summary="test", findings=[]),
            ),
            metadata=DataCoverageMetadata(),
        )
        assert is_coverage_result(result)

    def test_class_metadata_gap_model(self) -> None:
        gap = ClassMetadataGap(
            class_name="car",
            factor_name="time_of_day",
            factor_value="night",
            class_count=0,
            expected_count=15.0,
            deficit=1.0,
        )
        assert gap.class_name == "car"
        assert gap.deficit == 1.0

    def test_ontology_assessment_is_json_serializable(self) -> None:
        """Guards the tuple-key hazard: DataEval's raw results are not JSON-safe."""
        import json

        assessment = OntologyAssessment(
            source="inline",
            synthesized=False,
            representation=LabelSpaceCoverage(
                leaf_coverage=0.5,
                total_deficit=42,
                worklist=[
                    RepresentationRow(
                        concept="frog",
                        label="frog",
                        parent="amphibian",
                        action="acquire",
                        count=0,
                        target=25,
                        deficit=25,
                    )
                ],
                dark_branches=[DarkBranch(concept="amphibian", label="amphibian", leaves=1)],
                violations=[RepresentationViolation(concept="owl", label="owl", floor=0.05, actual=0.01, shortfall=8)],
                ignored_expected=["bogus"],
            ),
            conformance=LabelConformance(
                conforms=False,
                matched={"cat": "cat"},
                unmatched=["kitteh"],
                ambiguous={"bat": ["mammal_bat", "sports_bat"]},
            ),
            structure=OntologyStructure(
                concept_count=8,
                leaf_count=5,
                max_depth=3,
                roots=["subject"],
                isolated=[],
                external_ancestors={"cat": ["http://example.org/animal"]},
                redundant_edges=[["cat", "subject"]],
                ancestor_siblings=[["car", "vehicle"]],
                unary_parents=["amphibian"],
                label_collisions={"bat": ["mammal_bat", "sports_bat"]},
                nonconforming_labels={"Fighter Jet": "not lowercase_snake_case"},
            ),
        )
        # Must survive a round trip with no custom encoder.
        assert json.loads(assessment.model_dump_json())["source"] == "inline"

    def test_synthesized_assessment_omits_conformance_and_structure(self) -> None:
        assessment = OntologyAssessment(
            source="index2label",
            synthesized=True,
            representation=LabelSpaceCoverage(leaf_coverage=1.0, total_deficit=0),
        )
        assert assessment.conformance is None
        assert assessment.structure is None

    def test_raw_outputs_carry_ontology_fields(self) -> None:
        raw = _populated_raw()
        assert raw.ontology is None
        assert raw.ontology_skipped_reason is None


# ---------------------------------------------------------------------------
# TestWorkflowRegistration
# ---------------------------------------------------------------------------


class TestWorkflowRegistration:
    def test_registered_in_discovery(self) -> None:
        from dataeval_flow.workflow import list_workflows

        workflows = list_workflows()
        names = [w["name"] for w in workflows]
        assert "data-coverage" in names

    def test_get_workflow(self) -> None:
        from dataeval_flow.workflow import get_workflow

        wf = get_workflow("data-coverage")
        assert wf.name == "data-coverage"


# ---------------------------------------------------------------------------
# TestConfigSchemas
# ---------------------------------------------------------------------------


class TestConfigSchemas:
    def test_workflow_config(self) -> None:
        from dataeval_flow.config.schemas import DataCoverageWorkflowConfig

        cfg = DataCoverageWorkflowConfig(name="test_coverage")
        assert cfg.type == "data-coverage"
        assert cfg.name == "test_coverage"
        assert cfg.coverage_method == "adaptive"

    def test_task_config(self) -> None:
        from dataeval_flow.config.schemas import DataCoverageTaskConfig

        cfg = DataCoverageTaskConfig(
            name="cov_task",
            workflow="test_coverage",
            sources="my_source",
        )
        assert cfg.name == "cov_task"


# ---------------------------------------------------------------------------
# TestRunOntologyAnalysis
# ---------------------------------------------------------------------------


def _fixture_ontology() -> Any:
    """Vehicles and animals, with a whole amphibian branch nothing will populate."""
    from dataeval import Ontology

    return Ontology.from_hierarchy(
        {
            "subject": {
                "vehicle": {"wheeled": ["car", "truck"]},
                "animal": {"mammal": ["cat", "dog"], "amphibian": ["frog"]},
            }
        }
    )


class TestRunOntologyAnalysis:
    def test_acquire_and_augment(self) -> None:
        from dataeval_flow.workflows.coverage.ontology import run_ontology_analysis

        # 5 leaves, 200 samples -> uniform target 40 each.
        counts = {"car": 100, "truck": 60, "cat": 30, "dog": 10}
        result = run_ontology_analysis(_fixture_ontology(), source="inline", synthesized=False, class_counts=counts)

        by_concept = {row.concept: row for row in result.representation.worklist}
        assert by_concept["frog"].action == "acquire"
        assert by_concept["frog"].count == 0
        assert by_concept["dog"].action == "augment"
        assert by_concept["dog"].count == 10
        assert "car" not in by_concept  # above its uniform target
        # 4 of 5 leaves have examples.
        assert result.representation.leaf_coverage == 0.8
        assert result.representation.total_deficit == sum(r.deficit for r in result.representation.worklist)

    def test_dark_branch_rolls_up(self) -> None:
        from dataeval_flow.workflows.coverage.ontology import run_ontology_analysis

        counts = {"car": 100, "truck": 60, "cat": 30, "dog": 10}
        result = run_ontology_analysis(_fixture_ontology(), source="inline", synthesized=False, class_counts=counts)
        assert [b.concept for b in result.representation.dark_branches] == ["amphibian"]

    def test_violations_and_ignored_expected(self) -> None:
        from dataeval_flow.workflows.coverage.ontology import run_ontology_analysis

        counts = {"car": 100, "truck": 60, "cat": 30, "dog": 10}
        result = run_ontology_analysis(
            _fixture_ontology(),
            source="inline",
            synthesized=False,
            class_counts=counts,
            expected={"dog": 0.25, "nonexistent": 0.1},
        )
        # dog is 10/200 = 5%, below the asserted 25% floor.
        assert [v.concept for v in result.representation.violations] == ["dog"]
        assert result.representation.violations[0].shortfall == 40
        assert result.representation.ignored_expected == ["nonexistent"]

    def test_unmatched_class_name(self) -> None:
        from dataeval_flow.workflows.coverage.ontology import run_ontology_analysis

        counts = {"car": 100, "kitteh": 30}
        result = run_ontology_analysis(_fixture_ontology(), source="inline", synthesized=False, class_counts=counts)
        assert result.conformance is not None
        assert result.conformance.conforms is False
        assert result.conformance.unmatched == ["kitteh"]
        assert result.conformance.matched["car"] == "car"

    def test_structure_reported(self) -> None:
        from dataeval_flow.workflows.coverage.ontology import run_ontology_analysis

        result = run_ontology_analysis(
            _fixture_ontology(), source="inline", synthesized=False, class_counts={"car": 10}
        )
        assert result.structure is not None
        # 11 concepts: subject, vehicle, wheeled, car, truck, animal, mammal, cat, dog,
        # amphibian, frog.
        assert result.structure.concept_count == 11
        assert result.structure.leaf_count == 5
        assert result.structure.roots == ["subject"]
        assert result.structure.max_depth == 3
        # amphibian has exactly one child.
        assert "amphibian" in result.structure.unary_parents

    def test_label_pattern_flags_nonconforming(self) -> None:
        from dataeval import Ontology

        from dataeval_flow.workflows.coverage.ontology import run_ontology_analysis

        onto = Ontology.from_hierarchy({"Vehicle": ["car"]})
        result = run_ontology_analysis(
            onto,
            source="inline",
            synthesized=False,
            class_counts={"car": 10},
            label_pattern="^[a-z0-9_]+$",
        )
        assert result.structure is not None
        assert "Vehicle" in result.structure.nonconforming_labels

    def test_synthesized_skips_conformance_and_structure(self) -> None:
        from dataeval_flow.workflows._ontology import synthesize_ontology
        from dataeval_flow.workflows.coverage.ontology import run_ontology_analysis

        onto, source = synthesize_ontology({0: "cat", 1: "dog"})
        result = run_ontology_analysis(onto, source=source, synthesized=True, class_counts={"cat": 90, "dog": 10})
        assert result.synthesized is True
        assert result.source == "index2label"
        assert result.conformance is None
        assert result.structure is None
        # The worklist still works: uniform target is 50 each.
        assert [r.concept for r in result.representation.worklist] == ["dog"]
        assert result.representation.worklist[0].deficit == 40
        # Vacuous by construction.
        assert result.representation.leaf_coverage == 1.0
        assert result.representation.dark_branches == []


# ---------------------------------------------------------------------------
# TestOntologyWiring
# ---------------------------------------------------------------------------


class TestOntologyWiring:
    @patch("dataeval_flow.workflows.coverage.workflow.get_or_compute_metadata")
    @patch("dataeval_flow.workflows.coverage.workflow.label_stats")
    @patch("dataeval.bias.Balance")
    @patch("dataeval.bias.Diversity")
    def test_synthesized_by_default(
        self,
        mock_diversity: MagicMock,
        mock_balance: MagicMock,
        mock_label_stats: MagicMock,
        mock_get_metadata: MagicMock,
    ) -> None:
        """With no ontology configured, one is synthesized from index2label."""
        mock_get_metadata.return_value = _make_metadata(100)
        mock_label_stats.return_value = _make_label_stats()

        workflow = DataCoverageWorkflow()
        context = _make_context()
        result = workflow.execute(context, _make_params(run_gap_analysis=False))

        assert result.success
        assert result.data.raw.ontology is not None
        assert result.data.raw.ontology.synthesized is True
        assert result.data.raw.ontology.source == "index2label"
        assert result.data.raw.ontology.conformance is None

    @patch("dataeval_flow.workflows.coverage.workflow.get_or_compute_metadata")
    @patch("dataeval_flow.workflows.coverage.workflow.label_stats")
    @patch("dataeval.bias.Balance")
    @patch("dataeval.bias.Diversity")
    def test_configured_ontology_is_used(
        self,
        mock_diversity: MagicMock,
        mock_balance: MagicMock,
        mock_label_stats: MagicMock,
        mock_get_metadata: MagicMock,
    ) -> None:
        mock_get_metadata.return_value = _make_metadata(100)
        mock_label_stats.return_value = _make_label_stats()

        workflow = DataCoverageWorkflow()
        context = _make_context()
        params = _make_params(
            run_gap_analysis=False,
            ontology={"animal": {"mammal": ["cat", "dog"], "avian": ["bird"], "reptile": ["snake"]}},
        )
        result = workflow.execute(context, params)

        onto = result.data.raw.ontology
        assert onto is not None
        assert onto.synthesized is False
        assert onto.source == "inline"
        # snake is sanctioned but never collected.
        assert any(row.concept == "snake" and row.action == "acquire" for row in onto.representation.worklist)
        assert [b.concept for b in onto.representation.dark_branches] == ["reptile"]

    @patch("dataeval_flow.workflows.coverage.workflow.get_or_compute_metadata")
    @patch("dataeval_flow.workflows.coverage.workflow.label_stats")
    @patch("dataeval.bias.Balance")
    @patch("dataeval.bias.Diversity")
    def test_bad_ontology_skips_without_failing(
        self,
        mock_diversity: MagicMock,
        mock_balance: MagicMock,
        mock_label_stats: MagicMock,
        mock_get_metadata: MagicMock,
    ) -> None:
        mock_get_metadata.return_value = _make_metadata(100)
        mock_label_stats.return_value = _make_label_stats()

        workflow = DataCoverageWorkflow()
        context = _make_context()
        result = workflow.execute(context, _make_params(run_gap_analysis=False, ontology="does/not/exist.ttl"))

        assert result.success is True
        assert result.data.raw.ontology is None
        assert result.data.raw.ontology_skipped_reason is not None
        assert "does/not/exist.ttl" in result.data.raw.ontology_skipped_reason

    @patch("dataeval_flow.workflows.coverage.ontology.run_ontology_analysis")
    @patch("dataeval_flow.workflows.coverage.workflow.get_or_compute_metadata")
    @patch("dataeval_flow.workflows.coverage.workflow.label_stats")
    @patch("dataeval.bias.Balance")
    @patch("dataeval.bias.Diversity")
    def test_analysis_exception_skips_without_failing(
        self,
        mock_diversity: MagicMock,
        mock_balance: MagicMock,
        mock_label_stats: MagicMock,
        mock_get_metadata: MagicMock,
        mock_run_analysis: MagicMock,
    ) -> None:
        """A raise from run_ontology_analysis (e.g. Representation.evaluate) must not
        abort the whole workflow — it should degrade to ontology_skipped_reason like
        the OntologyLoadError branch above, leaving label/metadata/gap analysis intact.
        """
        mock_get_metadata.return_value = _make_metadata(100)
        mock_label_stats.return_value = _make_label_stats()
        mock_run_analysis.side_effect = RuntimeError("boom")

        workflow = DataCoverageWorkflow()
        context = _make_context()
        result = workflow.execute(context, _make_params(run_gap_analysis=False))

        assert result.success is True
        assert result.data.raw.ontology is None
        assert result.data.raw.ontology_skipped_reason is not None
        assert "boom" in result.data.raw.ontology_skipped_reason

    @patch("dataeval_flow.workflows.coverage.workflow.get_or_compute_metadata")
    @patch("dataeval_flow.workflows.coverage.workflow.label_stats")
    @patch("dataeval.bias.Balance")
    @patch("dataeval.bias.Diversity")
    def test_no_index2label_skips(
        self,
        mock_diversity: MagicMock,
        mock_balance: MagicMock,
        mock_label_stats: MagicMock,
        mock_get_metadata: MagicMock,
    ) -> None:
        mock_get_metadata.return_value = _make_metadata(100)
        mock_label_stats.return_value = _make_label_stats()

        workflow = DataCoverageWorkflow()
        context = _make_context()
        next(iter(context.dataset_contexts.values())).dataset.metadata = {}  # type: ignore
        result = workflow.execute(context, _make_params(run_gap_analysis=False))

        assert result.success is True
        assert result.data.raw.ontology is None
        assert "index2label" in (result.data.raw.ontology_skipped_reason or "")
