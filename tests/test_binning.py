"""Tests for the metadata factor and binning record."""

import itertools
import json
import logging
from typing import Any

import numpy as np
import pytest
from dataeval import Metadata

from dataeval_flow._logging import capture_diagnostics
from dataeval_flow.binning import attach_binning, describe_binning, mi_discrete_features
from dataeval_flow.config.schemas._metadata import ResultMetadata
from dataeval_flow.workflow.base import MetadataConfigMixin


def _metadata(**kwargs: Any) -> Metadata:
    """A small Metadata with one continuous, one categorical, one discrete factor."""
    rng = np.random.default_rng(0)
    n = 60
    return Metadata.from_factors(
        {
            "elevation": rng.normal(100.0, 25.0, n),
            "sensor": rng.choice(["a", "b", "c"], n),
            "count": rng.integers(0, 4, n),
        },
        **kwargs,
    )


class _Params(MetadataConfigMixin):
    """Minimal params carrier for attach_binning."""


class TestDescribeBinning:
    def test_records_factor_type_and_level(self):
        record = describe_binning(_metadata())
        assert record["factors"]["elevation"]["type"] == "continuous"
        assert record["factors"]["sensor"]["type"] == "categorical"
        # v1.1 metadata levels, not the retired "image"/"target" pair
        assert record["factors"]["elevation"]["level"] == "unit"

    def test_records_bin_ranges_for_continuous_factor(self):
        record = describe_binning(_metadata(continuous_factor_bins={"elevation": 4}))
        entry = record["factors"]["elevation"]

        assert entry["is_binned"] is True
        assert entry["bin_count"] == 4
        assert entry["bins_requested"] == 4

        bins = entry["bins"]
        assert sum(b["count"] for b in bins) == 60
        # Bins partition the line, so each one's range sits above the last.
        for lower, upper in itertools.pairwise(bins):
            assert lower["max"] <= upper["min"]

    def test_records_category_values_for_digitized_factor(self):
        record = describe_binning(_metadata())
        entry = record["factors"]["sensor"]

        assert entry["is_digitized"] is True
        assert entry["category_count"] == 3
        assert {c["value"] for c in entry["categories"]} == {"a", "b", "c"}
        assert sum(c["count"] for c in entry["categories"]) == 60

    def test_records_auto_bin_method_when_not_requested(self):
        record = describe_binning(_metadata(auto_bin_method="uniform_count"))
        assert record["auto_bin_method"] == "uniform_count"
        # A factor binned automatically names the method that placed its edges.
        assert record["factors"]["elevation"]["binned_by"] == "uniform_count"
        assert "bins_requested" not in record["factors"]["elevation"]

    def test_records_exclusions(self):
        record = describe_binning(_metadata(), excluded=["id", "width"])
        assert record["excluded"] == ["id", "width"]

    def test_flags_bin_request_naming_an_absent_factor(self):
        record = describe_binning(
            _metadata(),
            requested_bins={"elevation": 4, "nonexistent": 3},
        )
        assert record["unmatched_bin_requests"] == ["nonexistent"]

    def test_record_is_json_serializable(self):
        record = describe_binning(_metadata(continuous_factor_bins={"elevation": 4}))
        # The envelope is written as JSON, so numpy scalars must already be gone.
        assert json.loads(json.dumps(record)) == record

    def test_missing_companion_column_degrades_to_no_detail(self, monkeypatch: pytest.MonkeyPatch):
        """An upstream rename costs the per-bin detail, not the record."""
        monkeypatch.setattr("dataeval_flow.binning._BINNED_SUFFIX", "~renamed~")
        record = describe_binning(_metadata(continuous_factor_bins={"elevation": 4}))
        entry = record["factors"]["elevation"]

        assert entry["is_binned"] is True  # still stated
        assert "bins" not in entry  # detail simply absent


class TestAttachBinning:
    def test_attaches_single_record(self):
        meta = ResultMetadata()
        attach_binning(meta, _metadata(), _Params())
        assert meta.metadata_binning is not None
        assert "elevation" in meta.metadata_binning["factors"]

    def test_attaches_per_split_record(self):
        meta = ResultMetadata()
        attach_binning(meta, {"train": _metadata(), "test": _metadata()}, _Params())
        assert meta.metadata_binning is not None
        assert set(meta.metadata_binning["per_split"]) == {"train", "test"}

    def test_forwards_configured_exclusions_and_bins(self):
        meta = ResultMetadata()
        params = _Params(metadata_exclude=["id"], metadata_continuous_factor_bins={"elevation": 5})
        attach_binning(meta, _metadata(continuous_factor_bins={"elevation": 5}), params)
        assert meta.metadata_binning is not None
        assert meta.metadata_binning["excluded"] == ["id"]
        assert meta.metadata_binning["factors"]["elevation"]["bins_requested"] == 5

    def test_never_raises(self, caplog: pytest.LogCaptureFixture):
        """A broken Metadata costs the record, not the run."""
        meta = ResultMetadata()
        broken = object()
        with caplog.at_level(logging.WARNING):
            attach_binning(meta, broken, _Params())  # type: ignore[arg-type]
        assert meta.metadata_binning is None
        assert "Binning record unavailable" in caplog.text


class TestCaptureDiagnostics:
    def test_collects_library_warnings(self):
        with capture_diagnostics() as messages:
            logging.getLogger("dataeval").warning("pass value_range=(low, high)")
        assert any("value_range" in m for m in messages)

    def test_deduplicates_repeated_messages(self):
        with capture_diagnostics() as messages:
            for _ in range(3):
                logging.getLogger("dataeval").warning("same message")
        assert len(messages) == 1

    def test_ignores_records_below_warning(self):
        with capture_diagnostics() as messages:
            logging.getLogger("dataeval").info("routine chatter")
        assert messages == []

    def test_detaches_handler_on_exit(self):
        with capture_diagnostics() as messages:
            pass
        logging.getLogger("dataeval").warning("after the block")
        assert messages == []


class TestMiDiscreteFeatures:
    """What gets handed to mutual_info as discrete_features, per DataEval release."""

    def test_prefers_is_binned_when_present(self):
        """rc5 onward: the flag is only an entropy ceiling, so a binned factor is False."""

        class _Meta:
            is_binned = [True, False, True]
            is_discrete = [False, True, True]  # the wrong question — must not be read
            factor_names = ["a", "b", "c"]

        assert mi_discrete_features(_Meta()) == [False, True, False]  # type: ignore[arg-type]

    def test_falls_back_to_all_true(self):
        """rc4: the flag also picks the estimator, and factor_data is codes throughout.

        This is what rc4's Balance passes. Reading is_discrete here would mark a binned
        continuous factor continuous and move its class-to-factor score away from the
        value Balance reports for the same data.
        """

        class _Meta:
            is_discrete = [False, True, True]
            factor_names = ["a", "b", "c"]

        assert mi_discrete_features(_Meta()) == [True, True, True]  # type: ignore[arg-type]

    def test_non_iterable_is_binned_falls_back(self):
        class _Meta:
            is_binned = object()
            factor_names = ["a", "b"]

        assert mi_discrete_features(_Meta()) == [True, True]  # type: ignore[arg-type]

    def test_empty_when_there_are_no_factors(self):
        class _Meta:
            factor_names: list[str] = []

        assert mi_discrete_features(_Meta()) == []  # type: ignore[arg-type]


class TestAgreesWithBalance:
    """The fallback exists to match Balance; assert that on the installed release."""

    def test_class_to_factor_matches_balance(self):
        import warnings

        from dataeval.bias import Balance
        from dataeval.config import set_seed
        from dataeval.core import mutual_info

        set_seed(42)
        rng = np.random.default_rng(0)
        n = 600
        labels = rng.integers(0, 3, n)
        md = Metadata.from_factors(
            {
                "elevation": labels * 50.0 + rng.normal(0, 8, n),  # continuous -> binned
                "sensor": np.where(labels == 0, "a", rng.choice(list("bc"), n)),
                "rating": rng.integers(0, 5, n),
            },
            class_labels=labels,
        )
        names = list(md.factor_names)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            balance = {row["factor_name"]: float(row["mi_value"]) for row in Balance().evaluate(md).balance.to_dicts()}
            result = mutual_info(md.class_labels, md.factor_data, mi_discrete_features(md))

        class_to_factor = result["class_to_factor"]
        for i, name in enumerate(names):
            assert balance[name] == pytest.approx(float(class_to_factor[i + 1]), abs=1e-9), (
                f"{name}: coverage's own MI diverged from Balance's"
            )
