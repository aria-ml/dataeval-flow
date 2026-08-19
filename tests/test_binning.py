"""Tests for the metadata factor and binning record."""

import itertools
import json
import logging
from typing import Any

import numpy as np
import pytest
from dataeval import Metadata

from dataeval_flow._logging import capture_diagnostics
from dataeval_flow.binning import attach_binning, describe_binning
from dataeval_flow.config.schemas._metadata import ResultMetadata
from dataeval_flow.workflow.base import MetadataConfigMixin


def _metadata(n: int = 60, **kwargs: Any) -> Metadata:
    """A small Metadata with one continuous, one categorical, one discrete factor."""
    rng = np.random.default_rng(0)
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

    def test_records_the_cut_a_count_request_resolved_to(self):
        """A count says how many, not where. Where used to be derived and discarded."""
        record = describe_binning(_metadata(continuous_factor_bins={"elevation": 4}))
        entry = record["factors"]["elevation"]

        assert entry["encoding"]["kind"] == "bins"
        assert entry["encoding"]["provenance"] == "count"
        # Four intervals means five edges, the outer two open.
        edges = entry["encoding"]["edges"]
        assert len(edges) == 5
        assert (edges[0], edges[-1]) == ("-inf", "inf")
        assert all(isinstance(edge, float) for edge in edges[1:-1])

        bins = entry["fit"]["bins"]
        assert sum(b["count"] for b in bins) == 60
        # Bins partition the line, so each one's occupied span sits above the last.
        for lower, upper in itertools.pairwise(bins):
            assert lower["max"] <= upper["min"]

    def test_declared_edges_survive_verbatim(self):
        """The defect this replaces: a declared cutoff never reached its own record."""
        record = describe_binning(_metadata(continuous_factor_bins={"elevation": [-np.inf, 0.0, np.inf]}))
        entry = record["factors"]["elevation"]

        assert entry["encoding"]["provenance"] == "edges"
        assert entry["encoding"]["edges"] == ["-inf", 0.0, "inf"]
        assert entry["encoding"]["method"] is None

    def test_reports_declared_bins_nothing_reached(self):
        """Occupancy is a question about fit, and an empty declared bin is the answer."""
        record = describe_binning(_metadata(continuous_factor_bins={"elevation": [-np.inf, 0.0, 1.0, np.inf]}))
        fit = record["factors"]["elevation"]["fit"]

        # Every value is a positive elevation, so the two low bins hold nothing.
        assert fit["empty"] == [1, 2]
        assert sum(b["count"] for b in fit["bins"]) == 60

    def test_records_the_vocabulary_for_a_digitized_factor(self):
        record = describe_binning(_metadata())
        entry = record["factors"]["sensor"]

        assert entry["encoding"]["kind"] == "levels"
        assert entry["encoding"]["levels"] == ["a", "b", "c"]
        assert {level["value"] for level in entry["fit"]["levels"]} == {"a", "b", "c"}
        assert sum(level["count"] for level in entry["fit"]["levels"]) == 60

    def test_records_the_names_dataeval_gives_each_code(self):
        """Names travel with the record so an archived result re-renders identically.

        Asked of DataEval rather than derived from the edges, because these are the strings
        its own outputs use — `ParityOutput.insufficient_data` keys and `label=` axis
        groups — and a second renderer would disagree with them.
        """
        record = describe_binning(_metadata(continuous_factor_bins={"elevation": [-np.inf, 0.0, np.inf]}))
        assert record["factors"]["elevation"]["names"] == {"1": "< 0", "2": ">= 0"}
        assert record["factors"]["sensor"]["names"] == {"0": "a", "1": "b", "2": "c"}

    def test_names_cover_declared_bins_nothing_reached(self):
        """An empty bin is a finding, and reporting one means naming it."""
        record = describe_binning(_metadata(continuous_factor_bins={"elevation": [-np.inf, 0.0, 1.0, np.inf]}))
        entry = record["factors"]["elevation"]

        assert entry["fit"]["empty"] == [1, 2]
        assert set(entry["names"]) >= {"1", "2"}

    def test_names_keep_large_magnitudes_distinct(self):
        """Six significant figures collapses epoch-millisecond bins onto one label."""
        rng = np.random.default_rng(3)
        base = 1787011240000000
        md = Metadata.from_factors(
            {"capture_ms": (base + rng.integers(0, 1_200_000, 200)).astype(float)},
            continuous_factor_bins={"capture_ms": 6},
        )
        names = describe_binning(md)["factors"]["capture_ms"]["names"]
        assert len(set(names.values())) == len(names)

    def test_records_auto_bin_method_when_not_requested(self):
        record = describe_binning(_metadata(auto_bin_method="uniform_count"))
        assert record["auto_bin_method"] == "uniform_count"
        # A factor nobody declared names the method that placed its edges, and says so.
        encoding = record["factors"]["elevation"]["encoding"]
        assert encoding["provenance"] == "derived"
        assert encoding["method"] == "uniform_count"

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

    def test_missing_companion_column_costs_the_fit_and_keeps_the_policy(self, monkeypatch: pytest.MonkeyPatch):
        """An upstream rename costs the observation, not the decision.

        The suffixes are mirrored from a private module. They used to carry the record
        itself, so a rename lost it; now only occupancy reads them, and the policy comes
        from the public record.
        """
        monkeypatch.setattr("dataeval_flow.binning._BINNED_SUFFIX", "~renamed~")
        record = describe_binning(_metadata(continuous_factor_bins={"elevation": 4}))
        entry = record["factors"]["elevation"]

        assert entry["encoding"]["provenance"] == "count"  # the policy survives
        assert "fit" not in entry  # the observation simply absent


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
        # What was asked for, alongside what it resolved to.
        assert meta.metadata_binning["requested_bins"] == {"elevation": 5}
        assert meta.metadata_binning["factors"]["elevation"]["encoding"]["provenance"] == "count"

    def test_never_raises(self, caplog: pytest.LogCaptureFixture):
        """A broken Metadata costs the record, not the run."""
        meta = ResultMetadata()
        broken = object()
        with caplog.at_level(logging.WARNING):
            attach_binning(meta, broken, _Params())  # type: ignore[arg-type]
        assert meta.metadata_binning is None
        assert "Binning record unavailable" in caplog.text


class TestEncodingDigest:
    """A result that cannot say which cuts produced it cannot be compared with another."""

    def test_single_metadata_stamps_its_digest(self):
        meta = ResultMetadata()
        attach_binning(meta, _metadata(), MetadataConfigMixin())
        assert meta.metadata_binning is not None
        assert meta.encoding_digest
        assert meta.encoding_digest == meta.metadata_binning["encoding_digest"]

    def test_same_encoding_over_different_rows_keeps_one_digest(self):
        """The digest covers the policy, so it does not move when only the data does."""
        declared = {"elevation": [-np.inf, 90.0, 110.0, np.inf]}
        first, second = ResultMetadata(), ResultMetadata()
        attach_binning(first, _metadata(continuous_factor_bins=declared), MetadataConfigMixin())
        attach_binning(second, _metadata(n=120, continuous_factor_bins=declared), MetadataConfigMixin())

        assert first.encoding_digest == second.encoding_digest

    def test_a_declared_cut_changes_the_digest(self):
        plain, declared = ResultMetadata(), ResultMetadata()
        attach_binning(plain, _metadata(), MetadataConfigMixin())
        attach_binning(
            declared,
            _metadata(continuous_factor_bins={"elevation": [-np.inf, 0.0, np.inf]}),
            MetadataConfigMixin(),
        )
        assert plain.encoding_digest != declared.encoding_digest

    def test_splits_sharing_an_encoding_stamp_it_once(self):
        declared = {"elevation": [-np.inf, 90.0, 110.0, np.inf]}
        meta = ResultMetadata()
        attach_binning(
            meta,
            {
                "train": _metadata(continuous_factor_bins=declared),
                "test": _metadata(n=120, continuous_factor_bins=declared),
            },
            MetadataConfigMixin(),
        )
        assert meta.encoding_digest

    def test_splits_encoded_differently_stamp_nothing(self):
        """There is no single encoding to name, and saying one would be a lie."""
        meta = ResultMetadata()
        attach_binning(
            meta,
            {
                "train": _metadata(continuous_factor_bins={"elevation": [-np.inf, 0.0, np.inf]}),
                "test": _metadata(continuous_factor_bins={"elevation": [-np.inf, 50.0, np.inf]}),
            },
            MetadataConfigMixin(),
        )
        assert meta.encoding_digest is None
        # The per-split records still say what each one ran under.
        assert meta.metadata_binning is not None
        per_split = meta.metadata_binning["per_split"]
        assert per_split["train"]["encoding_digest"] != per_split["test"]["encoding_digest"]


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


class TestCapturesLibraryWarnings:
    """DataEval's most actionable advice is a warning, not a log record.

    A ``NullHandler`` on the ``dataeval`` root logger means log records reach only
    callers who configured logging, so the aggregated binning advice and the whole
    encoding-fit report are raised with ``warnings.warn``.  A handler-only collector
    archived the per-factor footnotes and dropped every one of those findings.
    """

    @staticmethod
    def _auto_binned() -> Metadata:
        md = _metadata()
        md.factor_info  # noqa: B018 - forces binning, which is what announces
        return md

    def test_collects_the_auto_binning_advice(self):
        with capture_diagnostics() as messages:
            self._auto_binned()
        assert any("binned automatically" in m for m in messages), messages

    def test_collects_the_occupancy_report(self):
        """A declared cut that no longer fits is stage 7's whole deliverable."""
        with capture_diagnostics() as messages:
            md = Metadata.from_factors(
                {"elevation": np.full(60, 100.0) + np.arange(60) * 1e-3},
                continuous_factor_bins={"elevation": [-np.inf, 0.0, 1.0, np.inf]},
            )
            md.factor_info  # noqa: B018

        assert any("left bins unused" in m for m in messages), messages

    def test_survives_the_warn_once_registry(self):
        """Two workflows through one call site must each be told, not just the first.

        ``warnings.warn`` keys its bookkeeping on the frame ``stacklevel`` selects, and
        DataEval points that at its caller — this package. Without an "always" filter the
        second workflow's diagnostic is suppressed before anything can record it.
        """
        with capture_diagnostics() as first:
            self._auto_binned()
        with capture_diagnostics() as second:
            self._auto_binned()

        assert any("binned automatically" in m for m in first), first
        assert any("binned automatically" in m for m in second), second

    def test_deduplicates_within_one_block(self):
        with capture_diagnostics() as messages:
            self._auto_binned()
            self._auto_binned()
        assert sum("binned automatically" in m for m in messages) == 1, messages

    def test_ignores_warnings_from_elsewhere(self):
        import warnings

        with capture_diagnostics() as messages:
            warnings.warn("not a library decision", UserWarning, stacklevel=1)
        assert messages == []

    def test_restores_the_warning_hook_on_exit(self):
        import warnings

        before = warnings.showwarning
        with capture_diagnostics():
            pass
        assert warnings.showwarning is before


class TestGapMiAgreesWithBalance:
    """Gap analysis compares against `gap_mi_threshold`, so its MI must be Balance's.

    A boolean `discrete_features` used to be able to imitate Balance. It cannot any more:
    `factor_source` decides per factor whether the codes or the measured values are read,
    consulting the encoding record's provenance, and that is two estimators chosen per
    column rather than one flag per column.
    """

    @staticmethod
    def _metadata_with_signal() -> Metadata:
        from dataeval.config import set_seed

        set_seed(42)
        rng = np.random.default_rng(0)
        n = 600
        labels = rng.integers(0, 3, n)
        return Metadata.from_factors(
            {
                "elevation": labels * 50.0 + rng.normal(0, 8, n),  # continuous -> binned
                "sensor": np.where(labels == 0, "a", rng.choice(list("bc"), n)),
                "rating": rng.integers(0, 5, n),
            },
            class_labels=labels,
        )

    @pytest.mark.parametrize("factor_source", [None, "auto", "coded", "values"])
    def test_matches_balance_under_every_factor_source(self, factor_source):
        import warnings

        from dataeval.bias import Balance

        from dataeval_flow.workflows.coverage.workflow import _balance_class_to_factor

        md = self._metadata_with_signal()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = {
                row["factor_name"]: float(row["mi_value"])
                for row in Balance(factor_source=factor_source).evaluate(md).balance.to_dicts()
            }
            actual = _balance_class_to_factor(md, factor_source)

        assert actual.keys() >= set(md.factor_names)
        for name in md.factor_names:
            assert actual[name] == pytest.approx(expected[name], abs=1e-9), (
                f"{name}: coverage's MI diverged from Balance's under factor_source={factor_source!r}"
            )

    def test_reuses_a_balance_result_rather_than_recomputing(self):
        """The precomputed path and the computed path must agree, or reuse changes numbers."""
        import warnings

        from dataeval_flow.workflows.coverage.workflow import _balance_class_to_factor, _mi_from_balance

        md = self._metadata_with_signal()
        names = list(md.factor_names)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from dataeval.bias import Balance

            summary = {"balance": Balance().evaluate(md).balance.to_dicts()}
            computed = _balance_class_to_factor(md, None)

        reused = _mi_from_balance(summary, names)
        assert reused is not None
        for name in names:
            assert reused[name] == pytest.approx(computed[name], abs=1e-9)


class TestRecordsFactorSource:
    """`factor_source` moves every bias number and leaves no other trace in the result."""

    def test_records_the_configured_source(self):
        record = describe_binning(_metadata(), factor_source="coded")
        assert record["factor_source"] == "coded"

    def test_records_the_default_when_unset(self):
        record = describe_binning(_metadata())
        # Whatever the installed release defaults to -- read off the evaluator, not assumed.
        from dataeval.bias import Balance

        assert record["factor_source"] == Balance().factor_source

    def test_attach_carries_it_from_params(self):
        result_metadata = ResultMetadata()
        params = MetadataConfigMixin(metadata_factor_source="values")
        attach_binning(result_metadata, _metadata(), params)
        assert result_metadata.metadata_binning is not None
        assert result_metadata.metadata_binning["factor_source"] == "values"
