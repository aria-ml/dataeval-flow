"""The metadata policy: resolved from the config, and checked before the data is read.

Every check here exists to fail early.  A descriptor that does not exist, a factor declared
through two channels, a vocabulary closed over levels nobody reviewed — each of them either
fails halfway through a run or, worse, succeeds while doing something other than what was
asked.  Catching them at config time costs a message instead of an hour.
"""

import json
from pathlib import Path

import pytest

from dataeval_flow.config._models import PipelineConfig
from dataeval_flow.policy import ResolvedPolicy, policy_for, policy_key, resolve_policy
from dataeval_flow.workflow.base import MetadataConfigMixin


def _descriptor(tmp_path: Path, factors: dict, name: str = "policy.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"version": 1, "factors": factors}), encoding="utf-8")
    return path


_DECLARED_BINS = {"temp_c": {"kind": "bins", "edges": ["-inf", 0.0, "inf"], "provenance": "edges", "method": None}}
_DERIVED_LEVELS = {"weather": {"kind": "levels", "levels": ["sun", "rain"], "provenance": "derived"}}
_REVIEWED_LEVELS = {"weather": {"kind": "levels", "levels": ["sun", "rain"], "provenance": "accepted"}}


def _config(**policy) -> PipelineConfig:
    return PipelineConfig.model_validate({"metadata": [{"name": "standard", **policy}]})


class TestResolvingTheOlderSpelling:
    """The per-workflow `metadata_*` fields keep working."""

    def test_reads_them_when_no_policy_is_named(self):
        params = MetadataConfigMixin(
            metadata_auto_bin_method="clusters",
            metadata_exclude=["id"],
            metadata_continuous_factor_bins={"temp_c": [0.0, 1.0]},
            metadata_factor_source="coded",
        )
        policy = resolve_policy(params)

        assert policy.auto_bin_method == "clusters"
        assert policy.exclude == ("id",)
        assert policy.continuous_factor_bins == {"temp_c": [0.0, 1.0]}
        assert policy.factor_source == "coded"

    def test_an_empty_workflow_resolves_to_defaults(self):
        assert resolve_policy(MetadataConfigMixin()) == ResolvedPolicy()


class TestResolvingANamedPolicy:
    def test_reads_the_pool_entry(self):
        config = _config(auto_bin_method="uniform_count", exclude=["id"], strict=False, factor_source="values")
        policy = resolve_policy(MetadataConfigMixin(metadata="standard"), config)

        assert policy.auto_bin_method == "uniform_count"
        assert policy.exclude == ("id",)
        assert policy.factor_source == "values"

    def test_an_unknown_name_is_refused(self):
        config = _config()
        with pytest.raises(ValueError, match="Unknown metadata policy"):
            resolve_policy(MetadataConfigMixin(metadata="nope"), config)

    def test_a_reference_without_a_pool_is_refused(self):
        with pytest.raises(ValueError, match="Unknown metadata policy|No metadata policy"):
            resolve_policy(MetadataConfigMixin(metadata="standard"), PipelineConfig())

    def test_a_reference_needs_a_config_to_resolve_against(self):
        with pytest.raises(ValueError, match="can only be resolved against a pipeline config"):
            resolve_policy(MetadataConfigMixin(metadata="standard"))

    def test_naming_a_policy_and_the_older_fields_is_refused(self):
        """Two sources disagreeing about one factor has no good resolution."""
        params = MetadataConfigMixin(metadata="standard", metadata_auto_bin_method="clusters")
        with pytest.raises(ValueError, match="also sets"):
            resolve_policy(params, _config())

    def test_an_untouched_legacy_field_does_not_trip_the_check(self):
        """`metadata_exclude` defaults to an empty list, which is not somebody setting it."""
        policy = resolve_policy(MetadataConfigMixin(metadata="standard"), _config(exclude=["id"]))
        assert policy.exclude == ("id",)


class TestApplyingADescriptor:
    def test_reads_the_factors_and_keeps_the_path(self, tmp_path: Path):
        path = _descriptor(tmp_path, _DECLARED_BINS)
        policy = resolve_policy(MetadataConfigMixin(metadata="standard"), _config(encoding=path.name), tmp_path)

        assert policy.encoding is not None
        assert set(policy.encoding) == {"temp_c"}
        # The path is what DataEval is handed: it owns the format and reads it itself.
        assert policy.metadata_kwargs()["encoding"] == path

    def test_a_missing_descriptor_is_refused(self, tmp_path: Path):
        """A descriptor that matches nothing is not a no-op — it is silent drift."""
        config = _config(encoding="absent.json")
        with pytest.raises(ValueError, match="does not exist"):
            resolve_policy(MetadataConfigMixin(metadata="standard"), config, tmp_path)

    def test_an_unreadable_descriptor_is_refused(self, tmp_path: Path):
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not readable JSON"):
            resolve_policy(MetadataConfigMixin(metadata="standard"), _config(encoding="broken.json"), tmp_path)

    def test_a_descriptor_without_factors_is_refused(self, tmp_path: Path):
        (tmp_path / "empty.json").write_text(json.dumps({"version": 1}), encoding="utf-8")
        with pytest.raises(ValueError, match="no 'factors' member"):
            resolve_policy(MetadataConfigMixin(metadata="standard"), _config(encoding="empty.json"), tmp_path)

    def test_declaring_one_factor_twice_is_refused(self, tmp_path: Path):
        _descriptor(tmp_path, _DECLARED_BINS)
        config = _config(encoding="policy.json", continuous_factor_bins={"temp_c": 4})
        with pytest.raises(ValueError, match="both `encoding` and `continuous_factor_bins`"):
            resolve_policy(MetadataConfigMixin(metadata="standard"), config, tmp_path)

    def test_declaring_different_factors_through_each_channel_is_fine(self, tmp_path: Path):
        _descriptor(tmp_path, _DECLARED_BINS)
        config = _config(encoding="policy.json", continuous_factor_bins={"elevation": 4})
        policy = resolve_policy(MetadataConfigMixin(metadata="standard"), config, tmp_path)
        assert set(policy.continuous_factor_bins) == {"elevation"}

    def test_an_absolute_descriptor_path_is_refused_by_the_schema(self):
        with pytest.raises(ValueError, match="must be relative"):
            _config(encoding="/etc/policy.json")


class TestStrictMustBeEarned:
    """`strict` does not consult provenance, so flow refuses to close what nobody reviewed."""

    def test_closing_a_derived_vocabulary_is_refused(self, tmp_path: Path):
        _descriptor(tmp_path, _DERIVED_LEVELS)
        config = _config(encoding="policy.json", strict=True)
        with pytest.raises(ValueError, match='still read provenance="derived"'):
            resolve_policy(MetadataConfigMixin(metadata="standard"), config, tmp_path)

    def test_closing_a_reviewed_vocabulary_is_allowed(self, tmp_path: Path):
        _descriptor(tmp_path, _REVIEWED_LEVELS)
        config = _config(encoding="policy.json", strict=True)
        policy = resolve_policy(MetadataConfigMixin(metadata="standard"), config, tmp_path)

        assert policy.strict is True
        assert policy.metadata_kwargs()["strict"] is True

    def test_a_derived_bin_spec_does_not_block_strict(self, tmp_path: Path):
        """Bin edges are unaffected by strict — an unseen magnitude lands in an end bin."""
        _descriptor(tmp_path, {**_DECLARED_BINS, **_REVIEWED_LEVELS})
        config = _config(encoding="policy.json", strict=True)
        assert resolve_policy(MetadataConfigMixin(metadata="standard"), config, tmp_path).strict is True

    def test_strict_without_a_descriptor_is_left_alone(self):
        policy = resolve_policy(MetadataConfigMixin(metadata="standard"), _config(strict=True))
        assert policy.strict is True


class TestPolicyKey:
    """The key decides which cache entry a run gets, so equal policies must key equally."""

    def test_the_same_policy_spelled_differently_keys_the_same(self):
        a = ResolvedPolicy(exclude=("a", "b"), continuous_factor_bins={"x": 4})
        b = ResolvedPolicy(exclude=("b", "a"), continuous_factor_bins={"x": 4})
        assert policy_key(a) == policy_key(b)

    def test_different_cuts_key_differently(self):
        a = ResolvedPolicy(continuous_factor_bins={"x": 4})
        b = ResolvedPolicy(continuous_factor_bins={"x": 8})
        assert policy_key(a) != policy_key(b)

    def test_the_descriptor_contents_are_in_the_key(self):
        """A descriptor edited in place under one path is a different policy."""
        a = ResolvedPolicy(encoding={"t": {"kind": "bins", "edges": ["-inf", 0.0, "inf"]}})
        b = ResolvedPolicy(encoding={"t": {"kind": "bins", "edges": ["-inf", 5.0, "inf"]}})
        assert policy_key(a) != policy_key(b)

    def test_strict_is_in_the_key(self):
        assert policy_key(ResolvedPolicy(strict=True)) != policy_key(ResolvedPolicy(strict=False))


class TestPolicyFor:
    """A workflow invoked directly still honours its own configured cuts."""

    def test_prefers_the_resolved_policy_on_the_context(self):
        class _Ctx:
            metadata_policy = ResolvedPolicy(auto_bin_method="clusters")

        params = MetadataConfigMixin(metadata_auto_bin_method="uniform_count")
        assert policy_for(_Ctx(), params).auto_bin_method == "clusters"

    def test_falls_back_to_the_parameters(self):
        class _Ctx:
            metadata_policy = None

        params = MetadataConfigMixin(metadata_auto_bin_method="uniform_count")
        assert policy_for(_Ctx(), params).auto_bin_method == "uniform_count"
