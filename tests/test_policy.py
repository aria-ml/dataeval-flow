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

    def test_strict_declaring_a_vocabulary_directly_is_allowed(self):
        """`factor_levels` is a declaration, so there is something strict can honestly close."""
        config = _config(strict=True, factor_levels={"weather": ["clear", "rain"]})
        policy = resolve_policy(MetadataConfigMixin(metadata="standard"), config)
        assert policy.strict is True

    def test_strict_declaring_nothing_at_all_is_refused(self):
        """The same rule as a derived descriptor, for the spelling that names no descriptor.

        With neither channel set, every vocabulary strict closes is one DataEval cut from
        this draw — the state `_check_strict_is_earned` exists to refuse. Left alone, it
        fails the run on the first category the sample happened to miss.
        """
        with pytest.raises(ValueError, match="declares no vocabulary"):
            resolve_policy(MetadataConfigMixin(metadata="standard"), _config(strict=True))


class TestDoubleDeclaration:
    """A factor declared through two channels is refused before the data is read."""

    def test_encoding_and_bins_conflict(self, tmp_path: Path):
        _descriptor(tmp_path, _DECLARED_BINS)
        config = _config(encoding="policy.json", continuous_factor_bins={"temp_c": 4})
        with pytest.raises(ValueError, match="both `encoding` and `continuous_factor_bins`"):
            resolve_policy(MetadataConfigMixin(metadata="standard"), config, tmp_path)

    def test_encoding_and_factor_levels_conflict(self, tmp_path: Path):
        """DataEval refuses this too, halfway through a run. Refused here instead."""
        _descriptor(tmp_path, _REVIEWED_LEVELS)
        config = _config(encoding="policy.json", factor_levels={"weather": ["clear", "rain"]})
        with pytest.raises(ValueError, match="both `encoding` and `factor_levels`"):
            resolve_policy(MetadataConfigMixin(metadata="standard"), config, tmp_path)

    def test_bins_and_factor_levels_conflict(self):
        """Refused with no descriptor in play at all — the check is not gated on one."""
        config = _config(continuous_factor_bins={"temp_c": 4}, factor_levels={"temp_c": [1, 2]})
        with pytest.raises(ValueError, match="both `continuous_factor_bins` and `factor_levels`"):
            resolve_policy(MetadataConfigMixin(metadata="standard"), config)

    def test_encoding_and_bins_conflict_through_a_level_prefix(self, tmp_path: Path):
        """The same collision, spelled the way injection actually produces it.

        A committed `unit_brightness` record and a bare `continuous_factor_bins:
        {"brightness": ...}` look unrelated by name, but `expand_declared_bins` carries the
        bare declaration onto the level-prefixed name before `Metadata` sees it — so this is
        the same factor declared twice, not two different ones, and must be refused exactly
        like the exact-name collision above rather than letting the bare declaration
        silently overwrite the committed record. `intrinsic_factors` must be declared for
        the collision to exist at all: `brightness` is a VISUAL statistic, so injection can
        only produce `unit_brightness` from it when `visual` is one of the families asked
        for — see `test_prefix_collision_does_not_fire_without_injection_declared` for the
        other half of that gate.
        """
        _descriptor(tmp_path, {"unit_brightness": _DECLARED_BINS["temp_c"]})
        config = _config(
            encoding="policy.json",
            continuous_factor_bins={"brightness": 4},
            intrinsic_factors=["visual"],
        )
        with pytest.raises(ValueError, match="both `encoding` and `continuous_factor_bins`") as exc:
            resolve_policy(MetadataConfigMixin(metadata="standard"), config, tmp_path)
        message = str(exc.value)
        assert "brightness" in message
        assert "unit_brightness" in message

    def test_a_level_prefixed_descriptor_factor_with_no_bare_match_is_not_a_conflict(self, tmp_path: Path):
        """The prefix check is scoped to an actual collision, not any level-prefixed name."""
        _descriptor(tmp_path, {"unit_brightness": _DECLARED_BINS["temp_c"]})
        config = _config(
            encoding="policy.json",
            continuous_factor_bins={"contrast": 4},
            intrinsic_factors=["visual"],
        )
        policy = resolve_policy(MetadataConfigMixin(metadata="standard"), config, tmp_path)
        assert policy.continuous_factor_bins == {"contrast": 4}

    def test_prefix_collision_does_not_fire_without_injection_declared(self, tmp_path: Path):
        """The collision depends on injection actually being on, not just on the spelling.

        With no `intrinsic_factors` declared, nothing can produce `unit_brightness` from a
        bare `brightness` declaration — so the two are just two differently-spelled,
        unrelated names, and the same setup that raises in
        `test_encoding_and_bins_conflict_through_a_level_prefix` must not raise here.
        """
        _descriptor(tmp_path, {"unit_brightness": _DECLARED_BINS["temp_c"]})
        config = _config(encoding="policy.json", continuous_factor_bins={"brightness": 4})
        policy = resolve_policy(MetadataConfigMixin(metadata="standard"), config, tmp_path)
        assert policy.continuous_factor_bins == {"brightness": 4}

    def test_unrelated_level_prefixed_column_is_not_a_conflict(self, tmp_path: Path):
        """A dataset-native column that merely looks level-prefixed is not a collision.

        `unit_price` is an ordinary column name, not something `add_factors` ever produces
        by splitting a statistic — no family injects a `price` statistic, `visual` included
        — so a bare `continuous_factor_bins: {"price": ...}` declaration is unrelated to it
        even though the two spellings share the `unit_` prefix.
        """
        _descriptor(tmp_path, {"unit_price": _DECLARED_BINS["temp_c"]})
        config = _config(
            encoding="policy.json",
            continuous_factor_bins={"price": 4},
            intrinsic_factors=["visual"],
        )
        policy = resolve_policy(MetadataConfigMixin(metadata="standard"), config, tmp_path)
        assert policy.continuous_factor_bins == {"price": 4}

    def test_prefix_collision_is_scoped_to_the_declared_family(self, tmp_path: Path):
        """A name only collides if the *declared* families could actually produce it.

        `brightness` is a VISUAL statistic, not a DIMENSION one, so declaring only
        `dimension` cannot make injection produce `unit_brightness` — the collision must not
        fire just because *some* family somewhere injects `brightness`.
        """
        _descriptor(tmp_path, {"unit_brightness": _DECLARED_BINS["temp_c"]})
        config = _config(
            encoding="policy.json",
            continuous_factor_bins={"brightness": 4},
            intrinsic_factors=["dimension"],
        )
        policy = resolve_policy(MetadataConfigMixin(metadata="standard"), config, tmp_path)
        assert policy.continuous_factor_bins == {"brightness": 4}


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


class TestDeriveFrom:
    """One split's encoding, applied to the rest, is what makes them comparable."""

    @staticmethod
    def _metadata(seed: int, n: int, **kwargs):
        import numpy as np
        from dataeval import Metadata

        rng = np.random.default_rng(seed)
        return Metadata.from_factors(
            {"elevation": rng.normal(100.0, 25.0, n), "sensor": rng.choice(list("ab"), n)}, **kwargs
        )

    def _derived(self, reference):
        from dataeval_flow.binning import _descriptor
        from dataeval_flow.policy import derive_from

        factors, _ = _descriptor(reference)
        return derive_from(ResolvedPolicy(), reference, factors)

    def test_the_next_dataset_gets_the_reference_cuts(self):
        reference = self._metadata(0, 400)
        derived = self._derived(reference)
        follower = self._metadata(7, 60, **derived.metadata_kwargs())

        assert follower.encoding("elevation").edges == reference.encoding("elevation").edges

    def test_an_independent_split_would_have_cut_differently(self):
        """Without this the two are not comparable, which is the whole point."""
        reference = self._metadata(0, 400)
        independent = self._metadata(7, 60)
        follower = self._metadata(7, 60, **self._derived(reference).metadata_kwargs())

        assert independent.encoding("elevation").edges != reference.encoding("elevation").edges
        assert follower.encoding("elevation").edges == reference.encoding("elevation").edges

    def test_a_new_category_appends_rather_than_renumbering(self):
        import numpy as np
        from dataeval import Metadata

        reference = self._metadata(0, 400)
        derived = self._derived(reference)
        rng = np.random.default_rng(3)
        follower = Metadata.from_factors(
            {"elevation": rng.normal(100.0, 25.0, 60), "sensor": rng.choice(list("abc"), 60)},
            **derived.metadata_kwargs(),
        )

        before = reference.encoding("sensor").levels
        after = follower.encoding("sensor").levels
        assert after[: len(before)] == before  # shared codes keep their meaning
        assert set(after) - set(before) == {"c"}

    def test_the_derived_policy_keys_differently(self):
        """A follower is a different cache entry from a split cut on its own draw."""
        reference = self._metadata(0, 400)
        assert policy_key(self._derived(reference)) != policy_key(ResolvedPolicy())

    def test_declaration_inputs_are_dropped_once_resolved(self):
        """The records already say where every declared cut fell; passing both is refused."""
        reference = self._metadata(0, 400, continuous_factor_bins={"elevation": 4})
        from dataeval_flow.binning import _descriptor
        from dataeval_flow.policy import derive_from

        factors, _ = _descriptor(reference)
        derived = derive_from(ResolvedPolicy(continuous_factor_bins={"elevation": 4}), reference, factors)
        assert derived.continuous_factor_bins == {}
        assert "continuous_factor_bins" not in derived.metadata_kwargs()

    def test_a_container_with_no_record_is_left_alone(self):
        assert self._derived(object()) == ResolvedPolicy()


class TestMetadataKwargsForLoad:
    """`Metadata.load` and `Metadata(...)` do not take the same arguments."""

    def test_factor_levels_is_construction_only(self):
        """Passing it to `load` is a TypeError the cache swallows as a permanent miss."""
        import inspect

        from dataeval import Metadata

        policy = ResolvedPolicy(factor_levels={"weather": ["clear", "rain"]})
        assert "factor_levels" in policy.metadata_kwargs()
        assert "factor_levels" not in policy.metadata_kwargs(for_load=True)
        assert "factor_levels" not in inspect.signature(Metadata.load).parameters

    def test_every_load_kwarg_is_one_load_accepts(self):
        """Derived rather than listed, so a future field cannot reintroduce the same break."""
        import inspect

        from dataeval import Metadata

        policy = ResolvedPolicy(
            auto_bin_method="clusters",
            exclude=("id",),
            continuous_factor_bins={"temp_c": 4},
            factor_levels={"weather": ["clear"]},
            strict=True,
        )
        accepted = set(inspect.signature(Metadata.load).parameters)
        assert set(policy.metadata_kwargs(for_load=True)) <= accepted


class TestDerivedPolicyDoesNotCloseUnreviewedVocabularies:
    """`strict` over cuts DataEval derived from the reference split's draw fails the run."""

    def test_strict_is_dropped_when_a_derived_vocabulary_rides_along(self, caplog):
        import logging as _logging

        import numpy as np
        from dataeval import Metadata

        from dataeval_flow.policy import derive_from

        rng = np.random.default_rng(0)
        reference = Metadata.from_factors(
            {"sensor": rng.choice(["a", "b"], 40), "temp_c": rng.normal(0.0, 1.0, 40)},
        )
        with caplog.at_level(_logging.WARNING, logger="dataeval_flow.policy"):
            derived = derive_from(ResolvedPolicy(strict=True), reference, None)

        assert derived.strict is False
        explained = next(r.getMessage() for r in caplog.records if r.name == "dataeval_flow.policy")
        assert "'sensor'" in explained  # the vocabulary nobody reviewed
        assert "temp_c" not in explained  # a cut, not a vocabulary

    def test_strict_survives_where_every_vocabulary_was_reviewed(self):
        import numpy as np
        from dataeval import Metadata

        from dataeval_flow.policy import derive_from

        rng = np.random.default_rng(0)
        reference = Metadata.from_factors(
            {"sensor": rng.choice(["a", "b"], 40)},
            factor_levels={"sensor": ["a", "b"]},
        )
        derived = derive_from(ResolvedPolicy(strict=True), reference, None)
        assert derived.strict is True


class TestIntrinsicFactors:
    """Which intrinsic statistics become factors is a policy decision, not a workflow param."""

    def test_defaults_to_nothing(self):
        assert resolve_policy(MetadataConfigMixin()).intrinsic_factors == ()

    def test_reads_the_policy_field(self):
        config = _config(intrinsic_factors=["visual", "pixel"])
        resolved = resolve_policy(MetadataConfigMixin(metadata="standard"), config)
        assert resolved.intrinsic_factors == ("visual", "pixel")

    def test_keys_the_policy(self):
        without = ResolvedPolicy()
        with_stats = ResolvedPolicy(intrinsic_factors=("visual",))
        assert policy_key(without) != policy_key(with_stats)

    def test_key_is_order_independent(self):
        # A set and a list of the same families describe one policy and must not
        # produce two cache entries.
        first = ResolvedPolicy(intrinsic_factors=("visual", "pixel"))
        second = ResolvedPolicy(intrinsic_factors=("pixel", "visual"))
        assert policy_key(first) == policy_key(second)

    def test_unknown_family_fails_at_config_time(self):
        config = _config(intrinsic_factors=["nonsense"])
        with pytest.raises(ValueError, match="nonsense"):
            resolve_policy(MetadataConfigMixin(metadata="standard"), config)

    def test_key_is_case_insensitive(self):
        lower = ResolvedPolicy(intrinsic_factors=("visual",))
        upper = ResolvedPolicy(intrinsic_factors=("VISUAL",))
        assert policy_key(lower) == policy_key(upper)


class TestValueRangeOnThePolicy:
    """Authored on the dataset, carried on the policy, because it changes the codes."""

    def test_defaults_to_none(self):
        assert ResolvedPolicy().value_range is None

    def test_keys_the_policy(self):
        assert policy_key(ResolvedPolicy()) != policy_key(ResolvedPolicy(value_range=(0.0, 1.0)))

    def test_different_ranges_key_differently(self):
        first = ResolvedPolicy(value_range=(0.0, 1.0))
        second = ResolvedPolicy(value_range=(0.0, 255.0))
        assert policy_key(first) != policy_key(second)
