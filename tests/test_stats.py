"""The stats policy: resolution, the view namespace, and the column filter."""

import numpy as np
import pytest
from dataeval.flags import ImageStats

from dataeval_flow.stats import (
    ResolvedStatsPolicy,
    check_consumers,
    columns_for,
    measurable_in,
    resolve_stats_policy,
    restrict_columns,
    stats_policy_for,
)


def _policy(**kwargs):
    base = {
        "measure": ((None, ImageStats.VISUAL | ImageStats.HASH), ("ir", ImageStats.PIXEL)),
        "channels": (("ir", (3,)),),
    }
    return ResolvedStatsPolicy(**{**base, **kwargs})


@pytest.mark.required
class TestColumnsFor:
    def test_unprefixed_view_gives_bare_names(self):
        assert columns_for([None], ImageStats.VISUAL) == {
            "brightness",
            "contrast",
            "darkness",
            "sharpness",
            "percentiles",
        }

    def test_named_view_prefixes_every_name(self):
        assert columns_for(["ir"], ImageStats.VISUAL) == {
            "ir_brightness",
            "ir_contrast",
            "ir_darkness",
            "ir_sharpness",
            "ir_percentiles",
        }

    def test_background_view_carries_the_fraction(self):
        assert "background_fraction" in columns_for(["background"], ImageStats.VISUAL)

    def test_a_band_background_does_not_carry_the_fraction(self):
        assert "background_fraction" not in columns_for(["background_ir"], ImageStats.VISUAL)

    def test_no_views_gives_no_columns(self):
        assert columns_for([], ImageStats.ALL) == set()


@pytest.mark.required
class TestMeasurableIn:
    def test_whole_image_carries_everything(self):
        assert measurable_in(None, ImageStats.ALL) == ImageStats.ALL

    def test_a_band_group_drops_geometry(self):
        assert not (measurable_in("ir", ImageStats.ALL) & ImageStats.DIMENSION)
        assert measurable_in("ir", ImageStats.ALL) & ImageStats.HASH

    def test_a_background_view_keeps_only_pixel_and_visual(self):
        kept = measurable_in("background", ImageStats.ALL)
        assert kept == ImageStats.PIXEL | ImageStats.VISUAL
        assert measurable_in("background_ir", ImageStats.ALL) == kept


@pytest.mark.required
class TestRestrictColumns:
    def _result(self):
        return {
            "source_index": [1, 2],
            "object_count": [1, 1],
            "invalid_box_count": [0, 0],
            "image_count": 2,
            "stats": {"mean": np.array([1.0, 2.0]), "brightness": np.array([3.0, 4.0])},
        }

    def test_keeps_only_the_named_columns(self):
        out = restrict_columns(self._result(), {"brightness"})
        assert set(out["stats"]) == {"brightness"}

    def test_carries_the_structural_fields_through(self):
        out = restrict_columns(self._result(), {"brightness"})
        assert out["image_count"] == 2
        assert list(out["source_index"]) == [1, 2]

    def test_does_not_mutate_the_input(self):
        result = self._result()
        restrict_columns(result, set())
        assert set(result["stats"]) == {"mean", "brightness"}

    def test_a_column_that_is_not_there_is_simply_absent(self):
        out = restrict_columns(self._result(), {"brightness", "ir_mean"})
        assert set(out["stats"]) == {"brightness"}


@pytest.mark.required
class TestResolvedStatsPolicy:
    def test_of_flags_is_one_unprefixed_view(self):
        policy = ResolvedStatsPolicy.of_flags(ImageStats.VISUAL)
        assert policy.request == {None: ImageStats.VISUAL}
        assert policy.channel_map is None
        assert policy.background is False

    def test_channel_map_is_none_when_no_group_is_measured(self):
        assert ResolvedStatsPolicy.of_flags(ImageStats.VISUAL).channel_map is None

    def test_channel_map_holds_the_selected_groups(self):
        assert _policy().channel_map == {"ir": [3]}

    def test_scope_fragment_is_empty_for_a_plain_policy(self):
        assert ResolvedStatsPolicy.of_flags(ImageStats.VISUAL).scope_fragment() == ""

    def test_scope_fragment_separates_two_definitions_of_one_name(self):
        a = _policy(channels=(("ir", (3,)),))
        b = _policy(channels=(("ir", (2,)),))
        assert a.scope_fragment() != b.scope_fragment()

    def test_scope_fragment_separates_background_from_none(self):
        assert _policy(background=True).scope_fragment() != _policy(background=False).scope_fragment()

    def test_scope_fragment_ignores_the_consumer_view_sets(self):
        a = _policy(outliers_from=(None,), factors_from=(None,))
        b = _policy(outliers_from=(None, "ir"), factors_from=())
        assert a.scope_fragment() == b.scope_fragment()

    def test_factor_identity_ignores_outliers_from(self):
        a = _policy(outliers_from=(None,))
        b = _policy(outliers_from=(None, "ir"))
        assert a.factor_identity() == b.factor_identity()

    def test_factor_identity_tracks_factors_from(self):
        a = _policy(factors_from=(None,))
        b = _policy(factors_from=(None, "ir"))
        assert a.factor_identity() != b.factor_identity()

    def test_families_of_never_reports_dimension_for_a_named_group(self):
        # `rgb` names DIMENSION alongside a real family. A band group cannot carry
        # geometry, so `families_of` must drop it rather than report it verbatim.
        policy = _policy(measure=((None, ImageStats.DIMENSION), ("rgb", ImageStats.VISUAL | ImageStats.DIMENSION)))
        assert not (policy.families_of("rgb") & ImageStats.DIMENSION)
        assert policy.families_of("rgb") == ImageStats.VISUAL


@pytest.mark.required
class TestNarrowedTo:
    """Narrowing a request narrows the bands to match, so `compute_stats` never sees a mismatch."""

    def test_narrowing_to_the_bare_view_drops_every_group(self):
        policy = _policy(channels=(("ir", (3,)),))
        narrowed = policy.narrowed_to({None: ImageStats.VISUAL})
        assert narrowed.request == {None: ImageStats.VISUAL}
        assert narrowed.channel_map is None

    def test_narrowing_to_a_group_alone_keeps_only_that_group(self):
        policy = _policy(channels=(("ir", (3,)), ("rgb", (0, 1, 2))))
        narrowed = policy.narrowed_to({"ir": ImageStats.PIXEL_MEAN})
        assert narrowed.request == {"ir": ImageStats.PIXEL_MEAN}
        assert narrowed.channel_map == {"ir": [3]}

    def test_background_and_consumer_view_sets_are_carried_through(self):
        policy = _policy(background=True, outliers_from=(None, "ir"), factors_from=("ir",))
        narrowed = policy.narrowed_to({None: ImageStats.VISUAL})
        assert narrowed.background is True
        assert narrowed.outliers_from == (None, "ir")
        assert narrowed.factors_from == ("ir",)


@pytest.mark.required
class TestResolveStatsPolicy:
    def _config(self, **policy):
        from dataeval_flow.config import PipelineConfig

        base = {"name": "p", "measure": [{"bands": None, "families": ["visual", "hash"]}]}
        return PipelineConfig(stats=[{**base, **policy}])  # type: ignore[arg-type]

    def _params(self, name):
        from dataeval_flow.workflows.cleaning.params import DataCleaningParameters

        return DataCleaningParameters(
            name="c",  # type: ignore[call-arg]
            type="data-cleaning",  # type: ignore[call-arg]
            outlier_method="modzscore",
            outlier_flags=["visual"],
            stats=name,
        )

    def test_none_when_no_policy_is_named(self):
        assert resolve_stats_policy(self._params(None), self._config(), None) is None

    def test_resolves_families_to_flags(self):
        policy = resolve_stats_policy(self._params("p"), self._config(), None)
        assert policy is not None
        assert policy.request == {None: ImageStats.VISUAL | ImageStats.HASH}

    def test_selects_only_the_groups_measure_names(self):
        config = self._config(
            measure=[{"bands": None, "families": ["visual", "hash"]}, {"bands": "ir", "families": ["pixel"]}]
        )
        policy = resolve_stats_policy(self._params("p"), config, {"rgb": (0, 1, 2), "ir": (3,)})
        assert policy is not None
        assert policy.channel_map == {"ir": [3]}

    def test_refuses_a_group_the_dataset_does_not_declare(self):
        config = self._config(
            measure=[{"bands": None, "families": ["visual", "hash"]}, {"bands": "swir", "families": ["pixel"]}]
        )
        with pytest.raises(ValueError, match="does not declare a channel group 'swir'"):
            resolve_stats_policy(self._params("p"), config, {"ir": (3,)})

    def test_refuses_a_name_that_is_not_in_the_pool(self):
        with pytest.raises(ValueError, match="stats policy"):
            resolve_stats_policy(self._params("absent"), self._config(), None)

    def test_refuses_a_named_policy_with_no_pipeline(self):
        with pytest.raises(ValueError, match="pipeline config"):
            resolve_stats_policy(self._params("p"), None, None)


@pytest.mark.required
class TestCheckConsumers:
    def test_passes_when_every_consumer_is_measured(self):
        check_consumers(
            _policy(outliers_from=(None,), factors_from=(None,)),
            outlier_flags=ImageStats.VISUAL,
            duplicate_flags=ImageStats.HASH_XXHASH,
            factor_flags=ImageStats.VISUAL,
        )

    def test_refuses_an_outlier_family_the_named_view_does_not_measure(self):
        with pytest.raises(ValueError, match="outlier_flags"):
            check_consumers(
                _policy(outliers_from=(None,)),
                outlier_flags=ImageStats.PIXEL,
                duplicate_flags=ImageStats.NONE,
                factor_flags=ImageStats.NONE,
            )

    def test_refuses_a_duplicate_flag_the_whole_image_does_not_measure(self):
        with pytest.raises(ValueError, match="duplicate_flags"):
            check_consumers(
                _policy(measure=((None, ImageStats.VISUAL),)),
                outlier_flags=ImageStats.VISUAL,
                duplicate_flags=ImageStats.HASH_XXHASH,
                factor_flags=ImageStats.NONE,
            )

    def test_refuses_a_factor_family_the_named_view_does_not_measure(self):
        with pytest.raises(ValueError, match="intrinsic_factors"):
            check_consumers(
                _policy(factors_from=("ir",)),
                outlier_flags=ImageStats.NONE,
                duplicate_flags=ImageStats.NONE,
                factor_flags=ImageStats.VISUAL,
            )

    def test_a_background_view_does_not_have_to_measure_geometry(self):
        check_consumers(
            _policy(
                measure=((None, ImageStats.VISUAL | ImageStats.DIMENSION),),
                background=True,
                outliers_from=(None, "background"),
            ),
            outlier_flags=ImageStats.VISUAL | ImageStats.DIMENSION,
            duplicate_flags=ImageStats.NONE,
            factor_flags=ImageStats.NONE,
        )

    def test_an_empty_consumer_list_needs_nothing(self):
        check_consumers(
            _policy(outliers_from=(), factors_from=()),
            outlier_flags=ImageStats.ALL,
            duplicate_flags=ImageStats.NONE,
            factor_flags=ImageStats.ALL,
        )

    def test_the_outlier_message_offers_the_view_list_as_a_lever(self):
        with pytest.raises(ValueError, match=r"drop '~' from `outliers_from`.*outliers_from: \[\]"):
            check_consumers(
                _policy(outliers_from=(None,)),
                outlier_flags=ImageStats.PIXEL,
                duplicate_flags=ImageStats.NONE,
                factor_flags=ImageStats.NONE,
            )

    def test_the_factor_message_offers_the_view_list_as_a_lever(self):
        with pytest.raises(ValueError, match=r"drop 'ir' from `factors_from`.*factors_from: \[\]"):
            check_consumers(
                _policy(factors_from=("ir",)),
                outlier_flags=ImageStats.NONE,
                duplicate_flags=ImageStats.NONE,
                factor_flags=ImageStats.VISUAL,
            )

    def test_the_duplicate_declaration_defaults_to_the_field_name(self):
        with pytest.raises(ValueError, match="`duplicate_flags`"):
            check_consumers(
                _policy(measure=((None, ImageStats.VISUAL),)),
                outlier_flags=ImageStats.NONE,
                duplicate_flags=ImageStats.HASH_XXHASH,
                factor_flags=ImageStats.NONE,
            )

    def test_the_duplicate_declaration_is_overridable(self):
        """A caller with no `duplicate_flags` field of its own must not have the error

        point at one. `duplicate_declaration` lets it name what actually drove the request.
        """
        with pytest.raises(ValueError, match="this workflow's duplicate detection") as excinfo:
            check_consumers(
                _policy(measure=((None, ImageStats.VISUAL),)),
                outlier_flags=ImageStats.NONE,
                duplicate_flags=ImageStats.HASH_XXHASH,
                factor_flags=ImageStats.NONE,
                duplicate_declaration="this workflow's duplicate detection",
            )
        assert "duplicate_flags" not in str(excinfo.value)


@pytest.mark.required
class TestStatsPolicyFor:
    def test_reads_the_context_when_it_carries_one(self):
        from types import SimpleNamespace

        declared = _policy()
        context = SimpleNamespace(stats_policy=declared)
        assert stats_policy_for(context, outlier_flags=ImageStats.VISUAL) is declared

    def test_falls_back_to_the_union_of_the_consumers(self):
        from types import SimpleNamespace

        policy = stats_policy_for(
            SimpleNamespace(stats_policy=None),
            outlier_flags=ImageStats.VISUAL,
            duplicate_flags=ImageStats.HASH_XXHASH,
        )
        assert policy.request == {None: ImageStats.VISUAL | ImageStats.HASH_XXHASH}

    def test_a_context_without_the_attribute_falls_back(self):
        from types import SimpleNamespace

        policy = stats_policy_for(SimpleNamespace(), outlier_flags=ImageStats.PIXEL)
        assert policy.request == {None: ImageStats.PIXEL}

    def test_checks_the_declared_policy_against_what_the_caller_reads(self):
        from types import SimpleNamespace

        context = SimpleNamespace(stats_policy=_policy(name="p", outliers_from=(None,)))
        with pytest.raises(ValueError, match="outlier_flags"):
            stats_policy_for(context, outlier_flags=ImageStats.PIXEL)

    def test_derive_flags_is_requested_in_place_of_the_union_when_no_policy_is_declared(self):
        from types import SimpleNamespace

        policy = stats_policy_for(
            SimpleNamespace(stats_policy=None),
            outlier_flags=ImageStats.PIXEL,
            derive_flags=ImageStats.ALL,
        )
        assert policy.request == {None: ImageStats.ALL}

    def test_derive_flags_runs_no_consumer_check_against_a_declared_policy(self):
        """A caller that reads whatever is measured has nothing to check a declared policy

        against — a band-group-only policy must not be refused for lacking `ImageStats.ALL`.
        """
        from types import SimpleNamespace

        declared = _policy(name="bands", outliers_from=(None,))
        context = SimpleNamespace(stats_policy=declared)
        policy = stats_policy_for(context, derive_flags=ImageStats.ALL)
        assert policy is declared

    def test_does_not_check_the_derived_policy(self):
        from types import SimpleNamespace

        policy = stats_policy_for(SimpleNamespace(stats_policy=None), outlier_flags=ImageStats.ALL)
        assert policy.request == {None: ImageStats.ALL}
