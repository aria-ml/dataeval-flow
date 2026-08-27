"""Family resolution and bin expansion — the pure half of intrinsic factor injection."""

import pytest
from dataeval.flags import ImageStats

from dataeval_flow.metadata import expand_declared_bins, resolve_families, stat_names_for


class TestResolvingFamilies:
    """Config names families; the enum is chosen by modality."""

    def test_resolves_a_single_family(self):
        assert resolve_families("image", ["visual"]) == ImageStats.VISUAL

    def test_ors_several_families(self):
        assert resolve_families("image", ["visual", "pixel"]) == ImageStats.VISUAL | ImageStats.PIXEL

    def test_is_case_insensitive(self):
        assert resolve_families("image", ["VISUAL"]) == ImageStats.VISUAL

    def test_empty_resolves_to_none(self):
        assert resolve_families("image", []) == ImageStats.NONE

    def test_unknown_family_names_both_sides(self):
        with pytest.raises(ValueError, match="nonsense") as exc:
            resolve_families("image", ["nonsense"])
        message = str(exc.value)
        assert "visual" in message
        assert "pixel" in message

    def test_individual_statistics_are_not_families(self):
        # getattr(ImageStats, "PIXEL_MEAN") resolves; the allowlist is what refuses it.
        with pytest.raises(ValueError, match="pixel_mean"):
            resolve_families("image", ["pixel_mean"])

    @pytest.mark.parametrize("degenerate", ["none", "all"])
    def test_degenerate_wholes_are_not_families(self, degenerate):
        with pytest.raises(ValueError, match=degenerate):
            resolve_families("image", [degenerate])

    def test_unknown_modality_names_the_known_ones(self):
        with pytest.raises(ValueError, match="video"):
            resolve_families("video", ["visual"])


class TestStatNames:
    """The column names a flag set produces, derived from the enum rather than listed."""

    def test_visual_and_pixel(self):
        assert stat_names_for(ImageStats.VISUAL | ImageStats.PIXEL) == {
            "brightness",
            "contrast",
            "darkness",
            "entropy",
            "histogram",
            "kurtosis",
            "mean",
            "missing",
            "percentiles",
            "sharpness",
            "skew",
            "std",
            "var",
            "zeros",
        }

    def test_hash_names(self):
        assert stat_names_for(ImageStats.HASH) == {"dhash", "dhash_d4", "phash", "phash_d4", "xxhash"}

    def test_none_produces_nothing(self):
        assert stat_names_for(ImageStats.NONE) == set()


class TestExpandingDeclaredBins:
    """A bin declared on a bare name has to reach the level-split factors."""

    LEVELS = ("unit", "instance")

    def test_single_level_is_the_identity(self):
        declared = {"brightness": 4}
        names = ["brightness", "weather"]
        assert expand_declared_bins(declared, names, ("unit",)) == {"brightness": 4}

    def test_expands_onto_both_levels(self):
        declared = {"brightness": 4}
        names = ["unit_brightness", "instance_brightness", "weather"]
        assert expand_declared_bins(declared, names, self.LEVELS) == {
            "unit_brightness": 4,
            "instance_brightness": 4,
        }

    def test_expands_onto_one_level_when_only_one_was_measured(self):
        declared = {"brightness": 4}
        names = ["unit_brightness", "weather"]
        assert expand_declared_bins(declared, names, self.LEVELS) == {"unit_brightness": 4}

    def test_explicit_edges_survive_expansion(self):
        declared = {"brightness": [0.0, 0.5, 1.0]}
        names = ["unit_brightness", "instance_brightness"]
        assert expand_declared_bins(declared, names, self.LEVELS) == {
            "unit_brightness": [0.0, 0.5, 1.0],
            "instance_brightness": [0.0, 0.5, 1.0],
        }

    def test_an_unmatched_name_falls_through_unchanged(self):
        # A typo must stay visible as an unmatched bin request, not be swallowed.
        declared = {"brightnes": 4}
        names = ["unit_brightness", "instance_brightness"]
        assert expand_declared_bins(declared, names, self.LEVELS) == {"brightnes": 4}

    def test_does_not_claim_a_lookalike_from_the_dataset(self):
        # `camera_brightness` is a dataset-native factor; `camera` is not a level.
        declared = {"brightness": 4}
        names = ["camera_brightness", "unit_brightness"]
        assert expand_declared_bins(declared, names, self.LEVELS) == {"unit_brightness": 4}

    def test_an_exact_match_wins_without_expanding(self):
        declared = {"brightness": 4}
        names = ["brightness", "unit_brightness"]
        assert expand_declared_bins(declared, names, self.LEVELS) == {
            "brightness": 4,
            "unit_brightness": 4,
        }

    def test_empty_declaration_stays_empty(self):
        assert expand_declared_bins({}, ["unit_brightness"], self.LEVELS) == {}
