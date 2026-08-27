"""Family resolution and bin expansion — the pure half of intrinsic factor injection."""

import numpy as np
import pytest
from dataeval.flags import ImageStats

from dataeval_flow.metadata import build_metadata, expand_declared_bins, resolve_families, stat_names_for
from dataeval_flow.policy import ResolvedPolicy


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


class _Target:
    def __init__(self, labels, boxes, scores):
        self.labels, self.boxes, self.scores = labels, boxes, scores


class _ICDataset:
    """Classification: one level, so factor names stay bare."""

    def __init__(self, n: int = 40) -> None:
        self._n = n
        self._rng = np.random.default_rng(0)

    @property
    def metadata(self) -> dict:
        return {"id": "inject-ic", "index2label": {0: "cat", 1: "dog"}}

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int):
        one_hot = np.zeros(2, dtype=np.float32)
        one_hot[index % 2] = 1.0
        image = (self._rng.random((3, 16, 16)) * (0.2 + 0.01 * index)).astype(np.float32)
        return image, one_hot, {"id": index, "weather": ["sun", "rain"][index % 2]}


class _ODDataset:
    """Detection: two levels, so `add_factors` splits every statistic in two."""

    def __init__(self, n: int = 40) -> None:
        self._n = n
        self._rng = np.random.default_rng(1)

    @property
    def metadata(self) -> dict:
        return {"id": "inject-od", "index2label": {0: "cat", 1: "dog"}}

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int):
        count = 1 + (index % 3)
        boxes = np.tile(np.array([0.0, 0.0, 8.0, 8.0], dtype=np.float32), (count, 1))
        boxes[:, 2] += np.arange(count)
        target = _Target(np.arange(count, dtype=np.intp) % 2, boxes, np.ones(count, dtype=np.float32))
        image = (self._rng.random((3, 32, 32)) * (0.2 + 0.01 * index)).astype(np.float32)
        return image, target, {"id": index, "weather": ["sun", "rain"][index % 2]}


class TestBuildMetadataInjects:
    """The capability reaches every caller of build_metadata, or it reaches none."""

    def test_no_policy_injects_nothing(self):
        metadata = build_metadata(_ICDataset())
        assert "brightness" not in metadata.factor_names

    def test_empty_intrinsic_factors_injects_nothing(self):
        metadata = build_metadata(_ICDataset(), ResolvedPolicy())
        assert "brightness" not in metadata.factor_names

    def test_injects_the_named_families(self):
        policy = ResolvedPolicy(intrinsic_factors=("visual",), value_range=(0.0, 1.0))
        metadata = build_metadata(_ICDataset(), policy)
        assert {"brightness", "contrast", "darkness", "sharpness"} <= set(metadata.factor_names)

    def test_does_not_inject_families_it_was_not_given(self):
        policy = ResolvedPolicy(intrinsic_factors=("visual",), value_range=(0.0, 1.0))
        metadata = build_metadata(_ICDataset(), policy)
        assert "mean" not in metadata.factor_names  # a PIXEL statistic

    def test_hashes_never_arrive(self):
        policy = ResolvedPolicy(intrinsic_factors=("hash",), value_range=(0.0, 1.0))
        metadata = build_metadata(_ICDataset(), policy)
        assert not {"xxhash", "phash", "dhash"} & set(metadata.factor_names)

    def test_declared_bin_binds_on_classification(self):
        policy = ResolvedPolicy(
            intrinsic_factors=("visual",),
            value_range=(0.0, 1.0),
            continuous_factor_bins={"brightness": 4},
        )
        metadata = build_metadata(_ICDataset(), policy)
        assert metadata.continuous_factor_bins == {"brightness": 4}
        assert metadata.factor_info["brightness"].factor_type == "continuous"

    def test_declared_bin_reaches_both_levels_on_detection(self):
        policy = ResolvedPolicy(
            intrinsic_factors=("visual",),
            value_range=(0.0, 1.0),
            continuous_factor_bins={"brightness": 4},
        )
        metadata = build_metadata(_ODDataset(), policy)
        assert metadata.continuous_factor_bins == {
            "unit_brightness": 4,
            "instance_brightness": 4,
        }

    def test_a_typo_is_left_unmatched(self):
        policy = ResolvedPolicy(
            intrinsic_factors=("visual",),
            value_range=(0.0, 1.0),
            continuous_factor_bins={"brightnes": 4},
        )
        metadata = build_metadata(_ODDataset(), policy)
        assert metadata.continuous_factor_bins == {"brightnes": 4}
