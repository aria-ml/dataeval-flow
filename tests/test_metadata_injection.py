"""Family resolution and bin expansion — the pure half of intrinsic factor injection."""

from unittest.mock import patch

import numpy as np
import pytest
from dataeval.flags import ImageStats
from dataeval.protocols import DatasetMetadata

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

    def test_convenience_groups_name_no_column(self):
        """``PIXEL_BASIC`` and ``NO_HASH`` are groups, so they contribute no name.

        Only Python 3.10 can fail this: 3.11 dropped composite members from
        ``iter(FlagClass)``, which hides them whether or not ``stat_names_for`` excludes
        them. The 3.10 leg of the matrix is what holds the guard in place.
        """
        assert stat_names_for(ImageStats.ALL) & {"basic", "distribution", "duplicates_basic", "hash"} == set()


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
    def metadata(self) -> DatasetMetadata:
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
    def metadata(self) -> DatasetMetadata:
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


class _WideRangeDataset:
    """Classification with float values outside any range dataeval can infer.

    Spans ``[0, 4000]`` rather than ``[0, 1]`` or ``[0, 255]``, so a ``VISUAL`` statistic
    needs the policy's declared ``value_range`` to mean anything — without it, dataeval
    cannot decode a bit depth and reports NaN. ``_ICDataset``'s values sit inside the
    ``[0, 1]`` convention dataeval infers on its own, so it cannot tell whether
    ``value_range`` was actually threaded through to ``compute_stats`` or silently dropped:
    both give the same answer. This fixture makes the wire observable.
    """

    def __init__(self, n: int = 40) -> None:
        self._n = n
        self._rng = np.random.default_rng(2)

    @property
    def metadata(self) -> DatasetMetadata:
        return {"id": "inject-wide", "index2label": {0: "cat", 1: "dog"}}

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int):
        one_hot = np.zeros(2, dtype=np.float32)
        one_hot[index % 2] = 1.0
        image = (self._rng.random((3, 16, 16)) * 4000.0).astype(np.float32)
        return image, one_hot, {"id": index, "weather": ["sun", "rain"][index % 2]}


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

    def test_declared_value_range_reaches_compute_stats(self):
        """``value_range`` must actually reach ``compute_stats``, not just sit on the policy.

        ``_WideRangeDataset``'s values fall outside any range dataeval can infer on its own,
        so a ``VISUAL`` statistic comes back NaN unless the declared ``value_range`` is the
        one ``compute_stats`` actually receives.
        """
        policy = ResolvedPolicy(intrinsic_factors=("visual",), value_range=(0.0, 4000.0))
        metadata = build_metadata(_WideRangeDataset(), policy)
        brightness = metadata.dataframe["brightness"].to_numpy()
        assert not np.isnan(brightness).any()

    def test_per_target_is_false_on_classification_data(self):
        """``per_target`` must come from ``metadata.multi_target``, not be hardcoded.

        Classification data has no boxes, so ``compute_stats``' own output cannot tell
        ``per_target=True`` from ``per_target=False`` apart on it — spying on the call is
        what actually pins the wire.
        """
        from dataeval_flow.cache import get_or_compute_stats as real_get_or_compute_stats

        policy = ResolvedPolicy(intrinsic_factors=("visual",), value_range=(0.0, 1.0))
        with patch("dataeval_flow.cache.get_or_compute_stats", wraps=real_get_or_compute_stats) as mock_stats:
            build_metadata(_ICDataset(), policy)
        assert mock_stats.call_args.kwargs["per_target"] is False
