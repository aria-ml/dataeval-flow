"""The columns reaching a consumer are the ones its config names, and nothing else.

Every test here is an invariant that fails on the release before this change: a warm cache
or a second workflow over one source widened what each consumer read.
"""

from typing import Any, Literal

import numpy as np
import polars as pl
import pytest
from dataeval.flags import ImageStats

from dataeval_flow._cache import DatasetCache, active_cache, get_or_compute_stats
from dataeval_flow._stats import ResolvedStatsPolicy, columns_for, restrict_columns, stats_policy_for
from dataeval_flow.evaluators import EvaluatorInputs
from dataeval_flow.evaluators.quality import DuplicatesConfig, OutliersConfig
from dataeval_flow.evaluators.quality._evaluator import find_duplicates, find_outliers

# `toy_images` and `toy_multiband_dataset` come from `tests/conftest.py`.


def _in_fresh_cache(work):
    """Run *work* under a cache nothing else has touched."""
    DatasetCache.clear_instances()
    cache = DatasetCache.get_or_create(None, "toy", "k")
    with active_cache(cache, "sel"):
        return work()


def _in_warmed_cache(warm, work):
    """Run *warm*, then *work*, under a cache nothing else has touched.

    Returns *warm*'s result alongside *work*'s, so a caller can check the cache entry was
    actually widened before trusting a comparison against it.
    """
    DatasetCache.clear_instances()
    cache = DatasetCache.get_or_create(None, "toy", "k")
    with active_cache(cache, "sel"):
        widened = warm()
        return widened, work()


# `duplicate_flags` spellings, keyed by the flags they resolve to. `DuplicatesConfig.flags` takes
# the config spelling; the tests reason in flags.
_DUPLICATE_FLAG_NAMES: dict[ImageStats, list[Literal["hash_basic", "hash_d4"]]] = {
    ImageStats.HASH_XXHASH: ["hash_basic"],
    ImageStats.HASH_DUPLICATES_D4: ["hash_d4"],
}


def _computed_as_the_engine_does(config: Any, dataset: Any) -> EvaluatorInputs:
    """`dataset`'s statistics as the engine prepares them for an evaluator step with no `stats:` policy named.

    The policy is the one `stats_policy_for` derives from the evaluator's own request, as the stats producer
    (`evaluators/_producers.py`) derives it when the step's context carries no policy. The value range is left unset,
    as the warm-ups leave it, so both land in one cache entry.
    """
    policy = stats_policy_for(None, **config.stats_request())
    return EvaluatorInputs(source="toy", stats=get_or_compute_stats(policy, dataset=dataset), stats_policy=policy)


def _duplicate_group_counts(output):
    """Return (exact, near) group counts from the Duplicates output `find_duplicates` returns.

    Each item-level row of its `data()` is one group, whose `dup_type` is `exact` or `near`.
    """
    kinds = output.data().filter(pl.col("level") == "item")["dup_type"].to_list()
    return kinds.count("exact"), kinds.count("near")


@pytest.mark.required
class TestOutlierColumnsAreDeclared:
    """What flags an outlier is what `outliers.flags` and `outliers_from` name.

    The cold/warm tests and `test_only_the_declared_families_flag` run `find_outliers`,
    the production call site quality's `outliers` step reaches, on statistics
    computed as the engine computes them for that step with no `stats:` block named — the
    derived-policy path. The regression was measured on that path.
    `test_a_band_group_does_not_flag_unless_outliers_from_names_it` and
    `test_background_fraction_does_not_flag_by_default` assert on `restrict_columns`
    directly and do not exercise an evaluator.
    """

    def _params(self):
        """The `outliers` evaluator that quality's `outliers: {flags: [visual], outlier_threshold: modzscore}`
        configures, with no `stats:` policy named."""
        return OutliersConfig(name="outliers", flags=["visual"], outlier_threshold="modzscore", per_target=True)

    def _issues(self, dataset):
        """Flag outliers as quality's `outliers` step does: statistics, then `find_outliers`."""
        config = self._params()
        return find_outliers(config, [_computed_as_the_engine_does(config, dataset)]).data()

    def _flagged_metrics(self, dataset):
        return sorted(set(self._issues(dataset)["metric_name"].cast(pl.Utf8).to_list()))

    def _flagged_items(self, dataset):
        return sorted(set(self._issues(dataset)["item_index"].to_list()))

    def _warm(self, dataset):
        """Widen the cache entry the `outliers` step will reuse, with families it does not name."""
        return lambda: get_or_compute_stats(
            ResolvedStatsPolicy.of_flags(ImageStats.PIXEL | ImageStats.DIMENSION),
            dataset=dataset,
        )

    def test_a_warm_cache_flags_the_same_metrics_as_a_cold_one(self, toy_images):
        cold = _in_fresh_cache(lambda: self._flagged_metrics(toy_images))
        widened, warm = _in_warmed_cache(self._warm(toy_images), lambda: self._flagged_metrics(toy_images))
        assert "mean" in widened["stats"]  # confirms the warm-up actually widened the shared entry
        assert cold == warm

    def test_a_warm_cache_flags_the_same_items_as_a_cold_one(self, toy_images):
        cold = _in_fresh_cache(lambda: self._flagged_items(toy_images))
        widened, warm = _in_warmed_cache(self._warm(toy_images), lambda: self._flagged_items(toy_images))
        assert "mean" in widened["stats"]
        assert cold == warm

    def test_only_the_declared_families_flag(self, toy_images):
        flagged = _in_fresh_cache(lambda: self._flagged_metrics(toy_images))
        assert set(flagged) <= {"brightness", "contrast", "darkness", "sharpness"}

    def test_a_band_group_does_not_flag_unless_outliers_from_names_it(self, toy_multiband_dataset):
        """Asserts on `restrict_columns` directly — the column filter, not a workflow."""
        policy = ResolvedStatsPolicy(
            measure=((None, ImageStats.VISUAL), ("ir", ImageStats.PIXEL)),
            channels=(("ir", (3,)),),
            outliers_from=(None,),
        )
        result = _in_fresh_cache(lambda: get_or_compute_stats(policy, dataset=toy_multiband_dataset, per_target=False))
        assert "ir_mean" in result["stats"]
        restricted = restrict_columns(result, columns_for(policy.outliers_from, ImageStats.VISUAL))
        assert "ir_mean" not in restricted["stats"]

    def test_background_fraction_does_not_flag_by_default(self, toy_multiband_dataset):
        """Asserts on `restrict_columns` directly — the column filter, not a workflow."""
        policy = ResolvedStatsPolicy(measure=((None, ImageStats.VISUAL),), background=True)
        result = _in_fresh_cache(lambda: get_or_compute_stats(policy, dataset=toy_multiband_dataset, per_target=False))
        assert "background_fraction" in result["stats"]
        restricted = restrict_columns(result, columns_for(policy.outliers_from, ImageStats.VISUAL))
        assert "background_fraction" not in restricted["stats"]


@pytest.mark.required
class TestDuplicateColumnsAreDeclared:
    """What detects a duplicate is what `duplicates.flags` names."""

    @pytest.fixture
    def paired_images(self):
        """A dataset holding one exact duplicate pair and one near-duplicate pair.

        The near pair is a 90-degree rotation of one base image, not a small pixel-value
        tweak. A tweak is absorbed identically by every hash family regardless of
        restriction, so cold and warm always agree for the wrong reason. A rotation
        separates the families: `phash`/`dhash` (regular) do not recognize a rotated image
        as similar, while `phash_d4`/`dhash_d4` (rotation/flip-invariant) do. Which family
        a config restricts to changes the answer, which is what this class needs to catch
        a defect.
        """
        rng = np.random.default_rng(1)

        class _Toy:
            def __init__(self):
                base = [rng.integers(0, 255, (3, 16, 16), dtype=np.uint8) for _ in range(6)]
                near = np.ascontiguousarray(np.rot90(base[1], k=1, axes=(1, 2)))
                self._images = [*base, base[0].copy(), near]
                self.metadata = {"id": "toy", "index2label": {0: "a"}}

            def __len__(self):
                return len(self._images)

            def __getitem__(self, index):
                return self._images[index], np.array([1.0]), {"id": index}

        return _Toy()

    def _groups(self, dataset, duplicate_flags):
        """Run the duplicate path through production code: quality's `dupes` step.

        Call `find_duplicates`, not a local reimplementation of the filter. A helper that
        calls `restrict_columns` itself tests the machinery only, and passes whether or
        not the evaluator was fixed; that defect reached review once already.

        No `stats:` policy is named, deliberately: it exercises the derived-policy path,
        which is what a config with no `stats:` block does, and that is the case this
        regression is about.
        """
        config = DuplicatesConfig(
            name="dupes", flags=_DUPLICATE_FLAG_NAMES[duplicate_flags], merge_near_duplicates=True
        )
        output = find_duplicates(config, [_computed_as_the_engine_does(config, dataset)])
        return _duplicate_group_counts(output)

    def test_a_warm_cache_finds_the_same_groups_as_a_cold_one(self, paired_images):
        """Same config, different cache state, same answer.

        Warm the entry through the same scope the `dupes` step uses, or the two calls land
        in different entries and the comparison proves nothing. Assert the widening
        happened before trusting the comparison.
        """

        def warm():
            return get_or_compute_stats(ResolvedStatsPolicy.of_flags(ImageStats.HASH), dataset=paired_images)

        cold = _in_fresh_cache(lambda: self._groups(paired_images, ImageStats.HASH_XXHASH))
        widened, warm_result = _in_warmed_cache(warm, lambda: self._groups(paired_images, ImageStats.HASH_XXHASH))
        assert "phash_d4" in widened["stats"]  # confirms the warm-up actually widened the shared entry
        assert cold == warm_result

    def test_xxhash_alone_finds_no_near_duplicates(self, paired_images):
        _exact, near = _in_fresh_cache(lambda: self._groups(paired_images, ImageStats.HASH_XXHASH))
        assert near == 0

    def test_a_perceptual_hash_finds_the_near_pair(self, paired_images):
        _exact, near = _in_fresh_cache(lambda: self._groups(paired_images, ImageStats.HASH_DUPLICATES_D4))
        assert near >= 1


@pytest.mark.required
class TestCrossSplitDuplicateColumnsAreDeclared:
    """Duplicates found across two splits restrict each split's statistics too.

    audit's `duplicates-cross` step runs the `duplicates` evaluator over train and the
    evaluation splits, so `find_duplicates` combines two stats results before detecting,
    and `dataeval`'s combine step refuses two results computed over different statistics.
    Two splits share one dataset but not necessarily one cache scope, so an unrelated
    consumer can widen one split's entry with a family the other split's entry never got.
    The unrestricted call then raises instead of running. Restricting each operand to the
    same declared set keeps them combinable regardless of what else shares either scope.

    The two tests below cover one operand each: each widens only *one* split's scope, so
    each fails when that split's statistics reach the combine unrestricted.
    """

    def _leakage(self, calc_a, calc_b):
        inputs = [EvaluatorInputs(source="train", stats=calc_a), EvaluatorInputs(source="test", stats=calc_b)]
        return _duplicate_group_counts(find_duplicates(DuplicatesConfig(name="dupes"), inputs))

    def _calc_pair(self, dataset):
        """A fresh (calc_a, calc_b) pair, each computed under its own scope.

        `toy_images` stands in for both splits under different `sel_key`s. The point is
        column-set consistency between the two calc results, not realistic cross-split
        duplicate content.
        """
        DatasetCache.clear_instances()
        cache = DatasetCache.get_or_create(None, "toy", "k")

        with active_cache(cache, "train"):
            calc_a = get_or_compute_stats(ResolvedStatsPolicy.of_flags(ImageStats.HASH), dataset=dataset)
        with active_cache(cache, "test"):
            calc_b = get_or_compute_stats(ResolvedStatsPolicy.of_flags(ImageStats.HASH), dataset=dataset)
        return cache, calc_a, calc_b

    def test_widening_the_train_split_does_not_change_the_cross_split_result(self, toy_images):
        """The defect this fixes: combining two splits raises once their cache entries
        diverge on anything other than the hash columns `Duplicates.from_stats` reads.

        Only `train`'s scope is widened here, so this fails when train's statistics reach
        the combine unrestricted.
        """
        cache, calc_a, calc_b = self._calc_pair(toy_images)
        baseline = self._leakage(calc_a, calc_b)

        with active_cache(cache, "train"):
            # A consumer sharing train's scope asks for a family duplicate detection never did.
            widened = get_or_compute_stats(ResolvedStatsPolicy.of_flags(ImageStats.VISUAL), dataset=toy_images)
            calc_a_widened = get_or_compute_stats(ResolvedStatsPolicy.of_flags(ImageStats.HASH), dataset=toy_images)

        assert "brightness" in widened["stats"]  # confirms the widening actually happened
        assert self._leakage(calc_a_widened, calc_b) == baseline

    def test_widening_the_test_split_does_not_change_the_cross_split_result(self, toy_images):
        """The symmetric case: only `test`'s scope is widened here, so this fails when
        test's statistics reach the combine unrestricted.
        """
        cache, calc_a, calc_b = self._calc_pair(toy_images)
        baseline = self._leakage(calc_a, calc_b)

        with active_cache(cache, "test"):
            # A consumer sharing test's scope asks for a family duplicate detection never did.
            widened = get_or_compute_stats(ResolvedStatsPolicy.of_flags(ImageStats.VISUAL), dataset=toy_images)
            calc_b_widened = get_or_compute_stats(ResolvedStatsPolicy.of_flags(ImageStats.HASH), dataset=toy_images)

        assert "brightness" in widened["stats"]  # confirms the widening actually happened
        assert self._leakage(calc_a, calc_b_widened) == baseline


@pytest.mark.required
class TestFactorColumnsAreDeclared:
    """The injected factor set is what `intrinsic_factors` and `factors_from` name."""

    def _factors(self, dataset, policy):
        from dataeval_flow._metadata import build_metadata

        return sorted(build_metadata(dataset, policy).factor_names)

    def _warm(self, dataset):
        return lambda: get_or_compute_stats(
            ResolvedStatsPolicy.of_flags(ImageStats.PIXEL | ImageStats.DIMENSION),
            dataset=dataset,
            per_target=False,
        )

    def test_a_warm_cache_injects_the_same_factors_as_a_cold_one(self, toy_images):
        from dataeval_flow._policy import ResolvedPolicy

        policy = ResolvedPolicy(intrinsic_factors=("visual",))
        cold = _in_fresh_cache(lambda: self._factors(toy_images, policy))
        widened, warm = _in_warmed_cache(self._warm(toy_images), lambda: self._factors(toy_images, policy))
        assert "mean" in widened["stats"]  # confirms the warm-up actually widened the shared entry
        assert cold == warm

    def test_only_the_declared_family_is_injected(self, toy_images):
        from dataeval_flow._policy import ResolvedPolicy

        factors = _in_fresh_cache(lambda: self._factors(toy_images, ResolvedPolicy(intrinsic_factors=("visual",))))
        assert "mean" not in factors
        assert "brightness" in factors

    def test_a_band_group_is_injected_only_when_factors_from_names_it(self, toy_multiband_dataset):
        from dataeval_flow._policy import ResolvedPolicy

        stats = ResolvedStatsPolicy(
            measure=((None, ImageStats.VISUAL), ("ir", ImageStats.VISUAL)),
            channels=(("ir", (3,)),),
            factors_from=(None,),
        )
        policy = ResolvedPolicy(intrinsic_factors=("visual",), stats=stats)
        factors = _in_fresh_cache(lambda: self._factors(toy_multiband_dataset, policy))
        assert not any(name.endswith("ir_brightness") for name in factors)

    def test_naming_the_group_injects_it(self, toy_multiband_dataset):
        from dataeval_flow._policy import ResolvedPolicy

        stats = ResolvedStatsPolicy(
            measure=((None, ImageStats.VISUAL), ("ir", ImageStats.VISUAL)),
            channels=(("ir", (3,)),),
            factors_from=(None, "ir"),
        )
        policy = ResolvedPolicy(intrinsic_factors=("visual",), stats=stats)
        factors = _in_fresh_cache(lambda: self._factors(toy_multiband_dataset, policy))
        assert any(name.endswith("ir_brightness") for name in factors)
