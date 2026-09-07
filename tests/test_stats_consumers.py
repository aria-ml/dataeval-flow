"""The columns reaching a consumer are the ones its config names, and nothing else.

Every test here is an invariant that fails on the release before this change: a warm cache
or a second workflow over one source widened what each consumer read.
"""

import numpy as np
import pytest
from dataeval.flags import ImageStats

from dataeval_flow.cache import DatasetCache, active_cache, get_or_compute_stats
from dataeval_flow.stats import ResolvedStatsPolicy, columns_for, restrict_columns

# `toy_images` and `toy_multiband_dataset` come from `tests/conftest.py` (Task 5).


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


#: `duplicate_flags` spellings, keyed by the flags they resolve to. `_run_cleaning` takes
#: the config spelling; the tests reason in flags.
_DUPLICATE_FLAG_NAMES = {
    ImageStats.HASH_XXHASH: ["hash_basic"],
    ImageStats.HASH_DUPLICATES_D4: ["hash_d4"],
}


def _duplicate_group_counts(outputs):
    """Return (exact, near) group counts from a cleaning run's raw outputs.

    `DataCleaningRawOutputs.duplicates` is a plain `{"items": {...}, "targets": {...}}`
    dict (see `_serialize_duplicates` in `workflows/cleaning/outputs.py`), not an object
    carrying `.items.exact` — hence the dict indexing here rather than attribute access.
    """
    items = outputs.duplicates["items"]
    return len(items.get("exact") or []), len(items.get("near") or [])


@pytest.mark.required
class TestOutlierColumnsAreDeclared:
    """What flags an outlier is what `outlier_flags` and `outliers_from` name.

    The cold/warm tests and `test_only_the_declared_families_flag` run `_run_cleaning`,
    the production call site this task fixes, with `context=None` — the derived-policy
    path a config with no `stats:` block takes, which is the case the regression this
    task fixes was measured on. `test_a_band_group_does_not_flag_unless_outliers_from_names_it`
    and `test_background_fraction_does_not_flag_by_default`, below, are different: they
    assert on `restrict_columns` directly and do not exercise a workflow.
    """

    def _params(self):
        """`outlier_flags: [visual]`, no `stats:` policy named."""
        from dataeval_flow.workflows.cleaning.params import DataCleaningParameters

        return DataCleaningParameters(
            name="c",  # type: ignore[call-arg]
            type="data-cleaning",  # type: ignore[call-arg]
            outlier_method="modzscore",
            outlier_flags=["visual"],
        )

    def _issues(self, dataset):
        """Run outlier detection exactly as `_run_cleaning` does."""
        from dataeval_flow.workflows.cleaning.workflow import _run_cleaning

        raw = _run_cleaning(dataset, self._params(), context=None)
        return raw.img_outliers["issues"]

    def _flagged_metrics(self, dataset):
        return sorted({issue["metric_name"] for issue in self._issues(dataset)})

    def _flagged_items(self, dataset):
        return sorted({issue["item_index"] for issue in self._issues(dataset)})

    def _warm(self, dataset):
        """Widen the cache entry `_run_cleaning` will reuse, with families it does not name."""
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
    """What detects a duplicate is what `duplicate_flags` names."""

    @pytest.fixture
    def paired_images(self):
        """A dataset holding one exact duplicate pair and one near-duplicate pair.

        The near pair is a 90-degree rotation of one base image, not a small pixel-value
        tweak. A tweak (tried first; see the task report) is absorbed identically by every
        hash family regardless of restriction, so cold and warm always agree for the wrong
        reason. A rotation genuinely separates the families: `phash`/`dhash` (regular) do
        not recognize a rotated image as similar, while `phash_d4`/`dhash_d4`
        (rotation/flip-invariant) do — so which family a config restricts to changes the
        answer, which is exactly what this class needs to catch a defect.
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
        """Run the duplicate path through production code.

        Call `_run_cleaning`, not a local reimplementation of the filter. A helper that
        calls `restrict_columns` itself tests Task 3's machinery and passes whether or not
        the workflow was ever fixed — that defect reached review once on Task 6 already.

        `context=None` is deliberate: it exercises the derived-policy path, which is what a
        config with no `stats:` block does, and that is the case this regression is about.
        """
        from dataeval_flow.workflows.cleaning.params import DataCleaningParameters
        from dataeval_flow.workflows.cleaning.workflow import _run_cleaning

        params = DataCleaningParameters(
            name="c",  # type: ignore[call-arg]
            type="data-cleaning",  # type: ignore[call-arg]
            outlier_method="modzscore",
            outlier_flags=["visual"],
            duplicate_flags=_DUPLICATE_FLAG_NAMES[duplicate_flags],
        )
        outputs = _run_cleaning(dataset, params)
        return _duplicate_group_counts(outputs)

    def test_a_warm_cache_finds_the_same_groups_as_a_cold_one(self, paired_images):
        """Same config, different cache state, same answer.

        Warm the entry through the same scope `_run_cleaning` uses, or the two calls land in
        different entries and this proves nothing. Assert the widening actually happened
        before trusting the comparison.
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
class TestAnalysisDuplicateColumnsAreDeclared:
    """The analysis workflow's duplicate-side entry points restrict too.

    `_assess_redundancy` restricts to `columns_for([None], ImageStats.HASH)` — already the
    full hash family superset, since the analysis workflow always asks for every hash family
    (there is no per-config `duplicate_flags` knob here). That makes the restriction a
    no-op for `_assess_redundancy`'s own near/exact group counts: nothing can widen a cache
    entry past the ceiling it already requested, and `Duplicates.from_stats` only ever reads
    the five hash columns by bare name, ignoring anything else regardless of restriction. A
    cold/warm group-count comparison through `_assess_redundancy` alone cannot fail at HEAD;
    see the task report for the empirical check.

    `_assess_cross_redundancy` is where the defect is real: it combines two stats results
    before detecting, and `dataeval`'s combine step refuses two results computed over
    different statistics. Two splits share one dataset but not necessarily one cache scope,
    so nothing stops an unrelated consumer from widening one split's entry with a family the
    other split's entry never got — at which point the unrestricted call raises instead of
    running, and restricting both operands to the same declared set is what keeps them
    combinable regardless of what else shares either scope.
    """

    def _leakage(self, calc_a, calc_b):
        from dataeval_flow.workflows.analysis.workflow import _assess_cross_redundancy

        result = _assess_cross_redundancy(calc_a, calc_b, "train", "test")
        return result.duplicate_leakage["exact_count"], result.duplicate_leakage["near_count"]

    def test_a_family_widened_on_one_split_does_not_change_the_cross_split_result(self, toy_images):
        """The defect this fixes: combining two splits raises once their cache entries
        diverge on anything other than the hash columns `Duplicates.from_stats` reads.

        `toy_images` stands in for both splits under different `sel_key`s — the point is
        column-set consistency between the two calc results, not realistic cross-split
        duplicate content.
        """
        DatasetCache.clear_instances()
        cache = DatasetCache.get_or_create(None, "toy", "k")

        with active_cache(cache, "train"):
            calc_a = get_or_compute_stats(ResolvedStatsPolicy.of_flags(ImageStats.HASH), dataset=toy_images)
        with active_cache(cache, "test"):
            calc_b = get_or_compute_stats(ResolvedStatsPolicy.of_flags(ImageStats.HASH), dataset=toy_images)

        baseline = self._leakage(calc_a, calc_b)

        with active_cache(cache, "train"):
            # A consumer sharing train's scope asks for a family cross-redundancy never did.
            widened = get_or_compute_stats(ResolvedStatsPolicy.of_flags(ImageStats.VISUAL), dataset=toy_images)
            calc_a_widened = get_or_compute_stats(ResolvedStatsPolicy.of_flags(ImageStats.HASH), dataset=toy_images)

        assert "brightness" in widened["stats"]  # confirms the widening actually happened
        assert self._leakage(calc_a_widened, calc_b) == baseline
