"""The columns reaching a consumer are the ones its config names, and nothing else.

Every test here is an invariant that fails on the release before this change: a warm cache
or a second workflow over one source widened what each consumer read.
"""

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
