"""Per-workflow reachability of policy-declared intrinsic factors.

Flow's own tests passed through the whole life of the D4 defect: each half worked, and
nothing asserted that a policy's declaration reached a workflow's result. Each test here
runs a real workflow over a real dataset and reads the envelope a user would read.
"""

from typing import Any

import pytest

from tests.test_metadata_injection import _ICDataset, _ODDataset

pytestmark = pytest.mark.required

WORKFLOWS = ["audit"]


# Minimal valid params per workflow: only the fields with no default.
_PARAMS = {
    "audit": {"outliers": {"flags": ["pixel"], "outlier_threshold": "adaptive"}},
}


def _run(workflow_type: str, dataset, policy_fields: dict) -> Any:
    """Run *workflow_type* over *dataset* under a policy, returning its ResultMetadata.

    Goes through ``run_tasks``, as a user does: a preset refuses a direct ``run``.
    """
    from dataeval_flow import run_tasks
    from dataeval_flow._cache import DatasetCache
    from tests.chain_toys import chain_pipeline

    DatasetCache.clear_instances()
    config = chain_pipeline(
        workflows=[{"name": "w", "type": workflow_type, "metadata": "p", **_PARAMS[workflow_type]}],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        datasets={"src": dataset},
        extra={"metadata": [{"name": "p", **policy_fields}]},
    )
    result = run_tasks(config)["t"]
    assert result.success, f"{workflow_type} failed: {result.errors}"
    return result.metadata


def _binning(result_metadata) -> dict:
    """The binning record of a run over one dataset."""
    record = result_metadata.metadata_binning
    assert record is not None
    return record


@pytest.mark.parametrize("workflow_type", WORKFLOWS)
def test_declared_bin_binds_on_classification(workflow_type):
    """The assertion whose absence let D4 ship — once per workflow."""
    result = _run(
        workflow_type,
        _ICDataset(16),
        {"intrinsic_factors": ("visual", "pixel"), "continuous_factor_bins": {"brightness": 4}},
    )
    record = _binning(result)
    assert not record.get("unmatched_bin_requests")
    assert len(record["factors"]["brightness"]["encoding"]["edges"]) - 1 == 4


@pytest.mark.parametrize("workflow_type", WORKFLOWS)
def test_declared_bin_binds_at_both_levels_on_detection(workflow_type):
    """Classification exercises only the identity case of the expansion."""
    result = _run(
        workflow_type,
        _ODDataset(16),
        {"intrinsic_factors": ("visual", "pixel"), "continuous_factor_bins": {"brightness": 4}},
    )
    record = _binning(result)
    assert not record.get("unmatched_bin_requests")
    for name in ("unit_brightness", "instance_brightness"):
        assert len(record["factors"][name]["encoding"]["edges"]) - 1 == 4


@pytest.mark.parametrize("workflow_type", WORKFLOWS)
def test_without_intrinsic_factors_the_bin_matches_nothing(workflow_type):
    """The negative: proves the tests above are sensitive to the mechanism."""
    result = _run(workflow_type, _ICDataset(16), {"continuous_factor_bins": {"brightness": 4}})
    assert _binning(result)["unmatched_bin_requests"] == ["brightness"]


@pytest.mark.parametrize("workflow_type", WORKFLOWS)
def test_a_misspelled_factor_stays_unmatched(workflow_type):
    """Expansion must not swallow a typo to make the envelope look clean."""
    result = _run(
        workflow_type,
        _ODDataset(16),
        {"intrinsic_factors": ("visual",), "continuous_factor_bins": {"brightnes": 4}},
    )
    assert _binning(result)["unmatched_bin_requests"] == ["brightnes"]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "The injection pass asks per_target=False on a classification dataset while the "
        "workflow's own pass asks per_target=True. `scope_key` includes per_target, so the "
        "two land in different cache entries and the whole PIXEL family is computed twice. "
        "Same failure mode the design named for value_range, on an axis no task closed."
    ),
)
def test_no_statistic_is_computed_twice(monkeypatch):
    """The Cost section's promise, which nothing else checks.

    Measured on the flags themselves, not on a call count. A workflow computing statistics
    anyway pays for one pass over each statistic; `load_or_compute_stats` may make a second
    *call* for metrics the first did not cover. It may not compute the same metric twice.
    """
    from dataeval_flow import _cache as cache_module

    calls = []
    original = cache_module._do_compute_stats

    def _spy(dataset, policy, per_image=True, per_target=True, value_range=None):
        calls.append(policy.families_of(None))
        return original(dataset, policy, per_image, per_target, value_range)

    monkeypatch.setattr(cache_module, "_do_compute_stats", _spy)
    _run(
        "audit",
        _ICDataset(),
        {"intrinsic_factors": ("visual", "pixel"), "continuous_factor_bins": {"brightness": 4}},
    )

    recomputed = [a & b for i, a in enumerate(calls) for b in calls[i + 1 :] if a & b]
    assert not recomputed, f"these statistics were computed more than once: {recomputed}"


def test_injection_and_no_injection_do_not_share_a_cache_entry():
    """Keyed by the factor set, or a warmed cache reintroduces the bug it closed."""
    with_stats = _run("audit", _ICDataset(), {"intrinsic_factors": ("visual",)})
    without = _run("audit", _ICDataset(), {})
    assert "brightness" in _binning(with_stats)["factors"]
    assert "brightness" not in _binning(without)["factors"]


def test_value_range_keys_the_metadata_cache():
    """Two ranges produce different injected values, so they must not share an entry.

    Asserted on the values, not on `policy_key`'s output: a differing key string proves the
    mechanism, not that the mechanism is wired to the cache. A dataset given in memory
    declares no `value_range`, so the two reads go through one active cache directly.
    """
    from dataeval_flow._binning import describe_binning
    from dataeval_flow._cache import DatasetCache, active_cache, get_or_compute_metadata
    from dataeval_flow._policy import ResolvedPolicy

    policy_fields: dict[str, Any] = {"intrinsic_factors": ("visual",), "continuous_factor_bins": {"brightness": 4}}
    DatasetCache.clear_instances()
    with active_cache(DatasetCache.get_or_create(None, "ic", "k"), "sel"):
        unit = get_or_compute_metadata(_ICDataset(), ResolvedPolicy(value_range=(0.0, 1.0), **policy_fields))
        byte = get_or_compute_metadata(_ICDataset(), ResolvedPolicy(value_range=(0.0, 255.0), **policy_fields))

    unit_edges = describe_binning(unit)["factors"]["brightness"]["encoding"]["edges"]
    byte_edges = describe_binning(byte)["factors"]["brightness"]["encoding"]["edges"]
    assert unit_edges != byte_edges, "the second read was served the first read's cached metadata"


def test_hashes_are_never_injected():
    result = _run("audit", _ICDataset(), {"intrinsic_factors": ("hash",)})
    factors = set(_binning(result)["factors"])
    assert not factors & {"xxhash", "phash", "dhash", "phash_d4", "dhash_d4"}
