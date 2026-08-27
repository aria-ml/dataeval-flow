"""Per-workflow reachability of policy-declared intrinsic factors.

The bar these set is the one D4 cleared: flow's own tests passed through the whole life of
the defect, because each half worked and nothing asserted that a policy's declaration
reached a given workflow's result.  So each assertion here runs a real workflow over a real
dataset and reads the envelope a user would read.
"""

import pytest

from tests.test_metadata_injection import _ICDataset, _ODDataset

pytestmark = pytest.mark.required

WORKFLOWS = ["data-cleaning", "data-coverage", "data-analysis"]


# Minimal valid params per workflow: only the fields with no default.
_PARAMS = {
    "data-cleaning": {"outlier_method": "adaptive", "outlier_flags": ["pixel"]},
    "data-coverage": {},
    "data-analysis": {"outlier_method": "adaptive", "outlier_flags": ["pixel"]},
}


def _run(workflow_type: str, dataset, policy_fields: dict, value_range=(0.0, 1.0), cache=None):
    """Execute *workflow_type* over *dataset* under a policy, returning its ResultMetadata.

    Goes through ``WorkflowContext`` + ``execute`` rather than the orchestrator, which is
    how every other workflow test in this suite runs a workflow.  The policy is placed on
    the context directly, standing in for the stamping the orchestrator does — the workflow
    reads it through ``policy_for`` either way.
    """
    from dataeval_flow.policy import ResolvedPolicy
    from dataeval_flow.workflow import DatasetContext, WorkflowContext, get_workflow

    workflow = get_workflow(workflow_type)
    params = workflow.params_schema(**_PARAMS[workflow_type])

    context = WorkflowContext(
        dataset_contexts={
            "default": DatasetContext(name="default", dataset=dataset, value_range=value_range, cache=cache),
        },
        metadata_policy=ResolvedPolicy(value_range=value_range, **policy_fields),
    )
    result = workflow.execute(context, params)
    assert result.success, f"{workflow_type} failed: {result}"
    return result.metadata


def _binning(result_metadata, split: str = "default") -> dict:
    """The binning record, whichever envelope shape the workflow reports.

    `data-analysis` is the one multi-split workflow of the three, so it reports a record
    per split; cleaning and coverage report one dataset's record flat.  Unwrapping here
    rather than asserting only the flat shape is what makes the parametrisation honest —
    the point is that every workflow carries the record, not that they carry it alike.
    """
    record = result_metadata.metadata_binning
    per_split = record.get("per_split")
    return per_split[split] if per_split is not None else record


@pytest.mark.parametrize("workflow_type", WORKFLOWS)
def test_declared_bin_binds_on_classification(workflow_type):
    """The assertion whose absence let D4 ship — once per workflow."""
    result = _run(
        workflow_type,
        _ICDataset(),
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
        _ODDataset(),
        {"intrinsic_factors": ("visual", "pixel"), "continuous_factor_bins": {"brightness": 4}},
    )
    record = _binning(result)
    assert not record.get("unmatched_bin_requests")
    for name in ("unit_brightness", "instance_brightness"):
        assert len(record["factors"][name]["encoding"]["edges"]) - 1 == 4


@pytest.mark.parametrize("workflow_type", WORKFLOWS)
def test_without_intrinsic_factors_the_bin_matches_nothing(workflow_type):
    """The negative: proves the tests above are sensitive to the mechanism."""
    result = _run(workflow_type, _ICDataset(), {"continuous_factor_bins": {"brightness": 4}})
    assert _binning(result)["unmatched_bin_requests"] == ["brightness"]


@pytest.mark.parametrize("workflow_type", WORKFLOWS)
def test_a_misspelled_factor_stays_unmatched(workflow_type):
    """Expansion must not swallow a typo to make the envelope look clean."""
    result = _run(
        workflow_type,
        _ODDataset(),
        {"intrinsic_factors": ("visual",), "continuous_factor_bins": {"brightnes": 4}},
    )
    assert _binning(result)["unmatched_bin_requests"] == ["brightnes"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The injection pass asks per_target=False on a classification dataset while the "
        "workflow's own pass asks per_target=True. `scope_key` includes per_target, so the "
        "two land in different cache entries and the whole PIXEL family is computed twice. "
        "Same failure mode the design named for value_range, on an axis no task closed."
    ),
)
def test_no_statistic_is_computed_twice(monkeypatch, tmp_path):
    """The Cost section's promise, which nothing else checks.

    Measured with a cache active and on the flags themselves, not on a call count: the
    claim is that a workflow computing statistics anyway pays for one pass over each
    statistic, and `load_or_compute_stats` is entitled to a second *call* for the metrics
    the first did not cover.  What it may not do is compute the same metric twice.
    """
    from dataeval_flow import cache as cache_module
    from dataeval_flow.cache import DatasetCache

    calls = []
    original = cache_module._do_compute_stats

    def _spy(dataset, desired_flags, per_image=True, per_target=True, value_range=None):
        calls.append(desired_flags)
        return original(dataset, desired_flags, per_image, per_target, value_range)

    monkeypatch.setattr(cache_module, "_do_compute_stats", _spy)
    _run(
        "data-cleaning",
        _ICDataset(),
        {"intrinsic_factors": ("visual", "pixel"), "continuous_factor_bins": {"brightness": 4}},
        cache=DatasetCache(cache_dir=tmp_path, dataset_name="ds"),
    )

    recomputed = [a & b for i, a in enumerate(calls) for b in calls[i + 1 :] if a & b]
    assert not recomputed, f"these statistics were computed more than once: {recomputed}"


def test_injection_and_no_injection_do_not_share_a_cache_entry():
    """Keyed by the factor set, or a warmed cache reintroduces the bug it closed."""
    with_stats = _run("data-cleaning", _ICDataset(), {"intrinsic_factors": ("visual",)})
    without = _run("data-cleaning", _ICDataset(), {})
    assert "brightness" in _binning(with_stats)["factors"]
    assert "brightness" not in _binning(without)["factors"]


def test_value_range_keys_the_metadata_cache():
    """Two ranges produce different injected values, so they must not share an entry.

    Asserted on the values rather than on `policy_key`'s output: a key string that differs
    proves the mechanism, not that the mechanism is wired to the cache.
    """
    policy_fields = {"intrinsic_factors": ("visual",), "continuous_factor_bins": {"brightness": 4}}
    unit = _run("data-cleaning", _ICDataset(), policy_fields, value_range=(0.0, 1.0))
    byte = _run("data-cleaning", _ICDataset(), policy_fields, value_range=(0.0, 255.0))

    unit_edges = _binning(unit)["factors"]["brightness"]["encoding"]["edges"]
    byte_edges = _binning(byte)["factors"]["brightness"]["encoding"]["edges"]
    assert unit_edges != byte_edges, "the second run was served the first run's cached metadata"


def test_hashes_are_never_injected():
    result = _run("data-cleaning", _ICDataset(), {"intrinsic_factors": ("hash",)})
    factors = set(_binning(result)["factors"])
    assert not factors & {"xxhash", "phash", "dhash", "phash_d4", "dhash_d4"}
