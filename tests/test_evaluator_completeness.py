"""Every attribute of a built-in evaluator's output reaches its JSON: in `data()`, as an extra, or named here.

A DataEval release that adds a result attribute then fails this test, rather than the attribute being left out of
every `to_dict()` and `export()` without anyone noticing.
"""

from collections.abc import Mapping
from typing import Any

import pytest

from dataeval_flow.evaluators import get_evaluator
from dataeval_flow.evaluators._registry import _BUILTINS
from tests.evaluator_toys import output_json, toy_run

# Attributes that restate `data()` in another form: filtered views of its rows, or the array it returns.
_VIEWS: dict[str, frozenset[str]] = {
    "duplicates": frozenset(
        {
            "crossing",
            "detections",
            "exact",
            "factor_groups",
            "frames",
            "items",
            "near",
            "sequences",
            "targets",
            "tracks",
        }
    ),
    "outliers": frozenset({"outliers"}),
    "prioritize": frozenset({"indices"}),
    # `alignment` is the same `LabelAlignment` `data()` already dumps, as a model rather than a dict.
    "label-alignment": frozenset({"alignment"}),
}

# Attributes that echo the inputs or settings back, most kept by DataEval for re-detection. Never serialized.
_ECHOES: dict[str, frozenset[str]] = {
    "duplicates": frozenset(
        {
            "annotation_digests",
            "calculation_results",
            "cluster_result",
            "cluster_sensitivity",
            "flags",
            "frame_map",
            "hash_radius",
            "item_count",
            "levels",
            "max_segment_gap",
            "merge_near_duplicates",
            "min_segment_frames",
            "min_track_frames",
            "redundancy_radius",
            "segment_offset_tolerance",
            "track_map",
            "verify_alignment",
        }
    ),
    "outliers": frozenset(
        {"calculation_results", "cluster_stats", "cluster_threshold", "dataset_steps", "outlier_threshold"}
    ),
    "label-health": frozenset(),
    "triage": frozenset(),
    "balance": frozenset({"plot_type"}),
    "diversity": frozenset({"plot_type"}),
    "parity": frozenset(),
    "representation": frozenset(),
    "coverage": frozenset({"class_axis"}),
    "prioritize": frozenset({"class_labels", "method", "num_bins", "order", "policy"}),
    "drift-domain-classifier": frozenset(),
    "drift-kneighbors": frozenset(),
    "drift-mmd": frozenset(),
    "drift-univariate": frozenset(),
    "drift-wasserstein": frozenset(),
    "ood-domain-classifier": frozenset(),
    "ood-kneighbors": frozenset(),
    # `ontology` is the config's own input, and `ontology_source` how the config named it, read back rather than
    # serialized: `conform` reaches both off the result.
    "label-alignment": frozenset({"ontology", "ontology_source"}),
}


def _public_attributes(output: Any) -> set[str]:
    return {name for name in dir(output) if not name.startswith("_") and not callable(getattr(output, name))}


def test_every_builtin_is_accounted_for():
    missing = sorted(set(_BUILTINS) - set(_ECHOES))
    assert not missing, f"list {missing}'s views and echoes here"


@pytest.mark.parametrize("name", sorted(_BUILTINS))
def test_every_output_attribute_reaches_the_json(name: str):
    result = toy_run(name)
    assert result.success, result.errors
    extras = set(get_evaluator(name).output_extras)
    views, echoes = _VIEWS.get(name, frozenset()), _ECHOES[name]
    assert not extras & (views | echoes), f"{name}: an attribute is listed twice"
    data = result.output.data()
    in_data = set(data) if isinstance(data, Mapping) else set()
    unaccounted = _public_attributes(result.output) - in_data - extras - views - echoes
    assert not unaccounted, f"{name}: {sorted(unaccounted)} are in neither data(), output_extras, _VIEWS nor _ECHOES"
    assert set(output_json(result).get("extras", {})) == extras
