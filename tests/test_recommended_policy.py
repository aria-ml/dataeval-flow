"""factor-triage's recommended policy: the suggested fixes plus a pin for every factor left unpinned, read from this
data, with the caveat that such a policy can mislead (docs/superpowers/specs/2026-09-30-recommended-policy-design.md).
"""

import json
import math
from typing import Any

import yaml

from dataeval_flow._recommend import CAVEAT, complete_stanza, pinned_count, recommend, render_recommendation

_INF = ["-inf", 265.8536348826109, 504.1481565221535, 742.442678161696, "inf"]


def _cut(
    edges: list[Any], provenance: str = "derived", span: tuple[float, float] | None = (27.5, 981.2)
) -> dict[str, Any]:
    bins = (
        []
        if span is None
        else [
            {"code": 1, "count": 3, "min": span[0], "max": span[0] + 1.0},
            {"code": 2, "count": 2, "min": span[1] - 1.0, "max": span[1]},
        ]
    )
    encoding = {"kind": "bins", "edges": edges, "provenance": provenance, "method": "uniform_width"}
    return {"type": "continuous", "level": "unit", "encoding": encoding, "fit": {"bins": bins, "empty": []}}


def _vocab(levels: list[Any], provenance: str = "derived") -> dict[str, Any]:
    fit = {"levels": [{"code": i, "value": v, "count": 1} for i, v in enumerate(levels)], "empty": []}
    return {
        "type": "categorical",
        "level": "unit",
        "encoding": {"kind": "levels", "levels": levels, "provenance": provenance},
        "fit": fit,
    }


def _canonical(value: Any) -> str:
    """`value` as sorted JSON, where NaN reads as `NaN` on both sides rather than never equalling itself."""
    return json.dumps(value, sort_keys=True)


def test_completing_answers_each_placeholder_with_a_drop() -> None:
    stanza = {
        "corrections": [
            {
                "kind": "remap",
                "factor": "latitude",
                "rules": [{"match": "N", "to": None}, {"match": "-999", "to": math.nan}],
            }
        ]
    }
    completed, dropped = complete_stanza(stanza)
    rules = completed["corrections"][0]["rules"]
    assert math.isnan(rules[0]["to"])
    assert math.isnan(rules[1]["to"])
    assert dropped == {"latitude": ["N"]}
    assert stanza["corrections"][0]["rules"][0]["to"] is None  # the suggestion is left as it was


def test_a_derived_cut_is_pinned_by_its_edges_and_a_vocabulary_by_its_levels() -> None:
    record = {"factors": {"altitude": _cut(_INF), "weather": _vocab(["clear", "fog", "rain"])}}
    assert recommend(record, {}) == {
        "continuous_factor_bins": {"altitude": [-math.inf, *_INF[1:4], math.inf]},
        "factor_levels": {"weather": ["clear", "fog", "rain"]},
    }


def test_a_cut_from_a_declared_count_is_pinned_and_declared_ones_are_not() -> None:
    record = {
        "factors": {
            "counted": _cut([0.0, 1.0, 2.0], "count"),
            "edged": _cut([0.0, 1.0], "edges"),
            "listed": _vocab(["a"], "declared"),
            "reviewed": _vocab(["b"], "accepted"),
        }
    }
    assert recommend(record, {}) == {"continuous_factor_bins": {"counted": [0.0, 1.0, 2.0]}}


def test_a_factor_the_descriptor_names_is_skipped_whatever_its_provenance() -> None:
    record = {"factors": {"altitude": _cut(_INF), "weather": _vocab(["clear"])}}
    assert recommend(record, {}, skip={"altitude"}) == {"factor_levels": {"weather": ["clear"]}}


def test_a_count_suggested_for_a_factor_the_descriptor_names_is_dropped() -> None:
    record = {"factors": {"altitude": _cut(_INF)}}
    assert recommend(record, {"continuous_factor_bins": {"altitude": 4}}, skip={"altitude"}) is None
    assert recommend(record, {"continuous_factor_bins": {"altitude": 4, "depth": 10}}, skip={"altitude"}) == {
        "continuous_factor_bins": {"depth": 10}
    }


def test_edges_replace_a_suggested_count_and_a_count_stays_where_nothing_pins() -> None:
    record = {"factors": {"altitude": _cut(_INF), "depth": {"type": "continuous", "encoding": {}}}}
    completed = {
        "corrections": [{"kind": "parse_value", "factor": "w", "drop": [","]}],
        "continuous_factor_bins": {"altitude": 4, "depth": 10},
    }
    assert recommend(record, completed) == {
        "corrections": [{"kind": "parse_value", "factor": "w", "drop": [","]}],
        "continuous_factor_bins": {"altitude": [-math.inf, *_INF[1:4], math.inf], "depth": 10},
    }


def test_there_is_nothing_to_recommend_where_everything_is_pinned_and_nothing_suggested() -> None:
    assert recommend({"factors": {"edged": _cut([0.0, 1.0], "edges")}}, {}) is None
    assert recommend({}, {}) is None


def test_pinned_count_counts_edges_and_vocabularies_not_counts() -> None:
    stanza = {"continuous_factor_bins": {"a": [0.0, 1.0], "b": 10}, "factor_levels": {"c": ["x"]}}
    assert pinned_count(stanza) == 2


def test_the_caveat_heads_the_yaml() -> None:
    text = render_recommendation({"factor_levels": {"weather": ["clear"]}}, {"factors": {}}, {})
    header = []
    for line in text.splitlines():
        if not line.startswith("# "):
            break
        header.append(line[2:])
    assert " ".join(header) == CAVEAT


def test_merging_into_a_named_policy_is_said_in_the_header_and_names_the_entry() -> None:
    text = render_recommendation(
        {"factor_levels": {"weather": ["clear"]}}, {"factors": {}}, {}, name="p", merge_into="p"
    )
    assert "# Merge these into your policy 'p'." in text
    assert yaml.safe_load(text)["metadata"][0]["name"] == "p"


def test_each_pin_says_what_it_assumes() -> None:
    record = {
        "factors": {
            "altitude": _cut(_INF),
            "counted": _cut([0.0, 1.0, 2.0], "count", span=None),
            "weather": _vocab(["clear", "fog", "rain"]),
        }
    }
    stanza = recommend(record, {})
    assert stanza is not None
    lines = render_recommendation(stanza, record, {}).splitlines()
    (altitude,) = [line for line in lines if line.lstrip().startswith("altitude:")]
    assert altitude.endswith("# seen 27.5 to 981.2; the outer bins are open, so values beyond this range join them")
    assert "altitude: [-.inf, 265.8536348826109, 504.1481565221535, 742.442678161696, .inf]" in altitude
    (counted,) = [line for line in lines if line.lstrip().startswith("counted:")]
    assert counted.endswith("# was a count of 2; these are the edges DataEval placed from this data")
    (weather,) = [line for line in lines if line.lstrip().startswith("weather:")]
    assert weather.endswith("# 3 levels seen; a category this data never saw takes a new code")


def test_a_dropped_value_says_it_was_dropped_by_default() -> None:
    completed, dropped = complete_stanza(
        {"corrections": [{"kind": "remap", "factor": "occlusion", "rules": [{"match": "high", "to": None}]}]}
    )
    text = render_recommendation(completed, {"factors": {}}, dropped)
    assert "to: .nan  # dropped by default: decide what 'high' means" in text


def test_a_sentinel_rule_is_not_said_to_be_dropped() -> None:
    completed, dropped = complete_stanza(
        {"corrections": [{"kind": "remap", "factor": "depth", "rules": [{"match": "-999", "to": math.nan}]}]}
    )
    text = render_recommendation(completed, {"factors": {}}, dropped)
    assert "to: .nan" in text
    assert "dropped by default" not in text


def test_a_factor_that_cannot_be_pinned_is_named() -> None:
    record = {
        "factors": {"depth": {"type": "continuous", "encoding": {}}},
        "unusable": {"serial": {}},
        "excluded": ["object_id"],
    }
    text = render_recommendation({"continuous_factor_bins": {"depth": 10}}, record, {})
    assert "# not pinned: depth has no encoding to read" in text
    assert "# not pinned: serial has no encoding to read" in text
    assert "object_id" not in text


def test_the_yaml_reads_back_as_the_stanza() -> None:
    record = {
        "factors": {
            "a: b": _cut([-1e-300, 0.1, 1 / 3]),
            "yes": _vocab(["null", "yes", "3", 3, "a: b"]),
        }
    }
    completed, dropped = complete_stanza(
        {"corrections": [{"kind": "remap", "factor": "no", "rules": [{"match": "N", "to": None}]}], "exclude": ["id"]}
    )
    stanza = recommend(record, completed)
    assert stanza is not None
    loaded = yaml.safe_load(render_recommendation(stanza, record, dropped))["metadata"][0]
    assert loaded.pop("name") == "standard"
    assert _canonical(loaded) == _canonical(stanza)
