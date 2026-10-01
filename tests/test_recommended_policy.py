"""factor-triage's recommended policy: the suggested fixes plus a pin for every factor left unpinned, read from this
data, with the caveat that such a policy can mislead (docs/superpowers/specs/2026-09-30-recommended-policy-design.md).
"""

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from dataeval import Metadata

from dataeval_flow import run_tasks
from dataeval_flow._binning import write_descriptor
from dataeval_flow._blocks import Code, Paragraph
from dataeval_flow._cache import DatasetCache
from dataeval_flow._policy import ResolvedPolicy
from dataeval_flow._recommend import CAVEAT, complete_stanza, pinned_count, recommend, render_recommendation
from dataeval_flow._triage_report import build_findings
from dataeval_flow.evaluators.quality._triage import read_back
from dataeval_flow.steps import ChainResult
from tests.chain_toys import chain_pipeline
from tests.triage_toys import (
    AltitudeDataset,
    AltitudeWeatherDataset,
    LatitudeDataset,
    MixedWeightDataset,
    OcclusionDataset,
    SpeedDataset,
    WeatherDataset,
)

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
        "continuous_factor_bins": {"altitude": _INF},
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
        "continuous_factor_bins": {"altitude": _INF, "depth": 10},
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


def test_a_held_value_says_it_was_not_dropped() -> None:
    text = render_recommendation({"factor_levels": {"weather": ["clear"]}}, {"factors": {}}, {}, held={"speed": [0.0]})
    assert "  # not dropped: decide whether speed's 0.0 is a marker or a reading" in text.splitlines()


def test_a_dropped_value_is_escaped_in_its_comment() -> None:
    completed, dropped = complete_stanza(
        {"corrections": [{"kind": "remap", "factor": "f", "rules": [{"match": "a\nb'c", "to": None}]}]}
    )
    text = render_recommendation(completed, {"factors": {}}, dropped)
    assert yaml.safe_load(text)["metadata"][0]["corrections"][0]["rules"][0]["match"] == "a\nb'c"
    assert '# dropped by default: decide what "a\\nb\'c" means' in text


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


def _floated(stanza: dict[str, Any]) -> dict[str, Any]:
    """`stanza` with each cut's edges as floats, which is how its YAML reads back."""
    bins = {k: [float(e) for e in v] if isinstance(v, list) else v for k, v in stanza["continuous_factor_bins"].items()}
    return {**stanza, "continuous_factor_bins": bins}


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
    assert _canonical(loaded) == _canonical(_floated(stanza))


_UNPINNED = {"derived", "count"}


def _triage(
    dataset: Any, policy: dict[str, Any] | None = None, *, verify: bool = True, data_dir: Path | None = None
) -> ChainResult:
    """One `factor-triage` step and its `metadata-issues` check over `dataset`, under `policy` where one is given."""
    DatasetCache.clear_instances()
    evaluator: dict[str, Any] = {"name": "triage", "type": "factor-triage", "verify": verify}
    if policy is not None:
        evaluator["metadata"] = "p"
    steps = [
        {"name": "triage", "evaluator": "triage", "input": "data"},
        {"name": "issues", "check": "metadata-issues", "input": "triage"},
    ]
    config = chain_pipeline(
        evaluators=[evaluator],
        workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        datasets={"src": dataset},
        extra={"metadata": [{"name": "p", **policy}]} if policy is not None else None,
    )
    result = run_tasks(config, data_dir=data_dir)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


def _data(result: ChainResult) -> dict[str, Any]:
    return result.steps["triage"].output.data()


def _record(result: ChainResult) -> dict[str, Any]:
    binning = result.metadata.metadata_binning
    assert binning is not None
    return binning


def test_the_recommendation_pins_a_derived_cut_by_its_edges() -> None:
    first = _triage(AltitudeDataset())
    edges = _record(first)["factors"]["altitude"]["encoding"]["edges"]
    data = _data(first)
    assert data["recommended_policy"] == {"continuous_factor_bins": {"altitude": edges}}
    assert "altitude: [-.inf, " in data["recommended_policy_yaml"]
    assert data["recommendation_error"] is None


@pytest.mark.parametrize(
    ("dataset", "still_unreadable"),
    [
        (AltitudeDataset, set()),
        (MixedWeightDataset, set()),
        (LatitudeDataset, set()),
        (OcclusionDataset, set()),
        (WeatherDataset, {"serial"}),
    ],
)
def test_a_run_under_the_recommendation_reads_every_factor_as_declared(
    dataset: Any, still_unreadable: set[str]
) -> None:
    first = _triage(dataset())
    again = _triage(dataset(), _data(first)["recommended_policy"])
    issues = _data(again)["findings"]
    assert not [f for f in issues if f.category in ("unbinned", "unreviewed")]
    assert {f.factor for f in issues if f.category == "unreadable"} == still_unreadable
    factors = _record(again)["factors"]
    assert factors
    assert not {name for name, info in factors.items() if (info.get("encoding") or {}).get("provenance") in _UNPINNED}


def test_a_readable_factor_keeps_its_bins_under_the_recommendation() -> None:
    first = _triage(AltitudeDataset())
    again = _triage(AltitudeDataset(), _data(first)["recommended_policy"])

    def counts(result: ChainResult) -> list[int]:
        return [b["count"] for b in _record(result)["factors"]["altitude"]["fit"]["bins"]]

    assert counts(again) == counts(first)


@pytest.mark.parametrize(
    ("dataset", "factor", "values"),
    [(LatitudeDataset, "latitude", ["N", "S"]), (OcclusionDataset, "occlusion", ["high"])],
)
def test_a_mixed_column_drops_its_strings_by_default(dataset: Any, factor: str, values: list[str]) -> None:
    data = _data(_triage(dataset()))
    (correction,) = [c for c in data["recommended_policy"]["corrections"] if c["factor"] == factor]
    assert [rule["match"] for rule in correction["rules"]] == values
    assert all(math.isnan(rule["to"]) for rule in correction["rules"])
    for value in values:
        assert f"# dropped by default: decide what '{value}' means" in data["recommended_policy_yaml"]
    (suggested,) = [c for c in data["suggested_policy"]["corrections"] if c["factor"] == factor]
    assert all(rule["to"] is None for rule in suggested["rules"])  # the suggestion still leaves them to the user


def test_a_cut_from_a_declared_count_is_pinned_by_the_edges_it_placed() -> None:
    result = _triage(AltitudeDataset(), {"continuous_factor_bins": {"altitude": 4}})
    encoding = _record(result)["factors"]["altitude"]["encoding"]
    assert encoding["provenance"] == "count"
    data = _data(result)
    assert data["recommended_policy"]["continuous_factor_bins"]["altitude"] == encoding["edges"]
    assert "# was a count of 4; these are the edges DataEval placed from this data" in data["recommended_policy_yaml"]
    assert "# Merge these into your policy 'p'." in data["recommended_policy_yaml"]


def test_nothing_is_recommended_where_the_policy_pins_every_factor() -> None:
    first = _triage(AltitudeDataset())
    edges = _record(first)["factors"]["altitude"]["encoding"]["edges"]
    data = _data(_triage(AltitudeDataset(), {"continuous_factor_bins": {"altitude": edges}}))
    assert (data["recommended_policy"], data["recommended_policy_yaml"], data["recommendation_error"]) == (
        None,
        None,
        None,
    )


def test_a_factor_the_descriptor_pins_is_not_named_again(tmp_path: Path) -> None:
    write_descriptor(_record(_triage(AltitudeDataset())), tmp_path / "bins.json")
    result = _triage(AltitudeDataset(), {"encoding": "bins.json"}, data_dir=tmp_path)
    assert _record(result)["factors"]["altitude"]["encoding"]["provenance"] == "derived"  # as the export wrote it
    assert _data(result)["recommended_policy"] is None


def test_a_descriptor_pinning_one_factor_leaves_the_other_to_the_recommendation(tmp_path: Path) -> None:
    write_descriptor(_record(_triage(AltitudeDataset())), tmp_path / "bins.json")
    result = _triage(AltitudeWeatherDataset(), {"encoding": "bins.json"}, data_dir=tmp_path)
    recommendation = _data(result)["recommended_policy"]
    assert recommendation == {"factor_levels": {"weather": ["clear", "fog", "rain"]}}
    _triage(AltitudeWeatherDataset(), {"encoding": "bins.json", **recommendation}, data_dir=tmp_path)  # no "both"


def test_triage_suggests_nothing_for_a_factor_the_descriptor_pins(tmp_path: Path) -> None:
    write_descriptor(_record(_triage(AltitudeDataset())), tmp_path / "bins.json")
    data = _data(_triage(AltitudeWeatherDataset(), {"encoding": "bins.json"}, data_dir=tmp_path))
    assert [(f.factor, f.category) for f in data["findings"]] == [("weather", "unreviewed")]
    assert "altitude" not in (data["suggested_policy"].get("continuous_factor_bins") or {})
    pasted = {"encoding": "bins.json", **data["suggested_policy"], **data["recommended_policy"]}
    _triage(AltitudeWeatherDataset(), pasted, data_dir=tmp_path)  # refused if anything names altitude twice


def test_the_recommendation_does_not_wait_for_verify() -> None:
    data = _data(_triage(AltitudeDataset(), verify=False))
    assert data["verification"] == []
    assert data["recommended_policy"] is not None


def test_a_failed_read_back_leaves_the_findings() -> None:
    with patch("dataeval_flow.evaluators.quality._triage.read_back", side_effect=RuntimeError("boom")):
        data = _data(_triage(MixedWeightDataset()))
    assert (data["recommended_policy"], data["recommended_policy_yaml"], data["recommendation_error"]) == (
        None,
        None,
        "boom",
    )
    assert [f.factor for f in data["findings"]] == ["weight"]
    assert [(v.factor, v.recovered) for v in data["verification"]] == [("weight", True)]


def test_the_chain_records_the_metadata_as_read_not_as_read_back() -> None:
    result = _triage(MixedWeightDataset())
    assert "weight" in (_record(result).get("unusable") or {})
    assert "weight" in _data(result)["recommended_policy"]["continuous_factor_bins"]


def test_read_back_never_changes_the_metadata_it_is_given() -> None:
    metadata = Metadata(WeatherDataset())
    names, excluded = list(metadata.factor_names), set(metadata.exclude)
    record = read_back(metadata, ResolvedPolicy(), {"exclude": ["weather"]})
    assert (list(metadata.factor_names), set(metadata.exclude)) == (names, excluded)
    assert "weather" not in (record.get("factors") or {})
    assert record["excluded"] == ["weather"]


def test_the_recommendation_is_json() -> None:
    body = json.loads(json.dumps(_triage(AltitudeDataset()).to_dict(), allow_nan=False))
    data = body["steps"]["triage"]["output"]["data"]
    assert list(data["recommended_policy"]["continuous_factor_bins"]) == ["altitude"]


def test_metadata_issues_shows_the_recommendation_last_with_its_caveat_first() -> None:
    result = _triage(AltitudeDataset())
    finding = result.findings[-1]
    assert (finding.severity, finding.title, finding.brief) == (
        "info",
        "Recommended policy",
        "pins 1 factor as read from this data",
    )
    caveat, code = finding.blocks
    assert isinstance(caveat, Paragraph)
    assert caveat.text == CAVEAT
    assert isinstance(code, Code)
    assert code.language == "yaml"
    assert code.text == _data(result)["recommended_policy_yaml"].rstrip("\n")


def test_a_recommendation_that_pins_nothing_is_briefed_as_completing_corrections() -> None:
    recommended = {"corrections": [{"kind": "remap", "factor": "no", "rules": [{"match": "N", "to": math.nan}]}]}
    data = {"recommended_policy": recommended, "recommended_policy_yaml": "metadata: []\n"}
    (finding,) = [f for f in build_findings(data, 3) if f.title == "Recommended policy"]
    assert finding.brief == "completes the suggested corrections"


def test_a_failed_recommendation_is_a_warning() -> None:
    with patch("dataeval_flow.evaluators.quality._triage.read_back", side_effect=RuntimeError("boom")):
        result = _triage(AltitudeDataset())
    finding = result.findings[-1]
    assert (finding.severity, finding.title, finding.brief) == ("warning", "Recommendation failed", "no recommendation")
    (paragraph,) = finding.blocks
    assert isinstance(paragraph, Paragraph)
    assert paragraph.text == "boom"


def test_nothing_to_recommend_makes_no_finding() -> None:
    first = _triage(AltitudeDataset())
    edges = _record(first)["factors"]["altitude"]["encoding"]["edges"]
    result = _triage(AltitudeDataset(), {"continuous_factor_bins": {"altitude": edges}})
    assert not [f for f in result.findings if f.title in ("Recommended policy", "Recommendation failed")]


def test_a_floor_mass_value_is_held_not_dropped() -> None:
    data = _data(_triage(SpeedDataset()))
    assert [f.category for f in data["findings"] if f.factor == "speed"][0] == "floor_mass"
    assert "corrections" not in data["recommended_policy"]
    assert "# not dropped: decide whether speed's 0.0 is a marker or a reading" in data["recommended_policy_yaml"]
    assert "dropped by default" not in data["recommended_policy_yaml"]
    assert data["recommended_policy"]["continuous_factor_bins"]["speed"] == ["-inf", 8.0, 16.0, "inf"]
    (suggested,) = data["suggested_policy"]["corrections"]
    assert suggested["rules"] == [{"match": 0.0, "to": None}]  # the suggestion still offers the remap
