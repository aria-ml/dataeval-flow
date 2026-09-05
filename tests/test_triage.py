"""Tests for the metadata triage engine, over hand-written binning records."""

from typing import Any

from dataeval_flow.triage import find_issues


def _record(**over: Any) -> dict[str, Any]:
    """A minimal describe_binning record: one clean continuous factor."""
    base: dict[str, Any] = {
        "factors": {
            "altitude": {
                "type": "continuous",
                "level": "unit",
                "encoding": {"kind": "bins", "provenance": "count", "edges": [0, 50, 100]},
                "fit": {
                    "bins": [
                        {"code": 1, "count": 30, "min": 0, "max": 49},
                        {"code": 2, "count": 30, "min": 50, "max": 99},
                    ],
                    "empty": [],
                },
            }
        },
        "dropped": {},
        "unusable": {},
        "unreviewed": [],
    }
    base.update(over)
    return base


def test_a_clean_record_yields_nothing():
    assert find_issues(_record()) == []


def test_unusable_mixed_types_is_blocking_and_repairable():
    record = _record(
        unusable={
            "weight": {
                "reasons": ["mixed_types"],
                "level": "unit",
                "repairable": True,
                "counts": {"numeric": 1842, "text": 58},
                "distinct": {"text": ["6,000", "12,400"]},
                "sampled": False,
            },
        }
    )
    (finding,) = find_issues(record)
    assert finding.factor == "weight"
    assert finding.category == "unreadable"
    assert finding.severity == "blocking"
    assert finding.repairable is True
    assert finding.detail["counts"] == {"numeric": 1842, "text": 58}
    assert finding.detail["distinct"]["text"] == ["6,000", "12,400"]


def test_unusable_that_cannot_be_repaired_is_a_note():
    record = _record(
        unusable={
            "histogram": {
                "reasons": ["multi_dimensional"],
                "level": None,
                "repairable": False,
                "counts": {},
                "distinct": {},
                "sampled": False,
            },
        }
    )
    (finding,) = find_issues(record)
    assert finding.severity == "note"
    assert finding.repairable is False


def test_distinct_values_are_never_truncated():
    # Task 2 builds a Remap from these; a truncated set would look complete and not be.
    values = [f"v{i}" for i in range(200)]
    record = _record(
        unusable={
            "code": {
                "reasons": ["mixed_types"],
                "level": "unit",
                "repairable": True,
                "counts": {"numeric": 1, "text": 200},
                "distinct": {"text": values},
                "sampled": False,
            },
        }
    )
    (finding,) = find_issues(record)
    assert finding.detail["distinct"]["text"] == values


def test_unmatched_bin_request_is_blocking_and_suggests_a_near_name():
    record = _record(unmatched_bin_requests=["altitud"])
    (finding,) = find_issues(record)
    assert finding.factor == "altitud"
    assert finding.category == "unbound_request"
    assert finding.severity == "blocking"
    assert finding.detail["near"] == ["altitude"]


def test_a_derived_continuous_cut_is_unbinned():
    record = _record()
    record["factors"]["altitude"]["encoding"]["provenance"] = "derived"
    record["factors"]["altitude"]["fit"]["empty"] = [3, 4, 5]
    (finding,) = find_issues(record)
    assert finding.category == "unbinned"
    assert finding.severity == "warning"


def test_a_derived_vocabulary_is_unreviewed_not_unbinned():
    record = _record(
        factors={
            "scene": {
                "type": "categorical",
                "level": "unit",
                "encoding": {"kind": "levels", "provenance": "derived", "levels": ["a", "b"]},
                "fit": {
                    "levels": [{"code": 0, "value": "a", "count": 30}, {"code": 1, "value": "b", "count": 30}],
                    "empty": [],
                },
            },
        }
    )
    (finding,) = find_issues(record)
    assert finding.category == "unreviewed"
    assert finding.factor == "scene"


def test_a_factor_with_no_encoding_at_all_is_blocking():
    record = _record(factors={"raw": {"type": "continuous", "level": "unit"}})
    (finding,) = find_issues(record)
    assert finding.category == "unbinned"
    assert finding.severity == "blocking"


def test_one_populated_bucket_is_degenerate():
    record = _record()
    record["factors"]["altitude"]["fit"]["bins"] = [{"code": 1, "count": 60, "min": 0, "max": 9}]
    findings = [f for f in find_issues(record) if f.category == "degenerate"]
    assert len(findings) == 1
    assert findings[0].severity == "note"


def test_high_missing_is_degenerate():
    record = _record()
    record["factors"]["altitude"]["fit"]["missing"] = 40
    findings = [f for f in find_issues(record) if f.category == "degenerate"]
    assert findings and "missing" in findings[0].remedy  # noqa: PT018


def test_a_wide_vocabulary_is_not_degenerate():
    # Thin levels are reported by ParityOutput.insufficient_data, not removed. Many
    # sparsely-populated levels is a normal factor, not a finding.
    record = _record(
        factors={
            "city": {
                "type": "categorical",
                "level": "unit",
                "encoding": {"kind": "levels", "provenance": "declared", "levels": [f"c{i}" for i in range(40)]},
                "fit": {"levels": [{"code": i, "value": f"c{i}", "count": 2} for i in range(40)], "empty": []},
            },
        }
    )
    assert find_issues(record) == []


def test_findings_are_ordered_by_severity_then_factor():
    record = _record(
        unusable={
            "histogram": {
                "reasons": ["multi_dimensional"],
                "level": None,
                "repairable": False,
                "counts": {},
                "distinct": {},
                "sampled": False,
            }
        },
        unmatched_bin_requests=["zzz"],
    )
    assert [f.severity for f in find_issues(record)] == ["blocking", "note"]


def test_a_malformed_record_yields_no_finding_rather_than_raising():
    # describe_binning degrades section by section upstream; a triage that dies because
    # one was unavailable is worse than one reporting the rest.
    assert find_issues({}) == []
    assert find_issues({"factors": None, "unusable": None}) == []


def _unusable(**over: Any) -> dict[str, Any]:
    """An unusable entry for a mixed column, with text values the caller names."""
    entry = {
        "reasons": ["mixed_types"],
        "level": "unit",
        "repairable": True,
        "counts": {"numeric": 10, "text": 3},
        "distinct": {"text": []},
        "sampled": False,
    }
    entry.update(over)
    return {"weight": entry}


def test_unit_cruft_becomes_a_parse_value():
    record = _record(unusable=_unusable(distinct={"text": ["6,000", "12,400", "1,203"]}))
    (finding,) = find_issues(record)
    assert finding.suggestion is not None
    assert finding.suggestion.complete is True
    assert finding.suggestion.corrections == [{"kind": "parse_value", "factor": "weight", "drop": [","]}]


def test_a_timestamp_becomes_a_parse_datetime():
    record = _record(
        unusable=_unusable(
            reasons=["cardinality_over_budget"],
            sampled=True,
            distinct={"text": ["2021-03-04", "2021-07-19", "2022-01-02"]},
        )
    )
    (finding,) = find_issues(record)
    assert finding.suggestion is not None
    assert finding.suggestion.corrections[0]["kind"] == "parse_datetime"
    # Coarsest absolute period still telling the values apart.
    assert finding.suggestion.corrections[0]["every"] == "year"
    assert finding.suggestion.complete is True


def test_granularity_goes_finer_only_when_it_has_to():
    record = _record(
        unusable=_unusable(
            reasons=["cardinality_over_budget"],
            sampled=True,
            distinct={"text": ["2021-03-04", "2021-03-19", "2021-03-28"]},
        )
    )
    (finding,) = find_issues(record)
    assert finding.suggestion is not None
    # One year and one month, so neither separates; week is the first that does.
    assert finding.suggestion.corrections[0]["every"] == "week"


def test_sentinels_alone_make_a_complete_remap():
    record = _record(unusable=_unusable(distinct={"text": ["N/A", "unknown", ""]}))
    (finding,) = find_issues(record)
    assert finding.suggestion is not None
    rules = finding.suggestion.corrections[0]["rules"]
    assert finding.suggestion.corrections[0]["kind"] == "remap"
    assert all(rule["to"] is None for rule in rules)
    assert finding.suggestion.complete is True


def test_a_semantic_value_is_enumerated_and_left_incomplete():
    record = _record(unusable=_unusable(distinct={"text": ["N", "NE", "unknown"]}))
    (finding,) = find_issues(record)
    assert finding.suggestion is not None
    rules = finding.suggestion.corrections[0]["rules"]
    assert [r["match"] for r in rules] == ["N", "NE", "unknown"]
    assert all(r["to"] is None for r in rules)
    # "N" is 0 degrees only if the column is a bearing, which flow cannot know.
    assert finding.suggestion.complete is False


def test_a_sampled_column_never_gets_a_remap():
    # Its values are near-unique by definition; no mapping could cover the column.
    record = _record(
        unusable=_unusable(reasons=["cardinality_over_budget"], sampled=True, distinct={"text": ["a7f", "b12", "c93"]})
    )
    (finding,) = find_issues(record)
    assert finding.suggestion is None


def test_an_ambiguous_date_format_is_not_guessed():
    # 03/04/2021 is two different days depending on the reading.
    record = _record(
        unusable=_unusable(
            reasons=["cardinality_over_budget"],
            sampled=True,
            distinct={"text": ["03/04/2021", "05/06/2021", "07/08/2021"]},
        )
    )
    (finding,) = find_issues(record)
    assert finding.suggestion is None
    assert "ambiguous" in finding.remedy


def test_a_non_repairable_finding_gets_no_suggestion():
    record = _record(
        unusable={
            "histogram": {
                "reasons": ["multi_dimensional"],
                "level": None,
                "repairable": False,
                "counts": {},
                "distinct": {},
                "sampled": False,
            }
        }
    )
    (finding,) = find_issues(record)
    assert finding.suggestion is None


def test_unreviewed_contributes_no_policy_fragment():
    # Its remedy is to export a descriptor; there is no path to emit, and inventing one
    # would put an invalid value in the stanza.
    record = _record(
        factors={
            "scene": {
                "type": "categorical",
                "level": "unit",
                "encoding": {"kind": "levels", "provenance": "derived", "levels": ["a", "b"]},
                "fit": {
                    "levels": [{"code": 0, "value": "a", "count": 30}, {"code": 1, "value": "b", "count": 30}],
                    "empty": [],
                },
            }
        }
    )
    (finding,) = find_issues(record)
    assert finding.suggestion is None


def test_unbinned_suggests_the_populated_bin_count():
    record = _record()
    record["factors"]["altitude"]["encoding"]["provenance"] = "derived"
    record["factors"]["altitude"]["fit"]["bins"] = [
        {"code": i, "count": 10, "min": i, "max": i + 1} for i in range(1, 8)
    ]
    record["factors"]["altitude"]["fit"]["empty"] = [8, 9, 10]
    (finding,) = find_issues(record)
    assert finding.suggestion is not None
    assert finding.suggestion.policy == {"continuous_factor_bins": {"altitude": 7}}
    assert finding.suggestion.complete is True


def test_a_no_encoding_categorical_factor_needs_a_vocabulary_not_bins():
    record = _record(factors={"scene": {"type": "categorical", "level": "unit"}})
    (finding,) = find_issues(record)
    assert finding.category == "unbinned"
    assert finding.severity == "blocking"
    assert "vocabulary" in finding.remedy or "levels" in finding.remedy
    assert "bin count" not in finding.remedy
