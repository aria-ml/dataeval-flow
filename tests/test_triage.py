"""Tests for the metadata triage engine, over hand-written binning records."""

from typing import Any

from dataeval_flow.config.schemas import MetadataPolicyConfig
from dataeval_flow.triage import (
    Finding,
    _numeric_drop,
    find_issues,
    incomplete_factors,
    render_stanza,
    suggest,
    to_policy_stanza,
)


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


def test_a_unit_suffix_without_a_separating_space_is_stripped():
    # "kg" is a trailing non-numeric run every value shares, not punctuation, so it can only
    # be found through `_common_suffix` — the letter-stripping fallback must not touch it.
    assert _numeric_drop(["12kg", "45kg", "7kg"]) == ["kg"]


def test_a_unit_suffix_with_a_separating_space_is_stripped():
    # The shared tail includes the space: dropping " kg" (not "kg" alone) is what leaves a
    # bare number.
    assert _numeric_drop(["12 kg", "45 kg", "7 kg"]) == [" kg"]


def test_an_identifier_that_already_ends_in_a_digit_gets_no_suffix_candidate():
    # Ruling A regression guard: `_common_suffix` must not invent a suffix out of "a7f" just
    # because the other values in the column already end in a digit, and the loose fallback
    # must not strip the letter either.
    assert _numeric_drop(["a7f", "b12", "c93"]) is None


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
    # "year" has no recurring counterpart in DataEval's registry, so nothing to name.
    assert "recurring" not in finding.remedy


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
    # "week" has no recurring counterpart either.
    assert "recurring" not in finding.remedy


def test_a_monthly_granularity_names_the_recurring_alternative():
    # The suggestion always emits the absolute reading; the remedy is where a reader learns
    # a recurring `month_of_year` period is also available and answers a different question.
    record = _record(
        unusable=_unusable(
            reasons=["cardinality_over_budget"],
            sampled=True,
            distinct={"text": ["2021-01-15", "2021-02-20", "2021-03-25"]},
        )
    )
    (finding,) = find_issues(record)
    assert finding.suggestion is not None
    assert finding.suggestion.corrections[0]["every"] == "month"
    assert "month_of_year" in finding.remedy
    # The suggestion itself is untouched -- only the remedy names the alternative.
    assert finding.suggestion.corrections[0] == {
        "kind": "parse_datetime",
        "factor": "weight",
        "every": "month",
        "format": "%Y-%m-%d",
    }


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


def test_a_no_encoding_continuous_factor_gets_a_default_bin_suggestion():
    # The sharpest `unbinned` case -- reached the evaluators as raw values -- must still
    # reach the stanza. There is no `fit` to read a populated count from, so this is the one
    # live path `default_bins` exists for.
    record = _record(factors={"raw": {"type": "continuous", "level": "unit"}})
    (finding,) = find_issues(record, default_bins=6)
    assert finding.category == "unbinned"
    assert finding.severity == "blocking"
    assert finding.suggestion is not None
    assert finding.suggestion.policy == {"continuous_factor_bins": {"raw": 6}}
    assert finding.suggestion.complete is True


def test_suggest_honours_its_contract_for_a_hand_built_unbinned_finding():
    # `suggest` is documented as "the config change that would address one finding" -- it
    # must not be an identity function that only echoes a suggestion built elsewhere.
    finding = Finding(
        factor="raw",
        category="unbinned",
        severity="blocking",
        detail={"type": "continuous"},
    )
    suggestion = suggest(finding, default_bins=4)
    assert suggestion is not None
    assert suggestion.policy == {"continuous_factor_bins": {"raw": 4}}


def test_a_degenerate_derived_cut_gets_no_bin_suggestion():
    # A skewed continuous factor where every value lands in one bin is both `unbinned`
    # (a derived cut nobody pinned) and `degenerate` (that cut groups nothing). Both
    # findings are true and both must appear, but the stanza must not pin a cut the
    # `degenerate` finding calls useless.
    record = _record()
    record["factors"]["altitude"]["encoding"]["provenance"] = "derived"
    record["factors"]["altitude"]["fit"]["bins"] = [{"code": 1, "count": 60, "min": 0, "max": 9}]
    findings = find_issues(record)
    by_category = {f.category: f for f in findings}
    assert set(by_category) == {"unbinned", "degenerate"}
    assert by_category["unbinned"].factor == "altitude"
    assert by_category["unbinned"].suggestion is None
    # The remedy -- what a bin declaration would look like -- is untouched; only the
    # machine-readable suggestion that would reach the stanza is suppressed.
    assert "continuous_factor_bins" in by_category["unbinned"].remedy
    assert by_category["degenerate"].factor == "altitude"
    stanza = to_policy_stanza(findings)
    assert "continuous_factor_bins" not in stanza


def test_the_stanza_merges_corrections_and_policy_edits():
    record = _record(unusable=_unusable(distinct={"text": ["6,000", "12,400"]}))
    record["factors"]["altitude"]["encoding"]["provenance"] = "derived"
    stanza = to_policy_stanza(find_issues(record))
    assert stanza["corrections"] == [{"kind": "parse_value", "factor": "weight", "drop": [","]}]
    assert stanza["continuous_factor_bins"] == {"altitude": 2}


def test_the_stanza_validates_as_a_policy():
    # A stanza that does not validate is a bug, not a suggestion.
    record = _record(unusable=_unusable(distinct={"text": ["6,000"]}))
    stanza = to_policy_stanza(find_issues(record))
    MetadataPolicyConfig.model_validate({"name": "standard", **stanza})


def test_an_incomplete_suggestion_still_reaches_the_stanza():
    # The user has to see the skeleton to fill it in; verification is what refuses to run it.
    record = _record(unusable=_unusable(distinct={"text": ["N", "NE"]}))
    stanza = to_policy_stanza(find_issues(record))
    assert stanza["corrections"][0]["rules"] == [
        {"match": "N", "to": None},
        {"match": "NE", "to": None},
    ]


def test_no_findings_makes_no_stanza():
    assert to_policy_stanza([]) == {}


def test_the_rendered_stanza_is_a_metadata_block():
    record = _record(unusable=_unusable(distinct={"text": ["6,000"]}))
    text = render_stanza(to_policy_stanza(find_issues(record)))
    assert text.startswith("metadata:")
    assert "- name: standard" in text
    assert "kind: parse_value" in text


def test_sentinel_remap_renders_with_no_todo_marker():
    # A sentinels-only remap is complete; the user's job is done.
    record = _record(unusable=_unusable(distinct={"text": ["N/A", "unknown"]}))
    findings = find_issues(record)
    stanza = to_policy_stanza(findings)
    text = render_stanza(stanza, incomplete=incomplete_factors(findings))
    # No incomplete factors, so no TODO markers at all
    assert "# TODO" not in text


def test_semantic_remap_renders_with_todo_marker():
    # A semantic remap is incomplete; the user must supply the mapping.
    record = _record(unusable=_unusable(distinct={"text": ["N", "NE"]}))
    findings = find_issues(record)
    stanza = to_policy_stanza(findings)
    text = render_stanza(stanza, incomplete=incomplete_factors(findings))
    # The incomplete semantic remap should have TODO markers
    assert "to: null        # TODO" in text


def test_mixed_complete_and_incomplete_factors_mark_only_incomplete():
    # Multiple factors: complete ones have bare nulls, incomplete ones have TODOs.
    record = _record(
        unusable={
            "bearing": {  # semantic, incomplete
                "reasons": ["mixed_types"],
                "level": "unit",
                "repairable": True,
                "counts": {"text": 3},
                "distinct": {"text": ["N", "NE"]},
                "sampled": False,
            },
            "quality": {  # sentinels, complete
                "reasons": ["mixed_types"],
                "level": "unit",
                "repairable": True,
                "counts": {"text": 3},
                "distinct": {"text": ["N/A", "unknown"]},
                "sampled": False,
            },
        }
    )
    findings = find_issues(record)
    stanza = to_policy_stanza(findings)
    text = render_stanza(stanza, incomplete=incomplete_factors(findings))

    # Verify the structure of marked vs unmarked nulls
    lines = text.split("\n")
    bearing_line_idx = None
    quality_line_idx = None

    for i, line in enumerate(lines):
        if "factor: bearing" in line:
            bearing_line_idx = i
        if "factor: quality" in line:
            quality_line_idx = i

    assert bearing_line_idx is not None, "bearing factor not found in output"
    assert quality_line_idx is not None, "quality factor not found in output"

    # Check lines in bearing section (between bearing and next factor or end)
    end_bearing = quality_line_idx if quality_line_idx > bearing_line_idx else len(lines)
    bearing_section = lines[bearing_line_idx:end_bearing]
    has_bearing_todo = any("to: null        # TODO" in line for line in bearing_section)
    assert has_bearing_todo, "bearing (incomplete) should have at least one TODO marker"

    # Check lines in quality section (between quality and end)
    quality_section = lines[quality_line_idx:]
    has_quality_bare_null = any("to: null" in line and "# TODO" not in line for line in quality_section)
    has_quality_todo = any("# TODO" in line for line in quality_section)
    assert has_quality_bare_null, "quality (complete) should have bare nulls"
    assert not has_quality_todo, "quality (complete) should not have TODO markers"


def test_distinct_values_with_factor_substring_dont_break_marking():
    # Regression: values like "safety factor: 2.0" contain "factor:" which could
    # confuse text-based factor tracking. All incomplete factor rules must be marked.
    record = _record(
        unusable={
            "bearing": {
                "reasons": ["mixed_types"],
                "level": "unit",
                "repairable": True,
                "counts": {"text": 3},
                "distinct": {"text": ["N", "safety factor: 2.0", "NW"]},
                "sampled": False,
            },
        }
    )
    findings = find_issues(record)
    stanza = to_policy_stanza(findings)
    text = render_stanza(stanza, incomplete=incomplete_factors(findings))

    # All three null values must be marked since bearing is incomplete
    marked_count = text.count("to: null        # TODO")
    assert marked_count == 3, f"Expected 3 marked nulls but found {marked_count}"


def test_values_containing_null_colon_dont_break_marking():
    # Values like "value: null" must not cause spurious marking issues
    record = _record(
        unusable={
            "config": {
                "reasons": ["mixed_types"],
                "level": "unit",
                "repairable": True,
                "counts": {"text": 2},
                "distinct": {"text": ["active", "value: null"]},
                "sampled": False,
            },
        }
    )
    findings = find_issues(record)
    stanza = to_policy_stanza(findings)
    text = render_stanza(stanza, incomplete=incomplete_factors(findings))

    # The config factor is incomplete, so both rules should be marked
    marked_count = text.count("to: null        # TODO")
    assert marked_count == 2, f"Expected 2 marked nulls but found {marked_count}"
