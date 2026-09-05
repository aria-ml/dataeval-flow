"""Tests for the metadata triage engine, over hand-written binning records."""

import math
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
    # The suggestion itself is untouched -- only the remedy names the alternative. No
    # `format`: these read as ISO-8601, which `ParseDateTime` infers without being told.
    assert finding.suggestion.corrections[0] == {
        "kind": "parse_datetime",
        "factor": "weight",
        "every": "month",
    }


def test_sentinels_alone_make_a_complete_remap():
    record = _record(unusable=_unusable(distinct={"text": ["N/A", "unknown", ""]}))
    (finding,) = find_issues(record)
    assert finding.suggestion is not None
    rules = finding.suggestion.corrections[0]["rules"]
    assert finding.suggestion.corrections[0]["kind"] == "remap"
    # Sentinels are answered with NaN, the marker that reads as "no value taken".
    assert all(math.isnan(rule["to"]) for rule in rules)
    assert finding.suggestion.complete is True


def test_a_semantic_value_is_enumerated_and_left_incomplete():
    record = _record(unusable=_unusable(distinct={"text": ["N", "NE", "unknown"]}))
    (finding,) = find_issues(record)
    assert finding.suggestion is not None
    rules = finding.suggestion.corrections[0]["rules"]
    assert [r["match"] for r in rules] == ["N", "NE", "unknown"]
    # The two bearings are left for the user; only the sentinel is answered.
    assert [r["to"] for r in rules[:2]] == [None, None]
    assert math.isnan(rules[2]["to"])
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
    # A complete factor's sentinels are answered with NaN, so it carries no null at all --
    # and therefore nothing a TODO marker could attach to.
    quality_section = lines[quality_line_idx:]
    assert any("to: .nan" in line for line in quality_section), "quality should answer its sentinels"
    assert not any("to: null" in line for line in quality_section), "quality should have no nulls"
    assert not any("# TODO" in line for line in quality_section), "quality should have no TODO markers"


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


def test_a_file_extension_is_not_stripped_to_make_a_number():
    """Dropping `jpg` from `1001.jpg` leaves `1001.`, which `float` accepts and nobody wrote.

    The recognizer is for a number wearing decoration — `6,000`, `12 kg` — not for an
    identifier with a numeric stem. Reading a filename as a number is the domain guess the
    design refuses, and it arrives marked complete, so verification would apply it.
    """
    assert _numeric_drop(["1001.jpg", "1002.jpg", "1015.jpg"]) is None


def test_a_trailing_separator_is_residue_rather_than_a_number():
    assert _numeric_drop(["12.abc", "34.abc"]) is None


def test_decorated_numbers_still_read():
    # The cases the recognizer exists for must keep working.
    assert _numeric_drop(["6,000", "12,400"]) == [","]
    assert _numeric_drop(["12 kg", "45 kg"]) == [" kg"]
    assert _numeric_drop(["3.5 kg", "4.25 kg"]) == [" kg"]


def test_a_filename_column_gets_no_correction_suggested():
    """The end-to-end shape of the same defect, as cppe5 produced it."""
    record = _record(
        unusable={
            "file_name": {
                "reasons": ["cardinality_over_budget"],
                "level": "unit",
                "repairable": True,
                "counts": {"text": 300},
                "distinct": {"text": ["1001.jpg", "1002.jpg", "1015.jpg"]},
                "sampled": True,
            },
        }
    )
    (finding,) = find_issues(record)
    assert finding.suggestion is None


def _derived(kind: str, ftype: str, **fit: Any) -> dict[str, Any]:
    """A factor entry whose encoding nobody pinned, of the given encoding kind."""
    enc: dict[str, Any] = {"kind": kind, "provenance": "derived"}
    enc.update({"edges": [0, 10, 20]} if kind == "bins" else {"levels": ["a", "b"]})
    default = (
        {"bins": [{"code": 1, "count": 5, "min": 0, "max": 9}], "empty": []}
        if kind == "bins"
        else {"levels": [{"code": 0, "value": "a", "count": 5}, {"code": 1, "value": "b", "count": 5}], "empty": []}
    )
    return {"type": ftype, "level": "unit", "encoding": enc, "fit": fit or default}


def test_a_binned_discrete_factor_needs_a_cut_not_a_descriptor():
    """SeaDrone's `altitude` comes back `discrete` yet is binned by `uniform_width`.

    Partitioning on `factor_type` filed every real numeric column under `unreviewed` and told
    it to export a descriptor, so `continuous_factor_bins` was never suggested for anything.
    The encoding's kind is what the remedy actually turns on.
    """
    record = _record(
        factors={
            "altitude": _derived(
                "bins",
                "discrete",
                bins=[{"code": i, "count": 4, "min": i, "max": i + 1} for i in range(1, 4)],
                empty=[],
            )
        }
    )
    (finding,) = find_issues(record)
    assert finding.category == "unbinned"
    assert finding.suggestion is not None
    assert finding.suggestion.policy == {"continuous_factor_bins": {"altitude": 3}}


def test_a_digitized_factor_still_wants_a_descriptor():
    record = _record(factors={"drone": _derived("levels", "categorical")})
    (finding,) = find_issues(record)
    assert finding.category == "unreviewed"
    assert finding.suggestion is None


def test_a_sentinel_does_not_veto_a_timestamp_reading():
    """SeaDrone's `date_time` is timestamps plus one empty string, and got no suggestion.

    A sentinel means "no reading", so it cannot be evidence against how the real values
    read. `ParseDateTime` tolerates it, giving the unrecorded rows a level of their own.
    """
    stamps = [f"2020-08-{d:02d}T10:00:00" for d in range(1, 9)]
    record = _record(
        unusable=_unusable(reasons=["cardinality_over_budget"], sampled=True, distinct={"text": ["", *stamps]})
    )
    (finding,) = find_issues(record)
    assert finding.suggestion is not None
    assert finding.suggestion.corrections[0]["kind"] == "parse_datetime"


def test_a_sentinel_does_not_veto_a_numeric_reading():
    record = _record(unusable=_unusable(distinct={"text": ["N/A", "6,000", "12,400"]}))
    (finding,) = find_issues(record)
    assert finding.suggestion is not None
    assert finding.suggestion.corrections[0]["kind"] == "parse_value"


def test_values_that_are_only_sentinels_still_remap():
    record = _record(unusable=_unusable(distinct={"text": ["N/A", "unknown"]}))
    (finding,) = find_issues(record)
    assert finding.suggestion is not None
    assert finding.suggestion.corrections[0]["kind"] == "remap"
    assert finding.suggestion.complete is True


def test_an_iso_timestamp_with_microseconds_is_read_without_pinning_a_format():
    """SeaDrone's `date_time` is `2020-08-25T14:19:24.650133`, which no `%H:%M:%S` matches.

    `ParseDateTime` infers ISO-8601 on its own, so the correction leaves `format` unset
    rather than pinning a pattern that would have to enumerate every ISO spelling.
    """
    stamps = [f"2020-08-{d:02d}T14:19:24.650133" for d in range(1, 9)]
    record = _record(
        unusable=_unusable(reasons=["cardinality_over_budget"], sampled=True, distinct={"text": ["", *stamps]})
    )
    (finding,) = find_issues(record)
    assert finding.suggestion is not None
    entry = finding.suggestion.corrections[0]
    assert entry["kind"] == "parse_datetime"
    assert "format" not in entry
    # August 1st-8th straddles two ISO weeks, and `week` is the coarsest period that tells
    # these apart -- the granularity is chosen from the values, format or no format.
    assert entry["every"] == "week"


def test_a_sentinel_maps_to_not_a_number_rather_than_to_null():
    """`None` is not how a reading is marked unrecorded — it leaves the column mixed.

    `Remap`'s `None` is the *key* catch-all; as a target it is simply a non-numeric value, so
    a column of 198 numbers and two nulls stays unusable and the correction that claimed to
    be complete recovers nothing. NaN is the marker that takes: it makes the column numeric
    and lands the row on the reserved missing code.
    """
    record = _record(unusable=_unusable(distinct={"text": ["N/A", "unknown"]}))
    (finding,) = find_issues(record)
    assert finding.suggestion is not None
    targets = [rule["to"] for rule in finding.suggestion.corrections[0]["rules"]]
    assert all(isinstance(t, float) and math.isnan(t) for t in targets)
    assert finding.suggestion.complete is True


def test_a_placeholder_stays_null_beside_a_sentinel():
    """Only the value flow refuses to code is left for the user; the sentinel is answered."""
    record = _record(unusable=_unusable(distinct={"text": ["N", "unknown"]}))
    (finding,) = find_issues(record)
    assert finding.suggestion is not None
    rules = {r["match"]: r["to"] for r in finding.suggestion.corrections[0]["rules"]}
    assert rules["N"] is None
    assert math.isnan(rules["unknown"])
    assert finding.suggestion.complete is False


def _numeric(
    name: str,
    lo: Any,
    hi: Any,
    rows: int,
    distinct: int,
    bins: int = 3,
    quantiles: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A binned numeric factor entry with a given span, row count and order statistics.

    ``quantiles`` default to an even spread, which is the shape that trips nothing.
    """
    per = rows // bins
    span = hi - lo
    spread = {str(q): lo + span * q for q in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)}
    return {
        name: {
            "type": "discrete",
            "level": "unit",
            "rows": rows,
            "n_distinct": distinct,
            "encoding": {"kind": "bins", "provenance": "derived", "edges": list(range(bins + 1))},
            "fit": {
                "bins": [{"code": i + 1, "count": per, "min": lo, "max": hi} for i in range(bins)],
                "empty": [],
            },
            "distribution": {
                "quantiles": {**spread, **(quantiles or {})},
                "histogram": [per] * 40,
                "cells": 40,
            },
        }
    }


def test_an_integer_column_that_never_repeats_is_an_identifier():
    """SeaDrone's `object_id`: 1305 whole numbers over 1305 detections, cut into 12 bins.

    Upstream drops this shape when the values are text; a numeric one is binned and kept, so
    an arbitrary label reaches the bias evaluators as a factor. Suggesting a bin count for it
    is the one recommendation that actively makes things worse.
    """
    record = _record(factors=_numeric("object_id", 988, 113566, rows=1305, distinct=1305))
    findings = find_issues(record)
    (finding,) = [f for f in findings if f.category == "degenerate"]
    assert finding.severity == "warning"
    assert finding.suggestion is not None
    assert finding.suggestion.policy == {"exclude": ["object_id"]}


def test_the_bin_count_is_withdrawn_from_an_identifier():
    record = _record(factors=_numeric("object_id", 988, 113566, rows=1305, distinct=1305))
    unbinned = [f for f in find_issues(record) if f.category == "unbinned"]
    assert unbinned, "the factor is still reported as cut from this draw"
    assert unbinned[0].suggestion is None, "but no cut is suggested for it"


def test_an_all_distinct_float_column_is_not_an_identifier():
    """The false positive this rule has to avoid: a measurement at any real precision never
    repeats either, and binning one of those is what binning is for."""
    record = _record(factors=_numeric("altitude", 0.4, 998.7, rows=200, distinct=200))
    assert [f for f in find_issues(record) if f.category == "degenerate"] == []


def test_a_repeating_integer_column_is_not_an_identifier():
    record = _record(factors=_numeric("frame", 178, 19800, rows=200, distinct=168))
    assert [f for f in find_issues(record) if f.category == "degenerate"] == []


def test_a_quarter_of_the_rows_on_one_extreme_is_reported():
    """SeaDrone's `-1`: `min == p25`, so a quarter of the column sits on its lowest value.

    Judged from the order statistics, not from the cut — the whole point. `-1` is a quarter of
    `altitude` while landing inside a single bin of its two-bin cut, where a chart drawn at
    that cut cannot show it at all.
    """
    factors: dict[str, Any] = {}
    for name in ("altitude", "compass_heading", "speed"):
        factors.update(_numeric(name, -1.0, 300.0, rows=200, distinct=78, quantiles={"0.0": -1.0, "0.25": -1.0}))
    findings = [f for f in find_issues(_record(factors=factors)) if f.category == "floor_mass"]
    assert sorted(f.factor for f in findings) == ["altitude", "compass_heading", "speed"]
    assert findings[0].detail["value"] == -1.0
    # Sharing the extreme is corroboration, carried in `detail` rather than permuted through
    # every remedy.
    assert sorted(findings[0].detail["shared_with"]) == ["compass_heading", "speed"]
    assert "2 other factors" in findings[0].remedy


def test_a_mass_at_one_extreme_is_reported_even_in_a_single_column():
    """The cross-column rule needed three columns to agree; this needs none."""
    record = _record(
        factors=_numeric("altitude", -1.0, 300.0, rows=200, distinct=78, quantiles={"0.0": -1.0, "0.25": -1.0})
    )
    (finding,) = [f for f in find_issues(record) if f.category == "floor_mass"]
    assert finding.detail["shared_with"] == []
    assert "marker" in finding.remedy


def test_a_mass_at_the_top_is_reported_too():
    record = _record(
        factors=_numeric("score", 0.0, 9999.0, rows=200, distinct=60, quantiles={"0.75": 9999.0, "1.0": 9999.0})
    )
    (finding,) = [f for f in find_issues(record) if f.category == "floor_mass"]
    assert finding.detail["value"] == 9999.0


def test_an_ordinary_spread_holds_no_mass():
    """`xspeed` reaches -11.5 and means it: its lowest value is not a quarter of the column."""
    factors = {
        **_numeric("xspeed", -11.5, 11.1, rows=200, distinct=46),
        **_numeric("yspeed", -7.4, 8.6, rows=200, distinct=49),
    }
    assert [f for f in find_issues(_record(factors=factors)) if f.category == "floor_mass"] == []


def test_a_constant_column_is_degenerate_rather_than_a_floor_mass():
    """It has no spread for a mass to be a mass *of*, and `degenerate` already says so."""
    record = _record(
        factors=_numeric(
            "flat",
            5.0,
            5.0,
            rows=200,
            distinct=1,
            quantiles={str(q): 5.0 for q in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)},
        )
    )
    categories = {f.category for f in find_issues(record)}
    assert "floor_mass" not in categories


def test_the_cut_is_withdrawn_from_a_column_with_a_floor_mass():
    record = _record(
        factors=_numeric("altitude", -1.0, 300.0, rows=200, distinct=78, quantiles={"0.0": -1.0, "0.25": -1.0})
    )
    unbinned = [f for f in find_issues(record) if f.category == "unbinned"]
    assert unbinned
    assert unbinned[0].suggestion is None


def test_two_identifiers_both_reach_the_exclude_list():
    record = _record(
        factors={
            **_numeric("object_id", 1, 9999, rows=1305, distinct=1305),
            **_numeric("track_id", 2, 8888, rows=1305, distinct=1305),
        }
    )
    stanza = to_policy_stanza(find_issues(record))
    assert sorted(stanza["exclude"]) == ["object_id", "track_id"]
