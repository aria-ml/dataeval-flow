"""Tests for the metadata-triage workflow."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from dataeval import Metadata

from dataeval_flow import run
from dataeval_flow._binning_report import distribution_blocks
from dataeval_flow._blocks import Code, Distribution, ItemRef, Proportion
from dataeval_flow._cache import DatasetCache
from dataeval_flow._policy import ResolvedPolicy, build_correction
from dataeval_flow._triage import TriageFinding, find_issues
from dataeval_flow._triage_report import build_findings, minority_kind, summarize
from dataeval_flow.config import ParseValueCorrectionConfig
from dataeval_flow.evaluators.quality import VerificationEntry
from dataeval_flow.evaluators.quality._triage import _factor_recovered, places, verify
from dataeval_flow.workflows.metadata_triage import MetadataTriageConfig, MetadataTriageWorkflow
from tests.finding_blocks import blocks_of, bullets, column, fields, paragraphs, rendered, sections, tables
from tests.test_triage import _numeric, _record
from tests.triage_toys import AltitudeDataset, LatitudeDataset, MixedWeightDataset, OcclusionDataset


def test_parameters_default_to_verifying():
    params = MetadataTriageConfig()
    assert params.verify is True
    assert params.checks.metadata_issues.max_examples == 20
    assert params.default_bins == 10
    assert params.min_missing_fraction == 0.2


def test_parameters_accept_a_named_policy():
    # MetadataConfigMixin is what makes `metadata: standard` resolve.
    assert MetadataTriageConfig(metadata="standard").metadata == "standard"


def _mixed_metadata(n: int = 60) -> Metadata:
    """Metadata whose ``weight`` factor mixes numerals with numerals wearing commas."""
    return Metadata(MixedWeightDataset(n))


def _altitude_metadata(n: int = 60) -> Metadata:
    """Metadata whose ``altitude`` factor is continuous and cut from this draw."""
    return Metadata(AltitudeDataset(n))


def _describe(metadata: Metadata) -> dict:
    from dataeval_flow._binning import describe_binning

    return describe_binning(metadata)


def test_the_workflow_reports_its_name():
    assert MetadataTriageWorkflow().name == "metadata-triage"


def test_a_mixed_column_is_found_and_a_correction_suggested():
    from dataeval_flow._triage import find_issues, to_policy_stanza

    findings = find_issues(_describe(_mixed_metadata()))
    weight = [f for f in findings if f.factor == "weight"]
    assert weight
    assert weight[0].category == "unreadable"
    assert to_policy_stanza(findings)["corrections"][0]["kind"] == "parse_value"


def test_the_preset_runs_end_to_end_on_a_real_dataset():
    result = run(MetadataTriageConfig(), MixedWeightDataset())

    assert result.success is True
    data = result.steps["triage"].output.data()
    assert any(f.factor == "weight" for f in data["findings"])
    assert "parse_value" in data["suggested_policy_yaml"]
    (weight,) = [v for v in data["verification"] if v.factor == "weight"]
    assert weight.recovered is True
    assert result.findings
    assert data["counts"]["blocking"] == 1


def test_build_correction_is_public():
    correction = build_correction(ParseValueCorrectionConfig(factor="weight", drop=[","]))
    assert correction.factor == "weight"


def test_verification_recovers_a_mixed_column():
    from dataeval_flow._triage import find_issues

    metadata = _mixed_metadata()
    record = _describe(metadata)
    findings = find_issues(record)
    entries = verify(metadata, ResolvedPolicy(), findings)
    weight = [e for e in entries if e.factor == "weight"]
    assert weight
    assert weight[0].applied is True
    assert weight[0].recovered is True


def test_an_incomplete_suggestion_is_never_applied():
    from dataeval_flow._triage import Suggestion, TriageFinding

    finding = TriageFinding(
        factor="direction",
        category="unreadable",
        severity="blocking",
        repairable=True,
        suggestion=Suggestion(
            corrections=[{"kind": "remap", "factor": "direction", "rules": [{"match": "N", "to": None}]}],
            complete=False,
        ),
    )
    entries = verify(_mixed_metadata(), ResolvedPolicy(), [finding])
    (entry,) = entries
    assert entry.applied is False
    assert entry.recovered is False
    assert "need codes" in entry.detail


def test_verification_pins_a_derived_bin_suggestion():
    """A bin suggestion's claim is different from a correction's: check it separately.

    ``altitude`` is already a factor before the suggestion runs — it is continuous and
    reads cleanly. A check reusing the correction test (present in ``factors``, absent
    from ``unusable``) would say ``recovered`` no matter what the suggested count did.
    The real check: applying the suggested count turns the encoding's ``provenance``
    from ``"derived"`` into a pinned value.
    """
    from dataeval_flow._triage import find_issues

    metadata = _altitude_metadata()
    record = _describe(metadata)
    findings = find_issues(record)
    altitude = [f for f in findings if f.factor == "altitude"]
    assert altitude
    assert altitude[0].category == "unbinned"
    assert altitude[0].suggestion is not None
    assert altitude[0].suggestion.policy.get("continuous_factor_bins")

    entries = verify(metadata, ResolvedPolicy(), findings)
    (entry,) = [e for e in entries if e.factor == "altitude"]
    assert entry.applied is True
    assert entry.recovered is True
    assert "bins" in entry.detail
    assert "empty" in entry.detail


def test_a_bin_suggestion_that_stays_derived_is_not_recovered():
    """The bin-recovery check can say no; this proves it at the predicate.

    Assigning ``Metadata.continuous_factor_bins`` routes through DataEval's
    ``digitize_data``, not its auto-cut ``bin_data``; the two set ``provenance`` to
    pinned versus ``"derived"`` respectively (see ``dataeval.core._bin``). A complete,
    correctly shaped bin suggestion therefore cannot come back ``"derived"`` through the
    real ``Metadata`` API: assigning the suggested count and reading the record back
    always pins it. The failure case is a suggestion whose factor never bound — a stale
    name, or a run where the assignment did not stick. The hand-built ``after`` record
    captures that case directly.
    """
    still_derived = {"unreviewed": ["altitude"], "factors": {"altitude": {}}, "unusable": {}}
    assert _factor_recovered(still_derived, "altitude", pinned=True) is False

    pinned_now = {"unreviewed": [], "factors": {"altitude": {}}, "unusable": {}}
    assert _factor_recovered(pinned_now, "altitude", pinned=True) is True


def test_a_vanished_factor_is_not_recovered_even_when_absent_from_unreviewed():
    """Absence must not read as success.

    A factor missing from the re-described record's `factors` is absent from `unreviewed`
    as well: `unreviewed` only names factors that exist. Recovery requires the factor to
    be present. Being un-listed as unreviewed is not being pinned.
    """
    vanished = {"unreviewed": [], "factors": {}, "unusable": {}}
    assert _factor_recovered(vanished, "altitude", pinned=True) is False


def test_a_factor_the_descriptor_pins_is_not_unpinned_whatever_its_provenance():
    """An exported descriptor still says ``derived``, but it pins the cut and the vocabulary across runs.

    Called unpinned, such a factor drew a bin count the policy then refuses ("named by both `encoding` and
    `continuous_factor_bins`"), or a remedy to export a vocabulary already exported.
    """
    weather = {"type": "categorical", "encoding": {"kind": "levels", "levels": ["clear"], "provenance": "derived"}}
    record = _describe(_altitude_metadata())
    record = {**record, "factors": {**record["factors"], "weather": weather}}
    unpinned = {f.factor for f in find_issues(record) if f.category in ("unbinned", "unreviewed")}
    assert unpinned == {"altitude", "weather"}
    pinned = find_issues(record, descriptor_factors={"altitude", "weather"})
    assert not [f for f in pinned if f.category in ("unbinned", "unreviewed")]


def test_the_workflow_is_discoverable():
    from dataeval_flow.workflows import get_workflow, list_workflows

    assert get_workflow("metadata-triage").name == "metadata-triage"
    assert any(w.name == "metadata-triage" for w in list_workflows())


def test_the_workflow_config_parses():
    from dataeval_flow.workflows.metadata_triage import MetadataTriageConfig

    cfg = MetadataTriageConfig(name="triage", metadata="standard")
    assert cfg.type == "metadata-triage"
    assert cfg.verify is True


def test_a_pipeline_config_accepts_a_triage_workflow():
    from dataeval_flow import PipelineConfig

    PipelineConfig.model_validate(
        {
            "workflows": [{"name": "triage", "type": "metadata-triage", "metadata": "standard"}],
        }
    )


def test_findings_render_a_stanza_and_a_report():
    """What `render_stanza` and `build_findings` add beyond the executed run.

    ``test_execute_runs_end_to_end_on_a_real_dataset`` drives ``execute`` and checks that
    the rendered YAML contains "parse_value" and that verification recovers ``weight``.
    It does not pin the exact stanza shape ``to_policy_stanza`` produces, or that
    ``build_findings`` groups an unreadable factor under a title naming it. Both are
    checked here, directly.
    """
    from dataeval_flow._triage import find_issues, incomplete_factors, render_stanza, to_policy_stanza

    metadata = _mixed_metadata()
    findings = find_issues(_describe(metadata))
    stanza = to_policy_stanza(findings)

    assert stanza["corrections"][0] == {"kind": "parse_value", "factor": "weight", "drop": [","]}
    assert "kind: parse_value" in render_stanza(stanza, incomplete=incomplete_factors(findings))

    raw = {"findings": findings}
    reportables = build_findings(raw, max_examples=20)
    assert any("Unreadable" in r.title for r in reportables)


# ---------------------------------------------------------------------------
# report.py — build_findings / summarize
# ---------------------------------------------------------------------------


def _bare_finding(category: str, severity: str, factor: str = "weight") -> Any:
    from dataeval_flow._triage import TriageFinding

    return TriageFinding(factor=factor, category=category, severity=severity, remedy="do something")  # type: ignore[arg-type]


def test_a_category_with_a_blocking_finding_is_a_warning_reportable():
    """`WorkflowResult.health` counts exactly the categories this marks `"warning"`."""
    raw = {
        "findings": [
            _bare_finding("unreadable", "blocking"),
            _bare_finding("unreadable", "note", factor="other"),
        ]
    }
    (reportable,) = build_findings(raw, max_examples=20)
    assert reportable.severity == "warning"


def test_a_category_with_only_warning_and_note_findings_is_an_info_reportable():
    # Hard-coding this branch to "info" would still pass; the blocking test above
    # catches that half.
    raw = {
        "findings": [
            _bare_finding("unreviewed", "warning"),
            _bare_finding("unreviewed", "note", factor="other"),
        ]
    }
    (reportable,) = build_findings(raw, max_examples=20)
    assert reportable.severity == "info"


def test_a_suggested_policy_yaml_becomes_a_reportable():
    raw = {"suggested_policy_yaml": "metadata:\n  - name: standard\n"}
    (reportable,) = build_findings(raw, max_examples=20)
    assert reportable.title == "Suggested policy"
    (code,) = blocks_of(reportable, Code)
    assert code.text == "metadata:\n  - name: standard"
    assert code.language == "yaml"


def test_a_verification_entry_becomes_a_reportable():
    raw = {
        "verification": [VerificationEntry(factor="weight", applied=True, recovered=True, detail="8 bins, 0 unread")]
    }
    (reportable,) = build_findings(raw, max_examples=20)
    assert reportable.title == "Verified"
    assert reportable.brief == "1 recovered"
    assert fields(reportable) == {"weight": "8 bins, 0 unread"}


def test_summarize_counts_by_category_and_by_severity():
    raw_findings = [
        _bare_finding("unreadable", "blocking"),
        _bare_finding("unreadable", "note", factor="other"),
        _bare_finding("degenerate", "note", factor="third"),
    ]
    counts = summarize(raw_findings)
    assert counts == {"unreadable": 2, "blocking": 1, "note": 2, "degenerate": 1}


# ---------------------------------------------------------------------------
# Verification failure, surfaced rather than silent
# ---------------------------------------------------------------------------


def test_a_verification_error_becomes_a_reportable_when_the_list_is_empty():
    # Distinct from `verify: false` and "nothing to verify", both of which also leave
    # `verification` empty but set no error -- and so add no Finding at all.
    raw = {"verification_error": "boom"}
    (reportable,) = build_findings(raw, max_examples=20)
    assert reportable.title == "Verification failed"
    assert reportable.severity == "warning"
    assert paragraphs(reportable) == ["boom"]


def test_a_long_verification_error_wraps_within_the_width():
    error = (
        "ValueError: could not convert string to float: '6,000' while re-reading factor 'weight' under the "
        "suggested policy; the parse_value correction dropped ',' but the column also holds '6 000'"
    )
    raw = {"verification_error": error}
    (reportable,) = build_findings(raw, max_examples=20)
    assert rendered(reportable, width=80).splitlines()[3:] == [
        "  ValueError: could not convert string to float: '6,000' while re-reading factor",
        "  'weight' under the suggested policy; the parse_value correction dropped ','",
        "  but the column also holds '6 000'",
    ]


def test_no_verification_and_no_error_adds_no_reportable():
    assert build_findings({}, max_examples=20) == []


# ---------------------------------------------------------------------------
# Report blocks: each finding's evidence, as the report draws it
# ---------------------------------------------------------------------------


def _unreadable_weight(values: list[str]) -> dict[str, Any]:
    """A `weight` column read as numbers on most rows and as numerals wearing commas on the rest."""
    return {
        "reasons": ["mixed_types"],
        "level": "unit",
        "repairable": True,
        "counts": {"numeric": 1842, "text": 58},
        "distinct": {"text": values},
        "sampled": False,
    }


def _floor(name: str) -> dict[str, Any]:
    """A column a quarter of whose rows sit on its lowest value, -1."""
    return _numeric(name, -1.0, 300.0, rows=200, distinct=78, quantiles={"0.0": -1.0, "0.25": -1.0})


def test_every_finding_carries_its_evidence_as_blocks():
    record = _record(
        factors={**_floor("altitude"), **_numeric("object_id", 988, 113566, rows=1305, distinct=1305)},
        unusable={"weight": _unreadable_weight(["6,000"])},
        unmatched_bin_requests=["altitud"],
    )
    raw = {
        "findings": find_issues(record),
        "suggested_policy_yaml": "metadata:\n  - name: standard\n",
        "verification": [VerificationEntry(factor="weight", applied=True, recovered=True, detail="8 bins, 0 unread")],
    }
    failed = {"verification_error": "boom"}
    findings = [*build_findings(raw, max_examples=20), *build_findings(failed, max_examples=20)]

    assert [f.title for f in findings] == [
        "Unreadable factors",
        "Unmatched bin requests",
        "Columns dominated by one value",
        "Degenerate factors",
        "Unpinned continuous bins",
        "Suggested policy",
        "Verified",
        "Verification failed",
    ]
    for finding in findings:
        assert finding.brief
        assert finding.blocks


def test_a_shared_floor_value_is_a_section_listing_its_factors():
    """One section per value, not per factor: the factors sharing it are the corroboration."""
    factors = {
        **_floor("speed"),
        **_floor("altitude"),
        **_floor("compass_heading"),
        **_numeric("score", 0.0, 9999.0, rows=200, distinct=60, quantiles={"0.75": 9999.0, "1.0": 9999.0}),
    }
    floors = [f for f in find_issues(_record(factors=factors)) if f.category == "floor_mass"]
    (finding,) = build_findings({"findings": floors}, max_examples=20)

    assert finding.brief == "4 factors"
    assert [(s.title, s.brief) for s in sections(finding)] == [
        ("-1.0", "appears in >=25% of rows across 3 factors"),
        ("9999.0", "appears in >=25% of rows across 1 factor"),
    ]
    assert bullets(finding) == ["altitude", "compass_heading", "speed", "score"]
    assert rendered(finding, width=80).splitlines()[3:] == [
        "  -1.0 — appears in >=25% of rows across 3 factors",
        "    - altitude",
        "    - compass_heading",
        "    - speed",
        "",
        "    A common extreme value across multiple factors may indicate a missing",
        "    reading sentinel. Verify and remap to `.nan` if appropriate.",
        "",
        "    If this is a valid measurement, note the high concentration at this value.",
        "    No automatic bin count is suggested for skewed distributions.",
        "",
        "  9999.0 — appears in >=25% of rows across 1 factor",
        "    - score",
        "",
        "    This may indicate a missing reading sentinel. Remap to `.nan` if",
        "    appropriate.",
        "",
        "    If this is a valid measurement, note the high concentration at this value.",
        "    No automatic bin count is suggested for skewed distributions.",
    ]


def test_each_unpinned_factor_is_a_section_under_the_remedy_they_share():
    """The remedy is stated once; each factor keeps its own heading and chart."""
    drone = {
        "type": "categorical",
        "level": "unit",
        "encoding": {"kind": "levels", "provenance": "derived", "levels": ["a", "b"]},
        "fit": {"levels": [{"code": 0, "value": "a", "count": 30}, {"code": 1, "value": "b", "count": 30}]},
    }
    factors = {
        **_numeric("temperature", -12.5, 41.0, rows=200, distinct=180, bins=8),
        **_floor("altitude"),
        "drone": drone,
    }
    findings = find_issues(_record(factors=factors))
    raw = {"findings": findings}
    by_title = {f.title: f for f in build_findings(raw, max_examples=20)}
    unbinned, unreviewed = by_title["Unpinned continuous bins"], by_title["Unpinned categorical vocabularies"]
    info = {f.factor: f.detail["info"] for f in findings if "info" in f.detail}

    assert paragraphs(unbinned) == [
        (
            "These cuts were derived from this sample. Declaring a bin count fixes how many bins there are; the "
            "recommended policy pins their edges, which is what keeps runs comparable."
        )
    ]
    assert [(s.title, s.brief) for s in sections(unbinned)] == [
        ("altitude", "no bin count suggested: single value appears in >=25% of rows"),
        ("temperature", "declare 8 bins"),
    ]
    assert [s.blocks for s in sections(unbinned)] == [
        distribution_blocks(info["altitude"]),
        distribution_blocks(info["temperature"]),
    ]
    assert [(s.title, s.brief) for s in sections(unreviewed)] == [("drone", "2 levels")]
    assert sections(unreviewed)[0].blocks == distribution_blocks(info["drone"])


def test_each_finding_is_a_section_holding_its_chart_examples_and_remedy():
    values = [f"{6000 + 400 * i:,}" for i in range(25)]
    histogram = {
        "reasons": ["multi_dimensional"],
        "level": None,
        "repairable": False,
        "counts": {},
        "distinct": {},
        "sampled": False,
    }
    record = _record(unusable={"weight": _unreadable_weight(values), "histogram": histogram})
    raw = {"findings": find_issues(record)}
    (finding,) = build_findings(raw, max_examples=20)

    weight, other = sections(finding)
    assert (weight.title, weight.brief, weight.severity) == ("weight", "[blocking] mixed_types @ unit", "warning")
    assert (other.title, other.brief, other.severity) == ("histogram", "[note] multi_dimensional", "info")
    assert blocks_of(finding, Proportion) == [Proportion(parts=[("numeric", 1842), ("text", 58)])]
    shown = ", ".join(repr(v) for v in values[:20])
    assert fields(finding) == {"text reads": f"{shown} (+5 more)"}
    assert paragraphs(finding) == [
        "-> mixed types; remap or cast values to a single type",
        "-> non-scalar data; cannot be processed as a metadata factor",
    ]


def test_an_identifier_draws_no_chart_where_a_thin_column_does():
    """The chart for an identifier would be the arbitrary cut its finding exists to reject."""
    thin = _numeric("mast", 0.0, 9.0, rows=200, distinct=10)
    thin["mast"]["fit"]["bins"] = [{"code": 1, "count": 200, "min": 0.0, "max": 9.0}]
    record = _record(factors={**_numeric("object_id", 988, 113566, rows=1305, distinct=1305), **thin})
    raw = {"findings": find_issues(record)}
    (finding,) = [f for f in build_findings(raw, max_examples=20) if f.title == "Degenerate factors"]

    charted = {s.title: [type(b) for b in s.blocks if isinstance(b, Distribution)] for s in sections(finding)}
    assert charted == {"mast": [Distribution], "object_id": []}


def test_a_box_plot_in_a_triage_finding_fits_the_width():
    """The charts used to be drawn at the full width and then indented by hand, so a box plot
    whose legend just fit beside it ran four columns past the line. Drawn in its section, it fits:
    the range sits beneath the box, and quartiles too long for forty cells' whiskers are named."""
    quartiles = {"0.25": 48.25, "0.5": 103.8, "0.75": 176.4}
    record = _record(factors=_numeric("altitude", 12.5, 298.6, rows=200, distinct=180, quantiles=quartiles))
    (finding,) = build_findings({"findings": find_issues(record)}, 20)

    lines = rendered(finding, width=80).splitlines()
    assert max(len(line) for line in lines) <= 80
    assert lines[-5:] == [
        "  altitude — declare 3 bins",
        "    " + "█" * 40,
        "    ├" + "─" * 4 + "█" * 7 + "│" + "█" * 10 + "─" * 16 + "┤",
        "    12.5                               298.6",
        "    p25 48.25 · p50 103.8 · p75 176.4",
    ]


# ---------------------------------------------------------------------------
# Where each problem value sits, for its items' thumbnails
# ---------------------------------------------------------------------------


def _unreadable(dataset: Any) -> Any:
    DatasetCache.clear_instances()
    result = run(MetadataTriageConfig(), dataset)
    return next(f for f in result.findings if f.title == "Unreadable factors")


def test_each_problem_value_is_listed_most_rows_first_with_up_to_eight_of_its_items():
    finding = _unreadable(LatitudeDataset())
    assert "Where the values that read as text are:" in paragraphs(finding)
    (table,) = tables(finding)
    assert [c.header for c in table.columns] == ["Value", "Count", "Items", ""]
    assert [(row["value"], row["count"], row["items"]) for row in table.rows] == [
        ("N", 9, "3, 10, 17, 24, 31, 38, 45, 52, … 1 more"),
        ("S", 1, "20"),
    ]
    assert column(table, "image")[1] == [ItemRef(source="data", index=20)]


def test_a_value_below_the_item_is_placed_by_its_box():
    (table,) = tables(_unreadable(OcclusionDataset()))
    assert column(table, "value") == ["high"]
    assert column(table, "image")[0] == [ItemRef(source="data", index=i, target=1) for i in (0, 5, 10, 15, 20, 25)]


def test_a_column_dropped_for_naming_its_rows_is_placed_nowhere():
    """Every value of an identifier is distinct, and none is a problem, so no row is looked up."""
    finding = TriageFinding(
        factor="serial",
        category="unreadable",
        severity="blocking",
        reasons=("cardinality_over_budget",),
        repairable=True,
        detail={"counts": {"numeric": 40, "text": 20}},
    )
    metadata = MagicMock()
    assert places(metadata, [finding], "train") == {}
    metadata.unusable_rows.assert_not_called()


@pytest.mark.parametrize("error", [ValueError("kept no values"), AttributeError("no unusable_rows")])
def test_a_factor_whose_rows_cannot_be_found_is_placed_nowhere_and_the_run_goes_on(
    error: Exception, caplog: pytest.LogCaptureFixture
):
    """The places are a sample for looking at: failing to find them costs that factor its table, never the run."""
    finding = TriageFinding(
        factor="latitude",
        category="unreadable",
        severity="blocking",
        reasons=("mixed_types",),
        repairable=True,
        detail={"counts": {"numeric": 50, "text": 10}},
    )
    metadata = MagicMock()
    metadata.unusable_rows.side_effect = error
    assert places(metadata, [finding], "train") == {}
    assert "latitude" in caplog.text


def test_a_tie_takes_text_for_the_problem_values():
    assert minority_kind({"numeric": 5, "text": 5}) == "text"
    assert minority_kind({"numeric": 2, "text": 5}) == "numeric"
    assert minority_kind({"text": 5}) is None


def test_past_500_problem_values_a_paragraph_counts_the_rest():
    record = _record(unusable={"weight": _unreadable_weight(["6,000"])})
    raw = {"findings": find_issues(record)}
    where = {"weight": [(f"{n:,}", 1, [ItemRef(source="data", index=n)]) for n in range(1000, 1502)]}
    (finding,) = build_findings({**raw, "places": where}, max_examples=20)
    (table,) = tables(finding)
    assert len(table.rows) == 500
    assert "502 values read as text; the 500 on the most rows are listed." in paragraphs(finding)
