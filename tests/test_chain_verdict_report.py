"""A preset chain's report: its verdict, its record, a section per question, and next steps (audit spec §3, §7.4, §11).

`ToyImages()` plants one exact duplicate (item 5 is item 0), so `image-duplicates` warns at `exact: 0.0` and passes at
`exact: 50.0`. With no extractor, `completeness` is skipped and `dimensional-completeness` is not assessed.
"""

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from PIL import Image
from pydantic import Field

from dataeval_flow import run_tasks
from dataeval_flow._blocks import Block, BulletList, Fields, Paragraph, Section, Table, Verdict
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._report import _record_table, _settings, question_status
from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow._result import LabelSpaceRecord, LineageRecord
from dataeval_flow.config import ImageFolderDatasetConfig, PipelineConfig
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin
from dataeval_flow.evaluators import EvaluatorResult
from dataeval_flow.evaluators._result import EvaluatorMetadata
from dataeval_flow.evaluators.bias import FactorSummaryConfig
from dataeval_flow.evaluators.quality import DuplicatesEvaluator
from dataeval_flow.steps import ChainResult, Finding, StepResult
from dataeval_flow.steps._result import ChainMetadata, ChainOutput
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.workflows import Workflow, WorkflowConfig
from dataeval_flow.workflows._preset import NextSteps, Preset, PresetChain, Record, ReportGroup
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyFactors, ToyImages
from tests.preset_toys import ToyVerdictPreset, ToyVerdictPresetConfig, register_presets

_TWO = {"a": ToyImages(), "b": ToyImages(seed=1)}


@pytest.fixture(autouse=True)
def _presets(plugins):
    register_presets(plugins)
    plugins["dataeval_flow.workflows"].extend(
        [
            ("toy-regrouped-preset", "tests.test_chain_verdict_report:_Regrouped"),
            ("toy-record-factors-preset", "tests.test_chain_verdict_report:_Factors"),
        ]
    )
    DatasetCache.clear_instances()
    yield plugins
    DatasetCache.clear_instances()


def _run(entry: dict[str, Any], *, extractor: bool = False, datasets: dict[str, Any] | None = None) -> ChainResult:
    task: dict[str, Any] = {"name": "t", "workflow": "w", "sources": list(datasets or ["src"])}
    if extractor:
        task["extractor"] = "flat"
    config = chain_pipeline(workflows=[{"name": "w", **entry}], tasks=[task], datasets=datasets, extractor=extractor)
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    return result


def _top(result: ChainResult, *, detailed: bool = True) -> list[Block]:
    """The report's blocks under its banner, less the envelope."""
    return result._document(detailed=detailed).blocks[1:]


def _outline(result: ChainResult, *, detailed: bool = True) -> list[str]:
    """Each top-level block: a section by its title, any other block by its type."""
    return [block.title if isinstance(block, Section) else block.type for block in _top(result, detailed=detailed)]


def _section(blocks: list[Block], title: str) -> Section:
    (section,) = [block for block in blocks if isinstance(block, Section) and block.title == title]
    return section


def _timeless(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not re.match(r"\s+(Timestamp|Duration):", line))


def test_the_detailed_outline_is_verdict_record_questions_next_steps_then_steps() -> None:
    result = _run({"type": "toy-verdict-preset"})
    assert _outline(result) == [
        "verdict",
        "Verdict",
        "What was run",
        "Clean",
        "Covered",
        "Next steps",
        "Steps",
        "Configuration",
    ]
    verdict = _section(_top(result), "Verdict")
    assert verdict.blocks == [
        Section(
            title="Blocking",
            blocks=[BulletList(items=["Image Duplicates (image-duplicates): 2 exact (16.7%), 0 near (0.0%)"])],
        ),
        Section(
            title="Not assessed",
            blocks=[
                BulletList(
                    items=[
                        (
                            "Dimensional Completeness (dimensional-completeness): `completeness` was skipped: "
                            "requires an extractor"
                        )
                    ]
                )
            ],
        ),
    ]
    assert _section(_top(result), "Next steps").blocks == [
        BulletList(
            items=[
                "Image Duplicates (image-duplicates): Remove them.",
                "Name an extractor. Not assessed: Dimensional Completeness.",
            ]
        )
    ]


def test_the_verdict_lists_each_acceptance_with_its_state_and_reason() -> None:
    result = _run({"type": "toy-verdict-preset", "accepted": {"image-duplicates": "Planted on purpose."}})
    verdict = _section(_top(result), "Verdict")
    assert _section(verdict.blocks, "Accepted risks").blocks == [
        BulletList(items=["Image Duplicates (warned): Planted on purpose."])
    ]


def test_each_question_holds_its_checks_findings_beside_their_evidence() -> None:
    top = _top(_run({"type": "toy-verdict-preset"}))
    clean, covered = _section(top, "Clean"), _section(top, "Covered")
    assert [(block.title, block.severity) for block in clean.blocks if isinstance(block, Section)] == [
        ("Image Duplicates", "warning")
    ]
    assert [(block.title, block.severity) for block in covered.blocks if isinstance(block, Section)] == [
        ("Dimensional Completeness", "info")
    ]
    (finding,) = clean.blocks
    assert isinstance(finding, Section)
    assert [block.title for block in finding.blocks if isinstance(block, Section)] == ["From Duplicates · dupes"]


def test_a_question_s_status_counts_its_warnings_and_unassessed_checks() -> None:
    top = _top(_run({"type": "toy-verdict-preset"}))
    assert _section(top, "Clean").brief == "1 warning"
    assert _section(top, "Covered").brief == "not assessed: requires an extractor"


def test_a_question_whose_checks_all_ran_and_none_warned_is_ok() -> None:
    top = _top(_run({"type": "toy-verdict-preset", "exact": 50.0}, extractor=True))
    assert [_section(top, title).brief for title in ("Clean", "Covered")] == ["ok", "ok"]


def _record(name: str, check: str, *findings: Finding, not_assessed: str | None = None, **fields: Any) -> StepResult:
    return StepResult(
        name=name,
        kind="check",
        type=check,
        inputs=[],
        status="ok",
        output=list(findings),
        not_assessed=not_assessed,
        **fields,
    )


@pytest.mark.parametrize(
    ("steps", "status"),
    [
        (
            {
                "a": _record("a", "x", Finding(severity="warning", title="X", step="a")),
                "b": _record("b", "y", not_assessed="`e` was skipped: requires an extractor"),
            },
            "1 warning, 1 not assessed",
        ),
        (
            {"a": _record("a", "x", Finding(severity="warning", title="X"), Finding(severity="warning", title="X"))},
            "2 warnings",
        ),
        (
            {
                "a": _record("a", "x", Finding(severity="info", title="X")),
                "b": _record("b", "y", not_assessed="no evaluation split given"),
            },
            "1 not assessed",
        ),
        (
            {
                "a": _record("a", "x", not_assessed="`e` was skipped: requires an extractor"),
                "b": _record("b", "y", not_assessed="`f` was skipped: requires an extractor"),
            },
            "not assessed: requires an extractor",
        ),
        (
            {
                "a": _record("a", "x", not_assessed="`e` was skipped: requires an extractor"),
                "b": _record("b", "y", not_assessed="no evaluation split given"),
            },
            "2 not assessed",
        ),
    ],
)
def test_a_question_s_status_names_one_reason_only_when_every_check_went_unassessed_for_it(steps, status) -> None:
    group = ReportGroup("Q", ("x", "y"))
    plan = NextSteps(by_reason={"requires an extractor": "Name one."})
    assert question_status(steps, group, plan) == status


def test_the_short_form_is_the_verdict_the_record_then_a_line_per_question_under_its_own_heading() -> None:
    result = _run({"type": "toy-verdict-preset"})
    assert _outline(result, detailed=False) == ["verdict", "What was run", "Questions", "Steps", "Configuration"]
    (verdict,) = [block for block in _top(result, detailed=False) if isinstance(block, Verdict)]
    assert verdict == Verdict(
        level="not-ready", label="Not ready", line="Not ready: Image Duplicates (2 exact (16.7%), 0 near (0.0%))"
    )
    assert _fields(_section(_top(result, detailed=False), "Questions")) == [
        ("Clean", "1 warning"),
        ("Covered", "not assessed: requires an extractor"),
    ]
    text = result.report(detailed=False)
    assert "  Verdict: Not ready: Image Duplicates (2 exact (16.7%), 0 near (0.0%))  [!!]" in text
    assert "  QUESTIONS\n" + "=" * 80 + "\n  Clean:   1 warning\n  Covered: not assessed: requires an extractor" in text


def test_a_verdict_report_never_claims_every_check_passed() -> None:
    result = _run({"type": "toy-verdict-preset", "exact": 50.0})
    assert result.verdict is not None
    assert result.verdict.level == "ready-with-caveats"
    for detailed in (True, False):
        text = result.report(detailed=detailed)
        assert "All checks passed" not in text
        assert "Health:" not in text
        assert "Verdict: Ready with caveats: 1 not assessed  [..]" in text
        page = result.to_html(detailed=detailed)
        assert ">passed<" not in page
        assert "All checks passed" not in page
        header = re.search(r'<header class="report-head">.*?</header>', page)
        assert header is not None
        assert '<span class="badge info">Ready with caveats</span> <span class="brief">1 not assessed</span>' in (
            header.group(0)
        )


def test_a_failed_task_with_no_verdict_keeps_its_failed_health() -> None:
    with patch.object(DuplicatesEvaluator, "run", side_effect=RuntimeError("no stats")):
        result = _run({"type": "toy-verdict-preset"}, extractor=True)
    assert result.verdict is None
    assert "Health: failed [!!]" in result.report()
    assert "Verdict:" not in result.report()
    assert '<span class="badge failed">failed: dupes</span>' in result.to_html()


def test_the_record_gives_each_split_its_digests() -> None:
    result = _run({"type": "toy-verdict-splits-preset"}, datasets=_TWO)
    record = _section(_top(result), "What was run")
    table, digests = record.blocks[:2]
    assert isinstance(table, Table)
    assert isinstance(digests, Fields)
    assert [column.header for column in table.columns] == ["", "a", "b"]
    assert [row[""] for row in table.rows] == ["Source", "Content digest", "Metadata digest", "How it was made"]
    rows = {row[""]: row for row in table.rows}
    train, evals = result.steps["content-digest-train"], result.steps["content-digest-evals"]
    assert evals.elements is not None
    a, b = train.output.data(), evals.elements["b"].output.data()
    assert (rows["Content digest"]["a"], rows["Content digest"]["b"]) == (
        f"{a['content'][:12]}…",
        f"{b['content'][:12]}…",
    )
    assert digests.items == [
        ("Content digest (a)", a["content"]),
        ("Metadata digest (a)", a["metadata"]),
        ("Content digest (b)", b["content"]),
        ("Metadata digest (b)", b["metadata"]),
    ]
    assert (rows["Source"]["a"], rows["Source"]["b"]) == ("a (a_data)", "b (b_data)")
    assert (rows["How it was made"]["a"], rows["How it was made"]["b"]) == ("`train` (a)", "`evals[b]` (b)")
    assert "content-digest" not in _outline(result)  # its results are in the record, not under "Other"


def test_a_text_report_over_three_splits_gives_every_full_digest_unbroken_on_one_line() -> None:
    result = _run({"type": "toy-verdict-splits-preset"}, datasets={**_TWO, "c": ToyImages(seed=2)})
    evals = result.steps["content-digest-evals"].elements
    assert evals is not None
    outputs = [result.steps["content-digest-train"].output.data(), *(evals[key].output.data() for key in "bc")]
    lines = result.report().splitlines()
    for digest in (output[kind] for output in outputs for kind in ("content", "metadata")):
        assert len(digest) == 64
        assert any(digest in line for line in lines), digest


def test_the_record_gives_each_split_its_own_encoding_and_factors() -> None:
    # Encoded on its own, ToyFactors(5)'s angles (0..4) cut differently from ToyFactors(60)'s, so the encodings differ.
    result = _run({"type": "toy-record-factors-preset"}, datasets={"train": ToyFactors(60), "test": ToyFactors(5)})
    assert result.success, result.errors
    assert result.metadata.metadata_binning is not None
    per_split = result.metadata.metadata_binning["per_split"]
    assert result.metadata.encoding_digest is None  # the splits disagree, so the run has no one encoding
    (table,) = [block for block in _section(_top(result), "What was run").blocks if isinstance(block, Table)]
    rows = {row[""]: row for row in table.rows}
    train, test = per_split["train"], per_split["evals[test]"]
    assert train["encoding_digest"] != test["encoding_digest"]
    assert (rows["Encoding"]["train"], rows["Encoding"]["test"]) == (train["encoding_digest"], test["encoding_digest"])
    names = ", ".join(train["factors"])
    assert (rows["Metadata factors"]["train"], rows["Metadata factors"]["test"]) == (f"2: {names}", f"2: {names}")


class _Rows(EvaluatorResult[Any]):
    """An evaluate step's result recording the rows given."""

    def __init__(self, *rows: tuple[str, str]) -> None:
        super().__init__(type="toy-rows", success=True, metadata=EvaluatorMetadata(), output=rows)
        self._rows = list(rows)

    def record_rows(self) -> list[tuple[str, str]]:
        return self._rows


def _ran(name: str, address: str, result: Any) -> StepResult:
    return StepResult(name=name, kind="evaluator", type="toy-rows", inputs=[address], status="ok", result=result)


def test_record_rows_keep_their_order_where_the_first_column_lacks_one_and_a_conformed_split_has_its_label_space():
    lineage = [
        LineageRecord(name="train", source="a", digest="0" * 12, items=4),
        LineageRecord(name="evals[b]", source="b", digest="1" * 12, items=4),
    ]
    metadata = ChainMetadata(workflow="w", lineage=lineage, label_space=[LabelSpaceRecord(source="b", digest="5" * 12)])
    elements = {"b": _ran("rows-evals", "evals[b]", _Rows(("Items", "4"), ("Labels", "6"), ("Classes", "2: x, y")))}
    steps = {
        "rows-train": _ran("rows-train", "train", _Rows(("Items", "4"), ("Classes", "2: x, y"))),
        "rows-evals": StepResult(
            name="rows-evals", kind="evaluator", type="toy-rows", inputs=["evals"], status="ok", elements=elements
        ),
    }
    result = ChainResult(type="w", success=True, metadata=metadata, output=ChainOutput(steps), steps=steps)
    (table,) = _record_table(result, ["toy-rows"])
    assert isinstance(table, Table)
    assert [row[""] for row in table.rows] == ["Items", "Labels", "Classes", "Label space", "How it was made"]
    assert {row[""]: (row["a"], row["b"]) for row in table.rows}["Label space"] == ("", "5" * 12)


def test_the_record_states_the_run_and_the_criteria() -> None:
    result = _run({"type": "toy-verdict-preset", "accepted": {"image-duplicates": "Planted on purpose."}})
    record = _section(_top(result), "What was run")
    run, criteria = (_fields(_section(record.blocks, title)) for title in ("Run", "Criteria"))
    assert [label for label, _ in run] == ["Flow", "Libraries", "Device", "Timestamp"]
    assert dict(run)["Flow"] == result.metadata.tool_version
    assert criteria == [("Blocking", "image-duplicates"), ("Accepted", "image-duplicates: Planted on purpose.")]


def _fields(section: Section) -> list[tuple[str, Any]]:
    (fields,) = section.blocks
    assert isinstance(fields, Fields)
    return fields.items


def test_the_record_shows_each_dataset_s_provenance(tmp_path: Path) -> None:
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        for index in range(4):
            Image.new("RGB", (8, 8), color=(index * 60, 0, 0)).save(tmp_path / name / f"{index}.png")
    facts: dict[str, str | int | float | bool] = {"owner": "Perception team", "license": "CC-BY-4.0"}
    config = PipelineConfig.model_validate(
        {
            "datasets": [
                ImageFolderDatasetConfig(name="a_data", path="a", provenance=facts).model_dump(),
                ImageFolderDatasetConfig(name="b_data", path="b").model_dump(),
            ],
            "sources": [{"name": "a", "dataset": "a_data"}, {"name": "b", "dataset": "b_data"}],
            "workflows": [{"name": "w", "type": "toy-verdict-splits-preset"}],
            "tasks": [{"name": "t", "workflow": "w", "sources": ["a", "b"]}],
        }
    )
    result = run_tasks(config, data_dir=tmp_path)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    (table,) = [block for block in _section(_top(result), "What was run").blocks if isinstance(block, Table)]
    (provenance,) = [row for row in table.rows if row[""] == "Provenance"]
    assert provenance == {"": "Provenance", "a": "owner: Perception team\nlicense: CC-BY-4.0", "b": ""}


def test_a_question_holds_its_splits_findings_under_their_keys_and_html_draws_every_finding_as_a_card() -> None:
    result = _run({"type": "toy-verdict-splits-preset"}, datasets=_TWO)
    clean = _section(_top(result), "Clean")
    assert [(block.title, block.severity) for block in clean.blocks if isinstance(block, Section)] == [
        ("Image Duplicates", "warning"),
        ("b", None),
    ]
    assert clean.brief == "2 warnings"
    page = result.to_html()
    assert '<details class="card warning" id="clean-image-duplicates" open>' in page
    assert '<section class="section" id="clean-b"><h3>b</h3>' in page
    assert '<details class="card warning" id="clean-b-image-duplicates" open>' in page


_LAYOUTS = {
    "one-group": (ReportGroup("Covered", ("dimensional-completeness",)),),
    "evidence-first": (
        ReportGroup("Is it covered?", ("dimensional-completeness",), ("duplicates",)),
        ReportGroup("Clean", ("image-duplicates",)),
    ),
    "absent": (
        ReportGroup("Clean", ("image-duplicates",)),
        ReportGroup("Absent", ("leakage",)),
        ReportGroup("Covered", ("dimensional-completeness",)),
    ),
}


class _RegroupedConfig(ToyVerdictPresetConfig):
    type: str = Field(default="toy-regrouped-preset", description="The workflow type this entry configures.")
    layout: str = Field(default="one-group", description="Which of `_LAYOUTS` the chain's headings are.")


class _Regrouped(Preset, Workflow[_RegroupedConfig, ChainResult]):
    """The toy verdict preset's chain, under the headings `layout` names."""

    name: ClassVar[str] = "toy-regrouped-preset"
    description: ClassVar[str] = "The toy verdict preset, regrouped."
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

    @classmethod
    def chain(cls, config: Any) -> PresetChain:
        return replace(ToyVerdictPreset.chain(config), groups=_LAYOUTS[config.layout])


class _FactorsConfig(WorkflowConfig[ChainResult], MetadataConfigMixin):
    type: str = Field(default="toy-record-factors-preset", description="The workflow type this entry configures.")
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.METADATA}), sources=SourceCount.TWO_OR_MORE)


class _Factors(Preset, Workflow[_FactorsConfig, ChainResult]):
    """Each split's factor summary, each split on its own encoding, and a record of what was read."""

    name: ClassVar[str] = "toy-record-factors-preset"
    description: ClassVar[str] = "Summarizes each split's factors."
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("train", InputSlot.model_validate({"name": "evals", "list": True}))

    @classmethod
    def chain(cls, config: Any) -> PresetChain:
        return PresetChain(
            steps=[
                {"name": "factor-summary-train", "evaluator": "summary", "input": "train"},
                {"name": "factor-summary-evals", "evaluator": "summary", "input": "evals"},
            ],
            evaluators=[FactorSummaryConfig(name="summary", metadata=config.metadata)],
            record=Record("What was run"),
        )


def test_a_check_no_question_names_shows_its_findings_after_the_next_steps() -> None:
    result = _run({"type": "toy-regrouped-preset", "layout": "one-group"})
    assert _outline(result)[3:6] == ["Covered", "Next steps", "Image Duplicates"]


def test_a_question_none_of_whose_checks_is_in_the_chain_is_left_out() -> None:
    result = _run({"type": "toy-regrouped-preset", "layout": "absent"})
    assert "Absent" not in _outline(result)
    assert [label for label, _ in _fields(_section(_top(result, detailed=False), "Questions"))] == ["Clean", "Covered"]


def test_evidence_a_question_shows_is_pointed_back_to_by_a_later_question_s_finding() -> None:
    top = _top(_run({"type": "toy-regrouped-preset", "layout": "evidence-first"}))
    covered = _section(top, "Is it covered?")
    assert [block.title for block in covered.blocks if isinstance(block, Section)] == [
        "Dimensional Completeness",
        "Duplicates · dupes",
    ]
    (finding,) = _section(top, "Clean").blocks
    assert isinstance(finding, Section)
    assert finding.blocks[-1] == Paragraph(text="Evidence: Duplicates · dupes, under Is it covered?")


@pytest.mark.parametrize(
    ("settings", "line"),
    [
        ({"warning": 3.0, "info": None}, "warning 3.0, info none"),
        ({"train": 20, "eval": 30}, "train 20, eval 30"),
        (
            {"near": {"rate": 0.1}, "factors": ["scene", "site"], "empty": False},
            "near (rate 0.1), factors [scene, site], empty false",
        ),
    ],
)
def test_a_check_s_criteria_read_as_words_not_python(settings, line) -> None:
    assert _settings(settings) == line


def test_a_chain_without_declarations_renders_as_before() -> None:
    result = _run({"type": "toy-preset"})
    assert _timeless(result.report(detailed=False)) == _TOY_SHORT
    assert _timeless(result.report(detailed=True)) == _TOY_DETAILED


_TOY_CONFIGURATION = """================================================================================
  CONFIGURATION
================================================================================
  sources:
    - name: src
      dataset: src_data
      dataset_config:
        name: src_data
        format: maite
        version: 1
        dataset: {type: protocol, class: ToyImages, id: toy-0-12}
  workflow: {name: w, type: toy-preset, exact: 0.0}

================================================================================"""

_TOY_SHORT = f"""
================================================================================
  TOY-PRESET
================================================================================
  Workflow:  w (toy-preset)
  Source:    src (src_data)

  Steps: 3 ran

================================================================================
  SUMMARY
================================================================================
  Image Duplicates ...................... 2 exact (16.7%), 0 near (0.0%)  [!!]

  Health: 1 warning(s) [!!] — review flagged findings

================================================================================
  STEPS
================================================================================
  Step   Status  Note
  -----  ------  ----
  dupes  ok
  rate   ok
  kept   ok

{_TOY_CONFIGURATION}"""

_TOY_DETAILED = f"""
================================================================================
  TOY-PRESET
================================================================================
  Workflow:  w (toy-preset)
  Source:    src (src_data)

  Steps: 3 ran

================================================================================
  SUMMARY
================================================================================
  Image Duplicates ...................... 2 exact (16.7%), 0 near (0.0%)  [!!]

  Health: 1 warning(s) [!!] — review flagged findings

================================================================================
  IMAGE DUPLICATES                                2 exact (16.7%), 0 near (0.0%)
================================================================================
  1 exact duplicate groups, 0 near-duplicate groups found.

  From Duplicates · dupes
    Group  Kind   Count  Items
    -----  -----  -----  -----
    0      exact      2  0, 5

================================================================================
  REMOVE · KEPT
================================================================================
  Kept 11 of 12 images. Removed 1 image: 1 named by `dupes`.

================================================================================
  STEPS
================================================================================
  Step   Type              Status  Reads         Note
  -----  ----------------  ------  ------------  ----
  dupes  duplicates        ok      `data` (src)
  rate   image-duplicates  ok      `dupes`
  kept   remove            ok      `data` (src)
                                   `dupes`

{_TOY_CONFIGURATION}"""
