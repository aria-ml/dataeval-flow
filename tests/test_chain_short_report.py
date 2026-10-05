"""A chain's short report (`detailed=False`): the summary, its health and one line per step; no finding, no evidence."""

import re

import pytest

from dataeval_flow import run
from dataeval_flow._cache import DatasetCache
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig
from tests.chain_toys import chain_pipeline, register_toys, run_toy_chain
from tests.evaluator_toys import ToyImages

_RULE = "=" * 80


@pytest.fixture
def cleaned():
    DatasetCache.clear_instances()
    config = DataCleaningConfig(name="clean", outliers={"flags": ["pixel", "visual"], "outlier_threshold": "zscore"})  # type: ignore[arg-type]
    yield run(config, ToyImages(count=24))
    DatasetCache.clear_instances()


def _timeless(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not re.match(r"\s+(Timestamp|Duration):", line))


def test_data_cleaning_short_text_is_its_summary_health_and_steps(cleaned) -> None:
    short = _timeless(cleaned.report(detailed=False))
    expected = """
================================================================================
  DATA CLEANING
================================================================================
  Workflow:  clean (data-cleaning)
  Source:    dataset (dataset)

  Steps: 10 ran

================================================================================
  SUMMARY
================================================================================
  Image Outliers ....................................... 1 images (4.2%)  [!!]
  Classwise Outliers ............ worst: b (8.3%), 1/1 classes over 3.0%  [!!]
  Image Duplicates ....................... 2 exact (8.3%), 0 near (0.0%)  [!!]
  Class Imbalance ................. 2 classes, 24 items, imbalance 1.0:1  [..]

  Health: 3 warning(s) [!!] — review flagged findings

================================================================================
  STEPS
================================================================================
  Step             Status  Note
  ---------------  ------  -----------
  outliers         ok
  labels           ok
  by-class         ok
  dupes            ok
  image-outliers   ok
  target-outliers  ok      no findings
  classwise        ok
  duplicates       ok
  imbalance        ok
  clean            ok

================================================================================
  METADATA FACTORS
================================================================================
  Encoding:        <digest>
  Auto-bin method: uniform_width

"""
    short = re.sub(r"Encoding:        [0-9a-f]{16}", "Encoding:        <digest>", short)
    assert short.split(f"{_RULE}\n  CONFIGURATION")[0] == expected
    assert "  CONFIGURATION" in short  # as a workflow result's short form: `_document` adds it whatever `detailed`


def test_the_short_form_is_shorter_and_names_every_step_once(cleaned) -> None:
    short, full = cleaned.report(detailed=False), cleaned.report(detailed=True)
    assert len(short.splitlines()) < len(full.splitlines())
    steps = short.split("  STEPS\n")[1].split("CONFIGURATION")[0]
    for name in cleaned.steps:
        assert len(re.findall(rf"^  {re.escape(name)} ", steps, re.MULTILINE)) == 1


def test_the_short_html_has_no_cards_and_a_compact_open_steps_table(cleaned) -> None:
    html = cleaned.to_html(detailed=False)
    assert 'class="card' not in html
    assert "From " not in html.split("<script>")[0]
    assert '<table class="summary">' in html
    assert '<section class="section"><h2>Steps</h2>' in html
    assert '<th class="left">Step</th><th class="left">Status</th><th class="left">Note</th></tr>' in html
    assert '<th class="left">Reads</th>' not in html
    assert '<th class="left">Type</th>' not in html
    assert html.count('data-value="target-outliers"') == 1
    assert "<h2>Configuration</h2>" in html
    full = cleaned.to_html(detailed=True)
    assert 'class="card' in full
    assert '<th class="left">Reads</th>' in full


@pytest.fixture
def toys(plugins):
    register_toys(plugins)
    DatasetCache.clear_instances()
    yield plugins
    DatasetCache.clear_instances()


@pytest.mark.usefixtures("toys")
def test_a_chain_with_a_failed_step_keeps_it_and_its_note_in_the_short_form() -> None:
    steps = [
        {"name": "few", "transform": "toy-first", "input": "a", "n": 4},
        {"name": "boom", "transform": "toy-explode", "input": "few"},
        {"name": "after", "transform": "toy-keep", "input": "boom"},
    ]
    workflow = {"name": "w", "inputs": ["a"], "steps": steps}
    config = chain_pipeline(
        workflows=[workflow], evaluators=[DuplicatesConfig(name="dupes")], datasets={"src": ToyImages()}
    )
    result = ChainResult.from_run("w", run_toy_chain(config, "w", ["src"]))
    short = result.report(detailed=False, width=100)
    assert "Steps: 3 (1 ran, 1 failed, 1 skipped)" in short
    assert "Health: failed [!!] — step `boom` failed" in short.split("  STEPS\n")[0]
    table = short.split("  STEPS\n")[1]
    assert re.search(r"^  boom\s+failed\s+RuntimeError: boom on few", table, re.MULTILINE)
    assert re.search(r"^  after\s+skipped\s+needs `boom`, which failed", table, re.MULTILINE)
