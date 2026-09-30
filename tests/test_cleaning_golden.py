"""data-cleaning agrees with what it found before its port: each finding's severity, title and brief (§10.3).

Deliberate differences from its legacy run (spec §10.3 item 3), each with its reason:

- **It records no encoding.** A data-cleaning result's `metadata_binning` and `encoding_digest` are null, where the
  legacy run recorded both, so `dataeval-flow encoding` finds none in it. data-cleaning's findings and its `clean`
  step never read the bins: it builds Metadata only for class labels. No evaluator result records a binning either,
  so carrying one into a chain's envelope is engine work, which the ports whose findings depend on bins
  (data-coverage, ood-detection, audit) will land. `test_a_data_cleaning_result_records_no_encoding` pins it.
- **It reads its dataset for statistics twice on a cold cache.** Its `outliers` and `dupes` steps each request their
  own statistics families from the same node, the outlier families and the hash families, and each is computed once;
  the legacy run computed their union in one pass. One pass over the union is future engine work.
  `tests/test_e2e.py::TestEndToEndCleaningWorkflow::test_config_to_output` pins the two requests.
- **It names its items by the chain's input, `data`, not by the source.** Spec §7.4 has an item reference name the
  node address it was read from.
  `tests/test_run.py::test_a_cleaning_run_carries_a_thumbnail_of_each_item_its_report_names` pins it.
"""

import json
from pathlib import Path

import pytest

from dataeval_flow import run
from dataeval_flow._cache import DatasetCache
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig
from tests.evaluator_toys import ToyImages
from tests.golden.cleaning import CASES

_GOLDEN = json.loads((Path(__file__).parent / "golden" / "cleaning_findings.json").read_text())


def test_every_case_is_recorded() -> None:
    assert sorted(_GOLDEN) == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_data_cleaning_gives_the_findings_it_gave_before_its_port(name: str) -> None:
    assert [[f.severity, f.title, f.brief] for f in CASES[name]()] == _GOLDEN[name]


def test_a_data_cleaning_result_records_no_encoding() -> None:
    """A deliberate difference: data-cleaning's result records no encoding, where its legacy run recorded one."""
    DatasetCache.clear_instances()
    result = run(DataCleaningConfig(outlier_method="zscore", outlier_flags=["pixel", "visual"]), ToyImages(count=12))
    assert result.success, result.errors
    assert result.metadata.metadata_binning is None
    assert result.metadata.encoding_digest is None
