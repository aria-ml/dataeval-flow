"""data-cleaning agrees with what it found before its port: each finding's severity, title and brief (§10.3).

Deliberate differences from its legacy run (spec §10.3 item 3), each with its reason:

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


def test_a_data_cleaning_result_records_the_encoding_its_steps_read() -> None:
    """`labels` and `by-class` read one encoding, so the envelope holds one record, as the legacy run's did."""
    DatasetCache.clear_instances()
    result = run(DataCleaningConfig(outlier_method="zscore", outlier_flags=["pixel", "visual"]), ToyImages(count=12))
    assert result.success, result.errors
    record = result.metadata.metadata_binning
    assert record is not None
    assert "per_split" not in record
    assert record["factors"] == {}
    assert result.metadata.encoding_digest == record["encoding_digest"]
