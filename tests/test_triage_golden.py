"""metadata-triage agrees with what it produced before its port: its findings' severity, title and brief, in order;
its suggested policy stanza; and the binning record on its envelope (spec §10.10)."""

import json
from pathlib import Path
from typing import Any

import pytest

from tests.golden.generate_triage import record
from tests.golden.triage import CASES

_GOLDEN = json.loads((Path(__file__).parent / "golden" / "triage_findings.json").read_text())


def _canonical(value: Any) -> str:
    """`value` as sorted JSON: a binning record can hold NaN, which never equals itself as a float."""
    return json.dumps(value, sort_keys=True)


def test_every_case_is_recorded() -> None:
    assert sorted(_GOLDEN) == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_metadata_triage_gives_what_it_gave_before_its_port(name: str) -> None:
    assert _canonical(record(name)) == _canonical(_GOLDEN[name])
