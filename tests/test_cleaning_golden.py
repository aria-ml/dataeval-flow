"""data-cleaning agrees with what it found before its port: each finding's severity, title and brief (§10.3)."""

import json
from pathlib import Path

import pytest

from tests.golden.cleaning import CASES

_GOLDEN = json.loads((Path(__file__).parent / "golden" / "cleaning_findings.json").read_text())


def test_every_case_is_recorded() -> None:
    assert sorted(_GOLDEN) == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_data_cleaning_gives_the_findings_it_gave_before_its_port(name: str) -> None:
    assert [[f.severity, f.title, f.brief] for f in CASES[name]()] == _GOLDEN[name]
