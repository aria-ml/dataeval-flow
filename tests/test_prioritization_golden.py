"""data-prioritization agrees with what it produced before its port: each pool's ranking, as indices into the pool,
and how many items cleaning removed from each source (spec §10.9)."""

import json
from pathlib import Path

import pytest

from tests.golden.generate_prioritization import record
from tests.golden.prioritization import CASES

_GOLDEN = json.loads((Path(__file__).parent / "golden" / "prioritization_rankings.json").read_text())


def test_every_case_is_recorded() -> None:
    assert sorted(_GOLDEN) == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_data_prioritization_gives_the_rankings_it_gave_before_its_port(name: str) -> None:
    assert record(name) == _GOLDEN[name]
