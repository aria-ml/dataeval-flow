"""parameter-sweep's counts, recorded as a golden before its removal (task-matrix spec §11.3)."""

import json
from pathlib import Path

from tests.golden import sweep

_GOLDEN = json.loads((Path(__file__).parent / "golden" / "sweep.json").read_text(encoding="utf-8"))


def test_the_legacy_sweep_reproduces_its_golden() -> None:
    assert sweep.legacy_counts() == _GOLDEN


def test_the_golden_exercises_both_counts() -> None:
    # A golden of zeros would agree with anything.
    assert len(_GOLDEN) == 16
    assert any(case["outlier_count"] for case in _GOLDEN)
    assert any(case["near_duplicate_groups"] for case in _GOLDEN)
