"""The legacy parameter-sweep's counts, recorded as a golden before its removal (task-matrix spec §11.3)."""

import json
from pathlib import Path

from tests.golden import sweep

_GOLDEN = json.loads((Path(__file__).parent / "golden" / "sweep.json").read_text(encoding="utf-8"))


def test_a_data_cleaning_matrix_gives_the_sweep_s_counts_for_every_combination() -> None:
    assert sweep.matrix_counts() == _GOLDEN


def test_the_golden_exercises_both_counts() -> None:
    # A golden of zeros would agree with anything.
    assert len(_GOLDEN) == 16
    assert any(case["outlier_count"] for case in _GOLDEN)
    assert any(case["near_duplicate_groups"] for case in _GOLDEN)
