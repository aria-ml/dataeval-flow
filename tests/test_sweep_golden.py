"""The legacy parameter-sweep's counts, recorded as a golden before its removal (task-matrix spec §11.3)."""

import json
from pathlib import Path
from typing import Any

from tests.golden import sweep

_GOLDEN = json.loads((Path(__file__).parent / "golden" / "sweep.json").read_text(encoding="utf-8"))


def _recorded(params: dict[str, Any]) -> dict[str, Any]:
    """A combination's values as the golden records them: the sweep's `outlier_method` and `outlier_threshold`, which
    `outliers.outlier_threshold` holds as one `method` or `[method, bound]`."""
    threshold = params["outliers.outlier_threshold"]
    method, bound = (threshold, None) if isinstance(threshold, str) else threshold
    return {
        "outlier_method": method,
        "outlier_threshold": bound,
        "outlier_cluster_threshold": params["outliers.cluster_threshold"],
        "duplicate_cluster_sensitivity": params["duplicates.cluster_sensitivity"],
    }


def test_a_data_cleaning_matrix_gives_the_sweep_s_counts_for_every_combination() -> None:
    mapped = [{**case, "params": _recorded(case["params"])} for case in sweep.matrix_counts()]
    assert mapped == _GOLDEN


def test_the_golden_exercises_both_counts() -> None:
    # A golden of zeros would agree with anything.
    assert len(_GOLDEN) == 16
    assert any(case["outlier_count"] for case in _GOLDEN)
    assert any(case["near_duplicate_groups"] for case in _GOLDEN)
