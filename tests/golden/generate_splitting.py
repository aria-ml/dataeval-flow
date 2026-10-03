"""Record `splitting.json` from the legacy data-splitting workflow, before its port to a preset.

Run once, on the legacy workflow: `.venv/bin/python -m tests.golden.generate_splitting`. It records each case's
findings, its indices, each part's label counts, the whole's balance and diversity rows and each part's uncovered
indices (data-splitting spec §9).
"""

import json
from pathlib import Path
from typing import Any

from dataeval_flow import run_tasks
from dataeval_flow.workflows.data_splitting._report import _normalize_label_counts
from tests.golden.splitting import CASES, pipeline


def _counts(stats: dict[str, Any], full: dict[str, Any]) -> dict[str, int]:
    """Label counts by class name, named with the whole's class table."""
    return _normalize_label_counts(stats.get("label_counts_per_class"), full.get("index2label"))


def _uncovered(coverage: dict[str, Any] | None) -> list[int] | None:
    return None if coverage is None else [int(index) for index in coverage["uncovered_indices"]]


def _record(name: str) -> dict[str, Any]:
    result = run_tasks(pipeline(name, legacy=True))["t"]
    assert result.success, result.errors
    raw = result.output.raw
    findings = result.output.report.findings
    full = raw.label_stats_full
    return {
        "findings": [{"severity": f.severity, "title": f.title, "description": f.description} for f in findings],
        "test": raw.test_indices,
        "test_counts": _counts(raw.label_stats_test, full),
        "test_uncovered": _uncovered(raw.coverage_test),
        "folds": [
            {
                "train": fold.train_indices,
                "val": fold.val_indices,
                "train_counts": _counts(fold.label_stats_train, full),
                "val_counts": _counts(fold.label_stats_val, full),
                "train_uncovered": _uncovered(fold.coverage_train),
                "val_uncovered": _uncovered(fold.coverage_val),
            }
            for fold in raw.folds
        ],
        "full_counts": _counts(full, full),
        "balance": raw.pre_split_balance.get("balance"),
        "diversity": raw.pre_split_diversity.get("factors"),
        "class_distribution": next(f.severity for f in findings if f.title == "Class distribution (full dataset)"),
        "stratification": next(
            {"severity": f.severity, "brief": f.brief} for f in findings if f.title == "Stratification quality"
        ),
    }


if __name__ == "__main__":
    golden = {name: _record(name) for name in CASES}
    path = Path(__file__).parent / "splitting.json"
    path.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path}")
