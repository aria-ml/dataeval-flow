"""The names the naming pass changed (naming spec §3.2, §5.3). A golden recorded before the pass is read through
these maps, so it still pins every value while the names move; the recorded JSON is never edited."""

__all__ = ["STEPS", "TITLES", "step", "title"]

# A finding's or section's title as recorded, and as the pass renamed it.
TITLES: dict[str, str] = {
    "Label Distribution": "Class Imbalance",
    "Label/Directory_Name Distribution": "Class Imbalance",
    "Duplicates": "Image Duplicates",
    "Embedding Coverage": "Class Coverage",
    "Class Balance Worklist": "Class Shortfall",
    "Label Space Coverage": "Leaf Coverage",
    "Metadata Coverage Gaps": "Factor Coverage Gaps",
    "Uncovered Rate": "Uncovered Items",
    "Label Alignment": "Mergeability",
    "Evaluation Coverage": "Eval Coverage",
    "OOD Factor Predictors": "Factor Predictors",
    "OOD Sample Metadata Deviations": "Factor Deviation",
}


def title(old: str) -> str:
    """`old` as the naming pass renamed it; a title it did not rename, unchanged."""
    return TITLES.get(old, old)


# Each preset's chain step names as recorded, and as the pass renamed them (naming spec §5.3).
STEPS: dict[str, dict[str, str]] = {
    "data-cleaning": {
        "labels": "label-health",
        "by-class": "outliers-by-class",
        "dupes": "duplicates",
        "classwise": "classwise-outliers",
        "duplicates": "image-duplicates",
        "imbalance": "class-imbalance",
    },
    "data-prioritization": {
        "rank": "prioritization",
        "reference-outliers": "outliers-reference",
        "pool-outliers": "outliers-pool",
        "reference-dupes": "duplicates-reference",
        "pool-dupes": "duplicates-pool",
    },
    "data-splitting": {
        "labels": "label-health",
        "labels-check": "class-imbalance",
        "labels-train": "label-health-train",
        "labels-val": "label-health-val",
        "labels-test": "label-health-test",
        "labels-rebalanced": "label-health-rebalanced",
        "rebalance": "rebalanced",
        "uncovered": "uncovered-items",
        "uncovered-train": "uncovered-items-train",
        "uncovered-val": "uncovered-items-val",
        "uncovered-test": "uncovered-items-test",
    },
    "data-coverage": {
        "labels": "label-health",
        "labels-check": "class-imbalance",
        "summary": "factor-summary",
        "worklist": "representation",
        "shortfall": "class-shortfall",
        "uncovered": "uncovered-items",
        "completeness-check": "dimensional-completeness",
        "gaps": "factor-gaps",
        "gaps-check": "factor-coverage-gaps",
    },
    "label-space": {
        "reconciliation": "label-reconciliation",
        "conformance": "label-conformance",
        "alignment": "label-alignment",
        "structure": "ontology-validation",
    },
    "metadata-triage": {"triage": "factor-triage", "issues": "metadata-issues"},
    "ood-detection": {"agreement": "ood-union", "agreement-check": "ood-agreement"},
}


def step(preset: str, old: str) -> str:
    """`old`, a step name `preset`'s chain recorded, as the pass renamed it, keeping an element suffix (`[train]`)."""
    base, bracket, rest = old.partition("[")
    renamed = STEPS.get(preset, {}).get(base, base)
    if base.endswith("-classes-check"):
        renamed = base.removesuffix("-classes-check") + "-by-class-check"
    elif base.endswith("-classes"):
        renamed = base.removesuffix("-classes") + "-by-class"
    return renamed + bracket + rest
