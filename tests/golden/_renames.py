"""The names the naming pass changed (naming spec §3.2, §5.3). A golden recorded before the pass is read through
these maps, so it still pins every value while the names move; the recorded JSON is never edited."""

__all__ = ["TITLES", "title"]

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
