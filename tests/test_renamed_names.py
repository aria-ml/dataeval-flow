"""The names the naming pass changed are registered, and the old ones fail as unknown (preset naming spec §4)."""

import pytest

from dataeval_flow.steps._registry import CHECKS, get_check

_CHECKS = {
    "classwise-outliers": ("class-outliers", "Class Outliers"),
    "stratification": ("class-stratification", "Class Stratification"),
    "metadata-issues": ("factor-issues", "Factor Issues"),
    "mergeability": ("label-mergeability", "Label Mergeability"),
    "distribution-shift": ("embedding-divergence", "Embedding Divergence"),
}


@pytest.mark.parametrize(("old", "new"), _CHECKS.items(), ids=list(_CHECKS))
def test_a_renamed_check_is_registered_under_its_new_name_only(old: str, new: tuple[str, str]) -> None:
    name, title = new
    assert get_check(name).title == title
    assert old not in CHECKS.names()
    with pytest.raises(ValueError, match=rf"Unknown check: '{old}'\. Installed: "):
        get_check(old)
