"""The names the naming pass changed are registered, and the old ones fail as unknown (preset naming spec §4)."""

import pytest

from dataeval_flow.steps._registry import CHECKS, get_check
from dataeval_flow.workflows._registry import WORKFLOWS, get_workflow

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


_PRESETS = {
    "data-bias": ("bias", "Bias"),
    "data-cleaning": ("quality", "Quality"),
    "data-coverage": ("scope", "Scope"),
    "data-prioritization": ("prioritization", "Prioritization"),
    "data-splitting": ("splits", "Splits"),
    "metadata-triage": ("triage", "Triage"),
    "label-space": ("taxonomy", "Taxonomy"),
}


@pytest.mark.parametrize(("old", "new"), _PRESETS.items(), ids=list(_PRESETS))
def test_a_renamed_preset_is_registered_under_its_new_name_only(old: str, new: tuple[str, str]) -> None:
    name, title = new
    assert get_workflow(name).title == title
    assert old not in WORKFLOWS.names()
    with pytest.raises(ValueError, match=rf"Unknown workflow: '{old}'\. Installed: "):
        get_workflow(old)


def test_the_label_space_concept_keeps_its_names() -> None:
    """The relabelled vocabulary's records and digest are a result concept, not the renamed preset."""
    from dataeval_flow import ResultMetadata
    from dataeval_flow._result import LabelSpaceRecord

    assert {"label_space", "label_space_digest"} <= set(ResultMetadata.model_fields)
    assert LabelSpaceRecord.__name__ == "LabelSpaceRecord"


def test_the_old_presets_are_unknown() -> None:
    for old in ("drift-monitoring", "ood-detection"):
        with pytest.raises(ValueError, match=rf"Unknown workflow: '{old}'"):
            get_workflow(old)
