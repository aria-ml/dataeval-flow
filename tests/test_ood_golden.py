"""The OOD golden, recorded from the legacy workflow before its port (ood-detection spec §10)."""

import json
from pathlib import Path

from tests.golden.ood import CASES

_GOLDEN = json.loads((Path(__file__).parent / "golden" / "ood.json").read_text())


def test_every_case_is_recorded() -> None:
    assert sorted(_GOLDEN) == sorted(CASES)


def test_the_two_detector_case_has_mutual_and_unique_images() -> None:
    both = _GOLDEN["both"]
    assert both["mutual"]
    assert any(both["unique"].values())
    assert "Unique OOD Samples (single-detector only)" in [finding["title"] for finding in both["findings"]]


def test_the_shifted_cases_flag_images_and_explain_them() -> None:
    for name in ("kneighbors", "domain_classifier", "both"):
        assert _GOLDEN[name]["union"], name
        assert _GOLDEN[name]["predictors"], name
        assert _GOLDEN[name]["deviations"], name
    assert _GOLDEN["insights_off"]["predictors"] is None


def test_the_like_for_like_case_flags_nothing() -> None:
    assert _GOLDEN["nothing_flagged"]["union"] == []
