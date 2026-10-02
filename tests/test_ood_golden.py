"""The ood-detection preset gives what the legacy workflow gave, recorded in `golden/ood.json` before its deletion
(ood-detection spec §10).

Deliberate differences, each with its reason:

- **The metadata insights are sections, not findings.** Legacy's "OOD Factor Predictors" and "OOD Sample Metadata
  Deviations" were `info` findings that judged nothing. The `factor-` steps' sections show them.
- **A detector's finding is titled by its evaluator entry,** "OOD (K-Neighbors)", not legacy's display name, which
  listed non-default settings ("K-Neighbors (k=5, distance_metric=euclidean)").
- **A brief counts images, not samples,** since on detection rows a sample could be a detection, and it drops
  legacy's `"{name}: "` prefix, since the title names the detector. The aggregate's sentence on scores moves to its
  description.
- **Findings are per test source.** Legacy joined the test sources end to end. `two_tests` compares each detector's
  flags and scores, joined in the same order, and no findings.
- **A detector that raises fails its step and the task,** where legacy recorded it and still succeeded. No case makes
  one raise, so `tests/test_ood_preset.py` pins it.
- **A domain-classifier detector thresholds on `n_std` unless `threshold_perc` is written,** where legacy always used
  the 95th percentile. The cases write `threshold_perc: 95`, as legacy ran.
- **With three detectors or more, an image some but not all flagged is partial,** where legacy listed it as unique to
  each detector that flagged it. The cases have at most two detectors, so `tests/test_ood_union.py` pins it.
- **An image no detector scored has no agreement score,** where legacy showed 0.0. Embeddings score every image.
- **A detector left out for a non-positive threshold adds no images to the union,** where legacy's `total_ood`, the
  denominator of its agreement description, still counted its flags. The golden's cases do not hit this.
- **Deviations are computed for the most out-of-distribution agreed images.** Legacy took the first
  `max_ood_insights` flagged images in index order, and showed the agreed ones among them. The deviations of the
  images both computed agree.
"""

import json
import re
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import PipelineConfig, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.steps import ChainResult
from tests.golden.ood import CASES, SINGLE_SOURCE, pipeline
from tests.golden.rerouting import approximately

_GOLDEN = json.loads((Path(__file__).parent / "golden" / "ood.json").read_text())
_INSIGHTS = ("OOD Factor Predictors", "OOD Sample Metadata Deviations")
_HEADINGS = {"ood-kneighbors": "OOD (K-Neighbors)", "ood-domain-classifier": "OOD (Domain Classifier)"}
_EXPLAINED = tuple(name for name in SINGLE_SOURCE if CASES[name].preset.get("metadata_insights", True))


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


@pytest.fixture(autouse=True)
def _fresh_caches() -> Iterator[None]:
    """Each run reads its own metadata, so each binning warning is raised in the test that causes it."""
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _run(name: str) -> ChainResult:
    """Case `name` through the preset, with the golden's seed."""
    case = CASES[name]
    workflow = {"name": "ood", "type": "ood-detection", **case.preset}
    task = {"name": "t", "workflow": "ood", "sources": list(case.datasets()), "extractor": "flat"}
    config = PipelineConfig.model_validate({**dict(pipeline(name)), "workflows": [workflow], "tasks": [task]})
    # The factor steps read metadata where anything was flagged, and DataEval bins the toys' continuous factors.
    explains = case.preset.get("metadata_insights", True) and _GOLDEN[name]["union"]
    with pytest.warns(UserWarning, match="binned automatically") if explains else nullcontext():
        result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


def _elements(result: ChainResult, step: str) -> dict[str, Any]:
    return {key: element.output for key, element in (result.steps[step].elements or {}).items()}


def _as_brief(description: str) -> str:
    """Legacy's description as the preset's brief: without the `{name}: ` prefix, images not samples, and the
    aggregate's first clause only."""
    text = re.sub(r"^[^:]+: ", "", description) if "samples OOD" in description else description
    text = text.split(", most out of distribution first")[0]
    return text.replace("sample(s)", "image(s)").replace("samples", "images")


@pytest.mark.parametrize("name", SINGLE_SOURCE)
def test_the_findings_agree_with_legacy(name: str) -> None:
    case, golden = CASES[name], _GOLDEN[name]
    legacy = [finding for finding in golden["findings"] if finding["title"] not in _INSIGHTS]
    titles = [_HEADINGS[step] for step in case.steps.values()]
    expected = [
        (
            finding["severity"],
            titles[index] if index < len(titles) else finding["title"],
            _as_brief(finding["description"]),
        )
        for index, finding in enumerate(legacy)
    ]
    found = [(finding.severity, finding.title, finding.brief) for finding in _run(name).findings]
    assert found == expected


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_flags_scores_and_agreement_agree_with_legacy(name: str) -> None:
    case, golden = CASES[name], _GOLDEN[name]
    result = _run(name)
    for key, step in case.steps.items():
        outputs = list(_elements(result, step).values())  # in source order, as legacy joined them
        assert [bool(flag) for output in outputs for flag in output.is_ood] == golden["detectors"][key]["is_ood"]
        scores = [float(score) for output in outputs for score in output.instance_score]
        assert scores == approximately(golden["detectors"][key]["scores"])
    if name not in SINGLE_SOURCE:
        return
    (union,) = _elements(result, "agreement").values()
    assert (union.union, union.mutual) == (golden["union"], golden["mutual"])
    assert union.unique == {case.steps[key]: indices for key, indices in golden["unique"].items()}
    assert union.scores == approximately(golden["normalized"])
    thresholds = {case.steps[key]: detector["threshold"] for key, detector in golden["detectors"].items()}
    assert union.thresholds == approximately(thresholds)


@pytest.mark.parametrize("name", _EXPLAINED)
def test_the_insights_agree_with_legacy(name: str) -> None:
    golden = _GOLDEN[name]
    result = _run(name)
    (predictors,) = _elements(result, "factor-predictors").values()
    (deviation,) = _elements(result, "factor-deviation").values()
    assert predictors.factors == approximately(golden["predictors"] or {})
    computed = {str(item.index): item.deviations for item in deviation.items}
    # Legacy computed every flagged image's deviations here (at most 50 were flagged), so each of the preset's is there.
    assert set(computed) <= set(golden["deviations"])
    assert computed == approximately({key: golden["deviations"][key] for key in computed})
