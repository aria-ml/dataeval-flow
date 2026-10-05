"""drift-monitoring agrees with what it produced before its port: each detector's verdict, distance, threshold and
chunks, each class's verdict, distance and p-value, and the findings' severities in order (spec §10.11).

`tests/golden/drift.py` maps each case's legacy config onto its preset config, and each legacy detector onto the step
that answers it. Deliberate differences from the legacy run (spec §10.3 item 3), each with its reason:

- **Each test source is tested on its own.** Legacy concatenated every test source into one test set. The user chose
  per-source tests (spec §10.7), so `two_tests` runs a custom workflow that `merge`s its test sources, then the
  detector and the `drift` check, which pins "merge with the verbs" against legacy's merged verdict.
- **Titles follow the evaluators', and settings move to Configuration.** A finding is titled as its step is, such as
  `Drift (MMD)`, where legacy wrote `MMD (n_permutations=50)`. This test compares no titles.
- **Whole-set and chunked findings gain briefs**: `drift`, `no drift`, or `3/10 chunks drifted`.
- **A classwise detector makes two findings, with the severity split between them**: a whole-set one and a by-class
  one. For an unchunked classwise detector, legacy's one severity is the worse of the two; for any other, it is the
  whole-set finding's.
- **`classwise_any_drift_is_warning` folds into `warn_on_drift`**, one `drift` check judging whole-set and per-class
  runs alike. At the defaults, which every case uses, both were true.
- **A chunked classwise detector's classes are now shown.** Legacy computed them and never showed them; the port runs
  an unchunked copy of the entry by class, and `classwise_chunked` compares its classes with legacy's.
- **Classwise on detection or unlabelled data is reported "not assessed"**, where legacy skipped it with only a log
  line. In `classwise_boxes`, the by-class run is skipped and its check makes an info finding.
- **`update_strategy`, `device` and the summary line are gone.** `update_strategy` was never applied, Flow sets
  the device, and the chain's findings replace the summary.

- **Step names follow the naming pass** (naming spec §5.3): recorded step names are read through
  `tests/golden/_renames.py`.

A detector that raises fails its step and the task, where legacy recorded it and still succeeded. No toy here makes
one raise, so `tests/test_drift_preset.py` pins it.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config import PipelineConfig
from dataeval_flow.steps import ChainResult
from tests.golden._renames import step as renamed
from tests.golden.drift import CASES, pipeline
from tests.golden.rerouting import approximately

_GOLDEN = json.loads((Path(__file__).parent / "golden" / "drift.json").read_text())
_ORDER = ("ok", "info", "warning")


def _run(name: str) -> ChainResult:
    """Case `name` through the preset, or through its custom workflow, with the golden's seed."""
    case = CASES[name]
    workflow = case.custom or {"name": "drift", "type": "drift-monitoring", **case.preset}
    task = {"name": "t", "workflow": workflow["name"], "sources": list(case.datasets()), "extractor": "flat"}
    evaluators = case.preset["detectors"] if case.custom else []
    config = PipelineConfig.model_validate(
        {**dict(pipeline(name)), "workflows": [workflow], "evaluators": evaluators, "tasks": [task]}
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    return result


def _output(result: ChainResult, step: str) -> Any:
    """What `step` gave the test source: the element named `test`, or, in the merged case, the step's own output."""
    record = result.steps[step]
    return record.elements["test"].output if record.elements is not None else record.output


def _severity(result: ChainResult, step: str) -> str:
    (finding,) = _output(result, step)
    return finding.severity


def test_every_case_is_recorded() -> None:
    assert sorted(_GOLDEN) == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_drift_monitoring_gives_what_it_gave_before_its_port(name: str) -> None:
    case, golden = CASES[name], _GOLDEN[name]
    result = _run(name)
    assert result.success, result.errors
    detectors, classes = {}, {}
    severities = []
    for key, step in case.steps.items():  # legacy's detector order, which its findings follow
        output = _output(result, step)
        chunks = output.details.iter_rows(named=True) if key in golden["chunked_detectors"] else []
        detectors[key] = {
            "drifted": output.drifted,
            "distance": float(output.distance),
            "threshold": float(output.threshold),
            "chunks": [{"value": float(row["value"]), "drifted": row["drifted"]} for row in chunks],
        }
        severity = _severity(result, f"{step}-check")
        if key in golden["classwise_detectors"]:
            per_class = _output(result, renamed("drift-monitoring", f"{step}-classes"))
            if per_class is not None:
                classes[key] = [
                    {
                        "class": label,
                        "drifted": inner.drifted,
                        "distance": inner.distance,
                        "p_val": inner.details["p_val"],
                    }
                    for label, inner in per_class.outputs.items()
                ]
            if key not in golden["chunked_detectors"]:
                severity = max(
                    severity, _severity(result, renamed("drift-monitoring", f"{step}-classes-check")), key=_ORDER.index
                )
        severities.append(severity)
    assert detectors == approximately(golden["detectors"])
    assert classes == approximately(golden["classwise"])
    assert severities == golden["severities"]


def test_classwise_on_detection_data_is_not_assessed() -> None:
    result = _run("classwise_boxes")
    assert (result.steps[renamed("drift-monitoring", "drift-kneighbors-classes")].elements or {})[
        "test"
    ].status == "skipped"
    (finding,) = _output(result, renamed("drift-monitoring", "drift-kneighbors-classes-check"))
    assert (finding.severity, finding.title, finding.brief) == ("info", "Drift (K-Neighbors) by class", "not assessed")
