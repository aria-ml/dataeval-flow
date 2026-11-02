"""The `performance-sufficiency` check: each target read at the largest size measured, and the size the curve projects.

The outputs are an exact power law, ``accuracy = 1 - 0.9 n^-0.35``, measured at 10, 20, 40, 80 and 160 items, which
DataEval's fit recovers (scale 0.9, exponent 0.35, asymptote 0, to six places). At 160 items the accuracy is 0.8477.
The curve reaches 0.85 at n = 167.22 and 0.90 at n = 532.61, so the smallest whole sizes are 168 and 533; it never
reaches 1.0. The training set holds 200 items.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from dataeval.performance import SufficiencyOutput
from pydantic import ValidationError

from dataeval_flow.steps import CheckContext, StepSkipped
from dataeval_flow.steps.checks import PerformanceSufficiencyCheck, PerformanceSufficiencyConfig, SufficiencyTarget

_STEPS = np.array([10, 20, 40, 80, 160])
_ACCURACY = 1 - 0.9 * _STEPS.astype(float) ** -0.35


def _output(**measures: Any) -> SufficiencyOutput:
    np.random.seed(0)  # noqa: NPY002 - DataEval's basin hopping draws from the global RNG
    return SufficiencyOutput(steps=_STEPS, measures={"accuracy": _ACCURACY[None, :], **measures})


def _judge(*targets: tuple[str, float], output: SufficiencyOutput | None = None, training: int = 200) -> Any:
    node = SimpleNamespace(value=output or _output(), computed_on=(SimpleNamespace(value=range(training)),))
    config = PerformanceSufficiencyConfig(input="s", targets=[SufficiencyTarget(metric=m, value=v) for m, v in targets])
    (finding,) = PerformanceSufficiencyCheck().run(config, {"input": node}, CheckContext("t", "s"))
    return finding


def _fields(block: Any) -> dict[str, Any]:
    return dict(block.blocks[0].items)


def test_a_target_reached_at_the_largest_size_is_ok() -> None:
    finding = _judge(("accuracy", 0.8))
    assert (finding.severity, finding.title) == ("ok", "Performance Sufficiency")
    assert finding.brief == "1 of 1 targets reached at 160 items"
    (block,) = finding.blocks
    assert (block.title, block.brief) == ("`accuracy` at least 0.8", "reached")
    assert _fields(block) == {"At 160 items": 0.8477, "Reached": "yes"}
    assert "extrapolation" in finding.description


def test_a_projection_within_the_training_set_informs() -> None:
    finding = _judge(("accuracy", 0.85))
    assert finding.severity == "info"
    (block,) = finding.blocks
    assert block.brief == "not reached; about 168 items"
    assert _fields(block) == {
        "At 160 items": 0.8477,
        "Reached": "no",
        "Projected size": "about 168 items",
        "Ratio to largest": "1.05",
    }


def test_a_projection_past_the_training_set_warns() -> None:
    finding = _judge(("accuracy", 0.8), ("accuracy", 0.9))
    assert finding.severity == "warning"
    assert finding.brief == "1 of 2 targets reached at 160 items"
    assert _fields(finding.blocks[1])["Projected size"] == "about 533 items"
    assert _fields(finding.blocks[1])["Ratio to largest"] == "3.33"
    assert _judge(("accuracy", 0.9), training=600).severity == "info"  # 533 items fit a training set of 600


@pytest.mark.parametrize("value", [1.0, 0.9999999])  # the curve's limit; about 7e19 items, past an int64
def test_a_target_the_curve_never_reaches_warns(value: float) -> None:
    finding = _judge(("accuracy", value))
    assert finding.severity == "warning"
    (block,) = finding.blocks
    assert block.brief == "not reached by the fitted curve"
    assert "Ratio to largest" not in _fields(block)


def test_a_curve_that_falls_as_the_training_set_grows_never_reaches_a_target_above_it() -> None:
    falling = (0.5 + 0.3 * _STEPS.astype(float) ** -0.5)[None, :]  # 0.595 at 10 items down to 0.524 at 160
    output = SufficiencyOutput(steps=_STEPS, measures={"accuracy": falling})
    assert 0 < int(output.inv_project({"accuracy": [0.55]})["accuracy"][0]) < 160  # where the curve passed 0.55
    finding = _judge(("accuracy", 0.55), output=output)
    assert finding.severity == "warning"
    assert finding.blocks[0].brief == "not reached by the fitted curve"


def test_the_size_is_right_whichever_way_inv_project_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """DataEval 1.2.0rc2 gives 167 for 0.85, where the curve is just under it; a later release rounds up to 168."""
    output = _output()
    assert int(output.inv_project({"accuracy": [0.85]})["accuracy"][0]) in (167, 168)
    assert _fields(_judge(("accuracy", 0.85), output=output).blocks[0])["Projected size"] == "about 168 items"
    rounded_down = output.inv_project

    def rounded_up(targets: Any, n_iter: int = 1000) -> Any:
        frame = rounded_down(targets, n_iter)
        return frame.with_columns(frame["accuracy"] + (frame["accuracy"] == 167))

    monkeypatch.setattr(output, "inv_project", rounded_up)
    assert _fields(_judge(("accuracy", 0.85), output=output).blocks[0])["Projected size"] == "about 168 items"


def test_a_per_class_metric_is_not_assessed() -> None:
    recall = np.stack([_ACCURACY, _ACCURACY], axis=-1)[None]  # one run, five sizes, two classes
    output = _output(recall=recall)
    finding = _judge(("accuracy", 0.8), ("recall", 0.5), output=output)
    assert finding.severity == "info"  # not assessed is never a pass
    assert finding.brief == "1 of 1 targets reached at 160 items, 1 not assessed"
    assert finding.blocks[1].brief == "not assessed: a per-class metric"
    with pytest.raises(StepSkipped, match=r"every target names a per-class metric \(recall\)"):
        _judge(("recall", 0.5), output=output)


def test_a_metric_the_run_did_not_measure_is_an_error_listing_those_it_did() -> None:
    with pytest.raises(ValueError, match=r"No metric \['f1'\] was measured; the run measured \['accuracy'\]"):
        _judge(("f1", 0.5))


def test_a_check_needs_a_target() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        PerformanceSufficiencyConfig(input="s", targets=[])
