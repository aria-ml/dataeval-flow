"""The `sufficiency-target` check: whether a model reached each metric's target, and how much data the fitted curve says
it would take."""

__all__ = ["SufficiencyTarget", "SufficiencyTargetCheck", "SufficiencyTargetConfig"]

from collections.abc import Mapping
from typing import Any, ClassVar

from dataeval.performance import SufficiencyOutput
from pydantic import BaseModel, ConfigDict, Field

from dataeval_flow._blocks import Block, Fields, Scalar, Section
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import StepSkipped
from dataeval_flow.steps.checks._limits import Severity
from dataeval_flow.workflows._base import Finding


class SufficiencyTarget(BaseModel):
    """One metric's target: the value the model should reach, where higher is better."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    metric: str = Field(description="A scalar metric the trainer's evaluation strategy returns, named as it names it.")
    value: float = Field(description="The value the metric should reach or pass; higher is better.")


class SufficiencyTargetConfig(CheckConfig):
    """A `sufficiency-target` step's input, and the target each metric should reach."""

    input: str = Field(description="A `sufficiency` Output.")
    targets: list[SufficiencyTarget] = Field(
        min_length=1,
        description=(
            "Each metric's target, as `{metric, value}`; higher is better. A metric the run did not measure is an "
            "error, and a per-class metric is not assessed."
        ),
    )


class SufficiencyTargetCheck(Check[SufficiencyTargetConfig]):
    """``sufficiency-target``: whether each metric reached its target at the largest training-set size measured, and,
    where not, the size the fitted curve projects for it (DataEval's ``inv_project``)."""

    name: ClassVar[str] = "sufficiency-target"
    description: ClassVar[str] = (
        "Judges `sufficiency`'s output: warns when a metric's target is not reached at the largest size measured and "
        "the fitted curve needs more data than the training set holds, or never reaches it."
    )
    experimental: ClassVar[bool] = True  # judges `sufficiency`, which is experimental
    title: ClassVar[str] = "Sufficiency Target"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(SufficiencyOutput,)),)

    def run(self, config: SufficiencyTargetConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """A block per target: its value at the largest size measured, whether it is reached, and where not, the size
        the curve projects and its ratio to that largest size.

        ``ok`` where every target is reached; ``warning`` where a projection exceeds the training set or the curve
        never reaches the target; ``info`` otherwise, as where a projection falls within the training set, or a
        target names a per-class metric, which is not assessed.

        Raises
        ------
        ValueError
            When a target names a metric the run did not measure, listing those it did.
        StepSkipped
            When every target names a per-class metric: recorded as not assessed.
        """
        node = inputs["input"]
        output: SufficiencyOutput = node.value
        measured = output.averaged_measures
        unknown = sorted({target.metric for target in config.targets} - set(measured))
        if unknown:
            raise ValueError(f"No metric {unknown} was measured; the run measured {sorted(measured)}.")
        per_class = [target.metric for target in config.targets if measured[target.metric].ndim > 1]
        if len(per_class) == len(config.targets):
            raise StepSkipped(f"every target names a per-class metric ({', '.join(per_class)}): only a scalar can be")
        largest = int(output.steps[-1])
        training = len(node.computed_on[0].value)
        blocks: list[Block] = []
        reached = short = beyond = 0
        for target in config.targets:
            label = f"`{target.metric}` at least {target.value:g}"
            if target.metric in per_class:
                blocks.append(Section(title=label, brief="not assessed: a per-class metric"))
                continue
            value = float(measured[target.metric][-1])
            items: list[tuple[str, Scalar]] = [(f"At {largest:,} items", round(value, 4))]
            if value >= target.value:
                reached += 1
                items.append(("Reached", "yes"))
                blocks.append(Section(title=label, brief="reached", blocks=[Fields(items=items)]))
                continue
            size = _projected_size(output, target.metric, target.value)
            if size is None or size > training:
                beyond += 1
            else:
                short += 1
            items.append(("Reached", "no"))
            if size is None:
                brief = "not reached by the fitted curve"
                items.append(("Projected size", brief))
            else:
                brief = f"not reached; about {size:,} items"
                items += [("Projected size", f"about {size:,} items"), ("Ratio to largest", f"{size / largest:.2f}")]
            blocks.append(Section(title=label, brief=brief, blocks=[Fields(items=items)]))
        severity: Severity = "warning" if beyond else "info" if short or per_class else "ok"
        assessed = len(config.targets) - len(per_class)
        brief = f"{reached} of {assessed} targets reached at {largest:,} items"
        if per_class:
            brief += f", {len(per_class)} not assessed"
        return [
            Finding(
                severity=severity,
                title=self.title,
                brief=brief,
                description=(
                    f"Each target is read at the largest size measured, {largest:,} of the training set's "
                    f"{training:,} items; higher is better. Where it is not reached, the size is where DataEval's "
                    "fitted power law reaches it, and a size past the largest measured is an extrapolation of that "
                    "curve, less certain the further it reaches."
                ),
                blocks=blocks,
            )
        ]


def _projected_size(output: SufficiencyOutput, metric: str, target: float) -> int | None:
    """The smallest size at which the fitted curve reaches `target`; ``None`` where it never does: where the target is
    at or past the curve's limit, or the curve is flat or falls as the training set grows (it then passes a target above
    its measurements only at smaller sizes), or the size is past what DataEval can count.

    DataEval 1.2.0rc2's ``inv_project`` rounds down, to a size where the curve is still under the target; a later
    release rounds up. Reading the curve at the size it gives, and adding one where it is short, is right on both.
    """
    # DataEval fits `scale * n**-exponent + asymptote` to 1 - metric: the metric rises only where scale and exponent are
    # positive, toward 1 - asymptote, which it never reaches.
    scale, exponent, asymptote = (float(value) for value in output.params[metric])
    if scale <= 0 or exponent <= 0 or target >= 1 - asymptote:
        return None
    size = int(output.inv_project({metric: [target]})[metric][0])
    if size < 0:
        return None
    size = max(size, 1)  # the curve is undefined at 0 items
    return size + 1 if round(float(output.project([size])[metric][0]), 9) < target else size
