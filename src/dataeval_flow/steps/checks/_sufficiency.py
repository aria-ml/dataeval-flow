"""The `performance-sufficiency` check: whether a model reached each metric's target, and how much data the fitted
curve says it would take."""

__all__ = ["SufficiencyTarget", "PerformanceSufficiencyCheck", "PerformanceSufficiencyConfig"]

from collections.abc import Mapping
from typing import Any, ClassVar

from dataeval.performance import SufficiencyOutput
from pydantic import BaseModel, ConfigDict, Field

from dataeval_flow._blocks import Cell, Column, Table
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


class PerformanceSufficiencyConfig(CheckConfig):
    """A `performance-sufficiency` step's input, and the target each metric should reach."""

    input: str = Field(description="A `sufficiency` Output.")
    targets: list[SufficiencyTarget] = Field(
        min_length=1,
        description=(
            "Each metric's target, as `{metric, value}`; higher is better. A metric the run did not measure is an "
            "error, and a per-class metric is not assessed."
        ),
    )


class PerformanceSufficiencyCheck(Check[PerformanceSufficiencyConfig]):
    """``performance-sufficiency``: whether each metric reached its target at the largest training-set size measured,
    and, where not, the size the fitted curve projects for it (DataEval's ``inv_project``)."""

    name: ClassVar[str] = "performance-sufficiency"
    description: ClassVar[str] = (
        "Judges `sufficiency`'s output: warns when a metric's target is not reached at the largest size measured and "
        "the fitted curve needs more data than the training set holds, or never reaches it."
    )
    experimental: ClassVar[bool] = True  # judges `sufficiency`, which is experimental
    title: ClassVar[str] = "Performance Sufficiency"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(SufficiencyOutput,)),)

    def run(
        self,
        config: PerformanceSufficiencyConfig,
        inputs: Mapping[str, Any],
        context: CheckContext,  # noqa: ARG002
    ) -> list[Finding]:
        """A learning curve per metric, drawn against its targets, then a row per target: whether it is reached, and
        where not, the size the fitted curve projects and its ratio to the largest size measured.

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
        rows: list[dict[str, Cell]] = []
        reached = short = beyond = 0
        for target in config.targets:
            row: dict[str, Cell] = {"target": f"{target.metric} ≥ {target.value:.10g}", "reached": "yes", "at": ""}
            rows.append(row)
            if target.metric in per_class:
                row.update(reached="n/a", at="not assessed: a per-class metric")
                continue
            if float(measured[target.metric][-1]) >= target.value:
                reached += 1
                continue
            size = _projected_size(output, target.metric, target.value)
            if size is None or size > training:
                beyond += 1
            else:
                short += 1
            row.update(reached="no", at=_never(output, target) if size is None else _reaches(size, largest))
        severity: Severity = "warning" if beyond else "info" if short or per_class else "ok"
        assessed = len(config.targets) - len(per_class)
        brief = f"{reached} of {assessed} targets reached at {largest:,} items"
        if per_class:
            brief += f", {len(per_class)} not assessed"
        columns = [
            Column(key="target", header="Target", align="left"),
            Column(key="reached", header="Reached"),
            Column(key="at", header="Fitted curve reaches it at", align="left"),
        ]
        curves = [
            _curve(output, metric, [t.value for t in config.targets if t.metric == metric])
            for metric in dict.fromkeys(t.metric for t in config.targets if t.metric not in per_class)
        ]
        return [
            Finding(
                severity=severity,
                title=self.title,
                brief=brief,
                blocks=[*curves, Table(columns=columns, rows=rows)],
            )
        ]


def _curve(output: SufficiencyOutput, metric: str, targets: list[float]) -> Table:
    """A metric at each size measured, as a bar scaled up to its highest target, with a marker at each target."""
    markers = [("target", value) for value in targets]
    columns = [
        Column(key="size", header="Items", align="right", format="{:,}"),
        Column(key="value", header=metric, format="{:.4f}"),
        Column(key="value", kind="bar", format="{:.10g}", markers=markers),
    ]
    values = output.averaged_measures[metric]
    rows: list[dict[str, Cell]] = [
        {"size": int(size), "value": round(float(value), 4)} for size, value in zip(output.steps, values, strict=True)
    ]
    return Table(columns=columns, rows=rows)


def _reaches(size: int, largest: int) -> str:
    """The size a curve reaches a target at, and its ratio to the largest size measured."""
    ratio = size / largest
    return f"~{size:,} items ({ratio:.1f}x)" if ratio < 10 else f"~{size:,} items ({ratio:,.0f}x)"


def _never(output: SufficiencyOutput, target: SufficiencyTarget) -> str:
    """Why the fitted curve never reaches `target`: it levels off below it, it does not rise, or it is too far out."""
    scale, exponent, asymptote = (float(value) for value in output.params[target.metric])
    if scale <= 0 or exponent <= 0:
        return "never (the curve does not rise)"
    if target.value >= 1 - asymptote:
        return f"never (levels off near {1 - asymptote:.3f})"
    return "never (too far out to project)"


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
