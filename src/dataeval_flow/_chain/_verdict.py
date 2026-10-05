"""A preset chain's verdict over the checks it names, and the next steps it gives (audit spec §6, §9.5, §21)."""

__all__ = [
    "LABELS",
    "Acceptance",
    "Level",
    "Unassessed",
    "Verdict",
    "VerdictItem",
    "judge",
    "next_step_lines",
    "reason_class",
]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from dataeval_flow.steps._result import StepResult
    from dataeval_flow.workflows._preset import NextSteps

Level = Literal["not-ready", "ready-with-caveats", "ready"]
LABELS: dict[Level, str] = {"not-ready": "Not ready", "ready-with-caveats": "Ready with caveats", "ready": "Ready"}


class VerdictItem(BaseModel):
    """A warning no acceptance covers: its check type, the step that made it, and its finding's title and brief."""

    check: str
    step: str
    title: str
    brief: str


class Acceptance(BaseModel):
    """An accepted check type, why, and whether its check warned on this run, passed, or could not run."""

    check: str
    reason: str
    state: Literal["warned", "did-not-warn", "not-assessed"]


class Unassessed(BaseModel):
    """A check, or element of one, that judged nothing, and why."""

    check: str
    step: str
    reason: str


class Verdict(BaseModel):
    """Whether the data is ready, with the warnings, acceptances and unassessed checks behind it."""

    level: Level
    blocking: list[VerdictItem]
    warnings: list[VerdictItem]
    accepted: list[Acceptance]
    not_assessed: list[Unassessed]

    @property
    def label(self) -> str:
        """The level as a report writes it: "Ready with caveats"."""
        return LABELS[self.level]

    def line(self) -> str:
        """The level and its reasons: each blocking warning, or how many caveats of each kind."""
        if self.level == "not-ready":
            return f"{self.label}: " + "; ".join(
                f"{item.title} ({item.brief})" if item.brief else item.title for item in self.blocking
            )
        counts = [
            (len(self.warnings), "warning", "warnings"),
            (sum(a.state == "warned" for a in self.accepted), "accepted risk", "accepted risks"),
            (len(self.not_assessed), "not assessed", "not assessed"),
        ]
        reasons = [f"{n} {one if n == 1 else many}" for n, one, many in counts if n]
        return f"{self.label}: {', '.join(reasons)}" if reasons else self.label


def judge(steps: "Mapping[str, StepResult]", *, blocking: Sequence[str], accepted: Mapping[str, str]) -> Verdict:
    """The verdict over `steps`' check records: an unaccepted warning of a `blocking` type makes it not ready; any
    other unaccepted warning, an acceptance that fired, or a check not assessed makes it ready with caveats."""
    blocked: list[VerdictItem] = []
    warnings: list[VerdictItem] = []
    warned: set[str] = set()
    unassessed: list[Unassessed] = []
    for record in steps.values():
        if record.kind != "check":
            continue
        runs = record.elements.items() if record.elements is not None else [(None, record)]
        for key, run in runs:
            if run.not_assessed is not None:
                step = record.name if key is None else f"{record.name}[{key}]"
                unassessed.append(Unassessed(check=record.type, step=step, reason=run.not_assessed))
            for finding in (run.output or []) if run.status == "ok" else []:
                if finding.severity != "warning":
                    continue
                if record.type in accepted:
                    warned.add(record.type)
                    continue
                item = VerdictItem(
                    check=record.type, step=finding.step or record.name, title=finding.title, brief=finding.brief or ""
                )
                (blocked if record.type in blocking else warnings).append(item)
    missed = {item.check for item in unassessed}
    acceptances = [
        Acceptance(
            check=check,
            reason=reason,
            state="warned" if check in warned else "not-assessed" if check in missed else "did-not-warn",
        )
        for check, reason in accepted.items()
    ]
    level: Level = "not-ready" if blocked else "ready-with-caveats" if warnings or warned or unassessed else "ready"
    return Verdict(level=level, blocking=blocked, warnings=warnings, accepted=acceptances, not_assessed=unassessed)


def next_step_lines(verdict: Verdict, plan: "NextSteps") -> list[str]:
    """One line per check type with unaccepted warnings, the blocking types before the others, then one per class of
    reason a check was not assessed, each with `plan`'s advice where it has some."""
    from dataeval_flow.steps._registry import get_check

    lines: list[str] = []
    warned: dict[str, list[VerdictItem]] = {}
    for item in [*verdict.blocking, *verdict.warnings]:
        warned.setdefault(item.check, []).append(item)
    for check, items in warned.items():
        steps = ", ".join(dict.fromkeys(item.step for item in items))
        advice = plan.by_check.get(check)
        lines.append(f"{items[0].title} ({steps})" + (f": {advice}" if advice else ""))
    classes: dict[str, list[str]] = {}
    for unassessed in verdict.not_assessed:
        classes.setdefault(reason_class(unassessed.reason, plan), []).append(get_check(unassessed.check).title)
    for reason, titles in classes.items():
        named = ", ".join(dict.fromkeys(titles))
        if reason in plan.by_reason:
            lines.append(f"{plan.by_reason[reason]} Not assessed: {named}.")
        else:
            lines.append(f"Not assessed ({named}): {reason.rstrip('.')}.")
    return lines


def reason_class(reason: str, plan: "NextSteps") -> str:
    """The class of reason `reason` is: the first of `plan`'s reason fragments it holds, or `reason` itself."""
    return next((fragment for fragment in plan.by_reason if fragment in reason), reason)
