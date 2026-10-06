"""A preset chain's verdict over the checks it names, and the next steps it gives (audit spec §6, §9.5, §21)."""

__all__ = [
    "LABELS",
    "Acceptance",
    "Level",
    "Unassessed",
    "Verdict",
    "VerdictItem",
    "judge",
    "moot_checks",
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
    """An accepted check type or check step, why, and whether its check warned on this run, passed, or could not run."""

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

    def meets(self, requirement: str) -> bool:
        """Whether this verdict passes `requirement`, the worst verdict `result: require:` lets through:
        ``ready-with-caveats`` passes anything but Not ready; ``ready-with-accepted-risks`` passes Ready, and Ready with
        caveats whose only caveats are fired acceptances; ``ready`` passes only Ready."""
        if requirement == "ready-with-caveats":
            return self.level != "not-ready"
        if requirement == "ready-with-accepted-risks":
            return self.level == "ready" or (
                self.level == "ready-with-caveats" and not self.warnings and not self.not_assessed
            )
        return self.level == "ready"

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


def judge(
    steps: "Mapping[str, StepResult]", *, blocking: Sequence[str], accepted: Mapping[str, str], prefix: str = ""
) -> Verdict:
    """The verdict over `steps`' check records: an unaccepted warning of a `blocking` type makes it not ready; any
    other unaccepted warning, an acceptance that fired, or a check not assessed makes it ready with caveats. A check
    run that did not complete is not assessed, for its skip reason or its errors; :func:`moot_checks` are left out. An
    acceptance keyed by a step covers that step's runs: all of them by its name, one by `name[element]`. A spliced
    chain's steps carry `prefix` (`audit/`): an acceptance, written in the entry's own names, covers them with it
    dropped."""
    blocked: list[VerdictItem] = []
    warnings: list[VerdictItem] = []
    warned: set[str] = set()
    unassessed: list[Unassessed] = []
    moot = moot_checks(steps)
    for record in steps.values():
        if record.kind != "check" or record.name in moot:
            continue
        runs = record.elements.items() if record.elements is not None else [(None, record)]
        for key, run in runs:
            # A check run that did not complete, such as an optional check that failed, judged nothing either.
            reason = run.not_assessed if run.status == "ok" else run.reason or "; ".join(run.errors)
            if reason is not None:
                step = record.name if key is None else f"{record.name}[{key}]"
                unassessed.append(Unassessed(check=record.type, step=step, reason=reason))
            for finding in (run.output or []) if run.status == "ok" else []:
                if finding.severity != "warning":
                    continue
                if covering := _covering(accepted, record, key, prefix):
                    warned.update(covering)
                    continue
                item = VerdictItem(
                    check=record.type, step=finding.step or record.name, title=finding.title, brief=finding.brief or ""
                )
                (blocked if record.type in blocking else warnings).append(item)
    missed = {
        name.removeprefix(prefix) for item in unassessed for name in (item.check, item.step, item.step.split("[", 1)[0])
    }
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


def _covering(accepted: Mapping[str, str], record: "StepResult", key: str | None, prefix: str) -> list[str]:
    """The `accepted` keys covering `record`'s run `key`: its step as the verdict names it, its bare step, its type;
    each step with `prefix` dropped."""
    name = record.name.removeprefix(prefix)
    step = name if key is None else f"{name}[{key}]"
    return [candidate for candidate in dict.fromkeys((step, name, record.type)) if candidate in accepted]


def moot_checks(steps: "Mapping[str, StepResult]") -> set[str]:
    """The check steps among `steps` that ran once, judged nothing, and whose check type another run assessed: a check
    over a list holding no element, such as the evaluation splits of a one-split task, beside the same check over train
    (audit spec §6.3). Its reason must be one an empty list carried, which only a non-check record's `not_assessed`
    holds. The verdict and a preset's report leave them out. An element left unassessed is kept, as is every run of a
    check type that no run assessed, and a run not assessed for another reason, such as a failed producer."""
    checks = [record for record in steps.values() if record.kind == "check"]
    assessed = {
        record.type
        for record in checks
        for run in (record.elements.values() if record.elements is not None else [record])
        if run.status == "ok" and run.not_assessed is None
    }
    empty = {r.not_assessed for r in steps.values() if r.kind != "check" and r.not_assessed is not None}
    return {
        record.name
        for record in checks
        if record.elements is None
        and record.not_assessed is not None
        and record.not_assessed in empty
        and record.type in assessed
    }


def next_step_lines(verdict: Verdict, plan: "NextSteps") -> list[str]:
    """One line per check type with unaccepted warnings, the blocking types before the others, then one per class of
    reason a check was not assessed, each with `plan`'s advice where it has some. Each line names its checks by title,
    with the steps, or elements, behind each: "Leakage (leakage)", "Not assessed: Eval Coverage (eval-coverage)"."""
    from dataeval_flow.steps._registry import get_check

    lines: list[str] = []
    warned: dict[str, list[VerdictItem]] = {}
    for item in [*verdict.blocking, *verdict.warnings]:
        warned.setdefault(item.check, []).append(item)
    for check, items in warned.items():
        steps = ", ".join(dict.fromkeys(item.step for item in items))
        advice = plan.by_check.get(check)
        lines.append(f"{items[0].title} ({steps})" + (f": {advice}" if advice else ""))
    classes: dict[str, dict[str, list[str]]] = {}
    for unassessed in verdict.not_assessed:
        titles = classes.setdefault(reason_class(unassessed.reason, plan), {})
        titles.setdefault(get_check(unassessed.check).title, []).append(unassessed.step)
    for reason, titles in classes.items():
        named = ", ".join(f"{title} ({', '.join(dict.fromkeys(steps))})" for title, steps in titles.items())
        if reason in plan.by_reason:
            lines.append(f"{plan.by_reason[reason]} Not assessed: {named}.")
        else:
            lines.append(f"Not assessed: {named}: {reason.rstrip('.')}.")
    return lines


def reason_class(reason: str, plan: "NextSteps") -> str:
    """The class of reason `reason` is: the first of `plan`'s reason fragments it holds, or `reason` itself."""
    return next((fragment for fragment in plan.by_reason if fragment in reason), reason)
