"""A run's results as a CI job reads them: a JUnit report for its test views, and a Markdown summary.

The JUnit report makes each task a test suite and each of its findings a test case, failing where the
finding is a warning, so a CI's test view shows which checks a dataset didn't pass. The Markdown summary
is each task's findings as a table, for a job summary or a merge-request comment. Both name the tasks
that failed, which the JSON, text and HTML files leave out.
"""

__all__ = ["junit_report", "markdown_summary"]

import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataeval_flow._result import Result
    from dataeval_flow.workflows._base import Finding

# Characters XML 1.0 can't hold, such as a terminal colour code's escape, which ElementTree writes as they are.
_NOT_XML = re.compile("[^\t\n\r\x20-퟿-�\U00010000-\U0010ffff]")
# Markdown's inline punctuation, escaped so a name or a finding's text shows as written.
_MARKDOWN = re.compile(r"([\\`*_\[\]<>#|~])")


def junit_report(results: Mapping[str, "Result[Any, Any]"]) -> str:
    """Each task as a test suite, and each of its findings as a test case, failing where it's a warning.

    A task that failed is one test case, ``run``, in error with its errors. A task with no findings, an
    evaluator's among them, is one passing test case, ``run``. A title a task's findings repeat is numbered,
    ``Outliers (2)``, since a CI tells its cases apart by name, and the task's time goes on its first case
    too, since a CI adds up its cases' times.
    """
    from dataeval_flow._matrix._result import MatrixResult

    root = ET.Element("testsuites", name="dataeval-flow")
    for task, result in results.items():
        if isinstance(result, MatrixResult):
            for run in result.runs:
                _suite(root, f"{task} · run {run.number}", run.result)
            continue
        _suite(root, task, result)
    _count(root, *root)
    for element in root.iter():
        element.text = element.text and _NOT_XML.sub("", element.text)
        element.attrib.update({key: _NOT_XML.sub("", value) for key, value in element.attrib.items()})
    ET.indent(root)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def _suite(root: ET.Element, task: str, result: "Result[Any, Any]") -> None:
    """One result as a test suite named *task*, under *root*."""
    from dataeval_flow.steps._result import ChainResult

    suite = ET.SubElement(root, "testsuite", name=task)
    if isinstance(result, ChainResult) and (result.success or result.failed_steps):
        for step in result.failed_steps:
            errors = result.steps[step].errors or [f"{step} failed"]
            error = ET.SubElement(ET.SubElement(suite, "testcase", classname=task, name=f"step: {step}"), "error")
            error.set("message", errors[0])
            error.text = "\n".join(errors)
        _finding_cases(suite, task, result.findings)
        if not len(suite):
            ET.SubElement(suite, "testcase", classname=task, name="run")
    elif not result.success:  # a failed task, a chain refused before any step ran among them
        error = ET.SubElement(ET.SubElement(suite, "testcase", classname=task, name="run"), "error")
        error.set("message", result.errors[0] if result.errors else "failed")
        error.text = "\n".join(result.errors) or None
    else:
        ET.SubElement(suite, "testcase", classname=task, name="run")
    seconds = getattr(result.metadata, "execution_time_s", None)
    if isinstance(seconds, int | float):
        for timed in (suite, suite[0]):
            timed.set("time", f"{seconds:.3f}")
    _count(suite, suite)


def _finding_cases(suite: ET.Element, task: str, findings: "Sequence[Finding]") -> None:
    """One test case per finding, numbered where a title repeats, failing where the finding is a warning."""
    titles: Counter[str] = Counter()
    for finding in findings:
        titles[finding.title] += 1
        name = finding.title if titles[finding.title] == 1 else f"{finding.title} ({titles[finding.title]})"
        case = ET.SubElement(suite, "testcase", classname=task, name=name)
        if finding.severity == "warning":
            failure = ET.SubElement(case, "failure", message=finding.brief or finding.title, type="warning")
            failure.text = finding.description or None


def _count(element: ET.Element, *suites: ET.Element) -> None:
    """Set *element*'s ``tests``, ``failures`` and ``errors`` from the test cases in *suites*."""
    cases = [case for suite in suites for case in suite.iter("testcase")]
    element.set("tests", str(len(cases)))
    element.set("failures", str(sum(case.find("failure") is not None for case in cases)))
    element.set("errors", str(sum(case.find("error") is not None for case in cases)))


def markdown_summary(results: Mapping[str, "Result[Any, Any]"]) -> str:
    """Each task's health, or a preset chain's verdict in its place, its findings as a table of severity, finding and
    result, and each failed task's errors."""
    from dataeval_flow._matrix._result import MatrixResult
    from dataeval_flow.steps._result import ChainResult

    lines = ["# dataeval-flow results", ""]
    for task, result in results.items():
        if isinstance(result, MatrixResult):
            from dataeval_flow._matrix._table import comparison_table

            health = result.health
            verdict = (
                "failed"
                if health["status"] == "failed"
                else "ran"  # an evaluator has no findings, so nothing to pass
                if result.runs[0].result.kind == "evaluator"
                else "passed"
                if not health["warnings"]
                else f"{health['warnings']} warning{'s' if health['warnings'] != 1 else ''}"
            )
            table = comparison_table(result)
            lines += [f"## {_inline(task)}", "", f"**Health:** {verdict}", ""]
            lines.append("| " + " | ".join(_inline(column.header) for column in table.columns) + " |")
            lines.append("| " + " | ".join("---" for _ in table.columns) + " |")
            lines += [
                "| " + " | ".join(_inline(str(row[column.key])) for column in table.columns) + " |"
                for row in table.rows
            ]
            if result.errors:
                lines += ["", *_fenced(result.errors)]
            lines.append("")
            continue
        if isinstance(result, ChainResult):
            warnings = result.warning_count
            health = (
                "failed"
                if result.health["status"] == "failed"
                else "passed"
                if not warnings
                else f"{warnings} warning{'s' if warnings != 1 else ''}"
            )
            # A preset's verdict stands in for the health line, which "passed" would contradict where it has caveats.
            judged = f"**Verdict:** {_inline(result.verdict.line())}" if result.verdict else f"**Health:** {health}"
            lines += [f"## {_inline(task)}", "", judged, ""]
            if result.failed_steps:
                lines += [f"**Failed steps:** {', '.join(_inline(step) for step in result.failed_steps)}", ""]
            elif not result.success:  # refused before any step ran
                lines += [*_fenced(result.errors), ""]
            lines += ["| Severity | Finding | Result |", "| --- | --- | --- |"]
            lines += [f"| {f.severity} | {_inline(f.title)} | {_inline(f.brief or '')} |" for f in result.findings]
            lines.append("")
            continue
        if not result.success:
            lines += [f"## {_inline(task)}: failed", "", *_fenced(result.errors), ""]
            continue
        lines += [f"## {_inline(task)}", "", f"`{result.type}` ran; an evaluator has no findings to list.", ""]
    return "\n".join(lines)


def _fenced(errors: Sequence[str]) -> list[str]:
    """*errors* as a code block, fenced longer than any run of backticks they hold, so they show as written."""
    text = "\n".join(errors)
    fence = "`" * max([3, *(len(run) + 1 for run in re.findall("`+", text))])
    return [fence, text, fence]


def _inline(text: str) -> str:
    """*text* on one line, as a heading or a table cell shows it: Markdown's punctuation escaped."""
    return _MARKDOWN.sub(r"\\\1", " ".join(text.split()))
