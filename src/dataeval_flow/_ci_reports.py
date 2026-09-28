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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataeval_flow._result import Result

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
    from dataeval_flow.workflows._result import WorkflowResult

    root = ET.Element("testsuites", name="dataeval-flow")
    for task, result in results.items():
        suite = ET.SubElement(root, "testsuite", name=task)
        if not result.success:
            error = ET.SubElement(ET.SubElement(suite, "testcase", classname=task, name="run"), "error")
            error.set("message", result.errors[0] if result.errors else "failed")
            error.text = "\n".join(result.errors) or None
        elif isinstance(result, WorkflowResult) and result.findings:
            titles: Counter[str] = Counter()
            for finding in result.findings:
                titles[finding.title] += 1
                name = finding.title if titles[finding.title] == 1 else f"{finding.title} ({titles[finding.title]})"
                case = ET.SubElement(suite, "testcase", classname=task, name=name)
                if finding.severity == "warning":
                    failure = ET.SubElement(case, "failure", message=finding.brief or finding.title, type="warning")
                    failure.text = finding.description or None
        else:
            ET.SubElement(suite, "testcase", classname=task, name="run")
        seconds = getattr(result.metadata, "execution_time_s", None)
        if isinstance(seconds, int | float):
            for timed in (suite, suite[0]):
                timed.set("time", f"{seconds:.3f}")
        _count(suite, suite)
    _count(root, *root)
    for element in root.iter():
        element.text = element.text and _NOT_XML.sub("", element.text)
        element.attrib.update({key: _NOT_XML.sub("", value) for key, value in element.attrib.items()})
    ET.indent(root)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def _count(element: ET.Element, *suites: ET.Element) -> None:
    """Set *element*'s ``tests``, ``failures`` and ``errors`` from the test cases in *suites*."""
    cases = [case for suite in suites for case in suite.iter("testcase")]
    element.set("tests", str(len(cases)))
    element.set("failures", str(sum(case.find("failure") is not None for case in cases)))
    element.set("errors", str(sum(case.find("error") is not None for case in cases)))


def markdown_summary(results: Mapping[str, "Result[Any, Any]"]) -> str:
    """Each task's findings as a table of severity, finding and result, and each failed task's errors."""
    from dataeval_flow.workflows._result import WorkflowResult

    lines = ["# dataeval-flow results", ""]
    for task, result in results.items():
        if not result.success:
            errors = "\n".join(result.errors)
            # A fence longer than any run of backticks the errors hold, so they show as written.
            fence = "`" * max([3, *(len(run) + 1 for run in re.findall("`+", errors))])
            lines += [f"## {_inline(task)}: failed", "", fence, errors, fence, ""]
            continue
        lines += [f"## {_inline(task)}", ""]
        if isinstance(result, WorkflowResult):
            warnings = result.warning_count
            health = "passed" if not warnings else f"{warnings} warning{'s' if warnings != 1 else ''}"
            lines += [f"**Health:** {health}", "", "| Severity | Finding | Result |", "| --- | --- | --- |"]
            lines += [f"| {f.severity} | {_inline(f.title)} | {_inline(f.brief or '')} |" for f in result.findings]
        else:
            lines.append(f"`{result.type}` ran; an evaluator has no findings to list.")
        lines.append("")
    return "\n".join(lines)


def _inline(text: str) -> str:
    """*text* on one line, as a heading or a table cell shows it: Markdown's punctuation escaped."""
    return _MARKDOWN.sub(r"\\\1", " ".join(text.split()))
