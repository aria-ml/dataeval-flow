"""Unit tests for the verification report's test-case status roll-up.

The roll-up is what keeps a documented gap visible in the exported VCRM, so the
precedence between a passing sibling test and an xfailed one is worth pinning.
"""

from verification.conftest import _tc_status


def _tests(*statuses: str) -> list[dict]:
    return [{"test": f"t{i}", "file": "f.py", "status": s} for i, s in enumerate(statuses)]


def test_an_xfail_keeps_its_passing_siblings_off_a_clean_pass() -> None:
    assert _tc_status(_tests("passed", "passed", "xfailed")) == "xfailed"


def test_a_real_failure_outranks_an_xfail() -> None:
    assert _tc_status(_tests("passed", "xfailed", "failed")) == "failed"


def test_all_passing_stays_passed() -> None:
    assert _tc_status(_tests("passed", "passed")) == "passed"


def test_all_skipped_stays_skipped() -> None:
    assert _tc_status(_tests("skipped")) == "skipped"
