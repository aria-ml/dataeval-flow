"""Tests for docker/promote-tags.sh — which floating tags a pipeline should apply.

The script is the release-line safety net: it decides whether a build gets the bare
``:cpu`` pointer or only its own series pointer.  Getting it wrong moves ``:cpu``
backwards onto an older patch, which is exactly what release branches make possible,
so the decision is tested here rather than trusted to a shell one-liner in CI.

Tags are fed on stdin rather than read from git, so these run anywhere.
"""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "docker" / "promote-tags.sh"

# A repo mid-way through the v0.3 line, with the v0.2 line still supported.
TAGS = "v0.1.0\nv0.1.1\nv0.1.2\nv0.2.0\nv0.2.1\nv0.2.2\nv0.3.0\n"


def promote(variant: str, tag: str | None, tags: str = TAGS, branch: str = "main") -> list[str]:
    """Run the script and return the floating tags it wants applied."""
    result = subprocess.run(  # noqa: S603 - fixed path to a repo script, args are test literals
        [str(SCRIPT), variant],
        input=tags,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "CI_COMMIT_TAG": tag or "", "CI_COMMIT_BRANCH": branch},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


@pytest.mark.required
def test_branch_pipeline_gets_edge_only() -> None:
    """No tag means a main-branch build: edge pointer, never a release pointer."""
    assert promote("cpu", None) == ["edge-cpu"]


@pytest.mark.required
def test_release_branch_build_claims_no_pointer() -> None:
    """``edge`` means "newest from main". An untagged v0.2 build must not seize it.

    Release branches build containers so an image can be scanned before its tag is
    pushed — that validation needs the immutable tag, not a floating one.
    """
    assert promote("cpu", None, branch="release/v0.2") == []


@pytest.mark.required
def test_release_branch_still_promotes_its_tags() -> None:
    """Suppressing ``edge`` must not suppress the tag pipeline that follows it.

    The series pointer alone, because v0.3.0 already holds the bare one — which is
    the ordinary case for a patch cut on a line main has moved past.
    """
    assert promote("cpu", "v0.2.3", branch="release/v0.2") == ["0.2-cpu"]


@pytest.mark.required
def test_patch_on_older_line_does_not_take_bare_tag() -> None:
    """The whole point: v0.2.3 cut after v0.3.0 must not steal ``:cpu``."""
    assert promote("cpu", "v0.2.3") == ["0.2-cpu"]


@pytest.mark.required
def test_highest_release_takes_both_pointers() -> None:
    """A release on the newest line moves ``:cpu`` as well as its series pointer."""
    assert promote("cpu", "v0.3.1") == ["0.3-cpu", "cpu"]


@pytest.mark.required
def test_version_ordering_is_numeric_not_lexical() -> None:
    """v0.10.0 outranks v0.2.0 — a plain string sort gets this backwards."""
    tags = "v0.2.0\nv0.10.0\n"
    assert promote("cpu", "v0.10.0", tags) == ["0.10-cpu", "cpu"]
    assert promote("cpu", "v0.2.1", tags) == ["0.2-cpu"]


@pytest.mark.required
def test_prerelease_gets_no_floating_tags() -> None:
    """An rc is reachable by its immutable tag only; nothing floating points at it."""
    assert promote("cpu", "v0.4.0-rc0", TAGS + "v0.4.0-rc0\n") == []


@pytest.mark.required
def test_pending_prerelease_does_not_block_a_stable_release() -> None:
    """v0.3.1 is still the highest *stable* tag even with a v0.4.0-rc0 in flight."""
    assert promote("cpu", "v0.3.1", TAGS + "v0.4.0-rc0\n") == ["0.3-cpu", "cpu"]


@pytest.mark.required
def test_first_release_takes_both_pointers() -> None:
    """A repo with one tag: that tag is the highest, so it gets everything."""
    assert promote("cpu", "v0.1.0", "v0.1.0\n") == ["0.1-cpu", "cpu"]


@pytest.mark.required
def test_variant_is_carried_through() -> None:
    """Every pointer is variant-scoped; cu130 must never be handed a cpu tag."""
    assert promote("cu130", "v0.3.1") == ["0.3-cu130", "cu130"]
    assert promote("cu126", None) == ["edge-cu126"]
