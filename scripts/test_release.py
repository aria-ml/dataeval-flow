#!/usr/bin/env python3
"""
Self-check for release's version arithmetic, categorization and changelog handling.

Run directly: `python scripts/test_release.py`. Deliberately not a pytest suite -
pyproject restricts testpaths to `tests`, and this needs to run anywhere git does.
"""

import contextlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import release
from release import TAG_CATEGORIES, is_version_tag, next_version, parse_version


def test_version_ladder() -> None:
    cases = {
        # (current, bump, prerelease kind): expected
        ("v1.1.1", "patch", None): "v1.1.2",
        ("v1.1.0", "minor", None): "v1.2.0",
        ("v1.9.3", "major", None): "v2.0.0",
        ("v9.0.0", "major", None): "v10.0.0",  # multi-digit majors keep counting
        ("v1.1.0", "minor", "rc"): "v1.2.0-rc0",
        ("v1.2.0-rc0", "minor", "rc"): "v1.2.0-rc1",
        ("v1.2.0-a3", "minor", "a"): "v1.2.0-a4",
        ("v1.2.0-a3", "minor", "rc"): "v1.2.0-rc0",  # promotion restarts the sequence
        ("v1.2.0-rc2", "minor", None): "v1.2.0",  # finalizing drops the suffix
        ("v1.2.0-rc2", "patch", None): "v1.2.0",  # ...and does not also bump
        ("v1.1.0", "major", "rc"): "v2.0.0-rc0",  # a prerelease carries the major bump with it
    }
    for (current, bump, kind), expected in cases.items():
        actual = next_version(current, bump, kind)
        assert actual == expected, f"{current} +{bump} kind={kind}: expected {expected}, got {actual}"


def test_prerelease_sorts_before_its_release() -> None:
    order = ["v1.1.0", "v1.2.0-a0", "v1.2.0-a1", "v1.2.0-rc0", "v1.2.0", "v1.2.1"]
    keys = [parse_version(tag) for tag in order]
    assert keys == sorted(keys), f"tags sort out of order: {order}"
    assert is_version_tag("not-a-tag") is False
    assert is_version_tag("v1.2.0-beta0") is False  # unknown kinds are not version tags
    try:
        parse_version("not-a-tag")
    except ValueError:
        pass
    else:
        raise AssertionError("parse_version should raise on a non-version tag")


def test_tag_vocabulary() -> None:
    assert TAG_CATEGORIES["fix"] == "fix"
    assert TAG_CATEGORIES["feat"] == "feature"
    assert TAG_CATEGORIES["depr"] == "deprecation"
    assert TAG_CATEGORIES["perf"] == "improvement"
    # housekeeping tags are valid but never move the public API
    for tag in ("docs", "test", "deps", "type", "devops", "devsecops", "lint", "misc"):
        assert TAG_CATEGORIES[tag] == "misc", tag
    assert "imrp" not in TAG_CATEGORIES  # the typo stays invalid on purpose


def test_the_changelog_rewrites_unchanged() -> None:
    # every section is rewritten on release, so parsing must lose nothing, the real file included
    text = (Path(__file__).resolve().parent.parent / "CHANGELOG.md").read_text()
    preamble, sections = release.changelog_sections(text, "CHANGELOG.md")
    assert f"{preamble}\n\n" + "\n\n".join(sections.values()) + "\n" == text


def test_prereleases_fold_by_subsection() -> None:
    sections = {
        "Unreleased": "## Unreleased\n\n### Fixed\n\n- c",
        "v1.2.0-rc1": "## v1.2.0-rc1\n\n### Added\n\n- b",
        "v1.2.0-rc0": "## v1.2.0-rc0\n\nLead.\n\n### Added\n\n- a\n  - a detail\n\n### Fixed\n\n- f",
    }
    folded = release.release_section("v1.2.0", sections)
    assert folded == "## v1.2.0\n\nLead.\n\n### Added\n\n- a\n  - a detail\n- b\n\n### Fixed\n\n- f\n- c", folded
    # a prerelease takes only the prose written since the tag before it
    assert release.release_section("v1.2.0-rc2", sections) == "## v1.2.0-rc2\n\n### Fixed\n\n- c"


def test_release_folds_prereleases_and_pulls_in_patches() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        changelog = repo / "CHANGELOG.md"

        def run(*args: str) -> None:
            subprocess.run(["git", "-C", tmp, *args], check=True, capture_output=True)

        def cut(subject: str, notes: str, *argv: str) -> None:
            text = changelog.read_text()
            changelog.write_text(text.replace("## v", f"## Unreleased\n\n{notes}\n\n## v", 1))
            run("commit", "-q", "-am", subject)
            sys.argv = ["release.py", *argv]
            with contextlib.redirect_stdout(io.StringIO()):
                release.main()

        run("init", "-q", "-b", "main")
        config = {"user.name": "t", "user.email": "t@t", "commit.gpgsign": "false", "tag.gpgsign": "false"}
        for key, value in config.items():
            run("config", key, value)
        changelog.write_text("# Changelog\n\n## v1.1.0\n\n### Fixed\n\n- Old\n")
        run("add", ".")
        run("commit", "-q", "-m", "Release v1.1.0")
        run("tag", "-a", "v1.1.0", "-m", "v1.1.0")
        release.REPO, release.CHANGELOG_FILE = repo, changelog

        run("switch", "-q", "-c", "release/v1.1")
        cut("[fix] Backport", "### Fixed\n\n- Backport")  # v1.1.1, whose section only release/v1.1 has
        run("switch", "-q", "main")
        cut("[fix] Repair", "### Fixed\n\n- Repair", "prerelease")
        cut("[feat] Add", "### Added\n\n- Add", "prerelease")
        assert "## v1.2.0-rc0" in changelog.read_text()  # a candidate keeps the ones before it
        cut("[misc] Tidy", "### Changed\n\n- Tidy")
        result = changelog.read_text()

    assert "-rc" not in result, result
    assert "Unreleased" not in result, result
    # one section, everything since v1.1.0, each subsection once in the order first written
    assert result.startswith("# Changelog\n\n## v1.2.0\n\n### Fixed\n\n- Repair\n\n### Added\n\n- Add\n"), result
    assert "### Changed\n\n- Tidy\n\n## v1.1.1\n" in result, result
    # the patch's section comes from its tag and sits in version order, below the release
    assert "## v1.1.1\n\n### Fixed\n\n- Backport\n\n## v1.1.0\n\n### Fixed\n\n- Old\n" in result, result
    assert result.count("Backport") == 1, result


def test_changelog_with_an_unknown_heading_is_refused() -> None:
    # rewriting the changelog section by section would otherwise drop or merge these lines
    for text in ("# Log\n\n## Next\n\n- x\n", "# Log\n\n## v1.1.0\n\n- x\n\n## v1.1.0\n\n- y\n"):
        try:
            release.changelog_sections(text, "CHANGELOG.md")
        except SystemExit:
            pass
        else:
            raise AssertionError(f"should refuse {text!r}")


def test_a_release_needs_notes() -> None:
    for sections in ({}, {"Unreleased": "## Unreleased\n"}):
        try:
            release.release_section("v1.2.0", sections)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"should refuse to release {sections!r} without notes")


def test_kind_pairs_only_with_prerelease() -> None:
    # a bare prerelease defaults to rc; an explicit kind is kept
    assert release.resolve_kind("prerelease", None) == "rc"
    assert release.resolve_kind("prerelease", "a") == "a"
    # a kind with a non-prerelease bump used to be silently dropped; now it fails loudly
    for kind in ("a", "rc"):
        try:
            release.resolve_kind("major", kind)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"major + {kind} should fail loudly")
    # a non-prerelease bump without a kind is left alone
    assert release.resolve_kind("major", None) is None
    assert release.resolve_kind(None, None) is None


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok - {name}")
