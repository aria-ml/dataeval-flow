"""Unit tests for the meta repo publishing script (.gitlab/scripts/push_verification.py)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDefaultBranchGuard:
    @pytest.fixture
    def push(self, monkeypatch):
        pytest.importorskip("requests")
        monkeypatch.syspath_prepend(str(ROOT / ".gitlab" / "scripts"))
        for var in ("CI_COMMIT_BRANCH", "CI_DEFAULT_BRANCH", "CI_COMMIT_TAG"):
            monkeypatch.delenv(var, raising=False)
        return _load("push_verification", ROOT / ".gitlab" / "scripts" / "push_verification.py")

    def test_default_branch_is_refused(self, push, monkeypatch):
        monkeypatch.setenv("CI_COMMIT_BRANCH", "main")
        monkeypatch.setenv("CI_DEFAULT_BRANCH", "main")
        with pytest.raises(SystemExit, match="default branch"):
            push.refuse_default_branch()

    def test_main_is_refused_even_if_the_default_differs(self, push, monkeypatch):
        monkeypatch.setenv("CI_COMMIT_BRANCH", "main")
        monkeypatch.setenv("CI_DEFAULT_BRANCH", "develop")
        with pytest.raises(SystemExit):
            push.refuse_default_branch()

    def test_release_branch_and_tag_are_allowed(self, push, monkeypatch):
        monkeypatch.setenv("CI_DEFAULT_BRANCH", "main")
        monkeypatch.setenv("CI_COMMIT_BRANCH", "release/v1.1")
        push.refuse_default_branch()
        monkeypatch.delenv("CI_COMMIT_BRANCH")
        monkeypatch.setenv("CI_COMMIT_TAG", "v1.1.5")
        push.refuse_default_branch()
