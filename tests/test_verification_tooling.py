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


class TestCollectCiJobs:
    @pytest.fixture
    def collect(self):
        return _load("collect_ci_jobs", ROOT / ".gitlab" / "scripts" / "collect_ci_jobs.py")

    def test_matrix_legs_fold_into_one_job_that_fails_if_any_leg_fails(self, collect):
        jobs = [{"name": f"test: [{v}]", "status": s} for v, s in (("3.11", "success"), ("3.12", "failed"))]
        assert collect.aggregate(jobs) == {"test": "failed"}

    def test_a_job_succeeds_only_when_every_leg_does(self, collect):
        jobs = [{"name": f"verify: [{v}]", "status": "success"} for v in ("3.11", "3.12")]
        assert collect.aggregate(jobs) == {"verify": "success"}

    def test_unfinished_jobs_are_left_out_so_their_evidence_stays_pending(self, collect):
        jobs = [{"name": "lint", "status": "running"}, {"name": "verify lock", "status": "skipped"}]
        assert collect.aggregate(jobs) == {"verify lock": "skipped"}

    def test_names_that_contain_a_colon_keep_it(self, collect):
        assert collect.job_key("release:check-pypi") == "release:check-pypi"
        assert collect.job_key("push:docker: [cpu]") == "push:docker"
