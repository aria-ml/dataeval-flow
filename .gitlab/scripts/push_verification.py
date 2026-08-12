#!/usr/bin/env python3
"""Push generated verification artifacts to the JATIC meta repo.

Commits the output of ``verification/generate_metarepo.py`` to the meta repo
via the GitLab API. Target project and directory come from
``verification/registry.yaml``:

    metarepo:
      project_id: 409
      path: DataEval-Flow

Only files this project generates are ever written:

    output/metarepo/vcrm.md            -> <path>/vcrm.md
    output/metarepo/test-cases/*.md    -> <path>/test-cases/*.md

The meta repo also holds hand-maintained content under the same directory (the
``FR-*.md`` / ``NFR-*.md`` requirement docs the VCRM links to). Nothing is ever
deleted by default: files that exist remotely but are no longer generated are
reported as stale and left alone. ``--prune`` opts into deleting stale
``test-cases/test-case-*.md`` files only, and never touches anything else.

Requires:
  - DATAEVAL_BUILD_PAT environment variable (GitLab personal access token)
  - Generated artifacts from verification/generate_metarepo.py

Usage:
  python3 .gitlab/scripts/push_verification.py [--dry-run] [--prune]
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import yaml
from requests import get, post
from rest import RestError, RestWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "verification" / "registry.yaml"
OUTPUT_DIR = PROJECT_ROOT / "output" / "metarepo"

# Generated test cases are named test-case-<major>-<minor>.md. Only files
# matching this pattern are eligible for --prune; anything else under the
# managed directory is assumed hand-maintained.
TEST_CASE_RE = re.compile(r"^test-case-[\w.-]+\.md$")


def load_metarepo_config() -> tuple[int, str]:
    """Return the meta repo project id and the directory this project owns."""
    with open(REGISTRY_PATH) as f:
        registry = yaml.safe_load(f)
    metarepo = registry["metarepo"]
    path = str(metarepo.get("path", "")).strip("/")
    if not path:
        raise SystemExit(f"error: metarepo.path is not set in {REGISTRY_PATH}")
    return int(metarepo["project_id"]), path


class MetaRepo(RestWrapper):
    """GitLab REST client scoped to the meta repo project."""

    def __init__(self, project_id: int) -> None:
        """Authenticate against the meta repo project using DATAEVAL_BUILD_PAT."""
        project_url = f"https://gitlab.jatic.net/api/v4/projects/{project_id}/"
        super().__init__(project_url, "DATAEVAL_BUILD_PAT", verbose=True)
        self.headers = {"PRIVATE-TOKEN": self.token}

    def list_tree(self, path: str = "", ref: str = "main") -> list[dict]:
        """List blobs under ``path``, recursively. Returns [] if it doesn't exist yet."""
        try:
            return self._request(
                get,
                "repository/tree",
                {"path": path, "ref": ref, "recursive": "true", "per_page": "100"},
            )
        except RestError as e:
            if e.status_code == 404:
                return []  # directory not created yet — every file is a create
            raise

    def commit(self, branch: str, message: str, actions: list[dict]) -> dict:
        """Apply ``actions`` to ``branch`` as a single commit."""
        return self._request(
            post,
            "repository/commits",
            None,
            {"branch": branch, "commit_message": message, "actions": actions},
        )


def collect_generated(base: str) -> dict[str, Path]:
    """Map meta repo file path -> local source file for everything CI generates."""
    generated: dict[str, Path] = {}

    tc_dir = OUTPUT_DIR / "test-cases"
    if tc_dir.exists():
        for f in sorted(tc_dir.glob("*.md")):
            generated[f"{base}/test-cases/{f.name}"] = f

    vcrm = OUTPUT_DIR / "vcrm.md"
    if vcrm.exists():
        generated[f"{base}/vcrm.md"] = vcrm

    return generated


def main() -> None:
    """Generate the commit plan and, unless ``--dry-run``, push it to the meta repo."""
    dry_run = "--dry-run" in sys.argv
    prune = "--prune" in sys.argv

    project_id, base = load_metarepo_config()
    generated = collect_generated(base)
    if not generated:
        print(f"No generated artifacts under {OUTPUT_DIR} — run `nox -s verify` first.")
        return

    repo = MetaRepo(project_id)
    existing = {item["path"] for item in repo.list_tree(base) if item.get("type") == "blob"}
    print(f"Meta repo project {project_id}, managing {base}/ ({len(existing)} existing file(s))")

    actions = [
        {
            "action": "update" if remote in existing else "create",
            "file_path": remote,
            "content": local.read_text(),
        }
        for remote, local in generated.items()
    ]

    # Anything present remotely that CI does not generate. Hand-written
    # requirement docs live here, so this is reported, not deleted.
    untracked = sorted(existing - set(generated))
    tc_prefix = f"{base}/test-cases/"
    stale_test_cases = [p for p in untracked if p.startswith(tc_prefix) and TEST_CASE_RE.match(Path(p).name)]
    preserved = [p for p in untracked if p not in stale_test_cases]

    version = os.environ.get("CI_COMMIT_TAG") or os.environ.get("DATAEVAL_FLOW_VERSION") or "dev"
    message = f"Update verification artifacts for dataeval-flow {version}"

    print(f"Commit message: {message}")
    print(f"Pushing {len(actions)} file(s):")
    for a in sorted(actions, key=lambda a: a["file_path"]):
        print(f"  {a['action']}: {a['file_path']}")

    if preserved:
        print(f"\nLeaving {len(preserved)} hand-maintained file(s) untouched:")
        for p in preserved:
            print(f"  keep: {p}")

    if stale_test_cases:
        verb = "Deleting" if prune else "Stale (use --prune to delete)"
        print(f"\n{verb}: {len(stale_test_cases)} test case file(s) no longer in registry.yaml:")
        for p in stale_test_cases:
            print(f"  {'delete' if prune else 'stale'}: {p}")
        if prune:
            actions += [{"action": "delete", "file_path": p} for p in stale_test_cases]

    if dry_run:
        print("\n--dry-run: skipping commit")
        return

    result = repo.commit("main", message, actions)
    print(f"\nCommitted: {result.get('id', 'unknown')}")


if __name__ == "__main__":
    main()
