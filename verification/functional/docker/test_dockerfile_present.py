"""TC-17-1 — Docker artifacts present and lint-clean."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = REPO_ROOT / "docker"


@pytest.mark.test_case("17-1")
class TestDockerfiles:
    @pytest.mark.parametrize("name", ["Dockerfile.cpu", "Dockerfile.cu118"])
    def test_dockerfile_exists(self, name: str) -> None:
        assert (DOCKER_DIR / name).is_file(), f"missing docker/{name}"

    @pytest.mark.parametrize("name", ["Dockerfile.cpu", "Dockerfile.cu118"])
    def test_dockerfile_starts_with_from(self, name: str) -> None:
        directive_lines = [
            line
            for line in (DOCKER_DIR / name).read_text().splitlines()
            if line.strip() and not line.startswith("#") and not line.upper().startswith("ARG ")
        ]
        assert directive_lines[0].upper().startswith("FROM"), f"docker/{name} does not start with FROM"

    @pytest.mark.parametrize("name", ["Dockerfile.cpu", "Dockerfile.cu118"])
    def test_dockerfile_has_nonroot_user(self, name: str) -> None:
        body = (DOCKER_DIR / name).read_text()
        user_directives = [
            line.strip().split(None, 1)[1]
            for line in body.splitlines()
            if (s := line.strip()) and not s.startswith("#") and s.upper().startswith("USER ")
        ]
        assert user_directives, f"docker/{name} declares no USER directive"
        last = user_directives[-1].strip().split()[0]
        assert last not in ("root", "0"), f"docker/{name} last USER directive is {last!r} (must be non-root)"
