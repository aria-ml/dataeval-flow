"""TC-16-1 — the Dockerfiles exist, come from one template, install the right extras and run as a non-root user."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from verification.helpers import REPO_ROOT

pytestmark = pytest.mark.required

DOCKER_DIR = REPO_ROOT / "docker"
VARIANTS = yaml.safe_load((DOCKER_DIR / "variants.yaml").read_text())["variants"]
NAMES = ["cpu", "cu126", "cu130"]


def _instructions(name: str) -> list[str]:
    """The Dockerfile's instructions, comments dropped and ``\\`` continuations joined."""
    text = (DOCKER_DIR / f"Dockerfile.{name}").read_text()
    lines = [line for line in text.replace("\\\n", " ").splitlines() if line.strip() and not line.startswith("#")]
    return [" ".join(line.split()) for line in lines]


def _stage(name: str, stage: str) -> list[str]:
    """The instructions of one build stage, from its ``FROM ... AS <stage>`` line to the next ``FROM``."""
    lines = _instructions(name)
    start = next(i for i, line in enumerate(lines) if re.match(rf"FROM \S+ AS {stage}$", line, re.IGNORECASE))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].upper().startswith("FROM ")), len(lines))
    return lines[start:end]


class TestDockerfiles:
    def test_there_is_one_variant_for_each_image_the_project_publishes(self) -> None:
        assert sorted(VARIANTS) == sorted(NAMES)

    @pytest.mark.parametrize("name", NAMES)
    def test_dockerfile_exists(self, name: str) -> None:
        assert (DOCKER_DIR / f"Dockerfile.{name}").is_file(), f"missing docker/Dockerfile.{name}"

    @pytest.mark.parametrize("name", NAMES)
    def test_dockerfile_starts_with_from(self, name: str) -> None:
        directives = [line for line in _instructions(name) if not line.upper().startswith("ARG ")]
        assert directives[0].upper().startswith("FROM "), f"docker/Dockerfile.{name} does not start with FROM"

    @pytest.mark.parametrize("name", NAMES)
    def test_dockerfile_has_nonroot_user(self, name: str) -> None:
        users = [line.split()[1] for line in _instructions(name) if line.upper().startswith("USER ")]
        assert users, f"docker/Dockerfile.{name} declares no USER directive"
        assert users[-1] not in ("root", "0"), f"docker/Dockerfile.{name} last USER directive is {users[-1]!r}"

    @pytest.mark.parametrize("name", NAMES)
    def test_the_published_stage_runs_as_the_non_root_user(self, name: str) -> None:
        """The stage CI publishes (``scanned``) adds files to ``prod``; neither may end as root."""
        prod = _stage(name, "prod")
        users = [line.split()[1] for line in prod if line.upper().startswith("USER ")]
        assert users[-1] == "dataeval"
        assert any(" useradd " in line for line in prod)
        scanned = _stage(name, "scanned")
        assert not [line for line in scanned if line.upper().startswith("USER ")]

    @pytest.mark.parametrize("name", NAMES)
    def test_the_image_starts_the_entrypoint_script_and_exposes_the_service_port(self, name: str) -> None:
        prod = _stage(name, "prod")

        assert 'ENTRYPOINT ["./entrypoint.sh"]' in prod
        assert 'CMD ["python", "-m", "dataeval_flow"]' in prod
        assert "EXPOSE 8001" in prod
        assert "ENV DATAEVAL_DATA=/dataeval" in prod
        assert "ENV DATAEVAL_OUTPUT=/output" in prod
        assert "ENV DATAEVAL_SERVICE_HOST=0.0.0.0" in prod

    @pytest.mark.parametrize("name", NAMES)
    def test_the_image_carries_no_secrets(self, name: str) -> None:
        names = [
            line.split()[1].split("=")[0] for line in _instructions(name) if line.upper().startswith(("ENV ", "ARG "))
        ]

        assert names
        assert not [n for n in names if re.search(r"SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY|CREDENTIAL", n, re.I)]


class TestGeneratedFromOneTemplate:
    @pytest.mark.parametrize("name", NAMES)
    def test_dockerfile_says_it_is_generated_from_the_template(self, name: str) -> None:
        head = (DOCKER_DIR / f"Dockerfile.{name}").read_text().splitlines()[:3]

        assert head[0] == "# AUTO-GENERATED from docker/Dockerfile.j2 — do not edit directly."
        assert head[2] == f"# Variant: {name}"

    def test_rendering_the_template_again_gives_the_committed_dockerfiles(self, tmp_path: Path) -> None:
        pytest.importorskip("jinja2", reason="the docker dependency group (jinja2) is not installed")
        scratch = tmp_path / "docker"
        scratch.mkdir()
        for source in ("generate.py", "Dockerfile.j2", "variants.yaml"):
            shutil.copy(DOCKER_DIR / source, scratch / source)

        run = subprocess.run(  # noqa: S603
            [sys.executable, str(scratch / "generate.py")], capture_output=True, text=True, check=False, cwd=tmp_path
        )

        assert run.returncode == 0, run.stderr
        for name in NAMES:
            assert (scratch / f"Dockerfile.{name}").read_text() == (DOCKER_DIR / f"Dockerfile.{name}").read_text(), (
                f"docker/Dockerfile.{name} is out of date: run `nox -s docker_gen`"
            )

    @pytest.mark.parametrize("name", NAMES)
    def test_each_variant_installs_the_extras_the_variants_file_lists(self, name: str) -> None:
        wanted = VARIANTS[name]["extras"]
        syncs = [line for line in _stage(name, "build") if line.startswith("RUN") and "uv sync" in line]

        assert syncs
        for line in syncs:
            assert re.findall(r"--extra (\S+)", line) == wanted

    @pytest.mark.parametrize("name", NAMES)
    def test_each_variant_is_built_on_the_base_image_the_variants_file_names(self, name: str) -> None:
        from_lines = [line.split() for line in _instructions(name) if line.upper().startswith("FROM ")]
        stages = [parts[3].lower() for parts in from_lines]
        external = [parts[1] for parts in from_lines if parts[1].lower() not in stages]

        assert stages == ["build", "test", "prod", "scanned"]
        assert external == [VARIANTS[name]["base_image"]] * 2
