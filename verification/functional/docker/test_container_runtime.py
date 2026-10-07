"""TC-16-1 and TC-27-1 — behavior of the built images.

These run against images that already exist; they never build one. Set ``DATAEVAL_FLOW_TEST_IMAGE`` to the CPU
image (a registry reference for the signature check) and ``DATAEVAL_FLOW_TEST_IMAGE_CUDA`` to a CUDA image.
Each test skips, saying why, when its image variable is unset or Docker (or cosign) is unavailable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from verification.fixtures import write_cli_project

pytestmark = pytest.mark.required

COSIGN_KEY = Path(__file__).resolve().parents[3] / "docker" / "cosign.pub"
SECURITY_DIR = "/usr/share/dataeval-flow/security/"


def _docker(*args: str, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["docker", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _require_image(variable: str) -> str:
    """Return the image named by *variable*, or skip the test if it cannot be used."""
    image = os.environ.get(variable)
    if not image:
        pytest.skip(f"{variable} is not set; no built image to test")
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")
    if _docker("info", timeout=60).returncode != 0:
        pytest.skip("the docker daemon is not reachable")
    if _docker("image", "inspect", image, timeout=60).returncode != 0:
        pytest.skip(f"image {image} is not available locally (pull or build it first)")
    return image


@pytest.fixture
def cpu_image() -> str:
    return _require_image("DATAEVAL_FLOW_TEST_IMAGE")


@pytest.fixture
def cuda_image() -> str:
    return _require_image("DATAEVAL_FLOW_TEST_IMAGE_CUDA")


@pytest.fixture
def mounts(tmp_path: Path) -> tuple[Path, Path]:
    """A data directory holding a config and images (read-only to the container) and a writable output directory."""
    data = tmp_path / "data"
    data.mkdir()
    write_cli_project(data)
    out = tmp_path / "out"
    out.mkdir()
    out.chmod(0o777)  # the image runs as its own non-root user, not as the test's user
    return data, out


def _mount(source: Path, target: str, *, readonly: bool = False) -> list[str]:
    return ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if readonly else "")]


class TestCpuImage:
    def test_pipeline_runs_against_mounted_directories_as_non_root(
        self, cpu_image: str, mounts: tuple[Path, Path]
    ) -> None:
        data, out = mounts

        user = _docker("run", "--rm", "--entrypoint", "id", cpu_image, "-u")
        assert user.returncode == 0, user.stderr
        assert user.stdout.strip() != "0"

        run = _docker("run", "--rm", *_mount(data, "/dataeval", readonly=True), *_mount(out, "/output"), cpu_image)

        assert run.returncode == 0, run.stdout + run.stderr
        results = json.loads((out / "results" / "result.json").read_text())
        assert set(results) == {"clean_task"}
        assert (out / "results" / "result.json").stat().st_uid != 0  # written by the non-root user

    @pytest.mark.parametrize("args", [[], ["--help"]], ids=["no-data-mounted", "help-flag"])
    def test_prints_usage_when_nothing_is_mounted_or_help_is_requested(self, cpu_image: str, args: list[str]) -> None:
        run = _docker("run", "--rm", cpu_image, *args)

        assert run.returncode == 0, run.stdout + run.stderr
        for section in ("USAGE:", "VOLUME MOUNTS", "ENVIRONMENT VARIABLES", "COMMAND-LINE OPTIONS", "No secret mounts"):
            assert section in run.stdout
        for variable in ("DATAEVAL_DATA", "DATAEVAL_OUTPUT", "DATAEVAL_CACHE", "DATAEVAL_MAX_PROCESSES"):
            assert variable in run.stdout

    def test_falls_back_to_in_memory_cache_when_cache_is_not_mounted(
        self, cpu_image: str, mounts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        data, out = mounts
        command = [cpu_image, "python", "-m", "dataeval_flow", "-vv"]

        unmounted = _docker("run", "--rm", *_mount(data, "/dataeval", readonly=True), *_mount(out, "/output"), *command)
        assert unmounted.returncode == 0, unmounted.stdout + unmounted.stderr
        assert (out / "results" / "result.json").is_file()
        assert "Cache enabled" not in unmounted.stdout

        # Control: a mounted cache is picked up, so the line above says something.
        cache = tmp_path / "cache"
        cache.mkdir()
        cache.chmod(0o777)
        mounted = _docker(
            "run",
            "--rm",
            *_mount(data, "/dataeval", readonly=True),
            *_mount(out, "/output"),
            *_mount(cache, "/cache"),
            *command,
        )
        assert mounted.returncode == 0, mounted.stdout + mounted.stderr
        assert "Cache enabled: /cache" in mounted.stdout
        assert any(cache.rglob("*.json"))


class TestCudaImage:
    def test_exits_with_one_and_names_the_gpu_flag_and_cpu_variant_without_gpu_access(
        self, cuda_image: str, mounts: tuple[Path, Path]
    ) -> None:
        data, out = mounts

        run = _docker("run", "--rm", *_mount(data, "/dataeval", readonly=True), *_mount(out, "/output"), cuda_image)

        assert run.returncode == 1, run.stdout + run.stderr
        assert "--gpus all" in run.stdout
        assert "dataeval-flow:cpu" in run.stdout


class TestPublishedImageSecurity:
    def test_signature_verifies_against_the_committed_public_key(self) -> None:
        image = os.environ.get("DATAEVAL_FLOW_TEST_IMAGE")
        if not image:
            pytest.skip("DATAEVAL_FLOW_TEST_IMAGE is not set; no published image to verify")
        if shutil.which("cosign") is None:
            pytest.skip("cosign is not installed")

        run = subprocess.run(  # noqa: S603
            ["cosign", "verify", "--key", str(COSIGN_KEY), image],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        assert run.returncode == 0, run.stderr

    def test_scan_reports_and_sbom_are_inside_the_image(self) -> None:
        image = _require_image("DATAEVAL_FLOW_TEST_IMAGE")

        listing = _docker("run", "--rm", "--entrypoint", "ls", image, SECURITY_DIR)

        assert listing.returncode == 0, listing.stderr
        files = set(listing.stdout.split())
        assert "sbom.cdx.json" in files
        assert {"vulnerability-scan.json", "vulnerability-scan.txt"} <= files

        sbom = _docker("run", "--rm", "--entrypoint", "cat", image, SECURITY_DIR + "sbom.cdx.json")
        assert json.loads(sbom.stdout)["bomFormat"] == "CycloneDX"
