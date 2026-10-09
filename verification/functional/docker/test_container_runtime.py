"""TC-16-1, TC-16-2 and TC-37-1 — behavior of the built images.

These run against images that already exist; they never build one. Set ``DATAEVAL_FLOW_TEST_IMAGE`` to the CPU
image and ``DATAEVAL_FLOW_TEST_IMAGE_CUDA`` to a CUDA image; set ``DATAEVAL_FLOW_TEST_IMAGE_PUBLISHED`` to a
published registry reference (the ``scanned`` build, which carries the scan reports) for the security checks.
Each test skips, saying why, when its image variable is unset or Docker (or cosign) is unavailable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from verification.helpers import REPO_ROOT
from verification.nonfunctional._support import write_project

pytestmark = pytest.mark.optional

COSIGN_KEY = REPO_ROOT / "docker" / "cosign.pub"
SECURITY_DIR = "/usr/share/dataeval-flow/security/"
SERVICE_PORT = "8001/tcp"


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
def mounts(tmp_path: Path, cpu_image: str) -> Iterator[tuple[Path, Path]]:
    """A data directory holding a config and images (read-only to the container) and a writable output directory."""
    data = tmp_path / "data"
    data.mkdir()
    write_project(data, n_tasks=2)
    out = tmp_path / "out"
    out.mkdir()
    out.chmod(0o777)  # the image runs as its own non-root user, not as the test's user
    yield data, out
    _hand_back(cpu_image, tmp_path)


def _hand_back(image: str, directory: Path) -> None:
    """Give files the container's user wrote back to the test's user, so pytest can remove them."""
    _docker(
        "run",
        "--rm",
        "--user",
        "0",
        "--entrypoint",
        "chown",
        *_mount(directory, "/hand-back"),
        image,
        "-R",
        f"{os.getuid()}:{os.getgid()}",
        "/hand-back",
    )


def _mount(source: Path, target: str, *, readonly: bool = False) -> list[str]:
    return ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if readonly else "")]


def _results(out: Path) -> dict:
    return json.loads((out / "results" / "result.json").read_text())


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
        assert set(_results(out)) == {"clean_task", "clean_task_2"}
        assert (out / "results" / "result.json").stat().st_uid != 0  # written by the non-root user

    def test_image_is_built_for_linux_amd64(self, cpu_image: str) -> None:
        inspect = _docker("image", "inspect", "--format", "{{.Os}}/{{.Architecture}}", cpu_image)

        assert inspect.returncode == 0, inspect.stderr
        assert inspect.stdout.strip() == "linux/amd64"

    @pytest.mark.parametrize("args", [[], ["--help"]], ids=["no-data-mounted", "help-flag"])
    def test_prints_usage_when_nothing_is_mounted_or_help_is_requested(self, cpu_image: str, args: list[str]) -> None:
        run = _docker("run", "--rm", cpu_image, *args)

        assert run.returncode == 0, run.stdout + run.stderr
        for section in ("USAGE:", "VOLUME MOUNTS", "ENVIRONMENT VARIABLES", "COMMAND-LINE OPTIONS", "No secret mounts"):
            assert section in run.stdout
        for variable in (
            "DATAEVAL_DATA",
            "DATAEVAL_OUTPUT",
            "DATAEVAL_CACHE",
            "DATAEVAL_MAX_PROCESSES",
            "DATAEVAL_LOG_FORMAT",
            "DATAEVAL_SERVICE_PORT",
        ):
            assert variable in run.stdout
        for command in ("workflows", "evaluators", "steps", "verify", "serve"):
            assert f"    {command} " in run.stdout

    def test_reports_an_output_directory_that_is_not_mounted(self, cpu_image: str, mounts: tuple[Path, Path]) -> None:
        data, _ = mounts

        run = _docker("run", "--rm", *_mount(data, "/dataeval", readonly=True), cpu_image)

        assert run.returncode == 1, run.stdout + run.stderr
        assert "ERROR: Output directory not mounted at /output" in run.stdout

    def test_reports_a_data_mount_that_is_empty(
        self, cpu_image: str, mounts: tuple[Path, Path], tmp_path: Path
    ) -> None:
        _, out = mounts
        empty = tmp_path / "empty"
        empty.mkdir()

        run = _docker("run", "--rm", *_mount(empty, "/dataeval", readonly=True), *_mount(out, "/output"), cpu_image)

        assert run.returncode == 1, run.stdout + run.stderr
        assert "ERROR: Data mount is empty at /dataeval" in run.stdout

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

    def test_honors_the_environment_variables_and_a_relocated_data_root(
        self, cpu_image: str, mounts: tuple[Path, Path]
    ) -> None:
        data, out = mounts

        run = _docker(
            "run",
            "--rm",
            "-e",
            "DATAEVAL_DATA=/elsewhere",
            "-e",
            "DATAEVAL_TASKS=clean_task_2",
            "-e",
            "DATAEVAL_LOG_FORMAT=plain",
            "-e",
            "DATAEVAL_VERBOSITY=2",
            "-e",
            "DATAEVAL_MAX_PROCESSES=2",
            *_mount(data, "/elsewhere", readonly=True),
            *_mount(out, "/output"),
            cpu_image,
            "python",
            "-m",
            "dataeval_flow",
        )

        assert run.returncode == 0, run.stdout + run.stderr
        assert set(_results(out)) == {"clean_task_2"}
        assert "OK: clean_task_2" in run.stdout
        assert _results(out)["clean_task_2"]["metadata"]["resolved_config"]["max_processes"] == 2
        assert "[INFO]" not in run.stdout  # plain format: no timestamp or level prefix

    def test_a_command_line_option_beats_its_environment_variable(
        self, cpu_image: str, mounts: tuple[Path, Path]
    ) -> None:
        data, out = mounts

        run = _docker(
            "run",
            "--rm",
            "-e",
            "DATAEVAL_TASKS=clean_task_2",
            *_mount(data, "/dataeval", readonly=True),
            *_mount(out, "/output"),
            cpu_image,
            "python",
            "-m",
            "dataeval_flow",
            "--task",
            "clean_task",
        )

        assert run.returncode == 0, run.stdout + run.stderr
        assert set(_results(out)) == {"clean_task"}


class TestCpuImageService:
    """`serve` in the container: the published port answers the three health endpoints."""

    @pytest.fixture
    def service(self, cpu_image: str, mounts: tuple[Path, Path]) -> Iterator[str]:
        data, out = mounts
        name = f"dataeval-flow-verify-{uuid.uuid4().hex[:8]}"
        started = _docker(
            "run",
            "-d",
            "--init",
            "--name",
            name,
            "-p",
            "127.0.0.1::8001",
            *_mount(data, "/dataeval", readonly=True),
            *_mount(out, "/output"),
            cpu_image,
            "python",
            "-m",
            "dataeval_flow",
            "serve",
        )
        assert started.returncode == 0, started.stderr
        try:
            published = _docker("port", name, SERVICE_PORT)
            assert published.returncode == 0, published.stderr
            yield f"http://{published.stdout.splitlines()[0].strip()}"
        finally:
            _docker("rm", "-f", name)

    @staticmethod
    def _get(url: str) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()
        except OSError:
            return 0, b""

    def test_the_published_port_answers_the_health_endpoints_and_describes_its_api(self, service: str) -> None:
        # The service imports Flow and PyTorch before it listens, so poll rather than assume a start-up time.
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            status, _ = self._get(f"{service}/healthz")
            if status == 200:
                break
            time.sleep(1)

        for probe in ("healthz", "readyz", "livez"):
            status, body = self._get(f"{service}/{probe}")
            assert (status, json.loads(body)) == (200, {"status": "ok"}), probe
        status, body = self._get(f"{service}/openapi.json")
        assert status == 200
        assert "/v1/runs" in json.loads(body)["paths"]


class TestCudaImage:
    def test_exits_with_one_and_names_the_gpu_flag_and_cpu_variant_without_gpu_access(
        self, cuda_image: str, tmp_path: Path
    ) -> None:
        data = tmp_path / "data"
        data.mkdir()
        write_project(data)
        out = tmp_path / "out"
        out.mkdir()
        out.chmod(0o777)

        run = _docker("run", "--rm", *_mount(data, "/dataeval", readonly=True), *_mount(out, "/output"), cuda_image)

        assert run.returncode == 1, run.stdout + run.stderr
        assert "--gpus all" in run.stdout
        assert "dataeval-flow:cpu" in run.stdout


class TestPublishedImageSecurity:
    def test_signature_verifies_against_the_committed_public_key(self) -> None:
        image = os.environ.get("DATAEVAL_FLOW_TEST_IMAGE_PUBLISHED")
        if not image:
            pytest.skip("DATAEVAL_FLOW_TEST_IMAGE_PUBLISHED is not set; no published image to verify")
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
        image = _require_image("DATAEVAL_FLOW_TEST_IMAGE_PUBLISHED")

        listing = _docker("run", "--rm", "--entrypoint", "ls", image, SECURITY_DIR)

        assert listing.returncode == 0, listing.stderr
        files = set(listing.stdout.split())
        assert "sbom.cdx.json" in files
        assert {"vulnerability-scan.json", "vulnerability-scan.txt"} <= files

        sbom = _docker("run", "--rm", "--entrypoint", "cat", image, SECURITY_DIR + "sbom.cdx.json")
        assert json.loads(sbom.stdout)["bomFormat"] == "CycloneDX"

        scan = _docker("run", "--rm", "--entrypoint", "cat", image, SECURITY_DIR + "vulnerability-scan.json")
        report = json.loads(scan.stdout)
        assert report["SchemaVersion"]
        assert report["Metadata"]["RepoTags"], "the scan report does not name the image it describes"
