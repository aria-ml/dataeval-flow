"""TC-37-1 (NFR-7) — the container build is wired to scan, sign and ship a software bill of materials.

These checks read the Dockerfiles and the pipeline definition. That the published images really carry the scan
reports and SBOM, and verify against the public key, is checked on the images themselves in
``verification/functional/docker/test_container_runtime.py``.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import yaml

from verification.helpers import REPO_ROOT

pytestmark = pytest.mark.required

NAMES = ["cpu", "cu126", "cu130"]


class _CiLoader(yaml.SafeLoader):
    """Reads ``.gitlab-ci.yml``: GitLab's ``!reference`` tags are kept as plain lists."""


_CiLoader.add_multi_constructor("!", lambda loader, _suffix, node: loader.construct_sequence(node, deep=True))


def _jobs() -> dict[str, Any]:
    return yaml.load((REPO_ROOT / ".gitlab-ci.yml").read_text(), Loader=_CiLoader)  # noqa: S506


def _script(job: str) -> list[str]:
    return [line if isinstance(line, str) else "" for line in _jobs()[job]["script"]]


def _first(lines: list[str], pattern: str) -> int:
    return next(i for i, line in enumerate(lines) if re.search(pattern, line))


class TestPipelineOrder:
    def test_the_image_is_scanned_before_it_is_pushed_under_its_published_tag(self) -> None:
        script = _script("push:docker")

        scan_gate = _first(script, r"trivy image .*--severity HIGH,CRITICAL --exit-code 1")
        push = _first(script, r"--target scanned .*--push")

        assert scan_gate < push

    def test_the_scan_gate_fails_on_high_and_critical_findings_and_the_report_is_kept(self) -> None:
        script = _script("push:docker")

        assert any("--format json --output docker/security/vulnerability-scan.json" in line for line in script)
        assert any("--format table --output docker/security/vulnerability-scan.txt" in line for line in script)
        assert any("--severity HIGH,CRITICAL --exit-code 1" in line for line in script)
        assert _jobs()["container_scanning"]["allow_failure"] is False
        assert _jobs()["container_scanning"]["variables"]["CS_SEVERITY_THRESHOLD"] == "HIGH"

    def test_the_pushed_image_is_signed_and_gets_a_cyclonedx_attestation_afterwards(self) -> None:
        script = _script("push:docker")

        push = _first(script, r"--target scanned .*--push")
        sign = _first(script, r"cosign sign --key cosign\.key")
        attest = _first(script, r"cosign attest --key cosign\.key --type cyclonedx")

        assert push < sign < attest
        assert any(
            "syft scan docker-archive:" in line and "cyclonedx-json=docker/security/sbom.cdx.json" in line
            for line in script
        )

    def test_the_pipeline_verifies_the_signature_and_attestation_with_the_committed_key(self) -> None:
        script = _script("verify:docker")

        assert any("cosign verify --key docker/cosign.pub" in line for line in script)
        assert any("cosign verify-attestation --key docker/cosign.pub --type cyclonedx" in line for line in script)
        assert any("vulnerability-scan.json" in line and "sbom.cdx.json" in line for line in script)

    def test_a_release_republishes_the_sbom_as_a_permanent_artifact(self) -> None:
        job = _jobs()["release:sbom"]

        assert any("cosign download attestation" in line for line in job["script"])
        assert job["artifacts"]["expire_in"] == "never"

    def test_the_scan_runs_at_the_pinned_trivy_version(self) -> None:
        assert re.fullmatch(r"\d+\.\d+\.\d+", _jobs()["variables"]["TRIVY_VERSION"])


class TestImageContents:
    def test_the_public_key_is_committed(self) -> None:
        key = (REPO_ROOT / "docker" / "cosign.pub").read_text()

        assert key.startswith("-----BEGIN PUBLIC KEY-----")
        assert "PRIVATE" not in key

    @pytest.mark.parametrize("name", NAMES)
    def test_the_published_stage_adds_the_scan_reports_and_sbom_at_the_documented_path(self, name: str) -> None:
        text = (REPO_ROOT / "docker" / f"Dockerfile.{name}").read_text()
        stage = text.split("FROM prod AS scanned", 1)[1]

        assert "COPY --chown=root:root docker/security/ /usr/share/dataeval-flow/security/" in stage

    @pytest.mark.parametrize("name", NAMES)
    def test_the_private_key_is_never_copied_into_an_image(self, name: str) -> None:
        text = (REPO_ROOT / "docker" / f"Dockerfile.{name}").read_text()

        assert "cosign.key" not in text
        assert "COSIGN" not in text
