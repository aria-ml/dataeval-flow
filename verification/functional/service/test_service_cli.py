"""TC-21-4 — `dataeval-flow serve`: its options, what it refuses before serving, and a service answering over HTTP."""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from dataeval_flow import __main__ as cli
from dataeval_flow._service import _app
from verification.helpers import run_cli

pytestmark = pytest.mark.required


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(data: Path, output: Path, port: int) -> subprocess.Popen[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("DATAEVAL_")}
    return subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "dataeval_flow", "serve", "-d", str(data), "-o", str(output), "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


class TestServeCommand:
    def test_help_documents_its_options_environment_variables_and_endpoints(self) -> None:
        proc = run_cli("serve", "--help")
        assert proc.returncode == 0, proc.stderr
        for documented in (
            "--data",
            "--output",
            "--cache",
            "--host",
            "--port",
            "DATAEVAL_DATA",
            "DATAEVAL_OUTPUT",
            "DATAEVAL_CACHE",
            "DATAEVAL_SERVICE_HOST",
            "DATAEVAL_SERVICE_PORT",
            "/healthz",
            "/livez",
            "/readyz",
            "/openapi.json",
        ):
            assert documented in proc.stdout, documented

    def test_serving_without_an_output_directory_is_refused(self, tmp_path: Path) -> None:
        proc = run_cli("serve", "--data", str(tmp_path))
        assert proc.returncode == 1
        assert "--output" in proc.stderr

    def test_serving_with_a_data_root_that_does_not_exist_is_refused(self, tmp_path: Path) -> None:
        proc = run_cli("serve", "--data", str(tmp_path / "missing"), "--output", str(tmp_path / "out"))
        assert proc.returncode == 1
        assert "Data root not found" in proc.stderr

    def test_a_command_line_option_overrides_its_environment_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple] = []
        monkeypatch.setattr(_app, "serve", lambda *args: calls.append(args))
        monkeypatch.setattr(logging, "captureWarnings", lambda _capture: None)
        for name in ("DATA", "OUTPUT", "CACHE", "LOG_FORMAT", "SERVICE_HOST", "SERVICE_PORT"):
            monkeypatch.delenv(f"DATAEVAL_{name}", raising=False)
        monkeypatch.setenv("DATAEVAL_SERVICE_HOST", "10.1.2.3")
        monkeypatch.setenv("DATAEVAL_SERVICE_PORT", "9000")
        monkeypatch.setenv("DATAEVAL_OUTPUT", str(tmp_path / "out"))
        monkeypatch.setattr(sys, "argv", ["dataeval-flow", "serve", "--data", str(tmp_path), "--port", "8123"])
        with pytest.raises(SystemExit) as exited:
            cli.main()
        assert exited.value.code == 0
        assert calls == [(tmp_path, tmp_path / "out", None, "10.1.2.3", 8123, "structured")]


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[str, Path, Path, subprocess.Popen[str]]]:
    data = tmp_path / "data"
    data.mkdir()
    output = tmp_path / "out"
    port = _free_port()
    process = _serve(data, output, port)
    base = f"http://127.0.0.1:{port}"
    end = time.monotonic() + 60
    while True:
        try:
            if httpx.get(f"{base}/healthz").status_code == 200:
                break
        except httpx.TransportError:
            pass
        assert process.poll() is None, "the service exited before it came up"
        assert time.monotonic() < end, "the service did not come up"
        time.sleep(0.2)
    try:
        yield base, data, output, process
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


class TestServingOverHTTP:
    def test_a_started_service_answers_its_probes_and_publishes_its_interface(self, server) -> None:
        base, *_ = server
        for probe in ("/healthz", "/livez", "/readyz"):
            assert httpx.get(base + probe).json() == {"status": "ok"}, probe
        assert "/v1/runs" in httpx.get(f"{base}/openapi.json").json()["paths"]
        assert httpx.get(f"{base}/v1/capabilities").json()["max_active_runs"] == 1

    def test_a_second_service_on_the_same_output_directory_exits_3(self, server) -> None:
        _, data, output, _ = server
        second = _serve(data, output, _free_port())
        try:
            log, _ = second.communicate(timeout=60)
        finally:
            if second.poll() is None:
                second.kill()
        assert second.returncode == 3
        assert "Another service is running the queue" in log

    def test_sigterm_shuts_the_service_down_gracefully(self, server) -> None:
        _, _, _, process = server
        process.send_signal(signal.SIGTERM)
        log, _ = process.communicate(timeout=30)
        assert "Application shutdown complete" in log
