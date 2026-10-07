"""``dataeval-flow serve``: options over environment, and what it refuses before it serves."""

import logging
import sys

import pytest

pytest.importorskip("fastapi", reason="needs the 'service' extra")

from dataeval_flow import __main__ as cli
from dataeval_flow._service import _app

pytestmark = pytest.mark.optional


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in ("DATA", "OUTPUT", "CACHE", "LOG_FORMAT", "SERVICE_HOST", "SERVICE_PORT"):
        monkeypatch.delenv(f"DATAEVAL_{name}", raising=False)
    monkeypatch.setattr(logging, "captureWarnings", lambda _capture: None)  # Leave the session's warnings to pytest.


def _main(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["dataeval-flow", *argv])
    with pytest.raises(SystemExit) as exited:
        cli.main()
    return exited.value.code


def test_options_take_precedence_over_the_environment(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(_app, "serve", lambda *args: calls.append(args))
    monkeypatch.setenv("DATAEVAL_SERVICE_HOST", "0.0.0.0")  # noqa: S104
    monkeypatch.setenv("DATAEVAL_SERVICE_PORT", "9000")
    monkeypatch.setenv("DATAEVAL_OUTPUT", str(tmp_path / "out"))
    assert _main(monkeypatch, "serve", "--data", str(tmp_path), "--port", "8123") == 0
    assert calls == [(tmp_path, tmp_path / "out", None, "0.0.0.0", 8123, "structured")]  # noqa: S104
    assert _main(monkeypatch, "--log-format", "plain", "serve", "-d", str(tmp_path), "-o", str(tmp_path)) == 0
    assert calls[-1] == (tmp_path, tmp_path, None, "0.0.0.0", 9000, "plain")  # noqa: S104


def test_refuses_without_an_output_or_a_data_root(tmp_path, monkeypatch, capsys):
    assert _main(monkeypatch, "serve", "--data", str(tmp_path)) == 1
    assert "--output" in capsys.readouterr().err
    assert _main(monkeypatch, "serve", "--data", str(tmp_path / "missing"), "--output", str(tmp_path)) == 1
    assert "Data root not found" in capsys.readouterr().err


def test_without_the_extra_says_how_to_install_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "dataeval_flow._service._app", None)
    assert _main(monkeypatch, "serve", "--output", str(tmp_path)) == 1
    assert "dataeval-flow[service]" in capsys.readouterr().err
