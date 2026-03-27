"""Tests for __main__.py — argument parsing and command dispatch."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestBuildParser:
    def test_headless_defaults(self) -> None:
        from dataeval_flow.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--config", "/cfg", "--output", "/out"])
        assert args.command is None
        assert args.config == Path("/cfg")
        assert args.output == Path("/out")

    def test_headless_no_args(self) -> None:
        from dataeval_flow.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args([])
        assert args.command is None
        assert args.output is None

    def test_app_subcommand(self) -> None:
        from dataeval_flow.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["app", "--config", "/cfg"])
        assert args.command == "app"
        assert args.config == Path("/cfg")

    def test_app_no_args(self) -> None:
        from dataeval_flow.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["app"])
        assert args.command == "app"
        assert args.config is None

    def test_config_subcommand(self) -> None:
        from dataeval_flow.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["config", "--config", "/cfg"])
        assert args.command == "config"
        assert args.config == Path("/cfg")

    def test_config_no_args(self) -> None:
        from dataeval_flow.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["config"])
        assert args.command == "config"
        assert args.config is None

    def test_env_var_defaults(self) -> None:
        """Environment variables provide defaults for headless execution."""
        env = {
            "DATAEVAL_DATA": "/env/data",
            "DATAEVAL_OUTPUT": "/env/output",
            "DATAEVAL_CACHE": "/env/cache",
        }
        with patch.dict(os.environ, env):
            from dataeval_flow.__main__ import _build_parser

            parser = _build_parser()
            args = parser.parse_args([])
            assert args.data == Path("/env/data")
            assert args.output == Path("/env/output")
            assert args.cache == Path("/env/cache")

    def test_explicit_args_override_env(self) -> None:
        """Explicit CLI args take precedence over env var defaults."""
        env = {"DATAEVAL_OUTPUT": "/env/output"}
        with patch.dict(os.environ, env):
            from dataeval_flow.__main__ import _build_parser

            parser = _build_parser()
            args = parser.parse_args(["--output", "/cli/output"])
            assert args.output == Path("/cli/output")


class TestParseArgs:
    def test_no_args_headless(self) -> None:
        from dataeval_flow.__main__ import parse_args

        with patch.object(sys, "argv", ["prog"]):
            args = parse_args()
            assert args.command is None

    def test_app_subcommand(self) -> None:
        from dataeval_flow.__main__ import parse_args

        with patch.object(sys, "argv", ["prog", "app"]):
            args = parse_args()
            assert args.command == "app"

    def test_config_subcommand(self) -> None:
        from dataeval_flow.__main__ import parse_args

        with patch.object(sys, "argv", ["prog", "config", "--config", "/cfg"]):
            args = parse_args()
            assert args.command == "config"
            assert args.config == Path("/cfg")

    def test_headless_with_flags(self) -> None:
        from dataeval_flow.__main__ import parse_args

        with patch.object(sys, "argv", ["prog", "--output", "/out"]):
            args = parse_args()
            assert args.command is None
            assert args.output == Path("/out")


class TestMain:
    def test_config_subcommand(self) -> None:
        from dataeval_flow.__main__ import main

        with (
            patch.object(sys, "argv", ["prog", "config"]),
            patch("dataeval_flow._app.cli.run_cli_builder") as mock_run,
            pytest.raises(SystemExit, match="0"),
        ):
            main()
        mock_run.assert_called_once_with(config_path=None)

    def test_config_with_path(self) -> None:
        from dataeval_flow.__main__ import main

        with (
            patch.object(sys, "argv", ["prog", "config", "--config", "/cfg"]),
            patch("dataeval_flow._app.cli.run_cli_builder") as mock_run,
            pytest.raises(SystemExit, match="0"),
        ):
            main()
        mock_run.assert_called_once_with(config_path=Path("/cfg"))

    def test_app_tui_available(self) -> None:
        from dataeval_flow.__main__ import main

        with (
            patch.object(sys, "argv", ["prog", "app"]),
            patch("dataeval_flow._app.app.run_builder") as mock_run,
            pytest.raises(SystemExit, match="0"),
        ):
            main()
        mock_run.assert_called_once_with(config_path=None, data_dir=None, cache_dir=None)

    def test_app_tui_import_error_shows_instructions(self, capsys: pytest.CaptureFixture) -> None:
        from dataeval_flow.__main__ import main

        args = MagicMock()
        args.command = "app"
        args.config = None

        import builtins

        original_import = builtins.__import__

        def patched_import(name: str, *a, **kw):
            if name == "dataeval_flow._app.app":
                raise ImportError("no textual")
            return original_import(name, *a, **kw)

        with (
            patch.object(sys, "argv", ["prog", "app"]),
            patch("dataeval_flow.__main__.parse_args", return_value=args),
            patch.object(builtins, "__import__", side_effect=patched_import),
            pytest.raises(SystemExit, match="1"),
        ):
            main()

        out = capsys.readouterr().out
        assert "dataeval-flow[app]" in out
        assert "dataeval-flow config" in out

    def test_headless_success(self) -> None:
        from dataeval_flow.__main__ import main

        with (
            patch.object(sys, "argv", ["prog", "--output", "/out"]),
            patch("dataeval_flow.runner.run", return_value=0) as mock_run,
            pytest.raises(SystemExit, match="0"),
        ):
            main()
        mock_run.assert_called_once()

    def test_headless_error(self) -> None:
        from dataeval_flow.__main__ import main

        with (
            patch.object(sys, "argv", ["prog", "--output", "/out"]),
            patch("dataeval_flow.runner.run", side_effect=FileNotFoundError("not found")),
            pytest.raises(SystemExit, match="1"),
        ):
            main()
