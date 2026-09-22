"""Environment-variable handling in the CLI entry point."""

import sys
from pathlib import Path

import pytest

from dataeval_flow.__main__ import _build_parser, apply_env_defaults, main

pytestmark = pytest.mark.required


class TestPathOptionsReadEnvironment:
    def test_config_defaults_from_environment(self, monkeypatch):
        monkeypatch.setenv("DATAEVAL_CONFIG", "/cfg/pipeline.yaml")
        args = _build_parser().parse_args([])
        assert args.config == Path("/cfg/pipeline.yaml")

    def test_command_line_beats_environment(self, monkeypatch):
        """CLI arguments take precedence over environment variables."""
        monkeypatch.setenv("DATAEVAL_CONFIG", "/cfg/from-env.yaml")
        args = _build_parser().parse_args(["--config", "/cfg/from-cli.yaml"])
        assert args.config == Path("/cfg/from-cli.yaml")

    @pytest.mark.parametrize(
        ("variable", "attribute"),
        [
            ("DATAEVAL_DATA", "data"),
            ("DATAEVAL_OUTPUT", "output"),
            ("DATAEVAL_CACHE", "cache"),
        ],
    )
    def test_existing_path_variables_still_apply(self, monkeypatch, variable, attribute):
        monkeypatch.setenv(variable, "/from/env")
        args = _build_parser().parse_args([])
        assert getattr(args, attribute) == Path("/from/env")


class TestLogFormatOption:
    def test_defaults_to_structured(self, monkeypatch):
        monkeypatch.delenv("DATAEVAL_LOG_FORMAT", raising=False)
        assert _build_parser().parse_args([]).log_format == "structured"

    def test_environment_selects_plain(self, monkeypatch):
        monkeypatch.setenv("DATAEVAL_LOG_FORMAT", "plain")
        assert _build_parser().parse_args([]).log_format == "plain"

    def test_command_line_beats_environment(self, monkeypatch):
        monkeypatch.setenv("DATAEVAL_LOG_FORMAT", "plain")
        args = _build_parser().parse_args(["--log-format", "structured"])
        assert args.log_format == "structured"

    def test_invalid_environment_value_raises(self, monkeypatch):
        monkeypatch.setenv("DATAEVAL_LOG_FORMAT", "json")
        with pytest.raises(ValueError, match="DATAEVAL_LOG_FORMAT"):
            _build_parser()


class TestMainReportsMalformedEnvironmentCleanly:
    def test_invalid_log_format_exits_one_without_a_traceback(self, monkeypatch, capsys):
        """Invalid environment values must exit cleanly without a traceback."""
        monkeypatch.setenv("DATAEVAL_LOG_FORMAT", "json")
        monkeypatch.setattr(sys, "argv", ["dataeval_flow"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "DATAEVAL_LOG_FORMAT" in err


class TestEncodingOutputIgnoresEnvironment:
    def test_encoding_output_stays_none(self, monkeypatch):
        """The encoding command ignores DATAEVAL_OUTPUT and defaults to stdout."""
        monkeypatch.setenv("DATAEVAL_OUTPUT", "/output")
        args = _build_parser().parse_args(["encoding", "result.json"])
        assert args.output is None


class TestEncodingTaskIgnoresEnvironment:
    def test_encoding_task_stays_none(self, monkeypatch):
        """DATAEVAL_TASKS must not apply to the encoding subcommand."""
        monkeypatch.setenv("DATAEVAL_TASKS", "drift,coverage")
        args = apply_env_defaults(_build_parser().parse_args(["encoding", "result.json"]))
        assert args.task is None


def _parsed(argv: list[str]):
    return apply_env_defaults(_build_parser().parse_args(argv))


class TestVerbosity:
    def test_default_is_zero_not_none(self, monkeypatch):
        """Verbosity defaults to 0 when unset."""
        monkeypatch.delenv("DATAEVAL_VERBOSITY", raising=False)
        assert _parsed([]).verbose == 0

    def test_environment_applies_when_no_flag_given(self, monkeypatch):
        monkeypatch.setenv("DATAEVAL_VERBOSITY", "2")
        assert _parsed([]).verbose == 2

    def test_command_line_replaces_rather_than_increments(self, monkeypatch):
        """CLI verbosity flags override rather than add to DATAEVAL_VERBOSITY."""
        monkeypatch.setenv("DATAEVAL_VERBOSITY", "3")
        assert _parsed(["-v"]).verbose == 1


class TestTaskSelection:
    def test_environment_applies_when_no_flag_given(self, monkeypatch):
        monkeypatch.setenv("DATAEVAL_TASKS", "drift,coverage")
        assert _parsed([]).task == ["drift", "coverage"]

    def test_command_line_replaces_rather_than_appends(self, monkeypatch):
        """CLI --task flags override rather than append to DATAEVAL_TASKS."""
        monkeypatch.setenv("DATAEVAL_TASKS", "drift,coverage")
        assert _parsed(["-t", "splitting"]).task == ["splitting"]


class TestFailOnWarning:
    def test_defaults_false(self, monkeypatch):
        monkeypatch.delenv("DATAEVAL_FAIL_ON_WARNING", raising=False)
        assert _parsed([]).fail_on_warning is False

    def test_environment_enables(self, monkeypatch):
        monkeypatch.setenv("DATAEVAL_FAIL_ON_WARNING", "true")
        assert _parsed([]).fail_on_warning is True

    def test_command_line_can_disable(self, monkeypatch):
        """--no-fail-on-warning overrides DATAEVAL_FAIL_ON_WARNING."""
        monkeypatch.setenv("DATAEVAL_FAIL_ON_WARNING", "true")
        assert _parsed(["--no-fail-on-warning"]).fail_on_warning is False
