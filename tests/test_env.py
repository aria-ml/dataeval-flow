"""Tests for dataeval_flow._env environment readers."""

from pathlib import Path

import pytest

from dataeval_flow._env import env_bool, env_choice, env_int, env_list, env_path

pytestmark = pytest.mark.required


class TestUnsetAndBlank:
    def test_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv("DE_TEST", raising=False)
        assert env_path("DE_TEST") is None
        assert env_int("DE_TEST") is None
        assert env_bool("DE_TEST") is None
        assert env_list("DE_TEST") is None
        assert env_choice("DE_TEST", ("a", "b")) is None

    def test_blank_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("DE_TEST", "   ")
        assert env_path("DE_TEST") is None
        assert env_int("DE_TEST") is None
        assert env_bool("DE_TEST") is None
        assert env_list("DE_TEST") is None
        assert env_choice("DE_TEST", ("a", "b")) is None


class TestValidValues:
    def test_env_path(self, monkeypatch):
        monkeypatch.setenv("DE_TEST", "/data/root")
        assert env_path("DE_TEST") == Path("/data/root")

    def test_env_int(self, monkeypatch):
        monkeypatch.setenv("DE_TEST", "3")
        assert env_int("DE_TEST") == 3

    def test_env_int_accepts_zero(self, monkeypatch):
        monkeypatch.setenv("DE_TEST", "0")
        assert env_int("DE_TEST") == 0

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "On"])
    def test_env_bool_true(self, monkeypatch, raw):
        monkeypatch.setenv("DE_TEST", raw)
        assert env_bool("DE_TEST") is True

    @pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "Off"])
    def test_env_bool_false(self, monkeypatch, raw):
        monkeypatch.setenv("DE_TEST", raw)
        assert env_bool("DE_TEST") is False

    def test_env_list_splits_and_strips(self, monkeypatch):
        monkeypatch.setenv("DE_TEST", " drift , coverage ")
        assert env_list("DE_TEST") == ["drift", "coverage"]

    def test_env_choice(self, monkeypatch):
        monkeypatch.setenv("DE_TEST", "plain")
        assert env_choice("DE_TEST", ("structured", "plain")) == "plain"


class TestMalformedRaises:
    """Invalid environment variable values must raise ValueError."""

    def test_env_int_rejects_non_integer(self, monkeypatch):
        monkeypatch.setenv("DE_TEST", "loud")
        with pytest.raises(ValueError, match="DE_TEST"):
            env_int("DE_TEST")

    def test_env_bool_rejects_unknown_word(self, monkeypatch):
        monkeypatch.setenv("DE_TEST", "maybe")
        with pytest.raises(ValueError, match="DE_TEST"):
            env_bool("DE_TEST")

    def test_env_list_rejects_separators_only(self, monkeypatch):
        monkeypatch.setenv("DE_TEST", " , , ")
        with pytest.raises(ValueError, match="DE_TEST"):
            env_list("DE_TEST")

    def test_env_choice_rejects_value_outside_choices(self, monkeypatch):
        monkeypatch.setenv("DE_TEST", "json")
        with pytest.raises(ValueError, match="DE_TEST"):
            env_choice("DE_TEST", ("structured", "plain"))
