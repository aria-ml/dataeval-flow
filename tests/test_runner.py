"""Tests for _resolve_config in runner.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dataeval_flow.config._models import PipelineConfig
from dataeval_flow.runner import _resolve_config


@pytest.fixture
def dummy_config() -> PipelineConfig:
    return PipelineConfig()


_LOADER = "dataeval_flow.config._loader"


class TestResolveConfig:
    def test_explicit_absolute_file(self, tmp_path: Path, dummy_config: PipelineConfig):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("tasks: []")

        with patch(f"{_LOADER}.load_config", return_value=dummy_config) as mock_load:
            result = _resolve_config(cfg_file, tmp_path)

        mock_load.assert_called_once_with(cfg_file)
        assert result is dummy_config

    def test_explicit_relative_file(self, tmp_path: Path, dummy_config: PipelineConfig):
        cfg_file = tmp_path / "sub" / "config.yaml"
        cfg_file.parent.mkdir()
        cfg_file.write_text("tasks: []")

        with patch(f"{_LOADER}.load_config", return_value=dummy_config) as mock_load:
            result = _resolve_config(Path("sub/config.yaml"), tmp_path)

        mock_load.assert_called_once_with(tmp_path / "sub" / "config.yaml")
        assert result is dummy_config

    def test_explicit_directory(self, tmp_path: Path, dummy_config: PipelineConfig):
        cfg_dir = tmp_path / "conf"
        cfg_dir.mkdir()

        with patch(f"{_LOADER}.load_config_folder", return_value=dummy_config) as mock_load:
            result = _resolve_config(cfg_dir, tmp_path)

        mock_load.assert_called_once_with(cfg_dir)
        assert result is dummy_config

    def test_none_config_uses_data_dir(self, tmp_path: Path, dummy_config: PipelineConfig):
        with patch(f"{_LOADER}.load_config_folder", return_value=dummy_config) as mock_load:
            result = _resolve_config(None, tmp_path)

        mock_load.assert_called_once_with(tmp_path)
        assert result is dummy_config

    def test_missing_path_raises(self, tmp_path: Path):
        missing = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="Config path not found"):
            _resolve_config(missing, tmp_path)

    def test_string_config_arg(self, tmp_path: Path, dummy_config: PipelineConfig):
        cfg_file = tmp_path / "my_config.yaml"
        cfg_file.write_text("tasks: []")

        with patch(f"{_LOADER}.load_config", return_value=dummy_config) as mock_load:
            result = _resolve_config("my_config.yaml", tmp_path)

        mock_load.assert_called_once_with(tmp_path / "my_config.yaml")
        assert result is dummy_config
