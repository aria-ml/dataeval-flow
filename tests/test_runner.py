"""Tests for _resolve_config in runner.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dataeval_flow.config._models import PipelineConfig
from dataeval_flow.runner import _resolve_config

pytestmark = pytest.mark.required


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


class TestWriteEncodingDescriptor:
    """A run writes the descriptor its results were computed under, beside them."""

    @staticmethod
    def _record(digest: str, edges: list) -> dict:
        return {
            "encoding_digest": digest,
            "descriptor_version": 1,
            "factors": {
                "temp_c": {"encoding": {"kind": "bins", "edges": edges, "provenance": "edges", "method": None}}
            },
        }

    def test_writes_one_when_the_tasks_agree(self, tmp_path: Path):
        from dataeval_flow.runner import _write_encoding_descriptor

        record = self._record("abc", ["-inf", 0.0, "inf"])
        _write_encoding_descriptor({"a": record, "b": record}, tmp_path)

        import json

        written = json.loads((tmp_path / "encoding.json").read_text())
        assert written["factors"]["temp_c"]["edges"] == ["-inf", 0.0, "inf"]
        assert written["version"] == 1

    def test_writes_nothing_when_the_tasks_disagree(self, tmp_path: Path, caplog):
        """Writing one of them would hand somebody a policy nobody chose."""
        from dataeval_flow.runner import _write_encoding_descriptor

        _write_encoding_descriptor(
            {"a": self._record("abc", ["-inf", 0.0, "inf"]), "b": self._record("def", ["-inf", 5.0, "inf"])},
            tmp_path,
        )

        assert not (tmp_path / "encoding.json").exists()
        assert "dataeval-flow encoding" in caplog.text

    def test_writes_nothing_when_no_task_built_metadata(self, tmp_path: Path):
        from dataeval_flow.runner import _write_encoding_descriptor

        _write_encoding_descriptor({}, tmp_path)
        assert not (tmp_path / "encoding.json").exists()

    def test_a_record_with_no_encodings_is_skipped_quietly(self, tmp_path: Path):
        """Never fatal: result.json already carries every record it is built from."""
        from dataeval_flow.runner import _write_encoding_descriptor

        _write_encoding_descriptor({"a": {"factors": {}}}, tmp_path)
        assert not (tmp_path / "encoding.json").exists()
