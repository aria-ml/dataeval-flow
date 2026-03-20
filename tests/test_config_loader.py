"""Tests for config_loader module."""

from pathlib import Path

import pytest

from dataeval_flow.config._merge import _deep_merge, merge_config_folder


class TestDeepMerge:
    """Tests for _deep_merge function."""

    def test_merge_dicts(self):
        """Deep merge of nested dicts."""
        base = {"a": {"b": 1}}
        overlay = {"a": {"c": 2}}
        _deep_merge(base, overlay)
        assert base == {"a": {"b": 1, "c": 2}}

    def test_merge_lists(self):
        """Lists are extended, not replaced."""
        base = {"items": [1, 2]}
        overlay = {"items": [3, 4]}
        _deep_merge(base, overlay)
        assert base == {"items": [1, 2, 3, 4]}

    def test_override_scalar(self):
        """Scalar values are overridden."""
        base = {"key": "old"}
        overlay = {"key": "new"}
        _deep_merge(base, overlay)
        assert base == {"key": "new"}

    def test_add_new_key(self):
        """New keys are added."""
        base = {"a": 1}
        overlay = {"b": 2}
        _deep_merge(base, overlay)
        assert base == {"a": 1, "b": 2}

    def test_none_value_overwrites(self):
        """None in overlay overwrites existing value."""
        base = {"key": "old"}
        overlay = {"key": None}
        _deep_merge(base, overlay)
        assert base == {"key": None}

    def test_nested_list_in_dict(self):
        """Lists nested inside dicts are extended recursively."""
        base = {"a": {"items": [1, 2]}}
        overlay = {"a": {"items": [3]}}
        _deep_merge(base, overlay)
        assert base == {"a": {"items": [1, 2, 3]}}


class TestMergeYamlFolder:
    """Tests for merge_config_folder function."""

    def test_merge_config_folder(self, tmp_path: Path):
        """Merge multiple YAML files from folder."""
        # Create test YAML files
        (tmp_path / "00-base.yaml").write_text("datasets:\n  - name: test1\n")
        (tmp_path / "01-tasks.yaml").write_text("tasks:\n  - name: task1\n")

        result = merge_config_folder(tmp_path)
        assert "datasets" in result
        assert "tasks" in result
        assert result["datasets"] == [{"name": "test1"}]
        assert result["tasks"] == [{"name": "task1"}]

    def test_merge_config_folder_alphabetical(self, tmp_path: Path):
        """Files are merged in alphabetical order."""
        (tmp_path / "02-second.yaml").write_text("logging:\n  app_level: WARNING\n")
        (tmp_path / "01-first.yaml").write_text("logging:\n  app_level: INFO\n")

        result = merge_config_folder(tmp_path)
        # 02-second.yaml comes after 01-first.yaml, so WARNING wins
        assert result["logging"]["app_level"] == "WARNING"

    def test_merge_config_folder_not_directory(self, tmp_path: Path):
        """Raises ValueError if path is not a directory."""
        file_path = tmp_path / "file.yaml"
        file_path.write_text("key: value\n")

        with pytest.raises(ValueError, match="not a directory"):
            merge_config_folder(file_path)

    def test_merge_config_folder_empty(self, tmp_path: Path):
        """Empty folder raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No valid pipeline config"):
            merge_config_folder(tmp_path)

    def test_merge_config_folder_yml_extension(self, tmp_path: Path):
        """Both .yaml and .yml extensions are supported."""
        (tmp_path / "config.yml").write_text("logging:\n  app_level: INFO\n")

        result = merge_config_folder(tmp_path)
        assert result == {"logging": {"app_level": "INFO"}}

    def test_merge_config_folder_mixed_extensions_sorted(self, tmp_path: Path):
        """Mixed .yaml and .yml files are sorted together alphabetically."""
        (tmp_path / "02-second.yaml").write_text("logging:\n  app_level: WARNING\n")
        (tmp_path / "01-first.yml").write_text("logging:\n  app_level: INFO\n")

        result = merge_config_folder(tmp_path)
        # 02-second.yaml comes after 01-first.yml alphabetically, so WARNING wins
        assert result["logging"]["app_level"] == "WARNING"

    def test_merge_config_folder_empty_file(self, tmp_path: Path):
        """YAML file with no content (safe_load returns None) is skipped."""
        (tmp_path / "00-base.yaml").write_text("logging:\n  app_level: INFO\n")
        (tmp_path / "01-empty.yaml").write_text("# just a comment\n")

        result = merge_config_folder(tmp_path)
        assert result == {"logging": {"app_level": "INFO"}}
