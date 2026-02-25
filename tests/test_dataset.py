"""Tests for the dataeval_app package."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.required
class TestImports:
    """Test that core imports work correctly."""

    def test_dataeval_import(self) -> None:
        """Verify DataEval can be imported successfully."""
        import dataeval

        assert dataeval is not None

    def test_package_import(self) -> None:
        """Verify the dataeval_app package can be imported."""
        import dataeval_app

        assert hasattr(dataeval_app, "__version__")

    def test_public_api_exports(self) -> None:
        """Verify __all__ exports are available from package root."""
        from dataeval_app import load_dataset, load_dataset_huggingface

        assert callable(load_dataset)
        assert callable(load_dataset_huggingface)

    def test_workflow_api_exports(self) -> None:
        """Verify workflow exports are available from package root."""
        from dataeval_app import get_workflow, list_workflows, run_task

        assert callable(get_workflow)
        assert callable(list_workflows)
        assert callable(run_task)


@pytest.mark.required
class TestDatasetModule:
    """Test the dataset library module."""

    def test_load_dataset_huggingface_datasetdict(self, mock_hf_dataset: MagicMock) -> None:
        """Test loading a DatasetDict with explicit split selection."""
        with (
            patch("datasets.load_from_disk") as mock_load,
            patch("maite_datasets.adapters.from_huggingface") as mock_adapter,
        ):
            mock_load.return_value = mock_hf_dataset
            mock_adapter.return_value = MagicMock()

            from dataeval_app import load_dataset_huggingface

            load_dataset_huggingface(Path("dummy/path"), split="train")

            # Should use the explicitly requested split
            mock_hf_dataset.__getitem__.assert_called_with("train")
            mock_adapter.assert_called()

    def test_load_dataset_delegates(self) -> None:
        """Test load_dataset delegates to load_dataset_huggingface."""
        with patch("dataeval_app.dataset.load_dataset_huggingface") as mock_hf:
            mock_hf.return_value = MagicMock()

            from dataeval_app.dataset import load_dataset

            load_dataset(Path("/some/path"))
            mock_hf.assert_called_once_with(Path("/some/path"), split=None)


@pytest.mark.required
class TestMainModule:
    """Test the __main__.py CLI module."""

    def test_parse_args_no_args(self) -> None:
        """Test parsing with no arguments succeeds (config defaults to None)."""
        from dataeval_app.__main__ import parse_args

        with patch("sys.argv", ["dataeval_app"]):
            args = parse_args()
            assert args.config is None

    def test_parse_args_with_config(self) -> None:
        """Test parsing with --config flag."""
        from dataeval_app.__main__ import parse_args

        with patch("sys.argv", ["dataeval_app", "--config", "/my/config"]):
            args = parse_args()
            assert args.config == Path("/my/config")

    def test_main_success(self) -> None:
        """Test main() with successful execution."""
        from dataeval_app.__main__ import main

        with (
            patch("sys.argv", ["dataeval_app"]),
            patch("dataeval_app.__main__._run_tasks"),
        ):
            main()

    def test_main_file_not_found(self) -> None:
        """Test main() with FileNotFoundError."""
        from dataeval_app.__main__ import main

        with (
            patch("sys.argv", ["dataeval_app"]),
            patch("dataeval_app.__main__._run_tasks", side_effect=FileNotFoundError("Not found")),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


@pytest.mark.optional
class TestMetadata:
    """Test the metadata output module [IR-3-H-12]."""

    def test_write_metadata_creates_file(self, tmp_path: Path) -> None:
        """Test that write_metadata creates metadata.json file."""
        from dataeval_app._jatic_metadata import write_metadata

        output_dir = tmp_path / "output"
        result = write_metadata(output_dir, "test_dataset", {"accuracy": 0.95})

        assert result.exists()
        assert result.name == "metadata.json"

    def test_write_metadata_content(self, tmp_path: Path) -> None:
        """Test that metadata.json contains expected fields."""
        import json

        from dataeval_app._jatic_metadata import write_metadata

        output_dir = tmp_path / "output"
        result = write_metadata(output_dir, "cifar10", {"score": 0.9})

        content = json.loads(result.read_text())
        assert content["dataset_id"] == "cifar10"
        assert content["tool"] == "dataeval-app"
        assert content["results"] == {"score": 0.9}
        assert "timestamp" in content
        assert "version" in content
        assert "tool_version" in content

    def test_write_metadata_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that write_metadata creates parent directories if needed."""
        from dataeval_app._jatic_metadata import write_metadata

        nested_dir = tmp_path / "deeply" / "nested" / "output"
        result = write_metadata(nested_dir, "test", {})

        assert result.exists()
        assert nested_dir.exists()


@pytest.mark.optional
class TestContainerRun:
    """Test the container landing script."""

    def test_get_dataset_path_default(self) -> None:
        """Test default container mount path."""
        from container_run import get_dataset_path

        with patch.dict("os.environ", {}, clear=True):
            path = get_dataset_path()
            assert path == Path("/data/dataset")

    def test_get_dataset_path_env_override(self) -> None:
        """Test environment variable override."""
        from container_run import get_dataset_path

        with patch.dict("os.environ", {"DATASET_PATH": "/custom/path"}):
            path = get_dataset_path()
            assert path == Path("/custom/path")

    def test_get_output_path_default(self) -> None:
        """Test default output mount path."""
        from container_run import get_output_path

        with patch.dict("os.environ", {}, clear=True):
            path = get_output_path()
            assert path == Path("/output")

    def test_get_output_path_env_override(self) -> None:
        """Test output path environment override."""
        from container_run import get_output_path

        with patch.dict("os.environ", {"OUTPUT_PATH": "/custom/output"}):
            path = get_output_path()
            assert path == Path("/custom/output")

    def test_main_config_not_found(self) -> None:
        """Test main() when config path doesn't exist."""
        from container_run import main

        result = main()
        assert result == 1
