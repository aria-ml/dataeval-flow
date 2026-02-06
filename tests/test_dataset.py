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
        from dataeval_app import inspect_dataset, load_dataset, load_dataset_huggingface

        assert callable(load_dataset)
        assert callable(load_dataset_huggingface)
        assert callable(inspect_dataset)


@pytest.mark.required
class TestDatasetModule:
    """Test the dataset library module."""

    def test_load_dataset_huggingface_datasetdict(self, mock_hf_dataset: MagicMock) -> None:
        """Test loading a DatasetDict uses first split automatically."""
        with (
            patch("datasets.load_from_disk") as mock_load,
            patch("maite_datasets.adapters.from_huggingface") as mock_adapter,
        ):
            mock_load.return_value = mock_hf_dataset
            mock_adapter.return_value = MagicMock()

            from dataeval_app import load_dataset_huggingface

            load_dataset_huggingface(Path("dummy/path"))

            # Should use first available split
            mock_hf_dataset.__getitem__.assert_called_with("train")
            mock_adapter.assert_called()

    def test_load_dataset_delegates(self) -> None:
        """Test load_dataset delegates to load_dataset_huggingface."""
        with patch("dataeval_app.dataset.load_dataset_huggingface") as mock_hf:
            mock_hf.return_value = MagicMock()

            from dataeval_app.dataset import load_dataset

            load_dataset(Path("/some/path"))
            mock_hf.assert_called_once_with(Path("/some/path"))

    def test_inspect_dataset_path_not_found(self) -> None:
        """Ensure FileNotFoundError is raised for missing path."""
        from dataeval_app import inspect_dataset

        with pytest.raises(FileNotFoundError):
            inspect_dataset(Path("/nonexistent/path"))

    def test_inspect_dataset_success(self, mock_dataset_path: Path, mock_maite_dataset: MagicMock) -> None:
        """Test successful dataset inspection."""
        with (
            patch("datasets.load_from_disk") as mock_load,
            patch("maite_datasets.adapters.from_huggingface") as mock_adapter,
        ):
            # Setup: single-split dataset (no keys method)
            mock_ds = MagicMock(spec=[])  # No keys attribute
            mock_load.return_value = mock_ds
            mock_adapter.return_value = mock_maite_dataset

            from dataeval_app import inspect_dataset

            result = inspect_dataset(mock_dataset_path)

            assert result == 0
            mock_load.assert_called_once()
            mock_adapter.assert_called_once()

    def test_inspect_dataset_non_tuple_item(self, mock_dataset_path: Path) -> None:
        """Test inspection handles non-tuple items gracefully."""
        mock_maite = MagicMock()
        mock_maite.__len__ = MagicMock(return_value=1)
        mock_maite.__getitem__ = MagicMock(return_value="not a tuple")
        mock_maite.metadata = {}

        with (
            patch("datasets.load_from_disk") as mock_load,
            patch("maite_datasets.adapters.from_huggingface") as mock_adapter,
        ):
            mock_ds = MagicMock(spec=[])
            mock_load.return_value = mock_ds
            mock_adapter.return_value = mock_maite

            from dataeval_app import inspect_dataset

            result = inspect_dataset(mock_dataset_path)
            assert result == 0

    def test_inspect_dataset_item_exception(self, mock_dataset_path: Path) -> None:
        """Test inspection handles item access exceptions gracefully."""
        mock_maite = MagicMock()
        mock_maite.__len__ = MagicMock(return_value=1)
        mock_maite.__getitem__ = MagicMock(side_effect=RuntimeError("Item error"))
        mock_maite.metadata = {}

        with (
            patch("datasets.load_from_disk") as mock_load,
            patch("maite_datasets.adapters.from_huggingface") as mock_adapter,
        ):
            mock_ds = MagicMock(spec=[])
            mock_load.return_value = mock_ds
            mock_adapter.return_value = mock_maite

            from dataeval_app import inspect_dataset

            result = inspect_dataset(mock_dataset_path)
            assert result == 0


@pytest.mark.required
class TestMainModule:
    """Test the __main__.py CLI module."""

    def test_parse_args_required(self) -> None:
        """Test that --dataset-path is required."""
        from dataeval_app.__main__ import parse_args

        with pytest.raises(SystemExit), patch("sys.argv", ["dataeval_app"]):
            parse_args()

    def test_parse_args_with_path(self) -> None:
        """Test parsing with dataset path."""
        from dataeval_app.__main__ import parse_args

        with patch("sys.argv", ["dataeval_app", "--dataset-path", "/some/path"]):
            args = parse_args()
            assert args.dataset_path == Path("/some/path")

    def test_main_success(self, mock_dataset_path: Path) -> None:
        """Test main() with successful execution."""
        from dataeval_app.__main__ import main

        with (
            patch("sys.argv", ["dataeval_app", "--dataset-path", str(mock_dataset_path)]),
            patch("dataeval_app.__main__.inspect_dataset", return_value=0),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_file_not_found(self) -> None:
        """Test main() with FileNotFoundError."""
        from dataeval_app.__main__ import main

        with (
            patch("sys.argv", ["dataeval_app", "--dataset-path", "/nonexistent"]),
            patch("dataeval_app.__main__.inspect_dataset", side_effect=FileNotFoundError("Not found")),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


@pytest.mark.optional
class TestMetadata:
    """Test the metadata output module [IR-3-H-12]."""

    def test_write_metadata_creates_file(self, tmp_path: Path) -> None:
        """Test that write_metadata creates metadata.json file."""
        from dataeval_app._metadata import write_metadata

        output_dir = tmp_path / "output"
        result = write_metadata(output_dir, "test_dataset", {"accuracy": 0.95})

        assert result.exists()
        assert result.name == "metadata.json"

    def test_write_metadata_content(self, tmp_path: Path) -> None:
        """Test that metadata.json contains expected fields."""
        import json

        from dataeval_app._metadata import write_metadata

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
        from dataeval_app._metadata import write_metadata

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

    def test_main_path_not_found(self) -> None:
        """Test main() when dataset path doesn't exist."""
        from container_run import main

        with patch.dict("os.environ", {"DATASET_PATH": "/nonexistent/path"}):
            result = main()
            assert result == 1

    def test_main_success(self, mock_dataset_path: Path) -> None:
        """Test main() successful execution."""
        from container_run import main

        with (
            patch.dict("os.environ", {"DATASET_PATH": str(mock_dataset_path)}),
            patch("dataeval_app.inspect_dataset", return_value=0) as mock_inspect,
        ):
            result = main()
            assert result == 0
            mock_inspect.assert_called_once()
