"""End-to-end integration tests for P1 config infrastructure.

These tests verify that components work together correctly:
- Multi-file config loading with all P1 sections
- Task reference resolution (task → dataset, preprocessor, selection)
- Container mount path constants

Note: True task execution tests require P2 TaskExecutor (not yet implemented).
"""

from pathlib import Path

from dataeval_app.ingest import load_config_folder


class TestConfigToFactoryIntegration:
    """Test config loading → factory instantiation flow."""

    def test_full_config_with_all_p1_sections(self, tmp_path: Path):
        """Load WorkflowConfig with datasets, preprocessors, selections, tasks."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Multi-file config simulating real deployment
        (config_dir / "00-datasets.yaml").write_text(
            "datasets:\n"
            "  - name: test_dataset\n"
            "    format: huggingface\n"
            "    path: ./test\n"
            "    splits:\n"
            "      - train\n"
            "      - test\n"
        )
        (config_dir / "01-preprocessors.yaml").write_text(
            "preprocessors:\n  - name: basic\n    steps:\n      - step: ToTensor\n"
        )
        (config_dir / "02-selections.yaml").write_text(
            "selections:\n  - name: subset\n    steps:\n      - type: Limit\n        params:\n          size: 100\n"
        )
        (config_dir / "03-tasks.yaml").write_text(
            "tasks:\n"
            "  - name: clean_task\n"
            "    dataset: test_dataset\n"
            "    preprocessor: basic\n"
            "    selection: subset\n"
            "    params:\n"
            "      outlier_method: modzscore\n"
            "    output_format: json\n"
        )

        config = load_config_folder(config_dir)

        # Verify all sections loaded
        assert config.datasets is not None
        assert config.preprocessors is not None
        assert config.selections is not None
        assert config.tasks is not None

        # Verify counts
        assert len(config.datasets) == 1
        assert len(config.preprocessors) == 1
        assert len(config.selections) == 1
        assert len(config.tasks) == 1

        # Verify task content
        task = config.tasks[0]
        assert task.name == "clean_task"
        assert task.dataset == "test_dataset"
        assert task.preprocessor == "basic"
        assert task.selection == "subset"
        assert task.params == {"outlier_method": "modzscore"}
        assert task.output_format == "json"

    def test_task_reference_resolution(self, tmp_path: Path):
        """Verify task references can be resolved to their configs."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        (config_dir / "config.yaml").write_text(
            "datasets:\n"
            "  - name: dataset_a\n"
            "    format: huggingface\n"
            "    path: ./a\n"
            "    splits: [train]\n"
            "  - name: dataset_b\n"
            "    format: coco\n"
            "    path: ./b\n"
            "    splits: [val]\n"
            "preprocessors:\n"
            "  - name: preproc_x\n"
            "    steps:\n"
            "      - step: ToTensor\n"
            "  - name: preproc_y\n"
            "    steps:\n"
            "      - step: Resize\n"
            "        params:\n"
            "          size: [224, 224]\n"
            "selections:\n"
            "  - name: sel_1\n"
            "    steps:\n"
            "      - type: Limit\n"
            "        params:\n"
            "          size: 50\n"
            "tasks:\n"
            "  - name: task1\n"
            "    dataset: dataset_a\n"
            "    preprocessor: preproc_x\n"
            "    selection: sel_1\n"
            "  - name: task2\n"
            "    dataset: dataset_b\n"
            "    preprocessor: preproc_y\n"
        )

        config = load_config_folder(config_dir)

        # Build lookup dicts
        datasets_by_name = {d.name: d for d in config.datasets or []}
        preprocessors_by_name = {p.name: p for p in config.preprocessors or []}
        selections_by_name = {s.name: s for s in config.selections or []}

        # Verify tasks loaded
        assert config.tasks is not None

        # Verify task1 references resolve
        task1 = config.tasks[0]
        assert task1.dataset in datasets_by_name
        assert task1.preprocessor in preprocessors_by_name
        assert task1.selection in selections_by_name

        # Verify resolved config details
        resolved_dataset = datasets_by_name[task1.dataset]
        assert resolved_dataset.format == "huggingface"
        assert resolved_dataset.path == "./a"

        resolved_preproc = preprocessors_by_name[task1.preprocessor]
        assert len(resolved_preproc.steps) == 1
        assert resolved_preproc.steps[0].step == "ToTensor"

        # Verify task2 references (no selection)
        task2 = config.tasks[1]
        assert task2.dataset in datasets_by_name
        assert task2.preprocessor in preprocessors_by_name
        assert task2.selection is None  # Optional, not set

    def test_multi_task_config(self, tmp_path: Path):
        """Multiple tasks sharing resources loaded correctly."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        (config_dir / "workflow.yaml").write_text(
            "datasets:\n"
            "  - name: shared_dataset\n"
            "    format: huggingface\n"
            "    path: ./data\n"
            "    splits: [train, test]\n"
            "preprocessors:\n"
            "  - name: shared_preproc\n"
            "    steps:\n"
            "      - step: ToTensor\n"
            "tasks:\n"
            "  - name: outlier_detection\n"
            "    dataset: shared_dataset\n"
            "    preprocessor: shared_preproc\n"
            "    params:\n"
            "      outlier_method: modzscore\n"
            "  - name: duplicate_detection\n"
            "    dataset: shared_dataset\n"
            "    preprocessor: shared_preproc\n"
            "  - name: analysis_only\n"
            "    dataset: shared_dataset\n"
            "    output_format: terminal\n"
        )

        config = load_config_folder(config_dir)

        assert config.tasks is not None
        assert len(config.tasks) == 3

        # All tasks reference the same dataset
        for task in config.tasks:
            assert task.dataset == "shared_dataset"

        # Verify different output formats
        assert config.tasks[0].output_format == "json"  # default
        assert config.tasks[2].output_format == "terminal"


class TestContainerMountPaths:
    """Test container mount path constants and alignment."""

    def test_container_mounts_defined(self):
        """CONTAINER_MOUNTS has all expected paths."""
        import sys

        # Add src to path for container_run import
        src_path = Path(__file__).parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from container_run import CONTAINER_MOUNTS

        assert CONTAINER_MOUNTS["config"] == Path("/data/config")
        assert CONTAINER_MOUNTS["dataset"] == Path("/data/dataset")
        assert CONTAINER_MOUNTS["model"] == Path("/data/model")
        assert CONTAINER_MOUNTS["output"] == Path("/output")

    def test_default_paths_match_container_mounts(self):
        """DEFAULT_CONFIG_FOLDER and DEFAULT_PARAMS_PATH align with container mounts."""
        from dataeval_app.ingest.params import DEFAULT_CONFIG_FOLDER, DEFAULT_PARAMS_PATH

        # Verify paths use /data/config pattern
        assert Path("/data/config") == DEFAULT_CONFIG_FOLDER
        assert Path("/data/config/params.yaml") == DEFAULT_PARAMS_PATH

        # Verify params.yaml is inside config folder
        assert DEFAULT_PARAMS_PATH.parent == DEFAULT_CONFIG_FOLDER

    def test_all_mount_points_follow_pattern(self):
        """All data mounts follow /data/<purpose>/ pattern except /output."""
        import sys

        src_path = Path(__file__).parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from container_run import CONTAINER_MOUNTS

        for name, path in CONTAINER_MOUNTS.items():
            if name == "output":
                assert path == Path("/output")
            else:
                assert str(path).startswith("/data/"), f"{name} should be under /data/"


class TestConfigMergeBehavior:
    """Test multi-file config merging behavior."""

    def test_list_sections_extend_not_replace(self, tmp_path: Path):
        """Lists from multiple files are extended, not replaced."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # First file defines dataset A
        (config_dir / "00-first.yaml").write_text(
            "datasets:\n  - name: dataset_a\n    format: huggingface\n    path: ./a\n    splits: [train]\n"
        )
        # Second file defines dataset B
        (config_dir / "01-second.yaml").write_text(
            "datasets:\n  - name: dataset_b\n    format: coco\n    path: ./b\n    splits: [test]\n"
        )

        config = load_config_folder(config_dir)

        # Both datasets should be present (extended, not replaced)
        assert config.datasets is not None
        assert len(config.datasets) == 2
        names = [d.name for d in config.datasets]
        assert "dataset_a" in names
        assert "dataset_b" in names

    def test_alphabetical_file_order(self, tmp_path: Path):
        """Files are processed in alphabetical order."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Create files with explicit ordering
        (config_dir / "02-second.yaml").write_text("tasks:\n  - name: task_second\n    dataset: x\n")
        (config_dir / "01-first.yaml").write_text("tasks:\n  - name: task_first\n    dataset: x\n")

        config = load_config_folder(config_dir)

        assert config.tasks is not None
        # First file (01-) processed first, so task_first is first in list
        assert config.tasks[0].name == "task_first"
        assert config.tasks[1].name == "task_second"
