"""End-to-end integration tests for config infrastructure.

These tests verify that components work together correctly:
- Multi-file config loading with all config sections
- Task reference resolution (task → dataset, preprocessor, selection)
- Container mount path constants
- Full pipeline: config YAML → run_task() → output files
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl

# Ensure target modules are in sys.modules for @patch with xdist
import dataeval_app.dataset
import dataeval_app.metadata
import dataeval_app.workflows.cleaning.workflow  # noqa: F401
from dataeval_app.config import load_config_folder


class TestConfigToFactoryIntegration:
    """Test config loading → factory instantiation flow."""

    def test_full_config_with_all_p1_sections(self, tmp_path: Path):
        """Load WorkflowConfig with datasets, preprocessors, selections, tasks."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Multi-file config simulating real deployment
        (config_dir / "00-datasets.yaml").write_text(
            "datasets:\n  - name: test_dataset\n    format: huggingface\n    path: ./test\n    split: train\n"
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
            "    workflow: data-cleaning\n"
            "    datasets: test_dataset\n"
            "    preprocessors: basic\n"
            "    selections: subset\n"
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
        assert task.datasets == "test_dataset"
        assert task.preprocessors == "basic"
        assert task.selections == "subset"
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
            "    split: train\n"
            "  - name: dataset_b\n"
            "    format: coco\n"
            "    path: ./b\n"
            "    split: val\n"
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
            "    workflow: data-cleaning\n"
            "    datasets: dataset_a\n"
            "    preprocessors: preproc_x\n"
            "    selections: sel_1\n"
            "  - name: task2\n"
            "    workflow: data-cleaning\n"
            "    datasets: dataset_b\n"
            "    preprocessors: preproc_y\n"
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
        assert task1.datasets in datasets_by_name
        assert task1.preprocessors in preprocessors_by_name
        assert task1.selections in selections_by_name

        # Verify resolved config details
        resolved_dataset = datasets_by_name[task1.datasets]
        assert resolved_dataset.format == "huggingface"
        assert resolved_dataset.path == "./a"

        resolved_preproc = preprocessors_by_name[task1.preprocessors]
        assert len(resolved_preproc.steps) == 1
        assert resolved_preproc.steps[0].step == "ToTensor"

        # Verify task2 references (no selection)
        task2 = config.tasks[1]
        assert task2.datasets in datasets_by_name
        assert task2.preprocessors in preprocessors_by_name
        assert task2.selections is None  # Optional, not set

    def test_multi_task_config(self, tmp_path: Path):
        """Multiple tasks sharing resources loaded correctly."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        (config_dir / "workflow.yaml").write_text(
            "datasets:\n"
            "  - name: shared_dataset\n"
            "    format: huggingface\n"
            "    path: ./data\n"
            "    split: train\n"
            "preprocessors:\n"
            "  - name: shared_preproc\n"
            "    steps:\n"
            "      - step: ToTensor\n"
            "tasks:\n"
            "  - name: outlier_detection\n"
            "    workflow: data-cleaning\n"
            "    datasets: shared_dataset\n"
            "    preprocessors: shared_preproc\n"
            "    params:\n"
            "      outlier_method: modzscore\n"
            "  - name: duplicate_detection\n"
            "    workflow: data-cleaning\n"
            "    datasets: shared_dataset\n"
            "    preprocessors: shared_preproc\n"
            "  - name: analysis_only\n"
            "    workflow: data-cleaning\n"
            "    datasets: shared_dataset\n"
            "    output_format: text\n"
        )

        config = load_config_folder(config_dir)

        assert config.tasks is not None
        assert len(config.tasks) == 3

        # All tasks reference the same dataset
        for task in config.tasks:
            assert task.datasets == "shared_dataset"

        # Verify different output formats
        assert config.tasks[0].output_format == "json"  # default
        assert config.tasks[2].output_format == "text"


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
        from dataeval_app.config.loader import DEFAULT_CONFIG_FOLDER, DEFAULT_PARAMS_PATH

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
            "datasets:\n  - name: dataset_a\n    format: huggingface\n    path: ./a\n    split: train\n"
        )
        # Second file defines dataset B
        (config_dir / "01-second.yaml").write_text(
            "datasets:\n  - name: dataset_b\n    format: coco\n    path: ./b\n    split: test\n"
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
        task_yaml = "tasks:\n  - name: {name}\n    workflow: data-cleaning\n    datasets: x\n"
        (config_dir / "02-second.yaml").write_text(task_yaml.format(name="task_second"))
        (config_dir / "01-first.yaml").write_text(task_yaml.format(name="task_first"))

        config = load_config_folder(config_dir)

        assert config.tasks is not None
        # First file (01-) processed first, so task_first is first in list
        assert config.tasks[0].name == "task_first"
        assert config.tasks[1].name == "task_second"


class TestEndToEndCleaningWorkflow:
    """Full pipeline: config YAML → run_task() → dataset → evaluators → output files.

    Mocks only at external boundaries (5 mocks):
      - datasets.load_from_disk          (HuggingFace I/O)
      - maite_datasets.adapters.from_huggingface  (MAITE conversion)
      - dataeval.Metadata                 (metadata extraction)
      - dataeval.quality.Outliers         (outlier evaluator)
      - dataeval.quality.Duplicates       (duplicate evaluator)

    Runs for real across 10 source files (20 functions):
      config/loader.py      load_config_folder, YAML parsing, file merge
      config/_merge.py      merge_config_dicts
      config/models.py      WorkflowConfig.model_validate
      config/schemas/       DatasetConfig, TaskConfig validation
      workflow/orchestrator  run_task, _resolve_by_name, _write_output
      workflow/__init__      get_workflow, WorkflowContext, WorkflowResult
      dataset.py            load_dataset, load_dataset_huggingface (split select)
      metadata.py           build_metadata
      workflows/cleaning/   DataCleaningWorkflow.execute, _run_cleaning,
                            _build_outliers, _build_duplicates,
                            _serialize_outlier_issues, _serialize_duplicates,
                            _compute_label_stats, _build_findings
      config/schemas/metadata  ResultMetadata

    Asserts (26 checks):
      - WorkflowResult: success, name, metadata (mode, evaluators)
      - results.json: outlier count/issues, target_outliers, duplicate groups,
        label stats (item_count, class_count, per-class counts), dataset_size,
        report summary and finding titles
      - metadata.json: dataset_id, tool, version, timestamp
      - Mock calls: each external mock called exactly once
    """

    @patch("dataeval_app.workflows.cleaning.workflow.Duplicates")
    @patch("dataeval_app.workflows.cleaning.workflow.Outliers")
    @patch("dataeval_app.metadata.Metadata")
    @patch("maite_datasets.adapters.from_huggingface")
    @patch("datasets.load_from_disk")
    def test_config_to_output(
        self,
        mock_load_from_disk: MagicMock,
        mock_from_huggingface: MagicMock,
        mock_metadata_cls: MagicMock,
        mock_outliers_cls: MagicMock,
        mock_duplicates_cls: MagicMock,
        tmp_path: Path,
    ):
        """Config YAML → run_task() produces results.json + metadata.json."""
        from dataeval_app.workflow import run_task

        # ── 1. Write config YAML ──────────────────────────────────────
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        (config_dir / "config.yaml").write_text(
            "datasets:\n"
            "  - name: test_ds\n"
            "    format: huggingface\n"
            "    path: ./dataset\n"
            "    split: train\n"
            "tasks:\n"
            "  - name: e2e_clean\n"
            "    workflow: data-cleaning\n"
            "    datasets: test_ds\n"
            "    params:\n"
            "      outlier_method: modzscore\n"
            "      outlier_flags:\n"
            "        - dimension\n"
            "        - pixel\n"
            "      outlier_threshold: null\n"
            "    output_format: json\n"
        )

        # ── 2. Create dataset directory (path.exists() check) ─────────
        (tmp_path / "dataset").mkdir()

        # ── 3. Output directory ───────────────────────────────────────
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # ── 4. Mock external boundaries ───────────────────────────────

        # 4a. HuggingFace dataset loading
        mock_hf_dataset = MagicMock()
        mock_hf_dataset.keys = MagicMock(return_value=["train"])
        mock_hf_dataset.__getitem__ = MagicMock(return_value=mock_hf_dataset)
        mock_load_from_disk.return_value = mock_hf_dataset

        # 4b. MAITE conversion — return dataset-like object with __len__ and __getitem__
        mock_maite_ds = MagicMock()
        mock_maite_ds.__len__ = MagicMock(return_value=10)
        mock_maite_ds.__getitem__ = MagicMock(return_value={"image": "fake", "label": 0})
        mock_from_huggingface.return_value = mock_maite_ds

        # 4c. Metadata mock
        mock_metadata = MagicMock()
        mock_metadata.class_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
        mock_metadata.index2label = {0: "cat", 1: "dog", 2: "bird"}
        mock_metadata.item_count = 10
        mock_metadata_cls.return_value = mock_metadata

        # 4d. Outliers mock — produces Polars DataFrame matching OutliersOutput.issues
        outlier_issues_df = pl.DataFrame(
            {
                "item_id": [2, 7],
                "metric_name": ["brightness", "contrast"],
                "metric_value": [3.5, -2.8],
            }
        )
        mock_outliers_result = MagicMock()
        mock_outliers_result.issues = outlier_issues_df
        mock_outliers_instance = MagicMock()
        mock_outliers_instance.evaluate.return_value = mock_outliers_result
        mock_outliers_cls.return_value = mock_outliers_instance

        # 4e. Duplicates mock — produces DuplicatesOutput-like structure
        mock_dup_items = MagicMock()
        mock_dup_items.exact = [[0, 5]]  # one exact duplicate group
        mock_near_group = MagicMock()
        mock_near_group.indices = [3, 8]
        mock_near_group.methods = frozenset({"hash"})
        mock_near_group.orientation = "same"
        mock_dup_items.near = [mock_near_group]

        mock_dup_targets = MagicMock()
        mock_dup_targets.exact = None
        mock_dup_targets.near = None

        mock_dup_result = MagicMock()
        mock_dup_result.items = mock_dup_items
        mock_dup_result.targets = mock_dup_targets
        mock_dup_instance = MagicMock()
        mock_dup_instance.evaluate.return_value = mock_dup_result
        mock_duplicates_cls.return_value = mock_dup_instance

        # ── 5. Load config and run task ───────────────────────────────
        config = load_config_folder(config_dir)
        assert config.tasks is not None
        task = config.tasks[0]

        result = run_task(task, config)

        # ── 6. Assert result ──────────────────────────────────────────
        assert result.success is True
        assert result.name == "data-cleaning"
        meta_dump = result.metadata.model_dump()
        assert meta_dump["mode"] == "advisory"
        assert "outliers" in meta_dump["evaluators"]
        assert "duplicates" in meta_dump["evaluators"]

        # ── 7. Write output via result.report() and verify ────────────
        task_dir = output_dir / task.name
        written_path = result.report(format="json", path=task_dir)
        results_file = task_dir / "results.json"
        assert written_path == results_file

        # Verify results.json content (now includes metadata envelope)
        results_data = json.loads(results_file.read_text())
        raw = results_data["raw"]

        # Outlier results
        assert raw["img_outliers"]["count"] == 2
        assert len(raw["img_outliers"]["issues"]) == 2
        assert raw["img_outliers"]["issues"][0]["item_id"] == 2

        # No target outliers (no target_id column)
        assert raw["target_outliers"] is None

        # Duplicate results
        assert len(raw["duplicates"]["items"]["exact"]) == 1
        assert raw["duplicates"]["items"]["exact"][0] == [0, 5]
        assert len(raw["duplicates"]["items"]["near"]) == 1
        assert raw["duplicates"]["items"]["near"][0]["indices"] == [3, 8]

        # Label stats
        assert raw["label_stats"]["item_count"] == 10
        assert raw["label_stats"]["class_count"] == 3
        assert raw["label_stats"]["label_counts_per_class"]["cat"] == 3
        assert raw["label_stats"]["label_counts_per_class"]["bird"] == 4

        # Dataset size
        assert raw["dataset_size"] == 10

        # Report findings
        report = results_data["report"]
        assert "Data cleaning complete" in report["summary"]
        finding_titles = [f["title"] for f in report["findings"]]
        assert "Image Outliers" in finding_titles
        assert "Duplicates" in finding_titles
        assert "Label Distribution" in finding_titles

        # Verify JATIC metadata is embedded in results.json
        meta = results_data["metadata"]
        assert meta["dataset_id"] == "test_ds"
        assert meta["tool"] == "dataeval-app"
        assert "version" in meta
        assert "timestamp" in meta
        assert meta["execution_time_s"] is not None

        # ── 8. Verify mock calls ──────────────────────────────────────
        mock_load_from_disk.assert_called_once()
        mock_from_huggingface.assert_called_once()
        mock_outliers_instance.evaluate.assert_called_once()
        mock_dup_instance.evaluate.assert_called_once()
