"""data-cleaning as a preset: its chain, what its task returns, and the cleaned Dataset it hands on (spec §10).

Measured on the toy datasets: 12 toy images hold one outlier, of class b, and one exact-duplicate pair, so `clean`
removes 2 and keeps 10; 24 toy images keep 22. Read through a view that shuffles them (seed 3) and keeps 16, the 24
hold nothing to remove.
"""

import re
from typing import Any

import pytest
from dataeval.data import View
from pydantic import ValidationError

import dataeval_flow._cache as cache_module
from dataeval_flow import PipelineConfig, run, run_task, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.config import DatasetProtocolConfig, SourceConfig, TaskConfig, ViewConfig, ViewOperation
from dataeval_flow.config.extractors import FlattenExtractorConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig, LabelHealthConfig, OutliersConfig
from dataeval_flow.steps import ChainResult, list_steps
from dataeval_flow.steps._port import DataType
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig, DataCleaningWorkflow
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages
from tests.example_plugin import MeanConfig

_BASE: dict[str, Any] = {
    "name": "cleaning",
    "type": "data-cleaning",
    "outlier_method": "zscore",
    "outlier_flags": ["pixel", "visual"],
}
_STEPS = [
    "outliers",
    "labels",
    "by_class",
    "dupes",
    "image_outliers",
    "target_outliers",
    "classwise",
    "duplicates",
    "imbalance",
    "clean",
]


@pytest.fixture(autouse=True)
def _fresh_caches():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _task(dataset: Any, **settings: Any) -> ChainResult:
    config = chain_pipeline(
        workflows=[{**_BASE, **settings}],
        tasks=[{"name": "t", "workflow": "cleaning", "sources": ["src"]}],
        datasets={"src": dataset},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


def _verdicts(result: ChainResult) -> list[tuple[str, str, str | None, str | None]]:
    return [(f.severity, f.title, f.brief, f.step) for f in result.findings]


def test_a_data_cleaning_task_returns_a_chain_result_of_its_steps() -> None:
    result = _task(ToyImages(count=12))
    assert result.type == "data-cleaning"
    assert list(result.steps) == _STEPS
    assert _verdicts(result) == [
        ("warning", "Image Outliers", "1 images (8.3%)", "image_outliers"),
        ("warning", "Classwise Outliers", "worst: b (16.7%), 1/1 classes over 3.0%", "classwise"),
        ("warning", "Duplicates", "2 exact (16.7%), 0 near (0.0%)", "duplicates"),
        ("info", "Label Distribution", "2 classes, 12 items, imbalance 1.0:1", "imbalance"),
    ]


def test_clean_removes_each_flagged_item_and_each_duplicate_but_the_first() -> None:
    clean = _task(ToyImages(count=12)).steps["clean"]
    assert len(clean.output) == 10
    assert clean.details == {"removed": {"items": 2, "detections": 0, "tracks": 0, "frames": 0}}


def test_run_returns_the_chain_and_the_cleaned_dataset() -> None:
    result = run(DataCleaningConfig(outlier_method="zscore", outlier_flags=["pixel", "visual"]), ToyImages(count=12))
    assert isinstance(result, ChainResult)
    clean = result.steps["clean"].output
    assert isinstance(clean, View)
    assert len(clean) == 10


def test_a_null_threshold_judges_nothing() -> None:
    result = _task(ToyImages(count=12), health_thresholds={"image_outliers": None})
    assert _verdicts(result)[0] == ("info", "Image Outliers", "1 images (8.3%)", "image_outliers")


def test_a_data_cleaning_task_reads_its_source_through_its_view() -> None:
    task = TaskConfig(name="t", workflow="cleaning", sources="src")
    config = chain_pipeline(
        workflows=[_BASE],
        tasks=[{"name": "t", "workflow": "cleaning", "sources": ["src"]}],
        datasets={"src": ToyImages(count=24)},
    )
    view = ViewConfig(
        name="shuffled",
        operations=[
            ViewOperation(type="Shuffle", params={"seed": 3}),
            ViewOperation(type="Limit", params={"size": 16}),
        ],
    )
    source = config.sources[0].model_copy(update={"view": "shuffled"})  # type: ignore[index]
    result = run_task(task, config.model_copy(update={"views": [view], "sources": [source]}))
    assert isinstance(result, ChainResult)
    assert result.metadata.lineage[0].items == 16
    assert ("info", "Label Distribution", "2 classes, 16 items, imbalance 1.0:1", "imbalance") in _verdicts(result)
    assert len(result.steps["clean"].output) == 16


def test_a_data_cleaning_step_cleans_each_split_of_a_list() -> None:
    config = chain_pipeline(
        workflows=[
            _BASE,
            {
                "name": "outer",
                "inputs": [{"name": "splits", "list": True}],
                "steps": [
                    {"name": "cleaning", "workflow": "cleaning", "input": "splits"},
                    {"name": "sizes", "evaluator": "labels", "input": "cleaning.clean"},
                ],
            },
        ],
        evaluators=[{"name": "labels", "type": "quality.label-health"}],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["s1", "s2"]}],
        datasets={"s1": ToyImages(count=12), "s2": ToyImages(count=24)},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert list(result.steps) == [*(f"cleaning/{step}" for step in _STEPS), "sizes"]
    assert _verdicts(result) == [
        ("warning", "Image Outliers", "1 images (8.3%)", "cleaning/image_outliers[s1]"),
        ("warning", "Image Outliers", "1 images (4.2%)", "cleaning/image_outliers[s2]"),
        ("warning", "Classwise Outliers", "worst: b (16.7%), 1/1 classes over 3.0%", "cleaning/classwise[s1]"),
        ("warning", "Classwise Outliers", "worst: b (8.3%), 1/1 classes over 3.0%", "cleaning/classwise[s2]"),
        ("warning", "Duplicates", "2 exact (16.7%), 0 near (0.0%)", "cleaning/duplicates[s1]"),
        ("warning", "Duplicates", "2 exact (8.3%), 0 near (0.0%)", "cleaning/duplicates[s2]"),
        ("info", "Label Distribution", "2 classes, 12 items, imbalance 1.0:1", "cleaning/imbalance[s1]"),
        ("info", "Label Distribution", "2 classes, 24 items, imbalance 1.0:1", "cleaning/imbalance[s2]"),
    ]
    clean = result.steps["cleaning/clean"].elements or {}
    assert {key: len(element.output) for key, element in clean.items()} == {"s1": 10, "s2": 22}
    assert result.steps["sizes"].inputs == ["cleaning.clean"]
    assert list(result.steps["sizes"].elements or {}) == ["s1", "s2"]


def test_the_settings_become_the_chain_s_evaluators_and_thresholds() -> None:
    config = DataCleaningConfig(
        outlier_method="modzscore",
        outlier_flags=["dimension"],
        outlier_threshold=4.0,
        duplicate_flags=["hash_d4"],
        duplicate_merge_near=False,
        metadata="policy",
        stats="measured",
        health_thresholds={"near_duplicates": 9.0, "class_label_imbalance": None},  # type: ignore[arg-type]
    )
    chain = DataCleaningWorkflow.chain(config)
    outliers, dupes, labels = chain.evaluators
    assert isinstance(outliers, OutliersConfig)
    assert isinstance(dupes, DuplicatesConfig)
    assert isinstance(labels, LabelHealthConfig)
    assert (outliers.name, outliers.flags, outliers.outlier_threshold, outliers.per_target, outliers.stats) == (
        "outliers",
        ["dimension"],
        ("modzscore", 4.0),
        True,
        "measured",
    )
    assert (dupes.name, dupes.flags, dupes.merge_near_duplicates, dupes.stats) == (
        "dupes",
        ["hash_d4"],
        False,
        "measured",
    )
    assert (labels.name, labels.metadata) == ("labels", "policy")
    steps: dict[str, Any] = {step["name"]: step for step in chain.steps}  # type: ignore[index]
    assert (steps["duplicates"]["exact"], steps["duplicates"]["near"], steps["imbalance"]["ratio"]) == (0.0, 9.0, None)


@pytest.mark.parametrize(
    ("field", "value"), [("value_range", [0.0, 255.0]), ("metadata_auto_bin_method", "uniform_width")]
)
def test_a_retired_field_is_refused(field: str, value: Any) -> None:
    with pytest.raises(ValidationError, match=re.escape(field)):
        DataCleaningConfig.model_validate({"outlier_method": "zscore", "outlier_flags": ["pixel"], field: value})


def test_the_catalog_lists_the_cleaned_dataset_as_data_cleaning_s_output() -> None:
    entry = next(e for e in list_steps(plugins=False).steps if e.type == "data-cleaning")
    assert [(port.port, port.type) for port in entry.outputs] == [("clean", DataType.DATASET)]


class TestClustersFollowTheirExtractor:
    """A cleaning run's clusters are keyed by the extractor whose embeddings they cluster."""

    def test_two_stateless_extractors_on_one_source_cluster_apart(
        self, plugins: dict[str, list[tuple[str, str]]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each extractor's clusters are computed from its own embeddings, keyed by it, and cached for it alone."""
        plugins["dataeval_flow.extractors"] = [("example.mean", "tests.example_plugin:MeanExtractor")]
        clustered: list[tuple[int, ...]] = []
        compute = cache_module._do_compute_clusters
        monkeypatch.setattr(
            cache_module, "_do_compute_clusters", lambda e, *a, **k: clustered.append(e.shape) or compute(e, *a, **k)
        )
        keys: list[str] = []
        load = DatasetCache.load_or_compute_cluster_result

        def spy(self: DatasetCache, sel_key: str, config_json: str, *args: Any, **kwargs: Any) -> Any:
            keys.append(config_json)
            return load(self, sel_key, config_json, *args, **kwargs)

        monkeypatch.setattr(DatasetCache, "load_or_compute_cluster_result", spy)
        DatasetCache.clear_instances()
        clean = DataCleaningConfig(
            name="clean",
            outlier_method="zscore",
            outlier_flags=["dimension"],
            outlier_cluster_threshold=2.0,
            outlier_cluster_algorithm="kmeans",
            outlier_n_clusters=2,
        )
        config = PipelineConfig(
            datasets=[DatasetProtocolConfig(name="toy", dataset=ToyImages())],
            sources=[SourceConfig(name="src", dataset="toy")],
            extractors=[FlattenExtractorConfig(name="flat", batch_size=8), MeanConfig(name="mean", batch_size=8)],
            workflows=[clean],
            tasks=[
                TaskConfig(name=f"clean_{x}", workflow="clean", sources="src", extractor=x) for x in ("flat", "mean")
            ],
        )

        results = run_tasks(config)
        assert all(result.success for result in results.values()), [result.errors for result in results.values()]
        assert clustered == [(12, 3 * 16 * 16), (12, 3)]
        flat_key, mean_key = keys
        assert flat_key != mean_key
        assert '"model":"flatten"' in flat_key
        assert '"model":"example.mean"' in mean_key

        run_tasks(config)
        assert len(clustered) == 2, "a stateless extractor's clusters are still cached"
        DatasetCache.clear_instances()
