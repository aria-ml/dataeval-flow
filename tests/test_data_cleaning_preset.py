"""data-cleaning as a preset: its chain, what its task returns, and the cleaned Dataset it hands on (spec §10).

Measured on the toy datasets: 12 toy images hold one outlier, of class b, and one exact-duplicate pair, so `clean`
removes 2 and keeps 10; 24 toy images keep 22. Read through a view that shuffles them (seed 3) and keeps 16, the 24
hold nothing to remove.
"""

import logging
import re
from typing import Any

import pytest
from dataeval.data import View
from pydantic import ValidationError

import dataeval_flow._cache as cache_module
from dataeval_flow import PipelineConfig, run, run_task, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._graph import GraphError
from dataeval_flow.config import (
    DatasetProtocolConfig,
    OntologyConfig,
    SourceConfig,
    TaskConfig,
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.config.extractors import FlattenExtractorConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig, LabelHealthConfig, OutliersConfig
from dataeval_flow.steps import ChainResult, list_steps
from dataeval_flow.steps._port import DataType
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig, DataCleaningWorkflow
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages
from tests.example_plugin import MeanConfig
from tests.workflow_toys import register_count

_BASE: dict[str, Any] = {
    "name": "cleaning",
    "type": "data-cleaning",
    "outliers": {"flags": ["pixel", "visual"], "outlier_threshold": "zscore"},
}
_STEPS = [
    "outliers",
    "label-health",
    "outliers-by-class",
    "duplicates",
    "image-outliers",
    "target-outliers",
    "classwise-outliers",
    "image-duplicates",
    "class-imbalance",
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
        ("warning", "Image Outliers", "1 images (8.3%)", "image-outliers"),
        ("warning", "Classwise Outliers", "worst: b (16.7%), 1/1 classes over 3.0%", "classwise-outliers"),
        ("warning", "Image Duplicates", "2 exact (16.7%), 0 near (0.0%)", "image-duplicates"),
        ("info", "Class Imbalance", "2 classes, 12 items, imbalance 1.0:1", "class-imbalance"),
    ]


def test_clean_removes_each_flagged_item_and_each_duplicate_but_the_first() -> None:
    clean = _task(ToyImages(count=12)).steps["clean"]
    assert len(clean.output) == 10
    assert clean.details == {
        "removed": {"items": 2, "detections": 0, "tracks": 0, "frames": 0},
        "by_plan": {"duplicates": {"items": 1}, "outliers": {"items": 1}},
    }


def test_clean_says_what_it_kept_and_what_each_plan_named() -> None:
    from dataeval_flow._blocks import Paragraph
    from dataeval_flow.steps.transforms import RemoveTransform

    clean = _task(ToyImages(count=24)).steps["clean"]
    assert RemoveTransform().section(clean) == [
        Paragraph(text="Kept 22 of 24 images. Removed 2 images: 1 named by `duplicates`, 1 by `outliers`.")
    ]


def test_run_returns_the_chain_and_the_cleaned_dataset() -> None:
    result = run(
        DataCleaningConfig(outliers={"flags": ["pixel", "visual"], "outlier_threshold": "zscore"}),  # type: ignore[arg-type]
        ToyImages(count=12),
    )
    assert isinstance(result, ChainResult)
    clean = result.steps["clean"].output
    assert isinstance(clean, View)
    assert len(clean) == 10


def _configuration(result: ChainResult) -> list[str]:
    text = result.report().splitlines()
    return [line.strip() for line in text[text.index("  CONFIGURATION") :]]


def test_the_report_keeps_a_threshold_the_user_set_to_null_and_drops_unset_defaults() -> None:
    shown = _configuration(_task(ToyImages(count=12), checks={"image-outliers": {"warning": None}}))
    assert "image-outliers: {warning: None}" in shown
    assert "stats: None" not in shown
    assert "metadata: None" not in shown
    assert not [line for line in shown if "None" in line and line != "image-outliers: {warning: None}"]


def test_the_report_keeps_every_threshold_the_user_nulled() -> None:
    nulls = {
        "image-duplicates": {"near": None},
        "image-outliers": {"warning": None},
        "target-outliers": {"warning": None},
        "classwise-outliers": {"warning": None},
        "class-imbalance": {"warning": None},
    }
    shown = _configuration(_task(ToyImages(count=12), checks=nulls))
    assert "checks:" in shown
    assert sorted(line for line in shown if "None" in line) == [
        "class-imbalance: {warning: None}",
        "classwise-outliers: {warning: None}",
        "image-duplicates: {exact: 0.0, near: None}",
        "image-outliers: {warning: None}",
        "target-outliers: {warning: None}",
    ]


def test_a_null_threshold_judges_nothing() -> None:
    result = _task(ToyImages(count=12), checks={"image-outliers": {"warning": None}})
    assert _verdicts(result)[0] == ("info", "Image Outliers", "1 images (8.3%)", "image-outliers")


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
    assert ("info", "Class Imbalance", "2 classes, 16 items, imbalance 1.0:1", "class-imbalance") in _verdicts(result)
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
        evaluators=[{"name": "labels", "type": "label-health"}],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["s1", "s2"]}],
        datasets={"s1": ToyImages(count=12), "s2": ToyImages(count=24)},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert list(result.steps) == [*(f"cleaning/{step}" for step in _STEPS), "sizes"]
    assert _verdicts(result) == [
        ("warning", "Image Outliers", "1 images (8.3%)", "cleaning/image-outliers[s1]"),
        ("warning", "Image Outliers", "1 images (4.2%)", "cleaning/image-outliers[s2]"),
        ("warning", "Classwise Outliers", "worst: b (16.7%), 1/1 classes over 3.0%", "cleaning/classwise-outliers[s1]"),
        ("warning", "Classwise Outliers", "worst: b (8.3%), 1/1 classes over 3.0%", "cleaning/classwise-outliers[s2]"),
        ("warning", "Image Duplicates", "2 exact (16.7%), 0 near (0.0%)", "cleaning/image-duplicates[s1]"),
        ("warning", "Image Duplicates", "2 exact (8.3%), 0 near (0.0%)", "cleaning/image-duplicates[s2]"),
        ("info", "Class Imbalance", "2 classes, 12 items, imbalance 1.0:1", "cleaning/class-imbalance[s1]"),
        ("info", "Class Imbalance", "2 classes, 24 items, imbalance 1.0:1", "cleaning/class-imbalance[s2]"),
    ]
    clean = result.steps["cleaning/clean"].elements or {}
    assert {key: len(element.output) for key, element in clean.items()} == {"s1": 10, "s2": 22}
    assert result.steps["sizes"].inputs == ["cleaning.clean"]
    assert list(result.steps["sizes"].elements or {}) == ["s1", "s2"]


def test_a_step_reading_the_cleaned_dataset_is_refused_a_kind_it_cannot_take_before_any_step_runs() -> None:
    """`cleaning.clean` carries the kind `cleaning/clean` made, classification, so preflight refuses a wrap that needs
    boxes. Without that kind, the wrap would fail only as a step, once it ran."""
    config = chain_pipeline(
        workflows=[
            _BASE,
            {
                "name": "outer",
                "inputs": ["data"],
                "steps": [
                    {"name": "cleaning", "workflow": "cleaning", "input": "data"},
                    {"name": "crops", "transform": "wrap", "input": "cleaning.clean", "wrapper": "DetectionCrops"},
                ],
            },
        ],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["src"]}],
        datasets={"src": ToyImages(count=12)},
    )
    message = "Step 'crops': DetectionCrops takes a object_detection Dataset, but its input is classification."
    with pytest.raises(GraphError, match=f"^{re.escape(message)}$"):
        run_tasks(config)


def test_the_settings_become_the_chain_s_evaluators_and_thresholds() -> None:
    """Each setting holds a value no other shares, so one routed to the wrong evaluator or check fails."""
    config = DataCleaningConfig(
        outliers={  # type: ignore[arg-type]
            "flags": ["dimension"],
            "outlier_threshold": ("modzscore", 4.0),
            "cluster_threshold": 2.5,
            "cluster_algorithm": "kmeans",
            "n_clusters": 3,
        },
        duplicates={  # type: ignore[arg-type]
            "flags": ["hash_d4"],
            "merge_near_duplicates": False,
            "cluster_sensitivity": 0.7,
            "cluster_algorithm": "hdbscan",
            "n_clusters": 5,
        },
        metadata="policy",
        stats="measured",
        checks={  # type: ignore[arg-type]
            "image-duplicates": {"exact": 1.0, "near": 9.0},
            "image-outliers": {"warning": 6.0},
            "target-outliers": {"warning": 7.0},
            "classwise-outliers": {"warning": 8.0},
            "class-imbalance": {"warning": None},
        },
    )
    chain = DataCleaningWorkflow.chain(config)
    outliers, dupes, labels = chain.evaluators
    assert isinstance(outliers, OutliersConfig)
    assert isinstance(dupes, DuplicatesConfig)
    assert isinstance(labels, LabelHealthConfig)
    assert (
        outliers.name,
        outliers.flags,
        outliers.outlier_threshold,
        outliers.cluster_threshold,
        outliers.cluster_algorithm,
        outliers.n_clusters,
        outliers.per_target,
        outliers.stats,
    ) == ("outliers", ["dimension"], ("modzscore", 4.0), 2.5, "kmeans", 3, True, "measured")
    assert (
        dupes.name,
        dupes.flags,
        dupes.merge_near_duplicates,
        dupes.cluster_sensitivity,
        dupes.cluster_algorithm,
        dupes.n_clusters,
        dupes.stats,
    ) == ("duplicates", ["hash_d4"], False, 0.7, "hdbscan", 5, "measured")
    assert (labels.name, labels.metadata) == ("label-health", "policy")
    steps: dict[str, Any] = {step["name"]: step for step in chain.steps}  # type: ignore[index]
    assert (
        steps["image-outliers"]["warning"],
        steps["target-outliers"]["warning"],
        steps["classwise-outliers"]["warning"],
        steps["image-duplicates"]["exact"],
        steps["image-duplicates"]["near"],
        steps["class-imbalance"]["warning"],
    ) == (6.0, 7.0, 8.0, 1.0, 9.0, None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("value_range", [0.0, 255.0]),
        ("metadata_auto_bin_method", "uniform_width"),
        ("metadata_exclude", ["id"]),
        ("metadata_continuous_factor_bins", {"brightness": 4}),
        ("metadata_factor_source", "coded"),
    ],
)
def test_a_retired_field_is_refused(field: str, value: Any) -> None:
    with pytest.raises(ValidationError, match=re.escape(field)):
        DataCleaningConfig.model_validate(
            {"outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}, field: value}
        )


def _conformed(ontology: str, plugins: dict[str, list[tuple[str, str]]]) -> PipelineConfig:
    """A data-cleaning task and a workflow-type task, `test.count`, each naming `ontology`, on one source read through
    a view that relabels `a` and `b` as `cat` and `dog`."""
    register_count(plugins)
    relabel = ViewOperation(type="Relabel", params={"class_remap": {"a": "cat", "b": "dog"}, "target": ["cat", "dog"]})
    animals = OntologyConfig(
        name="animals",
        concepts=[  # type: ignore[arg-type]
            {"id": "animal", "label": "animal"},
            {"id": "cat", "label": "cat", "parents": ["animal"]},
            {"id": "dog", "label": "dog", "parents": ["animal"]},
        ],
    )
    return chain_pipeline(
        workflows=[{**_BASE, "ontology": ontology}, {"name": "count", "type": "test.count", "ontology": ontology}],
        tasks=[
            {"name": "cleaning", "workflow": "cleaning", "sources": ["src"]},
            {"name": "count", "workflow": "count", "sources": ["src"]},
        ],
        datasets={"src": ToyImages(count=12)},
        extra={
            "sources": [SourceConfig(name="src", dataset="src_data", view="conform")],
            "views": [ViewConfig(name="conform", operations=[relabel])],
            "ontologies": [animals],
        },
    )


def test_a_data_cleaning_entry_s_ontology_is_recorded_as_a_workflow_type_task_s_is(
    plugins: dict[str, list[tuple[str, str]]],
) -> None:
    results = run_tasks(_conformed("animals", plugins))
    cleaning, count = results["cleaning"], results["count"]
    assert cleaning.success, cleaning.errors
    assert [(r.source, r.ontology, r.ontology_digest) for r in cleaning.metadata.label_space] == [
        ("src", "animals", "1a56aa26a662")
    ]
    assert cleaning.metadata.label_space_digest == "ae6c1114cbae"
    assert cleaning.metadata.label_space == count.metadata.label_space
    assert cleaning.metadata.label_space_digest == count.metadata.label_space_digest


def test_a_data_cleaning_entry_s_ontology_that_does_not_resolve_is_logged_as_a_workflow_type_task_s_is(
    plugins: dict[str, list[tuple[str, str]]], caplog: pytest.LogCaptureFixture
) -> None:
    message = (
        "Task ontology could not be resolved: could not read ontology file 'missing.ttl': [Errno 2] No such file or "
        "directory: 'missing.ttl' No ontology named 'missing.ttl' is declared under `ontologies:` either. Declared: "
        "animals."
    )
    with caplog.at_level(logging.WARNING, logger="dataeval_flow._orchestrator"):
        results = run_tasks(_conformed("missing.ttl", plugins))
    cleaning, count = results["cleaning"], results["count"]
    assert cleaning.success, cleaning.errors
    # One warning per task: data-cleaning's, then test.count's.
    assert [r.message for r in caplog.records if "ontology" in r.message] == [message, message]
    assert [(r.ontology, r.ontology_digest) for r in cleaning.metadata.label_space] == [(None, None)]
    assert cleaning.metadata.label_space == count.metadata.label_space


def test_the_catalog_lists_the_cleaned_dataset_as_data_cleaning_s_output() -> None:
    entry = next(e for e in list_steps(plugins=False).steps if e.type == "data-cleaning")
    assert [(port.port, port.type) for port in entry.outputs] == [("clean", DataType.DATASET)]


@pytest.mark.required
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
            outliers={  # type: ignore[arg-type]
                "flags": ["dimension"],
                "outlier_threshold": "zscore",
                "cluster_threshold": 2.0,
                "cluster_algorithm": "kmeans",
                "n_clusters": 2,
            },
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
