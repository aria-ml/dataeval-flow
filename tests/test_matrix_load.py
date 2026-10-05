"""A matrix expanded at load: what its keys name, what they may not vary, and every run validated (task-matrix spec
§3.3-§3.5)."""

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow._cache import DatasetCache
from dataeval_flow._matrix._build import expand_matrix
from tests.chain_toys import chain_pipeline, register_toys
from tests.evaluator_toys import ToyImages, shifted_sources

_CLEANING = {
    "name": "cleaning",
    "type": "data-cleaning",
    "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"},
}
_KNN = {"name": "knn", "type": "ood-kneighbors", "k": 5, "distance_metric": "euclidean"}
_OOD = {"name": "ood", "type": "ood-detection", "detectors": [_KNN, {"type": "ood-domain-classifier", "n_folds": 3}]}


@pytest.fixture(autouse=True)
def _toys(plugins: dict) -> Any:
    register_toys(plugins)
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _cleaning(matrix: Any, **task: Any) -> Any:
    return chain_pipeline(
        workflows=[_CLEANING],
        tasks=[{"name": "t", "workflow": "cleaning", "sources": "src", "matrix": matrix, **task}],
        extractor=True,
    )


def _runs(config: Any) -> list[Any]:
    """The runs of `config`'s one matrix task, loosely typed so a test reads each run's entries by their own fields."""
    return expand_matrix(config.tasks[0], config)


def _refusal(build: Any, *fragments: str) -> None:
    with pytest.raises(ValidationError) as caught:
        build()
    for fragment in fragments:
        assert fragment in str(caught.value)


def test_each_run_is_a_validated_pipeline_holding_only_its_task() -> None:
    config = _cleaning({"outliers.outlier_threshold": ["iqr", 3.0]})
    runs = _runs(config)
    assert [run.label for run in runs] == ["outliers.outlier_threshold=iqr", "outliers.outlier_threshold=3.0"]
    assert [run.number for run in runs] == [1, 2]
    for run in runs:
        assert [task.name for task in run.pipeline.tasks] == ["t"]
        assert run.task.matrix is None
    assert runs[1].pipeline.workflows[0].outliers.outlier_threshold == 3.0
    assert config.workflows[0].outliers.outlier_threshold == "zscore"


def test_a_path_into_a_nested_setting_left_at_its_default_sets_it() -> None:
    config = _cleaning({"checks.image-outliers.warning": [1.0, 5.0]})
    runs = _runs(config)
    assert [run.pipeline.workflows[0].checks.image_outliers.warning for run in runs] == [1.0, 5.0]


def test_an_untouched_entry_passes_through_as_the_same_instance() -> None:
    config = _cleaning({"outliers.outlier_threshold": [3.0]})
    (run,) = _runs(config)
    assert run.pipeline.extractors[0] is config.extractors[0]
    assert run.pipeline.datasets[0] is config.datasets[0]


def test_a_detector_is_reached_by_name_and_an_unnamed_one_by_type() -> None:
    config = chain_pipeline(
        workflows=[_OOD],
        tasks=[
            {
                "name": "t",
                "workflow": "ood",
                "sources": ["reference", "test"],
                "extractor": "flat",
                "matrix": {"detectors.knn.k": [3, 7], "detectors.ood-domain-classifier.n_folds": [2]},
            }
        ],
        datasets=shifted_sources(),
        extractor=True,
    )
    runs = _runs(config)
    assert [run.pipeline.workflows[0].detectors[0].k for run in runs] == [3, 7]
    assert runs[0].pipeline.workflows[0].detectors[1].n_folds == 2


def test_a_whole_list_item_keeps_its_name() -> None:
    config = chain_pipeline(
        workflows=[_OOD],
        tasks=[
            {
                "name": "t",
                "workflow": "ood",
                "sources": ["reference", "test"],
                "extractor": "flat",
                "matrix": {"detectors.knn": [{"type": "ood-kneighbors", "k": 9, "distance_metric": "euclidean"}]},
            }
        ],
        datasets=shifted_sources(),
        extractor=True,
    )
    (run,) = _runs(config)
    assert run.pipeline.workflows[0].detectors[0].name == "knn"
    assert run.pipeline.workflows[0].detectors[0].k == 9


def test_a_custom_workflow_is_swept_through_its_pool_entries_and_inline_steps() -> None:
    workflow = {
        "name": "w",
        "inputs": ["a"],
        "steps": [
            {"name": "few", "transform": "toy-first", "input": "a", "n": 3},
            {"name": "dupes", "evaluator": "dupes", "input": "few"},
        ],
    }
    config = chain_pipeline(
        workflows=[workflow],
        evaluators=[{"name": "dupes", "type": "duplicates"}],
        tasks=[
            {
                "name": "t",
                "workflow": "w",
                "sources": "src",
                "matrix": {"steps.few.n": [2, 4], "evaluators.dupes.merge_near_duplicates": [False]},
            }
        ],
    )
    runs = _runs(config)
    assert [run.pipeline.workflows[0].steps[0].settings["n"] for run in runs] == [2, 4]
    assert runs[0].pipeline.evaluators[0].merge_near_duplicates is False


def test_a_step_setting_left_at_its_default_is_set_and_one_the_step_lacks_is_refused() -> None:
    workflow = {
        "name": "w",
        "inputs": ["a"],
        "steps": [
            {"name": "out", "evaluator": "outl", "input": "a"},
            {"name": "gate", "check": "image-outliers", "input": "out"},
        ],
    }

    def build(matrix: Any) -> Any:
        return chain_pipeline(
            workflows=[workflow],
            evaluators=[{"name": "outl", "type": "outliers", "flags": ["pixel"]}],
            tasks=[{"name": "t", "workflow": "w", "sources": "src", "matrix": matrix}],
        )

    runs = _runs(build({"steps.gate.warning": [0.05, 0.2]}))
    assert [run.pipeline.workflows[0].steps[1].config.warning for run in runs] == [0.05, 0.2]
    _refusal(lambda: build({"steps.gate.imag": [0.05]}), "workflow 'w' has no setting `steps.gate.imag`")


def test_sources_and_extractor_are_the_task_s_own() -> None:
    config = chain_pipeline(
        workflows=[_CLEANING],
        datasets={"a": ToyImages(), "b": ToyImages(seed=1)},
        extractor=True,
        tasks=[
            {
                "name": "t",
                "workflow": "cleaning",
                "sources": "a",
                "matrix": {"sources": ["a", "b"], "extractor": [None, "flat"]},
            }
        ],
    )
    runs = _runs(config)
    assert [(run.task.source_names, run.task.extractor) for run in runs] == [
        (["a"], None),
        (["a"], "flat"),
        (["b"], None),
        (["b"], "flat"),
    ]


def test_a_matrix_task_as_written_skips_the_task_checks_when_its_runs_pass() -> None:
    # Cluster detection needs an extractor; the task names none, and every run supplies one.
    config = chain_pipeline(
        workflows=[{**_CLEANING, "outliers": {**_CLEANING["outliers"], "cluster_threshold": 1.0}}],
        extractor=True,
        tasks=[{"name": "t", "workflow": "cleaning", "sources": "src", "matrix": {"extractor": ["flat"]}}],
    )
    assert len(_runs(config)) == 1


def test_a_sibling_task_is_not_checked_against_the_matrix_s_overrides() -> None:
    # The matrix sets cluster detection on the shared entry; the sibling task, which has no extractor, never sees it.
    config = chain_pipeline(
        workflows=[_CLEANING],
        extractor=True,
        tasks=[
            {
                "name": "t",
                "workflow": "cleaning",
                "sources": "src",
                "extractor": "flat",
                "matrix": {"outliers.cluster_threshold": [1.0]},
            },
            {"name": "sibling", "workflow": "cleaning", "sources": "src"},
        ],
    )
    assert len(_runs(config)) == 1


@pytest.mark.parametrize(
    ("matrix", "fragment"),
    [
        ({"outlier_thresh": [1.0]}, "data-cleaning has no setting `outlier_thresh`"),
        ({"evaluators.nope.k": [1]}, "`evaluators:` has no entry named `nope`"),
        ({"steps.few.n": [1]}, "runs no custom workflow"),
        ({"name": ["other"]}, "a matrix varies settings, not identities"),
        ({"type": ["data-analysis"]}, "a matrix varies settings, not identities"),
        ({"sources": ["missing"]}, "names no source `missing`"),
        ({"extractor": ["missing"]}, "names no extractor `missing`"),
        ({"outliers.cluster_algorithm": ["bogus"]}, "run 1 (outliers.cluster_algorithm=bogus)"),
        ({"outliers.outlier_threshold": [3, 3.0]}, "runs 1 and 2"),
        ({"outliers.cluster_threshold.x": [1.0]}, "is unset, so `outliers.cluster_threshold.x` has nothing to set"),
        ([{"checks": [{}], "checks.image-duplicates.near": [1.0]}], "sets part of"),
        ([{"checks": [None], "checks.image-duplicates.near": [1.0]}], "sets part of"),
        ({"sources": [None]}, "`sources` value null is not a source name or a list of them"),
        ({"sources": [5]}, "`sources` value 5 is not a source name or a list of them"),
        ({"extractor": [["flat"]]}, "`extractor` value [flat] is not an extractor name or null"),
    ],
)
def test_a_key_or_run_that_cannot_run_is_refused_naming_it(matrix: Any, fragment: str) -> None:
    _refusal(lambda: _cleaning(matrix, extractor="flat"), "Task 't'", fragment)


def test_two_spellings_of_one_setting_in_a_grid_are_refused() -> None:
    config_args = {"evaluators": [_KNN], "datasets": shifted_sources(), "extractor": True}
    task = {
        "name": "t",
        "evaluator": "knn",
        "sources": ["reference", "test"],
        "extractor": "flat",
        "matrix": {"k": [3], "evaluators.knn.k": [4]},
    }
    _refusal(lambda: chain_pipeline(tasks=[task], **config_args), "name the same setting")


def test_a_prefix_is_judged_by_whole_segments() -> None:
    detectors = [_KNN, {**_KNN, "name": "knn2"}]
    task = {
        "name": "t",
        "workflow": "ood",
        "sources": ["reference", "test"],
        "extractor": "flat",
        "matrix": {
            "detectors.knn": [{"type": "ood-kneighbors", "k": 3, "distance_metric": "euclidean"}],
            "detectors.knn2.k": [4],
        },
    }
    config = chain_pipeline(
        workflows=[{**_OOD, "detectors": detectors}], tasks=[task], datasets=shifted_sources(), extractor=True
    )
    assert len(_runs(config)) == 1


def test_a_step_s_wiring_and_an_evaluator_step_s_settings_are_refused_with_the_way_to_vary_them() -> None:
    workflow = {
        "name": "w",
        "inputs": ["a"],
        "steps": [
            {"name": "few", "transform": "toy-first", "input": "a", "n": 3},
            {"name": "dupes", "evaluator": "dupes", "input": "few"},
        ],
    }

    def build(matrix: Any) -> Any:
        return chain_pipeline(
            workflows=[workflow],
            evaluators=[{"name": "dupes", "type": "duplicates"}],
            tasks=[{"name": "t", "workflow": "w", "sources": "src", "matrix": matrix}],
        )

    _refusal(lambda: build({"steps.few.input": ["a"]}), "wires the chain every run shares")
    _refusal(lambda: build({"steps.dupes.flags": [["hash_basic"]]}), "vary `evaluators.dupes.flags`")


def test_a_pool_key_a_run_does_not_read_is_refused_naming_the_run() -> None:
    from dataeval_flow.config.extractors import FlattenExtractorConfig
    from tests.evaluator_toys import FLAT

    other = FlattenExtractorConfig(name="flat2", batch_size=4)
    task = {
        "name": "t",
        "workflow": "cleaning",
        "sources": "src",
        "matrix": {"extractor": ["flat", "flat2"], "extractors.flat.batch_size": [2]},
    }
    _refusal(
        lambda: chain_pipeline(workflows=[_CLEANING], tasks=[task], extra={"extractors": [FLAT, other]}),
        "run 2 (extractor=flat2, extractors.flat.batch_size=2)",
        "reads no extractor `flat`",
    )


def test_a_null_in_a_list_is_a_run_of_its_own() -> None:
    config = _cleaning({"duplicates.flags": [None, ["hash_d4"]]})
    assert [run.values for run in _runs(config)] == [
        {"duplicates.flags": None},
        {"duplicates.flags": ["hash_d4"]},
    ]


def test_a_detector_s_own_extractor_is_read_by_the_run() -> None:
    from dataeval_flow.config.extractors import FlattenExtractorConfig
    from tests.evaluator_toys import FLAT

    detectors = [{**_KNN, "extractor": "flat2"}, {"type": "ood-domain-classifier", "n_folds": 3}]
    config = chain_pipeline(
        workflows=[{**_OOD, "detectors": detectors}],
        tasks=[
            {
                "name": "t",
                "workflow": "ood",
                "sources": ["reference", "test"],
                "extractor": "flat",
                "matrix": {"extractors.flat2.batch_size": [2, 4]},
            }
        ],
        datasets=shifted_sources(),
        extra={"extractors": [FLAT, FlattenExtractorConfig(name="flat2", batch_size=8)]},
    )
    assert [run.pipeline.extractors[1].batch_size for run in _runs(config)] == [2, 4]


def test_the_matrix_as_written_is_never_written_into() -> None:
    written = [{"checks": [{"image-duplicates": {"near": 2.0}}]}, {"checks.image-duplicates.near": [1.0]}]
    config = _cleaning(copy.deepcopy(written))
    assert [run.pipeline.workflows[0].checks.image_duplicates.near for run in _runs(config)] == [2.0, 1.0]
    assert config.tasks[0].matrix == written
