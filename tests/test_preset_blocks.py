"""A preset's other steps' settings sit under each step's type, spelled as the step spells them (naming spec §5)."""

from typing import Any, cast

import pytest
from pydantic import ValidationError

from dataeval_flow.workflows.bias import BiasConfig, BiasWorkflow
from dataeval_flow.workflows.prioritization import PrioritizationWorkflow, PrioritizationWorkflowConfig
from dataeval_flow.workflows.quality import QualityConfig, QualityWorkflow
from dataeval_flow.workflows.scope import ScopeConfig, ScopeWorkflow
from dataeval_flow.workflows.shift import ShiftConfig, ShiftWorkflow
from dataeval_flow.workflows.taxonomy import TaxonomyConfig, TaxonomyWorkflow

_DETECTORS = [{"name": "knn", "type": "ood-kneighbors", "distance_metric": "euclidean"}]
_OUTLIERS = {"flags": ["pixel", "visual"], "outlier_threshold": ["zscore", 2.5]}


def _entry(chain: Any, name: str) -> Any:
    return next(entry for entry in chain.evaluators if entry.name == name)


def _step(chain: Any, name: str) -> Any:
    return next(step for step in chain.steps if step["name"] == name)


def test_data_cleaning_hands_its_outliers_block_to_the_outliers_entry() -> None:
    config = QualityConfig.model_validate({"outliers": {**_OUTLIERS, "n_clusters": 4}})
    entry = _entry(QualityWorkflow.chain(config), "outliers")
    assert list(entry.flags) == ["pixel", "visual"]
    assert tuple(entry.outlier_threshold) == ("zscore", 2.5)
    assert entry.n_clusters == 4
    assert entry.per_target is True  # the preset fixes it


def test_data_cleaning_takes_a_bare_method_as_its_threshold() -> None:
    config = QualityConfig.model_validate({"outliers": {"flags": ["pixel"], "outlier_threshold": "iqr"}})
    assert _entry(QualityWorkflow.chain(config), "outliers").outlier_threshold == "iqr"


def test_data_cleaning_requires_its_outliers_flags_and_threshold() -> None:
    with pytest.raises(ValidationError, match="outliers"):
        QualityConfig.model_validate({})
    with pytest.raises(ValidationError, match="outlier_threshold"):
        QualityConfig.model_validate({"outliers": {"flags": ["pixel"]}})


def test_a_flat_outlier_key_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match="outlier_method"):
        QualityConfig.model_validate({"outliers": _OUTLIERS, "outlier_method": "zscore"})


def test_data_cleaning_hands_its_duplicates_block_to_the_duplicates_entry() -> None:
    config = QualityConfig.model_validate(
        {"outliers": _OUTLIERS, "duplicates": {"flags": ["hash_d4"], "merge_near_duplicates": False}}
    )
    entry = _entry(QualityWorkflow.chain(config), "duplicates")
    assert list(entry.flags) == ["hash_d4"]
    assert entry.merge_near_duplicates is False


def test_data_prioritization_ranks_by_its_prioritization_block() -> None:
    config = PrioritizationWorkflowConfig.model_validate(
        {"prioritization": {"method": "kmeans_distance", "c": 4, "n_init": 3, "policy": "stratified", "num_bins": 9}}
    )
    entry = _entry(PrioritizationWorkflow.chain(config), "prioritization")
    assert (entry.method, entry.c, entry.n_init) == ("kmeans_distance", 4, 3)
    assert (entry.order, entry.policy, entry.num_bins) == ("hard_first", "stratified", 9)


def test_data_prioritization_selects_by_its_select_block() -> None:
    config = PrioritizationWorkflowConfig.model_validate({"select": {"n": 7}})
    assert _step(PrioritizationWorkflow.chain(config), "selected")["n"] == 7


def test_a_matrix_varies_a_setting_inside_a_preset_block() -> None:
    from dataeval_flow import run_tasks
    from dataeval_flow._cache import DatasetCache
    from tests.chain_toys import chain_pipeline

    DatasetCache.clear_instances()
    entry = {"name": "clean", "type": "quality", "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}}
    grid = {"workflows.clean.outliers.outlier_threshold": ["zscore", "iqr"]}
    config = chain_pipeline(
        workflows=[entry], tasks=[{"name": "t", "workflow": "clean", "sources": ["src"], "matrix": grid}]
    )
    result = cast("Any", run_tasks(config)["t"])
    thresholds = sorted(str(run.values["workflows.clean.outliers.outlier_threshold"]) for run in result.runs)
    assert thresholds == ["iqr", "zscore"]


def _types(chain: Any) -> set[str | None]:
    return {
        step.get("check") or step.get("combine") or step.get("transform") or step.get("evaluator")
        for step in chain.steps
    }


def test_data_coverage_keys_its_step_settings_by_step_type() -> None:
    config = ScopeConfig.model_validate(
        {
            "representation": {"expected": {"cat": 0.3}},
            "wrap": {"params": {"padding": 0.1, "min_size": 4}},
        }
    )
    chain = cast("Any", ScopeWorkflow.chain(config))
    entries = {entry.type: entry for entry in chain.evaluators}
    assert entries["representation"].expected == {"cat": 0.3}
    (wrap,) = (step for step in chain.steps if step.get("transform") == "wrap")
    assert wrap["params"] == {"padding": 0.1, "min_size": 4}


def test_data_bias_keys_its_step_settings_by_step_type() -> None:
    config = BiasConfig.model_validate({"diversity": {"method": "shannon"}, "factor-gaps": {"mi_threshold": 0.2}})
    chain = cast("Any", BiasWorkflow.chain(config))
    entries = {entry.type: entry for entry in chain.evaluators}
    assert entries["diversity"].method == "shannon"
    (gaps,) = (step for step in chain.steps if step.get("combine") == "factor-gaps")
    assert gaps["mi_threshold"] == 0.2


def test_factor_gaps_false_drops_the_gap_steps() -> None:
    chain = BiasWorkflow.chain(BiasConfig.model_validate({"factor-gaps": False}))
    assert {"factor-gaps", "factor-coverage-gaps"} & _types(chain) == set()


@pytest.mark.parametrize("key", ["expected", "gaps", "crops"])
def test_data_coverages_old_keys_are_refused(key: str) -> None:
    with pytest.raises(ValidationError, match=key):
        ScopeConfig.model_validate({key: {} if key != "expected" else {"cat": 0.3}})


def test_label_space_keys_its_step_settings_by_step_type() -> None:
    config = TaxonomyConfig.model_validate(
        {
            "ontology": {"animal": {"cat": None}},
            "representation": {"expected": {"cat": 0.5}},
            "ontology-validation": {"label_pattern": "^[a-z]+$"},
        }
    )
    entries = {entry.type: entry for entry in cast("Any", TaxonomyWorkflow.chain(config)).evaluators}
    assert entries["representation"].expected == {"cat": 0.5}
    assert entries["ontology-validation"].label_pattern == "^[a-z]+$"


def test_shift_ood_switches_off_each_factor_step_by_its_key() -> None:
    off = ShiftConfig.model_validate({"detectors": _DETECTORS, "factor-predictors": False, "factor-deviation": False})
    assert {"factor-predictors", "factor-deviation"} & _types(ShiftWorkflow.chain(off)) == set()
    one = ShiftConfig.model_validate({"detectors": _DETECTORS, "factor-predictors": False})
    assert "factor-deviation" in _types(ShiftWorkflow.chain(one))
    assert "factor-predictors" not in _types(ShiftWorkflow.chain(one))


def test_shift_ood_takes_factor_deviations_settings_under_its_type() -> None:
    config = ShiftConfig.model_validate({"detectors": _DETECTORS, "factor-deviation": {"max_items": 9}})
    chain = cast("Any", ShiftWorkflow.chain(config))
    (step,) = (step for step in chain.steps if step.get("combine") == "factor-deviation")
    assert step["max_items"] == 9
    with pytest.raises(ValidationError, match="metadata_insights"):
        ShiftConfig.model_validate({"detectors": _DETECTORS, "metadata_insights": False})
