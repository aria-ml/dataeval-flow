"""A preset's other steps' settings sit under each step's type, spelled as the step spells them (naming spec §5)."""

from typing import Any, cast

import pytest
from pydantic import ValidationError

from dataeval_flow.workflows.data_cleaning import DataCleaningConfig, DataCleaningWorkflow
from dataeval_flow.workflows.data_prioritization import DataPrioritizationConfig, DataPrioritizationWorkflow

_OUTLIERS = {"flags": ["pixel", "visual"], "outlier_threshold": ["zscore", 2.5]}


def _entry(chain: Any, name: str) -> Any:
    return next(entry for entry in chain.evaluators if entry.name == name)


def _step(chain: Any, name: str) -> Any:
    return next(step for step in chain.steps if step["name"] == name)


def test_data_cleaning_hands_its_outliers_block_to_the_outliers_entry() -> None:
    config = DataCleaningConfig.model_validate({"outliers": {**_OUTLIERS, "n_clusters": 4}})
    entry = _entry(DataCleaningWorkflow.chain(config), "outliers")
    assert list(entry.flags) == ["pixel", "visual"]
    assert tuple(entry.outlier_threshold) == ("zscore", 2.5)
    assert entry.n_clusters == 4
    assert entry.per_target is True  # the preset fixes it


def test_data_cleaning_takes_a_bare_method_as_its_threshold() -> None:
    config = DataCleaningConfig.model_validate({"outliers": {"flags": ["pixel"], "outlier_threshold": "iqr"}})
    assert _entry(DataCleaningWorkflow.chain(config), "outliers").outlier_threshold == "iqr"


def test_data_cleaning_requires_its_outliers_flags_and_threshold() -> None:
    with pytest.raises(ValidationError, match="outliers"):
        DataCleaningConfig.model_validate({})
    with pytest.raises(ValidationError, match="outlier_threshold"):
        DataCleaningConfig.model_validate({"outliers": {"flags": ["pixel"]}})


def test_a_flat_outlier_key_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match="outlier_method"):
        DataCleaningConfig.model_validate({"outliers": _OUTLIERS, "outlier_method": "zscore"})


def test_data_cleaning_hands_its_duplicates_block_to_the_duplicates_entry() -> None:
    config = DataCleaningConfig.model_validate(
        {"outliers": _OUTLIERS, "duplicates": {"flags": ["hash_d4"], "merge_near_duplicates": False}}
    )
    entry = _entry(DataCleaningWorkflow.chain(config), "dupes")
    assert list(entry.flags) == ["hash_d4"]
    assert entry.merge_near_duplicates is False


def test_data_prioritization_selects_by_its_select_block() -> None:
    config = DataPrioritizationConfig.model_validate({"select": {"n": 7}})
    assert _step(DataPrioritizationWorkflow.chain(config), "selected")["n"] == 7


def test_data_prioritization_cleans_with_data_cleanings_blocks_and_its_dup_types() -> None:
    config = DataPrioritizationConfig.model_validate(
        {"cleaning": {"outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}, "dup_types": ["exact"]}}
    )
    chain = DataPrioritizationWorkflow.chain(config)
    plan = next(value for key, value in _step(chain, "reference-clean")["plans"].items() if "dupes" in key)
    assert plan["dup_types"] == ["exact"]
    assert config.cleaning is not None
    assert config.cleaning.duplicates.merge_near_duplicates is True


def test_a_matrix_varies_a_setting_inside_a_preset_block() -> None:
    from dataeval_flow import run_tasks
    from dataeval_flow._cache import DatasetCache
    from tests.chain_toys import chain_pipeline

    DatasetCache.clear_instances()
    entry = {"name": "clean", "type": "data-cleaning", "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}}
    grid = {"workflows.clean.outliers.outlier_threshold": ["zscore", "iqr"]}
    config = chain_pipeline(
        workflows=[entry], tasks=[{"name": "t", "workflow": "clean", "sources": ["src"], "matrix": grid}]
    )
    result = cast("Any", run_tasks(config)["t"])
    thresholds = sorted(str(run.values["workflows.clean.outliers.outlier_threshold"]) for run in result.runs)
    assert thresholds == ["iqr", "zscore"]
