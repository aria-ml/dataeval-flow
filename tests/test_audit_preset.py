"""audit: one chain over train and its evaluation splits, with a verdict, a record and five questions (audit spec §4,
§5, §6, §8, §11)."""

import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow._app._model._state import ConfigState
from dataeval_flow._blocks import Fields, Section
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._report import _record_section
from dataeval_flow.config import PipelineConfig
from dataeval_flow.config._loader import load_config
from dataeval_flow.steps import ChainResult
from dataeval_flow.steps._result import ChainMetadata, ChainOutput
from dataeval_flow.workflows._registry import get_workflow, list_workflows
from dataeval_flow.workflows.audit import AuditConfig, AuditWorkflow
from dataeval_flow.workflows.audit._config import CHECKS_MOVED, MOVED
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages
from tests.test_naming_conventions import _MINIMAL

_OUTLIERS = {"outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}}
_HEADINGS = [
    "Is the data clean?",
    "Are the labels sound?",
    "Does the data cover what the model must handle?",
    "Could the model learn a shortcut?",
    "Are the splits fit to evaluate on?",
]


def _config(entry: dict[str, Any] | None = None) -> AuditConfig:
    return AuditConfig.model_validate({"name": "w", **_OUTLIERS, **(entry or {})})


def _steps(config: AuditConfig) -> list[dict[str, Any]]:
    return [dict(step) for step in AuditWorkflow.chain(config).steps]


def _names(config: AuditConfig) -> list[str]:
    return [step["name"] for step in _steps(config)]


def test_the_chain_names_every_step_for_its_type_and_role() -> None:
    assert _names(AuditConfig.model_validate(_MINIMAL["audit"])) == [
        "label-health-train",
        "label-health-evals",
        "class-imbalance-train",
        "class-imbalance-evals",
        "outliers-train",
        "outliers-evals",
        "image-outliers-train",
        "image-outliers-evals",
        "duplicates-train",
        "duplicates-evals",
        "image-duplicates-train",
        "image-duplicates-evals",
        "factor-triage-train",
        "factor-triage-evals",
        "metadata-issues-train",
        "metadata-issues-evals",
        "content-digest-train",
        "content-digest-evals",
        "label-reconciliation-train",
        "label-reconciliation-evals",
        "label-conformance-train",
        "label-conformance-evals",
        "ood-kneighbors",
        "eval-coverage",
        "divergence",
        "distribution-shift",
        "duplicates-cross",
        "duplicates-pairs",
        "factor-leakage-cross",
        "factor-leakage-pairs",
        "leakage",
        "class-sufficiency",
        "untrained-classes",
        "stratification",
        "crops",
        "coverage",
        "class-coverage",
        "uncovered-items",
        "completeness",
        "dimensional-completeness",
        "factor-summary",
        "balance",
        "diversity",
        "shortcut-risk",
        "factor-gaps",
        "factor-coverage-gaps",
    ]


def test_optional_blocks_drop_their_steps() -> None:
    names = _names(_config())
    assert not [name for name in names if name.startswith(("label-reconciliation-", "label-conformance-"))]
    assert not [name for name in names if name.startswith("factor-leakage-")]
    assert "factors" not in next(step for step in _steps(_config()) if step["name"] == "leakage")
    assert not {"factor-gaps", "factor-coverage-gaps"} & set(_names(_config({"factor-gaps": False})))
    assert "uncovered-items" not in _names(_config({"coverage": {"method": "adaptive"}}))
    assert "uncovered-items" not in names  # adaptive is the default


def test_every_factor_reading_step_names_the_policy() -> None:
    config = _config({"metadata": "standard", "stats": "bands", "factor-leakage": {"factors": ["site"]}})
    chain = AuditWorkflow.chain(config)
    factor_types = {"label-health", "factor-summary", "balance", "diversity", "factor-triage", "factor-leakage"}
    entries = [entry for entry in chain.evaluators if entry.type in factor_types]
    assert sorted(entry.type for entry in entries) == sorted(factor_types)
    assert {entry.type: getattr(entry, "metadata", None) for entry in entries} == dict.fromkeys(
        factor_types, "standard"
    )
    (gaps,) = [step for step in _steps(config) if step.get("combine") == "factor-gaps"]
    assert gaps["metadata"] == "standard"
    stats = {
        entry.type: getattr(entry, "stats", None)
        for entry in chain.evaluators
        if entry.type in ("outliers", "duplicates")
    }
    assert stats == {"outliers": "bands", "duplicates": "bands"}


def test_the_chain_declares_the_five_questions_a_record_and_a_verdict() -> None:
    chain = AuditWorkflow.chain(_config())
    assert [group.heading for group in chain.groups] == _HEADINGS
    assert chain.blocking == ("leakage", "untrained-classes")
    assert chain.reference == "train"
    assert chain.record is not None
    assert chain.record.steps == ("label-health", "content-digest")


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (
            {"blocking": ["nope"]},
            (
                "`blocking` names `nope`, which this audit's chain has no check for. Its checks: class-coverage, "
                "class-imbalance, class-sufficiency, dimensional-completeness, distribution-shift, eval-coverage, "
                "factor-coverage-gaps, image-duplicates, image-outliers, leakage, metadata-issues, shortcut-risk, "
                "stratification, untrained-classes."
            ),
        ),
        (
            {"accepted": {"label-conformance": "x"}},
            "`accepted` names `label-conformance`, which this audit's chain has no check for.",
        ),
        ({"accepted": {"class-imbalance": "  "}}, "at least 1 character"),
        ({"accepted": {"class-imbalance": "Rare by design."}}, None),
    ],
    ids=["unknown-blocking", "no-ontology", "blank-reason", "loads"],
)
def test_blocking_and_accepted_name_only_checks_in_the_chain(entry: dict[str, Any], message: str | None) -> None:
    if message is None:
        assert _config(entry).accepted == {"class-imbalance": "Rare by design."}
        return
    with pytest.raises(ValidationError, match=re.escape(message)):
        _config(entry)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        *(({key: 1}, f"audit's `{key}` is refused: {MOVED[key]}.") for key in sorted(MOVED)),
        *(
            ({"health_thresholds": {key: 1}}, f"`health_thresholds.{key}` is refused: it is {CHECKS_MOVED[key]}.")
            for key in sorted(CHECKS_MOVED)
        ),
        ({"health_thresholds": {}}, "`health_thresholds` is now `checks:`, keyed by check type"),
    ],
    ids=[*sorted(MOVED), *(f"health_thresholds.{key}" for key in sorted(CHECKS_MOVED)), "health_thresholds"],
)
def test_each_data_analysis_field_is_refused_with_its_replacement(entry: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(message)):
        _config(entry)


def test_distribution_shift_derives_its_info_band() -> None:
    config = _config({"checks": {"distribution-shift": {"warning": 0.2}}})
    (shift,) = [step for step in _steps(config) if step["name"] == "distribution-shift"]
    assert (shift["warning"], shift["info"]) == (0.2, 0.08)
    with pytest.raises(ValidationError, match=re.escape("`info` (0.3) must not exceed `warning` (0.2).")) as caught:
        _config({"checks": {"distribution-shift": {"warning": 0.2, "info": 0.3}}})
    assert caught.value.errors()[0]["loc"][0] == "checks"
    assert AuditConfig.model_validate(config.model_dump()) == config
    assert AuditConfig.model_validate(config.model_dump(by_alias=False)) == config


def test_checks_take_hyphenated_and_snake_case_keys_and_dump_hyphenated() -> None:
    hyphenated = _config({"checks": {"image-outliers": {"warning": 5.0}}})
    snake = _config({"checks": {"image_outliers": {"warning": 5.0}}})
    assert hyphenated == snake
    dumped = hyphenated.model_dump()
    assert dumped["checks"]["image-outliers"] == {"warning": 5.0}
    assert "image_outliers" not in dumped["checks"]
    assert {"factor-gaps", "ood-kneighbors"} <= set(dumped)


def _pipeline(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "datasets": [{"name": "ds", "format": "huggingface", "path": "./d", "task": "image_classification"}],
        "sources": [{"name": "train", "dataset": "ds"}, {"name": "val", "dataset": "ds"}],
        "workflows": [{"name": "w", "type": "audit", **_OUTLIERS, **entry}],
        "tasks": [{"name": "t", "workflow": "w", "sources": ["train", "val"]}],
    }


def test_blocking_and_accepted_round_trip_through_load_and_save() -> None:
    entry = {"blocking": ["leakage"], "accepted": {"class-imbalance": "Rare class by design."}}
    config = PipelineConfig.model_validate(_pipeline(entry))
    reloaded = PipelineConfig.model_validate(config.model_dump(mode="json"))
    assert reloaded == config
    (audit,) = reloaded.workflows or []
    assert isinstance(audit, AuditConfig)
    assert (audit.blocking, audit.accepted) == (["leakage"], {"class-imbalance": "Rare class by design."})

    DatasetCache.clear_instances()
    result = run_tasks(
        chain_pipeline(
            workflows=[{"name": "w", "type": "audit", **_OUTLIERS, **entry}],
            tasks=[{"name": "t", "workflow": "w", "sources": ["train", "val"]}],
            datasets={"train": ToyImages(), "val": ToyImages(seed=1)},
        )
    )["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    recorded = result.metadata.resolved_config["workflow"]
    assert (recorded["blocking"], recorded["accepted"]) == (entry["blocking"], entry["accepted"])


def test_a_data_analysis_entry_is_refused_naming_audit_and_each_moved_key() -> None:
    pipeline = _pipeline({})
    pipeline["workflows"] = [{"name": "w", "type": "data-analysis", "outlier_method": "zscore"}]
    with pytest.raises(ValidationError) as caught:
        PipelineConfig.model_validate(pipeline)
    message = caught.value.errors()[0]["msg"].removeprefix("Value error, ")
    assert message.startswith("`data-analysis` is now `audit`. ")
    for key in [*MOVED, *(f"health_thresholds.{key}" for key in CHECKS_MOVED)]:
        assert f"`{key}` → " in message
    assert "`health_thresholds.image_outliers` → `checks.image-outliers.warning`" in message
    assert "`health_thresholds` → `checks:`" in message
    assert "`balance` → refused: `balance` always runs, and is skipped on metadata with no factors" in message
    assert "`outlier_flags` → `outliers.flags`;" in message


def test_data_analysis_is_no_workflow_type() -> None:
    assert "data-analysis" not in {workflow.name for workflow in list_workflows()}
    with pytest.raises(ValueError, match="Unknown workflow: 'data-analysis'"):
        get_workflow("data-analysis")


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_blocking_and_accepted_round_trip_through_the_builder(tmp_path: Path, suffix: str) -> None:
    entry = {"blocking": ["leakage"], "accepted": {"class-imbalance": "Rare class by design."}}
    state = ConfigState()
    state.load_dict(PipelineConfig.model_validate(_pipeline(entry)))
    path = tmp_path / f"pipeline{suffix}"
    state.save_file(path)
    (audit,) = load_config(path).workflows or []
    assert isinstance(audit, AuditConfig)
    assert (audit.blocking, audit.accepted) == (["leakage"], {"class-imbalance": "Rare class by design."})


def test_the_record_prints_each_checks_thresholds_as_its_criteria() -> None:
    config = _config({"accepted": {"class-imbalance": "Rare class by design."}})
    metadata = ChainMetadata(workflow="w", resolved_config={"workflow": config.model_dump(mode="json")})
    result = ChainResult(type="audit", success=True, metadata=metadata, output=ChainOutput({}), steps={})
    record = _record_section(result, AuditWorkflow.chain(config))
    (criteria,) = [block for block in record.blocks if isinstance(block, Section) and block.title == "Criteria"]
    (fields,) = criteria.blocks
    assert isinstance(fields, Fields)
    lines = dict(fields.items)
    assert lines["image-outliers"] == "warning 3.0"
    assert lines["image-duplicates"] == "exact 0.0, near 5.0"
    assert lines["class-imbalance"] == "warning 5.0, info none, empty false"
    assert lines["distribution-shift"] == "warning 0.5, info 0.2"
    assert lines["Blocking"] == "leakage, untrained-classes"
    assert lines["Accepted"] == "class-imbalance: Rare class by design."
