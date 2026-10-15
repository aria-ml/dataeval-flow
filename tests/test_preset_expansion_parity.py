"""A preset's expansion, as a custom workflow, declares and runs as the preset does (preset decomposition spec §3.3)."""

from collections.abc import Mapping
from typing import Any

import pytest

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._verdict import Level
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows._preset import Preset, PresetChain, expand_preset
from dataeval_flow.workflows._registry import get_workflow
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages
from tests.test_audit_preset import _OUTLIERS
from tests.test_audit_run import _leaky, _three
from tests.test_chain_verdict_report import _outline
from tests.test_label_space_preset import _ONTOLOGY
from tests.test_naming_conventions import _MINIMAL


@pytest.fixture(autouse=True)
def _fresh_caches():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _checks(chain: PresetChain) -> set[str]:
    return {
        check
        for step in chain.steps
        if (check := step.get("check") if isinstance(step, Mapping) else step.check) is not None
    }


def _expected(chain: PresetChain) -> PresetChain | None:
    """`chain`'s declarations as an expansion writes them: groups and next steps over the checks it runs."""
    ran = _checks(chain)
    groups = tuple(
        type(group)(group.heading, checks, group.evidence)
        for group in chain.groups
        if (checks := tuple(check for check in group.checks if check in ran))
    )
    declares = groups or chain.reference or chain.record or chain.blocking is not None or chain.next_steps.by_reason
    if not (declares or chain.preflight != type(chain.preflight)()):
        return None
    by_check = {key: value for key, value in chain.next_steps.by_check.items() if key in ran}
    return PresetChain(
        steps=(),
        groups=groups,
        reference=chain.reference,
        record=chain.record,
        blocking=chain.blocking,
        accepted=dict(chain.accepted),
        next_steps=type(chain.next_steps)(by_check, dict(chain.next_steps.by_reason)),
        preflight=chain.preflight,
    )


# The keys a preset's config requires; every other key takes its default.
_REQUIRED = ("outliers", "ontology", "detectors")


def _required(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if key in _REQUIRED}


@pytest.mark.parametrize("type_id", sorted(_MINIMAL))
@pytest.mark.parametrize("settings", ["minimal", "defaults"])
def test_an_expansion_declares_what_its_preset_s_chain_declares(type_id: str, settings: str) -> None:
    preset = get_workflow(type_id)
    assert issubclass(preset, Preset)
    entry = _MINIMAL[type_id] if settings == "minimal" else _required(_MINIMAL[type_id])
    config = preset.config_type.model_validate({"name": "w", **entry})
    workflow, _ = expand_preset(config, preset)
    declared = workflow.declared_chain()
    expected = _expected(preset.chain(config))
    assert (declared is None) == (expected is None)
    if declared is not None and expected is not None:
        assert PresetChain(**{**declared.__dict__, "steps": ()}) == expected
    assert workflow.ontology == config.ontology


def _both(
    type_id: str, entry: dict[str, Any], datasets: dict[str, Any], **task: Any
) -> tuple[ChainResult, ChainResult]:
    """The preset task's result and its expansion's, run as a custom task over the same data."""
    preset = get_workflow(type_id)
    assert issubclass(preset, Preset)
    config = preset.config_type.model_validate({"name": "w", **entry})
    workflow, evaluators = expand_preset(config, preset)
    results = []
    for workflows, pool in (([{"name": "w", "type": type_id, **entry}], []), ([workflow], list(evaluators))):
        DatasetCache.clear_instances()
        pipeline = chain_pipeline(
            workflows=workflows,
            evaluators=pool,
            tasks=[{"name": "t", "workflow": "w", "sources": list(datasets), **task}],
            datasets=datasets,
            extractor="extractor" in task,
        )
        result = run_tasks(pipeline)["t"]
        assert isinstance(result, ChainResult)
        results.append(result)
    return results[0], results[1]


def _outcome(result: ChainResult) -> dict[str, Any]:
    """What a run concluded, less its identity: the verdict, each step's status and each finding."""
    return {
        "success": result.success,
        "verdict": result.verdict.model_dump(mode="json") if result.verdict is not None else None,
        "no_verdict": result.no_verdict,
        "steps": {name: (record.status, record.reason) for name, record in result.steps.items()},
        "findings": [(finding.title, finding.severity, finding.brief) for finding in result.findings],
        "label_space_digest": result.metadata.label_space_digest,
    }


# Each case: the preset, its entry, the data, the task's keys, and the verdict level the preset run itself reaches (None
# for a preset that declares no verdict), which anchors the case so two runs failing alike cannot pass it.
_CASES: dict[str, tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], Level | None]] = {
    "audit-train-only": ("audit", _OUTLIERS, {"train": ToyImages()}, {}, "ready-with-caveats"),
    "audit-three-splits": ("audit", _OUTLIERS, _three(), {}, "not-ready"),
    "audit-leaky-blocked": ("audit", _OUTLIERS, _leaky(), {}, "not-ready"),
    "audit-accepted": (
        "audit",
        {**_OUTLIERS, "accepted": {"leakage": "Shared by design."}},
        _leaky(),
        {},
        "ready-with-caveats",
    ),
    "audit-extractor": ("audit", _OUTLIERS, _three(), {"extractor": "flat"}, "not-ready"),
    "quality": ("quality", _MINIMAL["quality"], {"src": ToyImages()}, {}, None),
    "splits": ("splits", _MINIMAL["splits"], {"src": ToyImages(count=40)}, {}, None),
    "taxonomy": ("taxonomy", {"ontology": _ONTOLOGY}, {"src": ToyImages(count=20)}, {}, None),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_an_expansion_runs_as_its_preset_does(case: str) -> None:
    type_id, entry, datasets, task, level = _CASES[case]
    preset_result, custom_result = _both(type_id, entry, datasets, **task)
    assert preset_result.success, preset_result.errors
    assert not [name for name, record in preset_result.steps.items() if record.status == "failed"]
    assert (preset_result.verdict.level if preset_result.verdict is not None else None) == level
    assert _outcome(custom_result) == _outcome(preset_result)
    assert _outline(custom_result) == _outline(preset_result)
