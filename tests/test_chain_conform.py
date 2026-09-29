"""Aligning a Dataset's labels to an ontology, and conforming it, gated by `allow:` (spec §6.2, §6.4)."""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.evaluators.scope import LabelAlignmentConfig, LabelAlignmentOutput
from dataeval_flow.steps import ChainResult, TransformContext
from dataeval_flow.steps.transforms import ConformConfig, ConformTransform
from tests.chain_toys import ToyDetections, chain_pipeline

_ONTOLOGY = {
    "name": "vehicles",
    "concepts": [
        {"id": "Vehicle", "label": "Vehicle", "synonyms": ["car", "truck"]},
        {"id": "Person", "label": "Person", "synonyms": ["person", "pedestrian"]},
    ],
}
_LOSSLESS = ToyDetections([[0], [1], [0, 1]], {0: "car", 1: "person"}, dataset_id="lossless")
_LOSSY = ToyDetections([[0], [1], [2]], {0: "car", 1: "truck", 2: "pedestrian"}, dataset_id="lossy")
_PARTIAL = ToyDetections([[0], [1], [0, 1]], {0: "car", 1: "boat"}, dataset_id="partial")


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _run(
    dataset: Any,
    conform: dict[str, Any],
    *,
    extra_steps: list[dict[str, Any]] | None = None,
    ontology: dict[str, Any] | str = "vehicles",
) -> ChainResult:
    steps = [
        {"name": "aligned", "evaluator": "align", "input": "a"},
        {"name": "c", "transform": "conform", "input": "a", "alignment": "aligned", **conform},
        *(extra_steps or []),
    ]
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["a"], "steps": steps}],
        evaluators=[LabelAlignmentConfig(name="align", ontology=ontology)],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        datasets={"src": dataset},
        extra={"ontologies": [_ONTOLOGY]},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    return result


def test_the_alignment_evaluator_reports_mergeability_and_the_remap() -> None:
    result = _run(_LOSSY, {"allow": "lossy"})
    output = result.steps["aligned"].output
    assert isinstance(output, LabelAlignmentOutput)
    assert output.alignment.mergeability == "lossy"
    assert output.alignment.paste_remap == {"car": "Vehicle", "truck": "Vehicle", "pedestrian": "Person"}


def test_a_lossless_alignment_conforms_under_the_default() -> None:
    result = _run(_LOSSLESS, {})
    assert result.steps["c"].status == "ok"
    conformed = result.steps["c"].output
    assert dict(conformed.metadata["index2label"]) == {0: "Vehicle", 1: "Person"}


def test_a_lossy_collapse_is_refused_until_the_config_allows_it() -> None:
    refused = _run(_LOSSY, {})
    assert refused.steps["c"].status == "failed"
    assert "is lossy, beyond `allow: lossless`" in refused.steps["c"].errors[0]
    allowed = _run(_LOSSY, {"allow": "lossy"})
    assert allowed.steps["c"].status == "ok"
    assert allowed.steps["c"].details["collapses"] == {"Vehicle": ["car", "truck"]}  # type: ignore[index]


def test_an_override_can_settle_an_unaligned_class() -> None:
    refused = _run(_PARTIAL, {"allow": "lossy"})
    assert "is partial, beyond `allow: lossy`" in refused.steps["c"].errors[0]
    settled = _run(_PARTIAL, {"allow": "lossy", "class_remap": {"boat": "Vehicle"}})
    assert settled.steps["c"].status == "ok"


@pytest.mark.parametrize("allow", ["lossless", "lossy"])
def test_an_override_for_a_class_the_input_does_not_have_fails_the_step(allow: str) -> None:
    overrides = {"bus": "Vehicle", "person": "Person", "van": "Vehicle"}
    result = _run(_LOSSLESS, {"allow": allow, "class_remap": overrides})
    assert result.steps["c"].status == "failed"
    assert result.steps["c"].errors == [
        "ValueError: `class_remap` overrides `bus`, `van`, which `a` does not have: its classes are car, person."
    ]


def test_partial_drops_the_unaligned_class_and_says_so() -> None:
    result = _run(_PARTIAL, {"allow": "partial"})
    details = result.steps["c"].details or {}
    assert details["dropped_classes"] == ["boat"]
    assert details["dropped_items"] == 1


def test_conform_counts_its_drops_from_the_relabel_it_built() -> None:
    """The drops come from the Relabel `run` applied, not from whatever the output handed back is made of."""
    alignment = _run(_PARTIAL, {"allow": "partial"}).steps["aligned"].output
    config = ConformConfig(input="a", alignment="aligned", allow="partial")
    inputs = {"input": SimpleNamespace(value=_PARTIAL, address="a"), "alignment": SimpleNamespace(value=alignment)}
    transform = ConformTransform()
    transform.run(config, inputs, TransformContext(task="t", step="c"))
    details = transform.details(config, inputs, {"output": _PARTIAL})
    assert (details["dropped_classes"], details["dropped_items"]) == (["boat"], 1)


def test_conform_records_the_label_space_it_applied() -> None:
    result = _run(_LOSSLESS, {})
    (record,) = [record for record in result.metadata.label_space if record.source == "c"]
    assert record.digest == result.steps["aligned"].output.alignment.label_space_digest
    assert result.metadata.label_space_digest == record.digest


def test_conform_records_the_ontology_by_the_name_the_alignment_resolved() -> None:
    (pooled,) = [record for record in _run(_LOSSLESS, {}).metadata.label_space if record.source == "c"]
    assert pooled.ontology == "vehicles"
    inline = _run(_LOSSLESS, {}, ontology={"thing": ["car", "person"]})
    (record,) = [record for record in inline.metadata.label_space if record.source == "c"]
    assert record.ontology == "inline"


def test_an_alignment_of_another_dataset_fails_the_load() -> None:
    steps = [
        {"name": "few", "transform": "view", "input": "a", "operations": [{"type": "Limit", "params": {"size": 2}}]},
        {"name": "aligned", "evaluator": "align", "input": "few"},
        {"name": "c", "transform": "conform", "input": "a", "alignment": "aligned"},
    ]
    with pytest.raises(ValidationError, match="reads `aligned`, which was computed on `few`, not on `a`"):
        chain_pipeline(
            workflows=[{"name": "w", "inputs": ["a"], "steps": steps}],
            evaluators=[LabelAlignmentConfig(name="align", ontology="vehicles")],
            datasets={"src": _LOSSLESS},
            extra={"ontologies": [_ONTOLOGY]},
        )


def test_an_alignment_computed_on_one_element_of_a_list_conforms_that_element() -> None:
    steps = [
        {"name": "aligned", "evaluator": "align", "input": "all"},
        {"name": "c", "transform": "conform", "input": "all[lossy]", "alignment": "aligned[lossy]", "allow": "lossy"},
    ]
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": [{"name": "all", "list": True}], "steps": steps}],
        evaluators=[LabelAlignmentConfig(name="align", ontology="vehicles")],
        tasks=[{"name": "t", "workflow": "w", "sources": ["lossless", "lossy"]}],
        datasets={"lossless": _LOSSLESS, "lossy": _LOSSY},
        extra={"ontologies": [_ONTOLOGY]},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    conformed = result.steps["c"]
    assert (conformed.status, conformed.inputs) == ("ok", ["all[lossy]", "aligned[lossy]"])
    assert (conformed.details or {})["collapses"] == {"Vehicle": ["car", "truck"]}


def test_conform_derives_its_remap_once_per_run() -> None:
    from unittest.mock import patch

    with patch.object(ConformTransform, "_remap", wraps=ConformTransform._remap) as derived:
        result = _run(_LOSSY, {"allow": "lossy"})
    assert result.steps["c"].status == "ok"
    assert derived.call_count == 1
