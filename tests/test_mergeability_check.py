"""The `mergeability` check: legacy data-coverage's Label Alignment finding, with its Relabel stanza (coverage spec
§3.4)."""

from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from dataeval_flow import run
from dataeval_flow._blocks import Code
from dataeval_flow.evaluators.scope import LabelAlignmentConfig
from dataeval_flow.steps.checks import MergeabilityCheck, MergeabilityConfig
from dataeval_flow.steps.checks._alignment import relabel_stanza, yaml_scalar
from tests.evaluator_toys import ToyImages


def _judge(ontology: Any) -> Any:
    result = run(LabelAlignmentConfig(ontology=ontology), ToyImages(count=10))
    assert result.success, result.errors
    node = SimpleNamespace(value=result.output)
    (finding,) = MergeabilityCheck().run(MergeabilityConfig(input="alignment"), {"input": node}, None)  # type: ignore[arg-type]
    return finding


def test_a_lossless_alignment_is_ok_and_carries_its_stanza() -> None:
    finding = _judge({"a": None, "b": None})
    assert (finding.severity, finding.title, finding.brief) == ("ok", "Mergeability", None)
    assert finding.description.startswith("Mergeability: lossless.")
    (code,) = [block for block in finding.blocks if isinstance(block, Code)]
    assert "class_remap" in code.text


def test_a_partial_alignment_warns() -> None:
    finding = _judge({"a": None})
    assert finding.severity == "warning"
    assert finding.description.startswith("Mergeability: partial.")


@pytest.mark.parametrize(("mergeability", "severity"), [("lossless", "ok"), ("lossy", "info"), ("partial", "warning")])
def test_mergeability_maps_to_severity_and_an_ambiguous_label_forces_a_warning(
    mergeability: str, severity: str
) -> None:
    def alignment(ambiguous: list[str]) -> Any:
        return SimpleNamespace(
            mergeability=mergeability,
            correspondences=[],
            unaligned_source=[],
            unaligned_target=[],
            paste_remap={},
            target_vocabulary=[],
            ambiguous_labels=ambiguous,
            label_space_digest=None,
        )

    def judge(ambiguous: list[str]) -> Any:
        node = SimpleNamespace(value=SimpleNamespace(alignment=alignment(ambiguous)))
        (finding,) = MergeabilityCheck().run(MergeabilityConfig(input="a"), {"input": node}, None)  # type: ignore[arg-type]
        return finding

    assert judge([]).severity == severity
    assert judge(["car"]).severity == "warning"


def test_the_stanza_round_trips_labels_yaml_would_misread() -> None:
    stanza = relabel_stanza({"0": "on", "car": "Car"}, ["Car", "on"])
    parsed = yaml.safe_load("view:\n    operations:\n" + stanza)["view"]
    (operation,) = parsed["operations"]
    assert operation["params"]["class_remap"] == {"0": "on", "car": "Car"}
    assert yaml_scalar("plain") == "plain"
    assert yaml_scalar("null") == '"null"'
