"""The ``label-space`` preset: a dataset's labels judged against a declared ontology (coverage spec §3.2)."""

__all__ = ["LabelSpaceWorkflow"]

from typing import Any, ClassVar

from dataeval_flow.evaluators.scope import (
    LabelAlignmentConfig,
    LabelReconciliationConfig,
    OntologyValidationConfig,
    RepresentationConfig,
)
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import Preset, PresetChain
from dataeval_flow.workflows.label_space._config import LabelSpaceConfig


class LabelSpaceWorkflow(Preset, Workflow[LabelSpaceConfig, ChainResult]):
    """Judges the task's one source, ``data``, against the entry's ontology.

    The settings expand to four evaluators, each judged by a check, in legacy data-coverage's order:

    - ``representation`` and ``leaf-coverage``: how many of the ontology's leaves have examples, the worklist, the
      empty branches and the unmet minimum shares;
    - ``reconciliation`` (``label-reconciliation``) and ``conformance`` (``label-conformance``): which class names
      resolve to one concept;
    - ``alignment`` (``label-alignment``) and ``mergeability``: whether the classes carry over, with the Relabel
      stanza;
    - ``structure`` (``ontology-validation``) and ``ontology-structure``: the ontology's own structure.

    No step is optional: the ontology is the preset's whole input. The result's ``label_space_digest`` is the
    alignment's, unless a source's ``Relabel`` already recorded a label space (coverage spec §3.5). It makes no
    Dataset, so it declares no outputs.
    """

    name: ClassVar[str] = "label-space"
    title: ClassVar[str] = "Label Space"
    description: ClassVar[str] = (
        "Judges a Dataset's labels against a declared ontology: leaf coverage, conformance, alignment and structure"
    )
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

    @classmethod
    def chain(cls, config: LabelSpaceConfig) -> PresetChain:
        """The four evaluators, each followed by its check."""
        limits = config.health_thresholds
        evaluators: list[Any] = [
            RepresentationConfig(name="representation", ontology=config.ontology, expected=config.expected),
            LabelReconciliationConfig(name="reconciliation", ontology=config.ontology),
            LabelAlignmentConfig(name="alignment", ontology=config.ontology),
            OntologyValidationConfig(name="structure", ontology=config.ontology, label_pattern=config.label_pattern),
        ]
        steps: list[dict[str, Any]] = [
            {"name": "representation", "evaluator": "representation", "input": "data"},
            {
                "name": "leaf-coverage",
                "check": "leaf-coverage",
                "input": "representation",
                **limits.leaf_coverage.model_dump(),
            },
            {"name": "reconciliation", "evaluator": "reconciliation", "input": "data"},
            {
                "name": "conformance",
                "check": "label-conformance",
                "input": "reconciliation",
                **limits.label_conformance.model_dump(),
            },
            {"name": "alignment", "evaluator": "alignment", "input": "data"},
            {"name": "mergeability", "check": "mergeability", "input": "alignment"},
            {"name": "structure", "evaluator": "structure", "input": "data"},
            {"name": "ontology-structure", "check": "ontology-structure", "input": "structure"},
        ]
        return PresetChain(steps=steps, evaluators=evaluators)
