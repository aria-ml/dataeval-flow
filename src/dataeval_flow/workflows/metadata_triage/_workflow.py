"""The ``metadata-triage`` preset: what a Dataset's metadata failed to read, and a policy that repairs it
(spec §10.10)."""

__all__ = ["MetadataTriageWorkflow"]

from typing import ClassVar

from dataeval_flow.evaluators.quality import TriageConfig
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import Preset, PresetChain
from dataeval_flow.workflows.metadata_triage._config import MetadataTriageConfig


class MetadataTriageWorkflow(Preset, Workflow[MetadataTriageConfig, ChainResult]):
    """Surface what a metadata run silently failed to read, and suggest how to fix it.

    The settings expand to two steps on the task's one source, ``data``:

    - ``triage`` (the ``triage`` evaluator): each factor the run could not read as configured, a policy stanza that
      repairs them, and, with ``verify``, what the repair recovers;
    - ``issues`` (the ``metadata-issues`` check): one finding per kind of issue, a warning where any is blocking, then
      the suggested policy, and what verification recovered or that it failed.

    It makes no Dataset, so it declares no outputs. Run as a step of a custom workflow, its findings are the chain's.
    """

    name: ClassVar[str] = "metadata-triage"
    title: ClassVar[str] = "Metadata Triage"
    description: ClassVar[str] = "Report unreadable and unpinned metadata factors, with suggested corrections"
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

    @classmethod
    def chain(cls, config: MetadataTriageConfig) -> PresetChain:
        """The ``triage`` entry these settings configure, and the check that makes its findings."""
        triage = TriageConfig(
            name="triage",
            metadata=config.metadata,
            verify=config.verify,
            default_bins=config.default_bins,
            min_missing_fraction=config.min_missing_fraction,
        )
        steps = [
            {"name": "triage", "evaluator": "triage", "input": "data"},
            {"name": "issues", "check": "metadata-issues", "input": "triage", "max_examples": config.max_examples},
        ]
        return PresetChain(steps=steps, evaluators=[triage])
