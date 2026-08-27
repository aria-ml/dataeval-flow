"""Metadata schemas: the policy a run is given, and the record it hands back."""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from dataeval_flow.config._paths import validate_config_path
from dataeval_flow.config.schemas._task import AutoBinMethod, FactorSource


class ResultMetadata(BaseModel):
    """Base metadata envelope for workflow results.

    Contains JATIC-required fields (version, timestamp, tool info,
    dataset identifiers).
    """

    version: str = "1.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_id: str | Sequence[str] = ""
    label_source: str | None = None
    model_id: str | None = None
    preprocessor_id: str | None = None
    selection_id: str | None = None
    source_descriptions: Sequence[str] = ()
    resolved_config: dict[str, Any] = Field(default_factory=dict)
    tool: str = "dataeval-flow"
    tool_version: str = ""
    execution_time_s: float | None = None

    #: How each metadata factor was encoded, and how well that encoding fits.
    #: Per factor: its type and level, the ``encoding`` applied (edges or
    #: vocabulary, who chose them, and how they were placed), the name each code
    #: reads as, and the ``fit`` this run's rows made against it — counts,
    #: occupied spans, and declared bins nothing reached.  ``None`` for workflows
    #: that build no metadata.  Recorded because the encoding decides what every
    #: evaluator reads, and it is otherwise reported only in logs no envelope
    #: references.
    metadata_binning: dict[str, Any] | None = None

    #: Fingerprint of the encoding every factor was read under, or ``None`` where
    #: the workflow built no metadata or its splits did not share one.  Comparing
    #: two runs is only sound if each can say which cuts produced it: without this,
    #: a bias score that moved is unattributable between *the override worked* and
    #: *the data changed*, which are the two readings a reader is trying to tell
    #: apart.  Same digest and same data means the numbers are comparable.
    encoding_digest: str | None = None

    #: Library diagnostics raised while the workflow ran — the decisions
    #: DataEval made on the caller's behalf and the ranges it could not resolve.
    #: Empty when the run raised none.
    diagnostics: Sequence[str] = ()


class MetadataPolicyConfig(BaseModel):
    """A named metadata policy, referenced by the workflows that share it.

    A bin edge is a claim about the world — *below 0 °C is freezing* — so where the cuts
    fell, and who chose them, is part of what a result means.  That makes the encoding a
    decision worth writing down once and pointing several workflows at, rather than a
    setting each of them spells out again: two workflows over one dataset that cut it
    differently produce numbers that merge into one result file and cannot be compared.

    Defined once under ``metadata:`` and referenced by name, like ``datasets``, ``views``,
    ``sources`` and ``extractors``.

    YAML example::

        metadata:
          - name: standard
            encoding: policy/factor_bins.json
            strict: true
            exclude: [id, filename]

        workflows:
          - name: coverage_check
            type: data-coverage
            metadata: standard
    """

    name: str = Field(description="Identifier for this policy")

    encoding: str | None = Field(
        default=None,
        description=(
            "Path, under the data root, to a committed encoding descriptor — the artifact "
            "`dataeval-flow encoding` writes. Applies the recorded cuts and vocabularies to "
            "this run instead of deriving them from its own draw, which is what makes two "
            "runs over different data comparable. Factors it does not name are encoded "
            "normally."
        ),
    )
    factor_levels: Mapping[str, Sequence[Any]] | None = Field(
        default=None,
        description=(
            "Vocabularies declared ahead of the data, one per factor: code i means "
            "levels[i], so two datasets declared against the same list share an alphabet "
            "without either having been structured first. The categorical counterpart to "
            "`continuous_factor_bins`."
        ),
    )
    strict: bool = Field(
        default=False,
        description=(
            "Whether a value no declared vocabulary holds is an error. The default admits "
            "it, which is what extension wants; true is for a closed taxonomy that should "
            "report the data leaving it rather than be widened to fit. Bin edges are "
            "unaffected — an unseen magnitude lands in an end bin either way."
        ),
    )
    reference_split: str | None = Field(
        default=None,
        description=(
            "Which split's encoding the whole run uses, for a workflow that reads several. "
            "Defaults to the first split the task names. Splits encoded independently land "
            "on different cuts for the same factor, so their per-factor statistics are not "
            "comparable; one reference makes them so."
        ),
    )

    auto_bin_method: AutoBinMethod | None = Field(
        default=None,
        description=(
            "How a continuous factor no declaration reaches is cut. Governs only those: a "
            "factor named by `encoding` or `continuous_factor_bins` is cut as declared."
        ),
    )
    exclude: Sequence[str] = Field(
        default_factory=list,
        description="Factor names removed before any evaluator sees them.",
    )
    continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = Field(
        default=None,
        description=(
            "Bin count (int) or explicit edges (list) per factor. Edges carry meaning and "
            "travel with the configuration; a count only says how many, and where the cuts "
            "land is still read off this sample. Mutually exclusive with `encoding` per "
            "factor."
        ),
    )
    factor_source: FactorSource | None = Field(
        default=None,
        description=(
            "Which representation of each factor the bias statistics read — `coded`, "
            "`values`, or `auto`. It moves every number they report, so workflows meant to "
            "be compared want one answer."
        ),
    )
    intrinsic_factors: Sequence[str] = Field(
        default_factory=list,
        description=(
            "Statistic families computed from the imagery and injected as metadata "
            "factors, so bias analysis can read them: `visual`, `pixel`, `dimension`. "
            "Named as families rather than individual statistics because that is the "
            "granularity worth deciding — `visual` and `pixel` are the bias workhorses, "
            "`dimension` is meaningful only for variable-size imagery. Hashes are never "
            "injected: they are near-unique per item, so a factor made from one "
            "correlates with everything and describes nothing. Empty means inject "
            "nothing, which costs exactly what a run costs today."
        ),
    )

    @field_validator("encoding")
    @classmethod
    def _check_encoding_path(cls, value: str | None) -> str | None:
        """Keep the descriptor path portable, like every other config path."""
        return None if value is None else validate_config_path(value)
