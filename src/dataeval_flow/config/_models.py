"""Pipeline and workflow composition models — SourceConfig, PipelineConfig."""

__all__ = [
    "PipelineConfig",
    "SourceConfig",
]

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from dataeval_flow.config.schemas import (
    DatasetConfig,
    DatasetProtocolConfig,
    ExtractorConfig,
    MetadataPolicyConfig,
    PreprocessorConfig,
    TaskConfig,
    ViewConfig,
    WorkflowConfig,
)

# ---------------------------------------------------------------------------
# Source — dataset + optional view
# ---------------------------------------------------------------------------


class SourceConfig(BaseModel):
    """Named source definition — bundles a dataset with an optional view.

    YAML example::

        sources:
          - name: cifar_train_subset
            dataset: cifar10_train
            view: first_5k

    The legacy ``selection`` key is accepted as a deprecated alias for ``view``.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True)

    name: str = Field(description="Identifier for the source")
    dataset: str = Field(description="Reference to a dataset name")
    view: str | None = Field(
        default=None,
        validation_alias=AliasChoices("view", "selection"),
        description="Reference to a view name (optional)",
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class LoggingConfig(BaseModel):
    """Logging level configuration."""

    app_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"
    lib_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "WARNING"


# ---------------------------------------------------------------------------
# Pipeline (top-level)
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration.

    All sections use a define-once, reference-by-name pattern.
    Sources compose datasets with optional views; extractors
    compose model type/params with optional preprocessors.
    Tasks reference workflows, sources, and extractors by name.

    The legacy ``selections`` key is accepted as a deprecated alias for
    ``views`` (with a :class:`DeprecationWarning`).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True)

    # Logging
    logging: LoggingConfig | None = None

    # Reproducibility [CR-7-S-1]
    seed: int | None = Field(
        default=None,
        description=(
            "Seed for every stochastic component of the run (clustering, random splits, "
            "shuffled views, sampling). Applied through DataEval's seed configuration before "
            "each task, so a task's result does not depend on what ran before it. "
            "None (the default) leaves randomness unseeded."
        ),
    )
    deterministic: bool = Field(
        default=False,
        description=(
            "Force PyTorch to use deterministic algorithms. Only meaningful alongside `seed`. "
            "Improves run-to-run reproducibility on GPU at some cost to performance."
        ),
    )

    # Named resource pools
    datasets: Sequence[DatasetConfig | DatasetProtocolConfig] | None = None
    preprocessors: Sequence[PreprocessorConfig] | None = None
    views: Sequence[ViewConfig] | None = Field(
        default=None,
        validation_alias=AliasChoices("views", "selections"),
        description="Named view pipeline definitions (dataset operations), referenced by sources",
    )

    metadata: Sequence[MetadataPolicyConfig] | None = Field(
        default=None,
        description=(
            "Named metadata policy definitions (encoding, vocabularies, exclusions), "
            "referenced by workflows. Defined once and shared so that workflows meant to "
            "be compared read their factors under one encoding."
        ),
    )

    # Composition layers
    sources: Sequence[SourceConfig] | None = Field(
        default=None,
        description="Named source definitions (dataset + optional view)",
    )
    extractors: Sequence[ExtractorConfig] | None = Field(
        default=None,
        description="Named extractor definitions (model type + params + optional preprocessor + batch_size)",
    )

    # Execution
    workflows: Sequence[WorkflowConfig] | None = Field(
        default=None,
        description="Named workflow configurations (type + params), referenced by tasks",
    )
    tasks: Sequence[TaskConfig] | None = None

    @model_validator(mode="before")
    @classmethod
    def _warn_legacy_selection_keys(cls, data: Any) -> Any:
        """Emit deprecation warnings for the legacy ``selections``/``selection`` keys."""
        if isinstance(data, Mapping):
            if "selections" in data and "views" not in data:
                warnings.warn(
                    "The 'selections' key is deprecated; use 'views' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            for source in data.get("sources") or []:
                if isinstance(source, Mapping) and "selection" in source and "view" not in source:
                    warnings.warn(
                        "The 'selection' key in a source is deprecated; use 'view' instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
        return data

    @model_validator(mode="after")
    def _check_unique_names(self) -> "PipelineConfig":
        """Raise if any named pool contains duplicate names.

        The pools are discovered rather than listed.  A reference resolves by first match,
        so a pool omitted from a hand-maintained list silently keeps the first definition
        and drops the second — most likely the one the user just edited — with the run
        reporting no problem.
        """
        for section_name in type(self).model_fields:
            items = getattr(self, section_name, None)
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                continue
            seen: set[str] = set()
            for item in items:
                name = getattr(item, "name", None)
                if name is None:
                    break
                if name in seen:
                    raise ValueError(f"Duplicate name '{name}' in {section_name}")
                seen.add(name)
        return self
