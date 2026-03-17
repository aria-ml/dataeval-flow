"""Pipeline and workflow composition models — SourceConfig, PipelineConfig."""

__all__ = [
    "PipelineConfig",
    "SourceConfig",
]

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from dataeval_flow.config.schemas import (
    DatasetConfig,
    DatasetProtocolConfig,
    ExtractorConfig,
    PreprocessorConfig,
    SelectionConfig,
    TaskConfig,
    WorkflowConfig,
)

# ---------------------------------------------------------------------------
# Source — dataset + optional selection
# ---------------------------------------------------------------------------


class SourceConfig(BaseModel):
    """Named source definition — bundles a dataset with an optional selection.

    YAML example::

        sources:
          - name: cifar_train_subset
            dataset: cifar10_train
            selection: first_5k
    """

    name: str = Field(description="Identifier for the source")
    dataset: str = Field(description="Reference to a dataset name")
    selection: str | None = Field(default=None, description="Reference to a selection name (optional)")


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
    Sources compose datasets with optional selections; extractors
    compose model type/params with optional preprocessors.
    Tasks reference workflows, sources, and extractors by name.
    """

    # Logging
    logging: LoggingConfig | None = None

    # Named resource pools
    datasets: Sequence[DatasetConfig | DatasetProtocolConfig] | None = None
    preprocessors: Sequence[PreprocessorConfig] | None = None
    selections: Sequence[SelectionConfig] | None = None

    # Composition layers
    sources: Sequence[SourceConfig] | None = Field(
        default=None,
        description="Named source definitions (dataset + optional selection)",
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

    @model_validator(mode="after")
    def _check_unique_names(self) -> "PipelineConfig":
        """Raise if any section contains duplicate names."""
        sections: dict[str, Sequence | None] = {
            "datasets": self.datasets,
            "preprocessors": self.preprocessors,
            "selections": self.selections,
            "sources": self.sources,
            "extractors": self.extractors,
            "workflows": self.workflows,
            "tasks": self.tasks,
        }
        for section_name, items in sections.items():
            if items is None:
                continue
            seen: set[str] = set()
            for item in items:
                if item.name in seen:
                    raise ValueError(f"Duplicate name '{item.name}' in {section_name}")
                seen.add(item.name)
        return self
