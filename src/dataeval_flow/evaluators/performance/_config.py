"""Config for the ``sufficiency`` evaluator: DataEval's Sufficiency, with the training code a trainer file supplies.

Field names are DataEval's argument names, except ``trainer``, which is Flow's. An unset field is not passed, so
DataEval's own default applies. Nothing here imports the trainer: it is loaded when the run starts.
"""

__all__ = ["SufficiencyConfig", "TrainerConfig"]

import os
from itertools import pairwise
from pathlib import PurePosixPath
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config._paths import validate_config_path
from dataeval_flow.evaluators._base import EvaluatorConfig
from dataeval_flow.evaluators.performance._result import SufficiencyResult

TRAINERS = "trainers"
"""The data root's folder a trainer file must sit in."""


class TrainerConfig(BaseModel):
    """Which :class:`~dataeval_flow.evaluators.performance.Trainer` a ``sufficiency`` run builds, and with what.

    ``class`` is a Python keyword, so the field is ``class_`` in Python and ``class`` in a config file; either name
    is accepted, and ``class`` is written back. The path is checked when the config loads, without reading the file;
    the file is read and run only when the run starts.

    Example YAML::

        trainer:
          path: trainers/centroids.py
          class: CentroidTrainer
          params: {size: 32}
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    path: str = Field(
        description=(
            "The trainer's Python file, relative to the data root and inside its `trainers/` folder, such as "
            "`trainers/centroids.py`. One file: importing a sibling file is not supported."
        )
    )
    # Split aliases rather than `alias=`, so a type checker takes `class_` as the constructor's keyword.
    class_: str = Field(
        validation_alias="class",
        serialization_alias="class",
        description="The `Trainer` subclass in the file to build.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments the class is constructed with. Free-form here: the trainer's own code checks them "
            "when the run starts."
        ),
    )

    @field_validator("path")
    @classmethod
    def _inside_trainers(cls, value: str) -> str:
        """Relative, ending in ``.py``, and inside ``trainers/`` once normalized: ``trainers/../x.py`` is ``x.py``."""
        validate_config_path(value)
        parts = PurePosixPath(os.path.normpath(value)).parts
        if len(parts) < 2 or parts[0] != TRAINERS or not value.endswith(".py"):
            raise ValueError(
                f"A trainer is a `.py` file in the data root's `{TRAINERS}/` folder, such as "
                f"`{TRAINERS}/my_trainer.py`; got '{value}'."
            )
        return value


class SufficiencyConfig(EvaluatorConfig[SufficiencyResult]):
    """Config for ``sufficiency``, DataEval's Sufficiency.

    Trains the trainer's model on growing subsets of the first source, the training set, evaluates it on the second,
    the test set, and fits a power law to each metric, from which it projects how much data a metric's target takes.
    The sources are told apart by position alone: the task's first source is the training set. A size is a number of
    items drawn with replacement, not of distinct items.

    Every parameter, its DataEval argument and its unset behavior is listed in the Evaluator Catalog
    (``reference/evaluators``), and ``dataeval-flow evaluators sufficiency`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: sufficiency
            type: sufficiency
            schedule: [100, 200, 400, 800]
            trainer:
              path: trainers/centroids.py
              class: CentroidTrainer
    """

    type: str = Field(default="sufficiency", description="The evaluator type this entry configures: `sufficiency`.")
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.DATASET}), sources=SourceCount.TWO)

    trainer: TrainerConfig = Field(
        description=(
            "(DataEval Flow) The trainer that supplies the model and the training, evaluation and reset strategies: "
            "a `Trainer` subclass in a file under the data root's `trainers/` folder, loaded when the run starts."
        )
    )
    runs: int | None = Field(
        default=None,
        ge=1,
        description="Independent training runs, each from a reset model, averaged. Unset uses DataEval's default (1).",
    )
    substeps: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Sizes the default schedule measures, spaced geometrically from 1% of the training set to all of it. Unset "
            "uses DataEval's default (5). Not with `schedule`, which DataEval would let override it silently."
        ),
    )
    unit_interval: bool | None = Field(
        default=None,
        description=(
            "Whether the metrics lie on [0, 1], such as accuracy, which bounds the fitted curve. Unset uses DataEval's "
            "default (true)."
        ),
    )
    schedule: list[int] | None = Field(
        default=None,
        description=(
            "The training-set sizes to measure, strictly increasing positive integers, none larger than the training "
            "set. Each is a number of items drawn with replacement. Unset uses the default schedule, which needs a "
            "training set of at least 100 items."
        ),
    )

    @field_validator("schedule")
    @classmethod
    def _increasing(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and (not value or value[0] < 1 or any(b <= a for a, b in pairwise(value))):
            raise ValueError(f"`schedule` takes strictly increasing positive integers; got {value}.")
        return value

    @model_validator(mode="after")
    def _schedule_or_substeps(self) -> Self:
        if self.schedule is not None and self.substeps is not None:
            raise ValueError("Set `schedule` or `substeps`, not both: DataEval ignores `substeps` under a schedule.")
        return self
