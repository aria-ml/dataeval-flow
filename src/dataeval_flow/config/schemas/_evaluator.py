"""Evaluator configuration schemas, one class per evaluator type.

An evaluator runs exactly one DataEval evaluator and reports its determinations, with no
health status. Each class adds ``name`` and ``type`` to that evaluator's parameters, as
the workflow configurations do.
"""

__all__ = ["DuplicatesEvaluatorConfig", "OutliersEvaluatorConfig"]

from typing import Literal

from pydantic import Field

from dataeval_flow.evaluators.quality.params import DuplicatesParameters, OutliersParameters


class DuplicatesEvaluatorConfig(DuplicatesParameters):
    """Typed evaluator configuration for ``quality.duplicates``.

    Example YAML::

        evaluators:
          - name: dupes
            type: quality.duplicates
            flags: [hash_basic, hash_d4]
    """

    name: str = Field(description="Identifier for this evaluator")
    type: Literal["quality.duplicates"] = "quality.duplicates"


class OutliersEvaluatorConfig(OutliersParameters):
    """Typed evaluator configuration for ``quality.outliers``.

    Example YAML::

        evaluators:
          - name: outliers
            type: quality.outliers
            flags: [pixel, visual]
            outlier_threshold: [zscore, 3.0]
    """

    name: str = Field(description="Identifier for this evaluator")
    type: Literal["quality.outliers"] = "quality.outliers"
