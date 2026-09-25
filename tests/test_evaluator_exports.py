"""The evaluator names a Python caller needs are importable from the evaluator packages."""

import dataeval_flow.evaluators
import dataeval_flow.evaluators.quality


def test_the_evaluator_framework_is_exported():
    for name in (
        "Evaluator",
        "EvaluatorConfig",
        "EvaluatorInputs",
        "EvaluatorResult",
        "get_evaluator",
        "list_evaluators",
    ):
        assert name in dataeval_flow.evaluators.__all__
        assert getattr(dataeval_flow.evaluators, name)


def test_the_quality_evaluators_are_exported():
    for name in ("DuplicatesConfig", "DuplicatesResult", "OutliersConfig", "OutliersResult"):
        assert name in dataeval_flow.evaluators.quality.__all__
        assert getattr(dataeval_flow.evaluators.quality, name)
