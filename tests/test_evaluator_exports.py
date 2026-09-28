"""The evaluator names a Python caller needs are importable from the evaluator packages."""

import dataeval_flow.evaluators
import dataeval_flow.evaluators.bias
import dataeval_flow.evaluators.quality
import dataeval_flow.evaluators.scope


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


def test_the_bias_evaluators_are_exported():
    for kind in ("Balance", "Diversity", "Parity"):
        for suffix in ("Config", "Evaluator", "Result"):
            assert f"{kind}{suffix}" in dataeval_flow.evaluators.bias.__all__
            assert getattr(dataeval_flow.evaluators.bias, f"{kind}{suffix}")


def test_the_scope_evaluators_are_exported():
    for kind in ("Representation", "Coverage", "Prioritize"):
        for suffix in ("Config", "Evaluator", "Result"):
            assert f"{kind}{suffix}" in dataeval_flow.evaluators.scope.__all__
            assert getattr(dataeval_flow.evaluators.scope, f"{kind}{suffix}")
