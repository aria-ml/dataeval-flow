"""The evaluator names a Python caller needs are importable from the top level."""

import dataeval_flow


def test_the_evaluator_api_is_exported():
    for name in (
        "EvaluatorConfig",
        "DuplicatesEvaluatorConfig",
        "OutliersEvaluatorConfig",
        "DuplicatesParameters",
        "OutliersParameters",
        "EvaluatorTaskConfig",
        "EvaluatorResult",
        "get_evaluator",
        "list_evaluators",
    ):
        assert name in dataeval_flow.__all__
        assert getattr(dataeval_flow, name)
