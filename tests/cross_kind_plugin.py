"""A workflow plugin that lists the evaluators while the workflow registry imports it: allowed, another registry."""

from dataeval_flow.evaluators import list_evaluators
from tests.example_plugin import CountWorkflow

__all__ = ["EVALUATORS_SEEN", "CountWorkflow"]

EVALUATORS_SEEN = [cls.name for cls in list_evaluators()]
