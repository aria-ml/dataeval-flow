"""A step type's friendly title, looked up by kind and type id, for a report's banner and step headings."""

__all__ = ["step_title"]


def step_title(kind: str, type_id: str) -> str:
    """The friendly title of step type `type_id` of `kind`, or the id itself when no such type is registered."""
    from dataeval_flow.evaluators._registry import EVALUATORS
    from dataeval_flow.steps._registry import CHECKS, COMBINES, TRANSFORMS
    from dataeval_flow.workflows._registry import WORKFLOWS

    registry = {
        "evaluator": EVALUATORS,
        "transform": TRANSFORMS,
        "combine": COMBINES,
        "check": CHECKS,
        "workflow": WORKFLOWS,
    }.get(kind)
    return registry.get(type_id).title if registry is not None and type_id in registry.names() else type_id
