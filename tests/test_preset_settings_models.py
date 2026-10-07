"""A preset's settings block for a step type is one model, shared, with the step's own defaults (preset naming spec
R8)."""

from dataeval_flow.steps.checks._labels import ClassImbalanceConfig


def test_one_class_imbalance_model_with_the_check_s_defaults() -> None:
    from dataeval_flow.workflows.audit._config import AuditChecks
    from dataeval_flow.workflows.bias import ClassImbalanceSettings
    from dataeval_flow.workflows.bias._config import BiasChecks

    step = ClassImbalanceConfig(input="x").model_dump(exclude={"input"})
    assert ClassImbalanceSettings().model_dump() == step == {"warning": 5.0, "info": None, "empty": True}
    assert BiasChecks.model_fields["class_imbalance"].annotation is ClassImbalanceSettings
    assert AuditChecks.model_fields["class_imbalance"].annotation is ClassImbalanceSettings


def test_one_representation_model_for_coverage_and_label_space() -> None:
    from dataeval_flow.workflows.scope import RepresentationSettings, ScopeConfig
    from dataeval_flow.workflows.taxonomy import TaxonomyConfig

    assert ScopeConfig.model_fields["representation"].annotation is RepresentationSettings
    assert TaxonomyConfig.model_fields["representation"].annotation is RepresentationSettings
    assert RepresentationSettings().expected is None
