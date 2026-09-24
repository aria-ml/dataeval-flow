"""Evaluators name a stats policy through `StatsPolicyRef`, which carries no deprecated fields."""

from pydantic import BaseModel

from dataeval_flow.config import PipelineConfig
from dataeval_flow.config.schemas import StatsPolicyConfig, StatsPolicyRef
from dataeval_flow.workflow.base import StatsConfigMixin
from dataeval_flow.workflow.orchestrator import _resolve_stats_policy


def test_the_workflow_mixin_extends_the_ref():
    assert issubclass(StatsConfigMixin, StatsPolicyRef)


def test_the_ref_carries_no_deprecated_field():
    assert set(StatsPolicyRef.model_fields) == {"stats"}


def test_a_bare_ref_resolves_its_named_policy():
    config = PipelineConfig(
        stats=[StatsPolicyConfig.model_validate({"name": "whole", "measure": [{"bands": None, "families": ["hash"]}]})]
    )
    resolved = _resolve_stats_policy(StatsPolicyRef(stats="whole"), config, {})
    assert resolved is not None
    assert resolved.name == "whole"


def test_a_model_without_the_ref_resolves_nothing():
    class SomeModel(BaseModel):
        name: str = "test"

    assert _resolve_stats_policy(SomeModel(), PipelineConfig(), {}) is None
