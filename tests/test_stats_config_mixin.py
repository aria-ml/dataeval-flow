"""Workflows and evaluators name a stats policy through one mixin, which carries no deprecated fields."""

import pytest
from pydantic import BaseModel

from dataeval_flow._orchestrator import _resolve_stats_policy
from dataeval_flow.config import PipelineConfig, StatsConfigMixin, StatsPolicyConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig, OutliersConfig


def test_the_mixin_carries_no_deprecated_field():
    assert set(StatsConfigMixin.model_fields) == {"stats"}


@pytest.mark.parametrize("config_type", [DuplicatesConfig, OutliersConfig])
def test_the_evaluators_never_took_value_range(config_type: type[BaseModel]):
    assert issubclass(config_type, StatsConfigMixin)
    assert "value_range" not in config_type.model_fields


def test_a_bare_mixin_resolves_its_named_policy():
    config = PipelineConfig(
        stats=[StatsPolicyConfig.model_validate({"name": "whole", "measure": [{"bands": None, "families": ["hash"]}]})]
    )
    resolved = _resolve_stats_policy(StatsConfigMixin(stats="whole"), config, {})
    assert resolved is not None
    assert resolved.name == "whole"


def test_a_model_without_the_mixin_resolves_nothing():
    class SomeModel(BaseModel):
        name: str = "test"

    assert _resolve_stats_policy(SomeModel(), PipelineConfig(), {}) is None
