"""`bias.*` against real DataEval, on metadata where `site` follows the class and `angle` does not."""

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import Result, run
from dataeval_flow._policy import ResolvedPolicy
from dataeval_flow.config import MetadataPolicyConfig
from dataeval_flow.evaluators.bias import (
    BalanceConfig,
    BalanceResult,
    DiversityConfig,
    DiversityResult,
    ParityConfig,
    ParityResult,
)
from dataeval_flow.evaluators.bias._evaluator import balance_arguments
from tests.evaluator_toys import ToyFactors, output_json


def _by_factor(result: "Result[Any, Any]", table: str) -> dict[str, dict[str, Any]]:
    """One table of a mapping output, keyed by factor name."""
    return {row["factor_name"]: row for row in output_json(result)["data"][table]["rows"]}


class TestBalance:
    def test_the_factor_that_follows_the_class_is_found(self):
        result = run(BalanceConfig(), ToyFactors())
        assert isinstance(result, BalanceResult)
        assert result.success, result.errors
        balance = _by_factor(result, "balance")
        assert balance["site"]["mi_value"] > 0.9
        assert balance["angle"]["mi_value"] < 0.1

    def test_the_named_policy_decides_the_factors(self):
        policy = MetadataPolicyConfig(name="no_site", exclude=["site"])
        result = run(BalanceConfig(metadata="no_site"), ToyFactors(), definitions=[policy])
        assert result.success, result.errors
        assert "site" not in _by_factor(result, "balance")
        assert "angle" in _by_factor(result, "balance")

    def test_the_older_metadata_fields_are_refused(self):
        with pytest.raises(ValidationError, match="metadata_exclude"):
            BalanceConfig.model_validate({"metadata_exclude": ["site"]})


class TestBalanceArguments:
    def test_unset_fields_are_left_to_dataeval(self):
        assert balance_arguments(BalanceConfig(), None) == {}

    def test_every_set_field_is_passed(self):
        config = BalanceConfig(
            num_neighbors=3, class_imbalance_threshold=0.2, factor_correlation_threshold=0.4, label="site"
        )
        assert balance_arguments(config, None) == {
            "num_neighbors": 3,
            "class_imbalance_threshold": 0.2,
            "factor_correlation_threshold": 0.4,
            "label": "site",
        }

    def test_the_policy_fills_an_unset_factor_source(self):
        assert balance_arguments(BalanceConfig(), ResolvedPolicy(factor_source="coded")) == {"factor_source": "coded"}

    def test_the_config_wins_over_the_policy(self):
        arguments = balance_arguments(BalanceConfig(factor_source="values"), ResolvedPolicy(factor_source="coded"))
        assert arguments == {"factor_source": "values"}


class TestDiversity:
    def test_every_factor_is_scored(self):
        result = run(DiversityConfig(), ToyFactors())
        assert isinstance(result, DiversityResult)
        assert result.success, result.errors
        assert {"site", "angle"} <= set(_by_factor(result, "factors"))

    def test_the_method_reaches_dataeval(self):
        simpson = _by_factor(run(DiversityConfig(method="simpson"), ToyFactors()), "factors")
        shannon = _by_factor(run(DiversityConfig(method="shannon"), ToyFactors()), "factors")
        assert simpson["angle"]["diversity_value"] != shannon["angle"]["diversity_value"]


class TestParity:
    def test_the_factor_that_follows_the_class_is_significant(self):
        result = run(ParityConfig(), ToyFactors())
        assert isinstance(result, ParityResult)
        assert result.success, result.errors
        factors = _by_factor(result, "factors")
        assert factors["site"]["is_significant"]
        assert not factors["angle"]["is_significant"]
