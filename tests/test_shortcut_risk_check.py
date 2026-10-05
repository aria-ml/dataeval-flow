"""The `shortcut-risk` check: factors that tell much about the class (audit spec §10.2, §19 I7)."""

from types import SimpleNamespace
from typing import Any

from dataeval_flow import run
from dataeval_flow.evaluators.bias import BalanceConfig
from dataeval_flow.steps import CheckContext
from dataeval_flow.steps.checks import ShortcutRiskCheck, ShortcutRiskConfig
from tests.evaluator_toys import ToyFactors


def _judge(**limits: Any) -> Any:
    output = run(BalanceConfig(), ToyFactors(60)).output  # `site` follows the class; `angle` does not
    node = SimpleNamespace(value=output, computed_on=(SimpleNamespace(address="train"),), address="balance")
    (finding,) = ShortcutRiskCheck().run(
        ShortcutRiskConfig(input="balance", **limits), {"input": node}, CheckContext("t", "s")
    )
    return finding


def test_a_factor_tied_to_the_class_warns() -> None:
    finding = _judge()
    assert finding.severity == "warning"
    assert finding.title == "Shortcut Risk"
    assert finding.brief.startswith("1 of 2 factors tied to the class: site (MI=")


def test_balance_s_class_label_row_is_never_a_factor() -> None:
    assert "class_label" not in _judge().brief


def test_the_limit_decides() -> None:
    assert _judge(warning=None).severity == "info"
    assert _judge(warning=1.0).severity == "ok"
