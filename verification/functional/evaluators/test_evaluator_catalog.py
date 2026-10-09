"""TC-20-1 — evaluator catalog: the types DataEval Flow lists, and the schema each one's parameters follow."""

from __future__ import annotations

import json

import pytest

from dataeval_flow.evaluators import EvaluatorResult, get_evaluator, list_evaluators
from verification.functional.chains._cli import invoke
from verification.functional.chains._toys import EVALUATOR_DATA
from verification.helpers import run_cli

pytestmark = pytest.mark.required


def _output(proc) -> str:
    return proc.stdout + proc.stderr


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """``cli(*args)``: the command in this process, with its exit code and output."""
    return lambda *args: invoke(monkeypatch, capsys, *args)


class TestEvaluatorCatalog:
    def test_list_evaluators_returns_every_type_with_a_description(self) -> None:
        listed = list_evaluators()
        assert {cls.name for cls in listed} == set(EVALUATOR_DATA)
        assert len(listed) == len(EVALUATOR_DATA) == 26
        assert all(cls.description.strip() for cls in listed)

    def test_get_evaluator_returns_the_class_named_and_refuses_an_unknown_type(self) -> None:
        assert get_evaluator("duplicates").name == "duplicates"
        with pytest.raises(ValueError, match="Unknown evaluator"):
            get_evaluator("does-not-exist")

    def test_each_config_declares_its_type_and_the_result_class_it_returns(self) -> None:
        for cls in list_evaluators():
            config = cls.config_type
            assert config.model_fields["type"].default == cls.name
            assert config.result_type is not None
            assert issubclass(config.result_type, EvaluatorResult)

    def test_evaluators_command_lists_every_type(self) -> None:
        proc = run_cli("evaluators")
        assert proc.returncode == 0, _output(proc)
        listed = {line.split()[0] for line in proc.stdout.splitlines() if line.strip() and not line.startswith(" " * 8)}
        assert listed == set(EVALUATOR_DATA)

    def test_evaluators_json_lists_name_description_inputs_and_source_count(self, cli) -> None:
        proc = cli("evaluators", "--json")
        assert proc.code == 0, proc.output
        entries = json.loads(proc.stdout)
        assert {entry["name"] for entry in entries} == set(EVALUATOR_DATA)
        by_name = {entry["name"]: entry for entry in entries}
        assert by_name["duplicates"]["sources"] == "1+"
        assert by_name["drift-mmd"]["sources"] == "2"
        assert by_name["drift-wasserstein"]["sources"] == "3"
        assert by_name["drift-mmd"]["consumes"] == "embeddings"
        assert all(entry["description"] for entry in entries)

    def test_evaluators_command_with_a_type_prints_its_parameter_schema(self, cli) -> None:
        proc = cli("evaluators", "duplicates")
        assert proc.code == 0, proc.output
        schema = json.loads(proc.stdout)
        assert schema["properties"]["type"]["const"] == "duplicates"
        assert "cluster_sensitivity" in schema["properties"]
        assert schema["additionalProperties"] is False

    def test_evaluators_command_refuses_an_unknown_type_and_lists_the_installed_ones(self, cli) -> None:
        proc = cli("evaluators", "does-not-exist")
        assert "Unknown evaluator: 'does-not-exist'" in proc.output
        assert "duplicates" in proc.output
