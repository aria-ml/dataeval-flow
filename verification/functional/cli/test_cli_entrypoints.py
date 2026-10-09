"""TC-15-1 — command-line entry points: `--help` and `--version`, and the discovery subcommands."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

import dataeval_flow
from verification.functional.reporting._project import Invocation

pytestmark = pytest.mark.required

Cli = Callable[..., Invocation]

PRESETS = {"audit", "bias", "prioritization", "quality", "scope", "shift", "splits", "taxonomy", "triage"}
EVALUATORS = {
    "balance",
    "completeness",
    "content-digest",
    "coverage",
    "divergence",
    "diversity",
    "drift-domain-classifier",
    "drift-kneighbors",
    "drift-mmd",
    "drift-univariate",
    "drift-wasserstein",
    "duplicates",
    "factor-leakage",
    "factor-summary",
    "factor-triage",
    "label-alignment",
    "label-health",
    "label-reconciliation",
    "ontology-validation",
    "ood-domain-classifier",
    "ood-kneighbors",
    "outliers",
    "parity",
    "prioritization",
    "profile",
    "representation",
}
SUBCOMMANDS = ["workflows", "evaluators", "steps", "encoding", "config", "app", "verify", "serve"]
ENVIRONMENT_VARIABLES = [
    "DATAEVAL_CONFIG",
    "DATAEVAL_DATA",
    "DATAEVAL_OUTPUT",
    "DATAEVAL_CACHE",
    "DATAEVAL_LOG_FORMAT",
    "DATAEVAL_REPORT_WIDTH",
    "DATAEVAL_REPORT_IMAGES",
    "DATAEVAL_FAIL_ON_WARNING",
    "DATAEVAL_REQUIRE",
    "DATAEVAL_MAX_PROCESSES",
]
OPTIONS = [
    "--config",
    "--data",
    "--output",
    "--cache",
    "--task",
    "--max-processes",
    "--fail-on-warning",
    "--no-fail-on-warning",
    "--log-format",
    "--report-width",
    "--report-images",
    "--require",
    "--verbose",
    "--version",
]


def _python_m(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "dataeval_flow", *args], capture_output=True, text=True, check=False
    )


class TestEntryPoints:
    def test_console_script_is_on_the_path_and_reports_the_installed_build(self) -> None:
        # The interpreter's own bin directory first: an environment need not be activated for its script to exist.
        search = os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])
        script = shutil.which("dataeval-flow", path=search)
        assert script is not None
        done = subprocess.run([script, "--version"], capture_output=True, text=True, check=False)  # noqa: S603
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == f"dataeval-flow {dataeval_flow.__version__}"

    def test_module_form_runs_without_the_console_script(self) -> None:
        done = _python_m("--help")
        assert done.returncode == 0, done.stderr
        assert done.stdout.startswith("usage: dataeval_flow")
        assert _python_m("--version").stdout.strip() == f"dataeval-flow {dataeval_flow.__version__}"


class TestHelp:
    def test_version_prints_the_package_version_and_exits_0(self, cli: Cli) -> None:
        done = cli("--version")
        assert done.code == 0
        assert done.stdout.strip() == f"dataeval-flow {dataeval_flow.__version__}"
        assert dataeval_flow.__version__ not in ("", "unknown")

    def test_top_level_help_lists_every_subcommand_option_and_environment_variable(self, cli: Cli) -> None:
        done = cli("--help")
        assert done.code == 0
        text = " ".join(done.stdout.split())
        for subcommand in SUBCOMMANDS:
            assert subcommand in text, subcommand
        for option in OPTIONS:
            assert option in text, option
        for variable in ENVIRONMENT_VARIABLES:
            assert f"${variable}" in text, variable

    @pytest.mark.parametrize("subcommand", SUBCOMMANDS)
    def test_each_subcommand_answers_help_with_its_usage_and_exit_0(self, cli: Cli, subcommand: str) -> None:
        done = cli(subcommand, "--help")
        assert done.code == 0
        assert done.stdout.startswith(f"usage: dataeval_flow {subcommand}")
        assert "options:" in done.stdout

    def test_serve_help_documents_its_environment_variables_and_endpoints(self, cli: Cli) -> None:
        text = " ".join(cli("serve", "--help").stdout.split())
        for variable in (
            "DATAEVAL_DATA",
            "DATAEVAL_OUTPUT",
            "DATAEVAL_CACHE",
            "DATAEVAL_SERVICE_HOST",
            "DATAEVAL_SERVICE_PORT",
        ):
            assert variable in text, variable
        assert "/healthz" in text
        assert "--host" in text
        assert "--port" in text

    def test_verify_and_encoding_help_name_their_arguments(self, cli: Cli) -> None:
        verify = " ".join(cli("verify", "--help").stdout.split())
        assert "manifest" in verify
        assert "--config" in verify
        assert "--source" in verify
        encoding = " ".join(cli("encoding", "--help").stdout.split())
        assert "result" in encoding
        assert "--task" in encoding


class TestWorkflowsCommand:
    def test_lists_the_nine_presets_with_a_description_each(self, cli: Cli) -> None:
        done = cli("workflows")
        assert done.code == 0
        names = {line.split()[0] for line in done.stdout.splitlines() if line.strip()}
        assert names >= PRESETS
        assert "quality" in done.stdout
        assert "Outlier and duplicate detection" in done.stdout

    def test_json_listing_is_machine_readable(self, cli: Cli) -> None:
        entries = json.loads(cli("workflows", "--json").stdout)
        assert {entry["name"] for entry in entries} >= PRESETS
        assert all(entry["description"] for entry in entries)

    def test_the_workflows_listing_holds_no_retired_workflow_and_no_evaluator(self, cli: Cli) -> None:
        names = {entry["name"] for entry in json.loads(cli("workflows", "--json").stdout)}
        assert not names & {"data-cleaning", "data-analysis", "parameter-sweep", "duplicates", "outliers"}

    def test_a_named_workflow_prints_the_schema_of_its_settings(self, cli: Cli) -> None:
        schema = json.loads(cli("workflows", "quality").stdout)
        assert schema["type"] == "object"
        assert {"outliers", "duplicates", "checks"} <= set(schema["properties"])

    def test_an_unknown_workflow_exits_1_and_lists_the_installed_ones(self, cli: Cli) -> None:
        done = cli("workflows", "nosuch")
        assert done.code == 1
        assert "Unknown workflow: 'nosuch'" in done.stderr
        assert "quality" in done.stderr


class TestEvaluatorsCommand:
    def test_lists_every_evaluator_with_what_it_consumes_and_how_many_sources_it_reads(self, cli: Cli) -> None:
        entries = json.loads(cli("evaluators", "--json").stdout)
        assert {entry["name"] for entry in entries} >= EVALUATORS
        duplicates = next(entry for entry in entries if entry["name"] == "duplicates")
        assert duplicates["description"]
        assert {"consumes", "sources"} <= set(duplicates)
        drift = next(entry for entry in entries if entry["name"] == "drift-mmd")
        assert drift["sources"] == "2"
        assert "embeddings" in drift["consumes"]

    def test_the_text_listing_shows_what_each_consumes(self, cli: Cli) -> None:
        text = cli("evaluators").stdout
        assert "balance" in text
        assert "consumes: metadata; sources: 1" in text

    def test_a_named_evaluator_prints_its_parameter_schema(self, cli: Cli) -> None:
        schema = json.loads(cli("evaluators", "duplicates").stdout)
        assert schema["additionalProperties"] is False
        assert "cluster_sensitivity" in schema["properties"]

    def test_an_unknown_evaluator_exits_1(self, cli: Cli) -> None:
        done = cli("evaluators", "nosuch")
        assert done.code == 1
        assert "Unknown evaluator" in done.stderr

    def test_the_evaluator_listing_holds_no_preset(self, cli: Cli) -> None:
        names = {entry["name"] for entry in json.loads(cli("evaluators", "--json").stdout)}
        assert not names & (PRESETS - {"prioritization"})  # prioritization names both a preset and an evaluator


class TestStepsCommand:
    def test_the_json_catalog_lists_every_kind_of_step_with_its_ports(self, cli: Cli) -> None:
        catalog = json.loads(cli("steps", "--json").stdout)
        assert catalog["flow_version"] == dataeval_flow.__version__
        kinds = {step["kind"] for step in catalog["steps"]}
        assert kinds == {"evaluator", "check", "transform", "combine", "workflow"}
        remove = next(step for step in catalog["steps"] if step["type"] == "remove")
        assert remove["kind"] == "transform"
        assert remove["inputs"][0]["port"] == "input"
        assert remove["outputs"]

    def test_the_text_catalog_has_a_row_per_step_with_its_ports(self, cli: Cli) -> None:
        text = cli("steps").stdout
        assert "transform  remove" in text
        assert "evaluator  duplicates" in text
        assert "->" in text

    def test_a_named_step_prints_its_entry_and_settings_schema(self, cli: Cli) -> None:
        entry = json.loads(cli("steps", "remove").stdout)
        assert (entry["kind"], entry["type"]) == ("transform", "remove")
        assert entry["config_schema"]["type"] == "object"

    def test_a_name_two_kinds_share_needs_kind_name(self, cli: Cli) -> None:
        ambiguous = cli("steps", "prioritization")
        assert ambiguous.code == 1
        assert "name one as KIND:NAME" in ambiguous.stderr
        chosen = json.loads(cli("steps", "workflow:prioritization").stdout)
        assert chosen["kind"] == "workflow"

    def test_an_unknown_step_exits_1(self, cli: Cli) -> None:
        done = cli("steps", "nosuch")
        assert done.code == 1
        assert "'nosuch' names no step" in done.stderr
