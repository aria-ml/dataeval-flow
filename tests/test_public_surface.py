"""The public surface: pinned by a snapshot, one path per name, private elsewhere, documented."""

import importlib
import inspect
import pkgutil
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import dataeval_flow

SNAPSHOT = Path(__file__).with_name("public_api.txt")
EXTENSIBLE = [
    "dataeval_flow.workflows:Workflow",
    "dataeval_flow.workflows:WorkflowConfig",
    "dataeval_flow.workflows:WorkflowOutput",
    "dataeval_flow.workflows:WorkflowRawOutput",
    "dataeval_flow.workflows:WorkflowReport",
    "dataeval_flow.workflows:WorkflowResult",
    "dataeval_flow.evaluators:Evaluator",
    "dataeval_flow.evaluators:EvaluatorConfig",
    "dataeval_flow.evaluators:EvaluatorResult",
    "dataeval_flow.config.extractors:Extractor",
    "dataeval_flow.config.extractors:ExtractorConfig",
    "dataeval_flow.config.transforms:Transform",
    "dataeval_flow:Result",
    "dataeval_flow:ResultMetadata",
]


def _public_modules() -> list[str]:
    names = ["dataeval_flow"]
    names.extend(
        info.name
        for info in pkgutil.walk_packages(dataeval_flow.__path__, "dataeval_flow.")
        if not any(part.startswith("_") for part in info.name.split(".")[1:])
    )
    return sorted(names)


def _surface() -> list[str]:
    lines = []
    for module_name in _public_modules():
        module = importlib.import_module(module_name)
        lines.extend(f"{module_name}:{name}" for name in sorted(getattr(module, "__all__", [])))
    return lines


def test_the_surface_matches_the_snapshot() -> None:
    assert _surface() == SNAPSHOT.read_text().splitlines(), "update tests/public_api.txt deliberately"


def test_each_public_name_has_one_path() -> None:
    owners: dict[int, list[str]] = {}
    for line in _surface():
        module_name, name = line.split(":")
        obj = getattr(importlib.import_module(module_name), name)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            owners.setdefault(id(obj), []).append(line)
    assert [paths for paths in owners.values() if len(paths) > 1] == []


def test_every_other_module_is_private() -> None:
    public = set(_public_modules())
    for info in pkgutil.walk_packages(dataeval_flow.__path__, "dataeval_flow."):
        if info.name not in public and info.name != "dataeval_flow.__main__":
            assert any(part.startswith("_") for part in info.name.split(".")[1:]), info.name


@pytest.mark.parametrize("line", _surface())
def test_every_public_name_is_documented(line: str) -> None:
    module_name, name = line.split(":")
    obj = getattr(importlib.import_module(module_name), name)
    if inspect.isclass(obj) or inspect.isfunction(obj):
        assert (obj.__doc__ or "").strip(), line


@pytest.mark.parametrize("path", EXTENSIBLE)
def test_extensible_bases_explain_subclassing(path: str) -> None:
    module_name, name = path.split(":")
    doc = getattr(importlib.import_module(module_name), name).__doc__ or ""
    assert "Subclassing" in doc, path
    assert "Examples" in doc, path


@pytest.mark.parametrize("line", _surface())
def test_every_public_config_field_is_described(line: str) -> None:
    module_name, name = line.split(":")
    obj = getattr(importlib.import_module(module_name), name)
    if inspect.isclass(obj) and issubclass(obj, BaseModel):
        missing = [field for field, info in obj.model_fields.items() if not info.description]
        assert missing == [], f"{line}: {missing}"


def _result_classes() -> list[str]:
    """Every per-type result on the surface: each workflow's and each evaluator's ``<X>Result``."""
    from dataeval_flow.evaluators import EvaluatorResult
    from dataeval_flow.workflows import WorkflowResult

    lines = []
    for line in _surface():
        module_name, name = line.split(":")
        obj = getattr(importlib.import_module(module_name), name)
        bases = (WorkflowResult, EvaluatorResult)
        if inspect.isclass(obj) and issubclass(obj, bases) and obj not in bases:
            lines.append(line)
    return lines


def _documented_fields(doc: str) -> dict[str, str]:
    """The ``Fields`` section of a docstring: each entry's name and its description, whitespace folded."""
    lines = inspect.cleandoc(doc).splitlines()
    start = lines.index("Fields") + 2
    fields: dict[str, list[str]] = {}
    name = ""
    for i, line in enumerate(lines[start:], start):
        if i + 1 < len(lines) and lines[i + 1] and set(lines[i + 1]) == {"-"}:
            break  # the next section's title
        if line and not line.startswith(" "):
            name = line
            fields[name] = []
        elif line.strip():
            fields[name].append(line.strip())
    return {name: " ".join(parts) for name, parts in fields.items()}


def _own_fields(model: type[BaseModel], base: type[BaseModel], prefix: str) -> dict[str, str]:
    """`model`'s fields that `base` does not declare, under `prefix`, each with its description in reST quoting."""
    return {
        f"{prefix}{name}": re.sub(r"(?<!`)`([^`]+)`(?!`)", r"``\1``", info.description or "")
        for name, info in model.model_fields.items()
        if name not in base.model_fields
    }


def _typed_fields(result: Any) -> dict[str, str]:
    """What typed code reads beyond the bases: the output's and the metadata's own fields, as their models say."""
    from dataeval_flow import ResultMetadata
    from dataeval_flow._kind import type_arguments
    from dataeval_flow.evaluators import EvaluatorResult
    from dataeval_flow.workflows import WorkflowRawOutput, WorkflowReport, WorkflowResult

    if issubclass(result, EvaluatorResult):
        return _own_fields(result.metadata_type, ResultMetadata, "metadata.")
    metadata, output = type_arguments(result, WorkflowResult)
    raw, report = output.model_fields["raw"].annotation, output.model_fields["report"].annotation
    return {
        **_own_fields(raw, WorkflowRawOutput, "output.raw."),
        **_own_fields(report, WorkflowReport, "output.report."),
        **_own_fields(metadata, ResultMetadata, "metadata."),
    }


@pytest.mark.parametrize("line", _result_classes())
def test_each_result_documents_the_fields_typed_code_reads(line: str) -> None:
    """The docstring's ``Fields`` section is the models' own fields and descriptions, so the API pages cannot drift.

    The models are private (typed code reaches them only by attribute), so their result's page is where a reader
    finds them. Regenerate the section from the models' ``Field`` descriptions when this fails.
    """
    from dataeval_flow.evaluators import EvaluatorResult

    module_name, name = line.split(":")
    result = getattr(importlib.import_module(module_name), name)
    documented = _documented_fields(result.__doc__ or "")
    expected = _typed_fields(result)
    assert all(expected.values()), f"{line}: give every field a description"
    if issubclass(result, EvaluatorResult):
        assert documented.pop("output", ""), f"{line}: document the DataEval output"
    assert documented == expected, line
