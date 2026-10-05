"""The naming rules every built-in step follows (naming spec §3.1, §4.2, §5, §6). A failure names the rule; the
rules are written out for plugin authors in reference/naming.md."""

import re
from collections import Counter

import pytest

from dataeval_flow.evaluators._registry import EVALUATORS
from dataeval_flow.steps._registry import CHECKS, COMBINES, TRANSFORMS
from dataeval_flow.workflows._registry import WORKFLOWS

_STEPS = [
    cls for registry in (EVALUATORS, TRANSFORMS, COMBINES, CHECKS, WORKFLOWS) for cls in registry.list(plugins=False)
]
# data-analysis is deleted by audit MR 3; until then it keeps its legacy surface (naming spec §9.1).
_LEGACY = {"data-analysis"}
# The two step configs whose `<Type>Config` name a pipeline pool entry already holds (`views:`, `exports:`).
_POOL_NAMED = {"view": "ViewTransformConfig", "export": "ExportTransformConfig"}


def _squashed(text: str) -> str:
    """`text` lowercased, with everything but letters and digits dropped: `Drift (K-Neighbors)` → `driftkneighbors`."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _id(cls: type) -> str:
    return f"{cls.kind}:{cls.name}"


@pytest.mark.parametrize("cls", _STEPS, ids=_id)
def test_a_step_type_and_its_title_name_one_thing(cls: type) -> None:
    assert _squashed(cls.title) == _squashed(cls.name), (
        f"{cls.kind} `{cls.name}` is titled {cls.title!r}: a type and its title agree once case, spaces, hyphens and "
        "parentheses are dropped (reference/naming.md)"
    )


def test_no_two_step_types_share_a_name_across_kinds() -> None:
    assert sorted(name for name, n in Counter(cls.name for cls in _STEPS).items() if n > 1) == []


@pytest.mark.parametrize("cls", _STEPS, ids=_id)
def test_a_step_class_is_named_for_its_type_and_kind(cls: type) -> None:
    assert cls.__name__.lower() == _squashed(cls.name) + cls.kind, f"`{cls.__name__}` is not `<Type><Kind>`"


@pytest.mark.parametrize("cls", [cls for cls in _STEPS if cls.name not in _LEGACY], ids=_id)
def test_a_step_config_is_named_for_its_type(cls: type) -> None:
    expected = _POOL_NAMED.get(cls.name)
    if expected is not None:
        assert cls.config_type.__name__ == expected
    else:
        assert cls.config_type.__name__.lower() == _squashed(cls.name) + "config", (
            f"`{cls.config_type.__name__}` is not `<Type>Config`"
        )


def _own_outputs() -> dict[type, list[str]]:
    """Each output class Flow defines, and the types whose output ports carry it."""
    found: dict[type, list[str]] = {}
    for cls in _STEPS:
        if cls.name in _LEGACY:
            continue
        for port in cls.output_ports():
            for output in port.classes:
                if output.__module__.startswith("dataeval_flow"):
                    found.setdefault(output, []).append(cls.name)
    return found


@pytest.mark.parametrize("output", list(_own_outputs()), ids=lambda output: output.__name__)
def test_an_output_flow_defines_is_named_for_its_type(output: type) -> None:
    types = _own_outputs()[output]
    if len(types) == 1:
        assert output.__name__.lower() == _squashed(types[0]) + "output", f"`{output.__name__}` is not `<Type>Output`"


def test_an_old_check_type_is_refused_at_load() -> None:
    from pydantic import ValidationError

    from tests.chain_toys import chain_pipeline

    steps = [{"name": "o", "evaluator": "o", "input": "data"}, {"name": "c", "check": "outlier-rate", "input": "o"}]
    with pytest.raises(ValidationError, match=r"Unknown check: 'outlier-rate'\. Installed: \[.*'image-outliers'"):
        chain_pipeline(
            workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
            evaluators=[{"name": "o", "type": "outliers", "flags": ["pixel"]}],
            tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        )
