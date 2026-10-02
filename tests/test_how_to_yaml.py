"""Every YAML snippet in the evaluator and custom-workflow pages loads, assembled in order as a reader builds it.

A renamed parameter or evaluator type then fails CI instead of a reader's config.
"""

import copy
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from dataeval_flow import PipelineConfig

_DOCS = Path(__file__).resolve().parents[1] / "docs" / "source"
_YAML_BLOCK = re.compile(r"```yaml\n(.*?)```", re.S)

# What the how-tos assume a reader's pipeline already defines: a dataset, the sources they name, and one extractor.
_BASE: dict[str, Any] = {
    "datasets": [{"name": "ds", "format": "huggingface", "path": "./d", "task": "image_classification"}],
    "sources": [
        {"name": name, "dataset": "ds"}
        for name in ("train", "test", "validation", "operational", "labeled", "unlabeled")
    ],
    "extractors": [{"name": "bovw_ext", "model": "bovw", "vocab_size": 512, "batch_size": 32}],
}

# What the chain concept page's snippets assume besides: the cameras a list input binds, and the evaluators its
# steps name.
_CHAINS_BASE: dict[str, Any] = {
    **_BASE,
    "sources": [*_BASE["sources"], *({"name": name, "dataset": "ds"} for name in ("cam1", "cam2"))],
    "evaluators": [
        {"name": "dupes", "type": "duplicates"},
        {"name": "balance", "type": "balance"},
        {"name": "mmd", "type": "drift-mmd"},
    ],
}

_PAGES: dict[str, dict[str, Any]] = {
    "how_to/run_a_single_evaluator.md": _BASE,
    "how_to/evaluator_recipes.md": _BASE,
    "how_to/write_a_custom_workflow.md": _BASE,
    "how_to/reuse_a_workflow.md": _BASE,
    "how_to/run_a_matrix.md": _BASE,
    "concepts/WorkflowsAsChains.md": _CHAINS_BASE,
    "reference/checks.md": _BASE,
}


def _merge(config: dict[str, Any], block: dict[str, Any]) -> dict[str, Any]:
    """`block` added to `config` as a reader adds it: each list section extended, a same-named entry replaced."""
    for key, entries in block.items():
        if isinstance(entries, list):
            names = {entry.get("name") for entry in entries}
            config[key] = [entry for entry in config.get(key, []) if entry.get("name") not in names] + entries
        else:
            config[key] = entries
    return config


@pytest.mark.parametrize("page", _PAGES)
def test_every_yaml_snippet_loads(page: str):
    blocks = _YAML_BLOCK.findall((_DOCS / page).read_text(encoding="utf-8"))
    assert blocks, f"{page} has no yaml snippets"
    config = copy.deepcopy(_PAGES[page])
    for block in blocks:
        config = _merge(config, yaml.safe_load(block))
        PipelineConfig.model_validate(config)
