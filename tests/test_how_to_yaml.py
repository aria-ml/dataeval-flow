"""Every YAML snippet in the evaluator how-tos loads, assembled in order the way a reader builds a pipeline.

A renamed parameter or evaluator type then fails CI instead of a reader's config.
"""

import copy
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from dataeval_flow import PipelineConfig

_HOW_TO = Path(__file__).resolve().parents[1] / "docs" / "source" / "how_to"
_PAGES = ["run_a_single_evaluator.md", "evaluator_recipes.md"]
_YAML_BLOCK = re.compile(r"```yaml\n(.*?)```", re.S)

# What the pages assume a reader's pipeline already defines: a dataset, the sources they name, and one extractor.
_BASE: dict[str, Any] = {
    "datasets": [{"name": "ds", "format": "huggingface", "path": "./d", "task": "image_classification"}],
    "sources": [
        {"name": name, "dataset": "ds"}
        for name in ("train", "test", "validation", "operational", "labeled", "unlabeled")
    ],
    "extractors": [{"name": "bovw_ext", "model": "bovw", "vocab_size": 512, "batch_size": 32}],
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
    blocks = _YAML_BLOCK.findall((_HOW_TO / page).read_text(encoding="utf-8"))
    assert blocks, f"{page} has no yaml snippets"
    config = copy.deepcopy(_BASE)
    for block in blocks:
        config = _merge(config, yaml.safe_load(block))
        PipelineConfig.model_validate(config)
