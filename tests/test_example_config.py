"""`config/params.example.yaml` parses, and its stats policy actually works.

The file is all-comments by design (see its own header), so nothing has ever loaded it.
Three separate defects have shipped in its `stats:` block for exactly that reason: nobody
ran it against the code it demonstrates. These tests un-comment the relevant sections and
run them through the real config and stats machinery, so a fourth defect fails a test
instead of shipping.
"""

from pathlib import Path
from typing import Any

import pytest
from dataeval.flags import ImageStats

from dataeval_flow.config import DataCleaningWorkflowConfig, PipelineConfig
from dataeval_flow.config.schemas._stats import StatsPolicyConfig
from dataeval_flow.metadata import resolve_families
from dataeval_flow.stats import (
    OUTLIER_FLAG_MAP,
    ResolvedStatsPolicy,
    check_consumers,
    resolve_stats_policy,
)

_EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "config" / "params.example.yaml"


def _extract_top_level_block(lines: list[str], key: str) -> str:
    """Return the raw YAML for a `# {key}: ...` block commented out in the example config.

    Every line of the file is a comment, so "un-commenting" means finding the line that
    introduces *key* at the top level and stripping the leading `#` off it and every line
    that follows, until a line that is not a comment at all — a genuinely blank line, which
    is how the file separates one top-level section from the next. A blank *comment* line
    (bare `#`) is a spacer the file uses *within* a section (e.g. between list entries) and
    must not end the block early.
    """
    out: list[str] = []
    capturing = False
    for line in lines:
        if not capturing:
            if line == f"# {key}:":
                capturing = True
                out.append(line[1:])
            continue
        if line == "" or not line.startswith("#"):
            break
        out.append(line[1:])
    return "\n".join(out)


def _load_example_sections() -> dict[str, list[dict[str, Any]]]:
    """Parse the `datasets:`, `stats:`, and `workflows:` blocks out of the example config."""
    import yaml

    lines = _EXAMPLE_PATH.read_text().splitlines()
    return {
        key: yaml.safe_load(_extract_top_level_block(lines, key))[key] for key in ("datasets", "stats", "workflows")
    }


def _channel_groups_of(dataset: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """The `channel_groups` a parsed dataset dict declares, in `resolve_stats_policy`'s shape."""
    return {
        name: (bands,) if isinstance(bands, int) else tuple(bands) for name, bands in dataset["channel_groups"].items()
    }


def _build_clean_config() -> tuple[PipelineConfig, DataCleaningWorkflowConfig, dict[str, tuple[int, ...]]]:
    """Build a `PipelineConfig` scoped to just `m3fd`, `multispectral`, and the `clean` workflow."""
    sections = _load_example_sections()
    m3fd = next(d for d in sections["datasets"] if d["name"] == "m3fd")
    clean = next(w for w in sections["workflows"] if w["name"] == "clean")
    # `model_validate` rather than the constructor: these are raw dicts off the YAML, not
    # already-typed config models.
    config = PipelineConfig.model_validate({"datasets": [m3fd], "stats": sections["stats"], "workflows": [clean]})
    assert config.workflows is not None
    (clean_params,) = config.workflows
    assert isinstance(clean_params, DataCleaningWorkflowConfig)
    return config, clean_params, _channel_groups_of(m3fd)


@pytest.mark.required
class TestExampleStatsPolicyParses:
    """The `stats:` block on its own is valid `StatsPolicyConfig`."""

    def test_the_multispectral_policy_parses(self):
        sections = _load_example_sections()
        (declared,) = sections["stats"]
        policy = StatsPolicyConfig(**declared)
        assert policy.name == "multispectral"

    def test_it_produces_all_six_views(self):
        """Two band groups plus `background: true` reach the full `2n + 2` vocabulary."""
        sections = _load_example_sections()
        (declared,) = sections["stats"]
        policy = StatsPolicyConfig(**declared)
        produced = {"~" if v is None else v for v in policy.produced_views()}
        assert produced == {"~", "rgb", "ir", "background", "background_rgb", "background_ir"}

    def test_ir_asks_for_more_than_rgb(self):
        """The feature's headline claim: two groups can carry different families."""
        sections = _load_example_sections()
        (declared,) = sections["stats"]
        by_bands = {entry["bands"]: set(entry["families"]) for entry in declared["measure"]}
        assert by_bands["rgb"] < by_bands["ir"]


@pytest.mark.required
class TestExampleCleaningWorkflowCanReadItsOwnPolicy:
    """The `clean` workflow's `outlier_flags` must be satisfied by `multispectral`.

    This is the exact failure mode of the shipped defect: `check_consumers` refused the
    policy because the whole-image entry did not measure what `outlier_flags: [visual]`
    needed on `~`. Nothing parsed this file, so nothing caught it before review.
    """

    def _resolve(self) -> tuple[ResolvedStatsPolicy, DataCleaningWorkflowConfig]:
        config, clean_params, channel_groups = _build_clean_config()
        resolved = resolve_stats_policy(clean_params, config, channel_groups)
        assert resolved is not None
        return resolved, clean_params

    def test_resolves_without_error(self):
        self._resolve()

    def test_the_declared_outlier_flags_are_satisfied(self):
        resolved, clean_params = self._resolve()
        outlier_flags = ImageStats.NONE
        for name in clean_params.outlier_flags:
            outlier_flags |= OUTLIER_FLAG_MAP[name]

        # Must not raise: `outlier_flags: [visual]` is exactly what `~` has to measure.
        check_consumers(
            resolved,
            outlier_flags=outlier_flags,
            duplicate_flags=ImageStats.NONE,
            factor_flags=ImageStats.NONE,
        )


@pytest.mark.required
class TestExamplePolicyAlsoSatisfiesADataAnalysisStyleConsumer:
    """`data-analysis` is not in the example, but the guide's advice covers it.

    `measure_band_groups.md` says a policy used by `data-analysis` must give `~` the full
    `hash` family, because analysis always runs duplicate detection over the whole image.
    Check that claim directly against the shipped policy, so the doc and the config cannot
    drift apart again.
    """

    def test_the_whole_image_entry_carries_the_full_hash_family_and_visual(self):
        sections = _load_example_sections()
        (declared,) = sections["stats"]
        whole_image = next(entry for entry in declared["measure"] if entry["bands"] is None)
        assert set(whole_image["families"]) >= {"hash", "visual"}

    def test_duplicate_and_visual_factor_checks_pass(self):
        config, clean_params, channel_groups = _build_clean_config()
        resolved = resolve_stats_policy(clean_params, config, channel_groups)
        assert resolved is not None

        # Duplicate detection always reads the whole image, unconditionally, on
        # `data-analysis`; there is no field to name in the error, hence the override.
        check_consumers(
            resolved,
            outlier_flags=ImageStats.NONE,
            duplicate_flags=ImageStats.HASH,
            factor_flags=ImageStats.NONE,
            duplicate_declaration="this workflow's duplicate detection, which always runs",
        )
        # `intrinsic_factors: [visual]` read from `factors_from: [~, rgb, ir]`.
        check_consumers(
            resolved,
            outlier_flags=ImageStats.NONE,
            duplicate_flags=ImageStats.NONE,
            factor_flags=ImageStats(resolve_families("image", ["visual"])),
        )
