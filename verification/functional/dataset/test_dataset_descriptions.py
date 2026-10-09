"""TC-3-2 — what describes a dataset: value range, channel groups and provenance."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow.config import (
    ChannelGroupConfig,
    CocoDatasetConfig,
    ImageFolderDatasetConfig,
    PipelineConfig,
)
from verification.fixtures import write_image_folder
from verification.functional.dataset.formats import FORMATS, entry
from verification.functional.orchestration.support import pipeline_dict

pytestmark = pytest.mark.required


class TestDatasetDescriptions:
    """`value_range`, `channel_groups` and `provenance` describe a dataset; they sit on any file-backed format."""

    def test_the_description_fields_sit_on_every_file_backed_format(self) -> None:
        for fmt, (cls, _) in FORMATS.items():
            config = cls(
                **entry(fmt),
                value_range=(0, 1),
                channel_groups={"rgb": [0, 1, 2], "ir": 3},
                provenance={"owner": "me"},
            )
            assert config.value_range == (0.0, 1.0), fmt
            assert dict(config.channel_groups) == {"rgb": [0, 1, 2], "ir": 3}, fmt
            assert config.provenance == {"owner": "me"}, fmt

    @pytest.mark.parametrize("name", ["brightness", "background", "instance"])
    def test_a_channel_group_may_not_take_a_statistic_or_reserved_name(self, name: str) -> None:
        with pytest.raises(ValidationError, match="collides with a reserved name"):
            CocoDatasetConfig(name="d", path="d", channel_groups={name: [0, 1]})

    def test_a_channel_group_needs_bands_that_exist(self) -> None:
        with pytest.raises(ValidationError, match="names no bands"):
            CocoDatasetConfig(name="d", path="d", channel_groups={"rgb": []})
        with pytest.raises(ValidationError, match="negative band index"):
            CocoDatasetConfig(name="d", path="d", channel_groups={"rgb": [0, -1]})

    def test_a_channel_group_can_carry_its_own_value_range(self) -> None:
        config = CocoDatasetConfig(
            name="d",
            path="d",
            channel_groups={"thermal": {"bands": 3, "value_range": [-40, 120]}},  # type: ignore[dict-item]
        )
        assert config.channel_groups == {"thermal": ChannelGroupConfig(bands=3, value_range=(-40.0, 120.0))}
        with pytest.raises(ValidationError, match="low < high"):
            ChannelGroupConfig(bands=3, value_range=(5, 5))

    def test_channel_groups_measure_their_bands_separately(self, tmp_path: Path) -> None:
        """A stats policy names the groups; their statistics are `<group>_<statistic>` beside the whole image's."""
        write_image_folder(tmp_path / "imgs", n_per_class=4, n_classes=2)
        data = pipeline_dict(
            datasets=[
                {
                    "name": "ds",
                    "format": "image_folder",
                    "path": "imgs",
                    "infer_labels": True,
                    "channel_groups": {"first": 0, "rest": [1, 2]},
                }
            ],
            stats=[
                {
                    "name": "banded",
                    "measure": [
                        {"bands": None, "families": ["visual"]},
                        {"bands": "first", "families": ["visual"]},
                        {"bands": "rest", "families": ["pixel"]},
                    ],
                }
            ],
            evaluators=[{"name": "prof", "type": "profile", "stats": "banded"}],
            tasks=[{"name": "t", "evaluator": "prof", "sources": "main"}],
        )
        result = run_tasks(PipelineConfig.model_validate(data), data_dir=tmp_path)["t"]
        assert result.success, result.errors
        fields = {field["name"]: field["group"] for field in result.to_dict()["output"]["data"]["fields"]}
        assert fields["brightness"] is None
        assert fields["first_brightness"] == "first"
        assert fields["rest_mean"] == "rest"
        assert "first_mean" not in fields  # `first` was asked for visual statistics only

    def test_a_stats_policy_naming_an_undeclared_group_is_refused_before_the_data_is_read(self, tmp_path: Path) -> None:
        write_image_folder(tmp_path / "imgs", n_per_class=2, n_classes=2)
        data = pipeline_dict(
            stats=[
                {
                    "name": "banded",
                    "measure": [{"bands": None, "families": ["pixel"]}, {"bands": "ir", "families": ["pixel"]}],
                }
            ],
            evaluators=[{"name": "prof", "type": "profile", "stats": "banded"}],
            tasks=[{"name": "t", "evaluator": "prof", "sources": "main"}],
        )
        with pytest.raises(ValueError, match="does not declare a channel group 'ir'"):
            run_tasks(PipelineConfig.model_validate(data), data_dir=tmp_path)

    def test_provenance_takes_names_and_plain_values(self) -> None:
        facts = {"owner": "Perception team", "license": "CC-BY-4.0", "frames": 1200, "public": False, "ratio": 0.5}
        assert ImageFolderDatasetConfig(name="d", path="p", provenance=facts).provenance == facts
        assert ImageFolderDatasetConfig(name="d", path="p").provenance is None

    def test_a_yaml_date_in_provenance_is_kept_as_iso_text(self) -> None:
        text = "datasets:\n  - {name: d, format: image_folder, path: p, provenance: {collected: 2025-06-01}}\n"
        raw = yaml.safe_load(text)
        entry = PipelineConfig.model_validate(raw).datasets[0]  # type: ignore[index]
        assert entry.provenance == {"collected": "2025-06-01"}  # type: ignore[union-attr]

    def test_a_nested_provenance_value_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="provenance"):
            ImageFolderDatasetConfig(name="d", path="p", provenance={"owner": {"team": "x"}})  # type: ignore[dict-item]

    def test_provenance_and_description_are_recorded_in_the_results_resolved_config(self, tmp_path: Path) -> None:
        write_image_folder(tmp_path / "imgs", n_per_class=3, n_classes=2)
        facts = {"owner": "Perception team", "collected": "2025-06-01"}
        data = pipeline_dict(
            datasets=[
                {
                    "name": "ds",
                    "format": "image_folder",
                    "path": "imgs",
                    "infer_labels": True,
                    "value_range": [0, 255],
                    "provenance": facts,
                }
            ],
            evaluators=[{"name": "labels", "type": "label-health"}],
            tasks=[{"name": "t", "evaluator": "labels", "sources": "main"}],
        )
        result = run_tasks(PipelineConfig.model_validate(data), data_dir=tmp_path)["t"]
        assert result.success, result.errors
        (source,) = result.metadata.resolved_config["sources"]
        assert source["dataset_config"]["provenance"] == facts
        assert source["dataset_config"]["value_range"] == [0.0, 255.0]

    def test_provenance_does_not_change_the_cached_identity_of_a_dataset(self) -> None:
        from dataeval_flow._dataset import _config_key

        plain = ImageFolderDatasetConfig(name="d", path="p")
        described = ImageFolderDatasetConfig(name="d", path="p", provenance={"owner": "me"})
        assert _config_key(described) == _config_key(plain)
        assert _config_key(ImageFolderDatasetConfig(name="d", path="p", infer_labels=True)) != _config_key(plain)
