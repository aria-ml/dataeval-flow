"""TC-3-2 — sources: merging datasets and loading a source as a run reads it."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import dataset_digest, load_source, run_tasks
from dataeval_flow._sources import MergeConfigError
from dataeval_flow.config import PipelineConfig, SourceConfig
from verification.fixtures import write_image_folder

pytestmark = pytest.mark.required

CAT_DOG = ["cat", "dog"]


def _relabel(name: str, class_remap: dict[str, str], target: list[str] = CAT_DOG) -> dict[str, Any]:
    return {"name": name, "operations": [{"type": "Relabel", "params": {"class_remap": class_remap, "target": target}}]}


@pytest.fixture
def two_datasets(tmp_path: Path) -> Path:
    """`a` holds six images (classes `class_0`, `class_1`), `b` four (classes `cat`, `dog`), numbered differently."""
    write_image_folder(tmp_path / "a", n_per_class=3, n_classes=2, seed=1)
    write_image_folder(tmp_path / "b", n_per_class=2, n_classes=2, seed=2)
    shutil.move(tmp_path / "b" / "class_0", tmp_path / "b" / "cat")
    shutil.move(tmp_path / "b" / "class_1", tmp_path / "b" / "dog")
    return tmp_path


def _merge_config(**sections: Any) -> PipelineConfig:
    data: dict[str, Any] = {
        "datasets": [
            {"name": "da", "format": "image_folder", "path": "a", "infer_labels": True},
            {"name": "db", "format": "image_folder", "path": "b", "infer_labels": True},
        ],
        "views": [
            _relabel("va", {"class_0": "cat", "class_1": "dog"}),
            _relabel("vb", {"cat": "cat", "dog": "dog"}),
            _relabel("vb_swapped", {"cat": "cat", "dog": "dog"}, target=["dog", "cat"]),
            {"name": "first4", "operations": [{"type": "Limit", "params": {"size": 4}}]},
        ],
        "sources": [
            {"name": "sa", "dataset": "da", "view": "va"},
            {"name": "sb", "dataset": "db", "view": "vb"},
            {"name": "sb_swapped", "dataset": "db", "view": "vb_swapped"},
            {"name": "merged", "merge": ["sa", "sb"]},
            {"name": "merged_first4", "merge": ["sa", "sb"], "view": "first4"},
            {"name": "mismatched", "merge": ["sa", "sb_swapped"]},
            {"name": "loop_a", "merge": ["sa", "loop_b"]},
            {"name": "loop_b", "merge": ["sb", "loop_a"]},
        ],
    }
    data.update(sections)
    return PipelineConfig.model_validate(data)


class TestSourceConfig:
    def test_a_source_names_a_dataset_or_a_merge(self) -> None:
        assert SourceConfig(name="s", dataset="d").merge is None
        assert list(SourceConfig(name="s", merge=["a", "b"]).merge or ()) == ["a", "b"]

    def test_a_source_naming_both_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="names both `dataset` and `merge`"):
            SourceConfig(name="s", dataset="d", merge=["a", "b"])

    def test_a_source_naming_neither_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="names neither `dataset` nor `merge`"):
            SourceConfig(name="s")

    def test_a_merge_of_one_source_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="merges at least two sources"):
            SourceConfig(name="s", merge=["a"])


class TestMerge:
    def test_a_merged_source_concatenates_its_operands_in_order(self, two_datasets: Path) -> None:
        merged = load_source(_merge_config(), "merged", data_dir=two_datasets)
        assert len(merged) == 10
        assert merged.metadata["index2label"] == {0: "cat", 1: "dog"}
        # Operand a's six items come first, then operand b's four; both use the one shared numbering.
        targets = [int(merged[i][1].argmax()) for i in range(len(merged))]
        assert targets == [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]

    def test_a_merged_datums_id_names_its_operand_and_its_own_id(self, two_datasets: Path) -> None:
        merged = load_source(_merge_config(), "merged", data_dir=two_datasets)
        ids = [merged[i][2]["id"] for i in range(len(merged))]
        assert ids[:3] == ["0:0", "0:1", "0:2"]
        assert ids[6:] == ["1:0", "1:1", "1:2", "1:3"]
        assert len(set(ids)) == len(ids)

    def test_the_merged_source_s_own_view_applies_after_the_merge(self, two_datasets: Path) -> None:
        merged = load_source(_merge_config(), "merged_first4", data_dir=two_datasets)
        assert len(merged) == 4
        assert [merged[i][2]["id"] for i in range(4)] == ["0:0", "0:1", "0:2", "0:3"]

    def test_operands_that_number_their_classes_differently_are_refused(self, two_datasets: Path) -> None:
        with pytest.raises(ValueError, match="share the same 'index2label'"):
            load_source(_merge_config(), "mismatched", data_dir=two_datasets)

    def test_a_merge_that_names_itself_through_another_is_refused(self, two_datasets: Path) -> None:
        with pytest.raises(
            MergeConfigError, match=r"Source 'loop_a' merges itself through: loop_a -> loop_b -> loop_a"
        ):
            load_source(_merge_config(), "loop_a", data_dir=two_datasets)

    def test_a_merge_may_hold_another_merge(self, two_datasets: Path) -> None:
        config = _merge_config(
            sources=[
                {"name": "sa", "dataset": "da", "view": "va"},
                {"name": "sb", "dataset": "db", "view": "vb"},
                {"name": "inner", "merge": ["sa", "sb"]},
                {"name": "outer", "merge": ["inner", "sa"]},
            ]
        )
        assert len(load_source(config, "outer", data_dir=two_datasets)) == 16

    def test_merges_nested_deeper_than_eight_are_refused(self, two_datasets: Path) -> None:
        sources: list[dict[str, Any]] = [
            {"name": "sa", "dataset": "da", "view": "va"},
            {"name": "lvl0", "merge": ["sa", "sa"]},
            *({"name": f"lvl{n}", "merge": [f"lvl{n - 1}", "sa"]} for n in range(1, 11)),
        ]
        with pytest.raises(MergeConfigError, match="nests merges more than 8 deep"):
            load_source(_merge_config(sources=sources), "lvl10", data_dir=two_datasets)

    def test_a_merge_naming_an_unknown_source_lists_the_known_ones(self, two_datasets: Path) -> None:
        sources = [{"name": "sa", "dataset": "da", "view": "va"}, {"name": "m", "merge": ["sa", "nope"]}]
        with pytest.raises(ValueError, match=r"Unknown source: 'nope'. Available: \['sa', 'm'\]"):
            load_source(_merge_config(sources=sources), "m", data_dir=two_datasets)

    def test_a_task_runs_over_a_merged_source_and_records_what_was_merged(self, two_datasets: Path) -> None:
        config = _merge_config(
            evaluators=[{"name": "labels", "type": "label-health"}],
            tasks=[{"name": "t", "evaluator": "labels", "sources": "merged"}],
        )
        result = run_tasks(config, data_dir=two_datasets)["t"]
        assert result.success, result.errors
        assert result.output.data()["label_counts_per_class"] == {"cat": 5, "dog": 5}
        (source,) = result.metadata.resolved_config["sources"]
        assert [operand["dataset"] for operand in source["merge"]] == ["da", "db"]
        assert [operand["view"] for operand in source["merge"]] == ["va", "vb"]
        assert result.metadata.dataset_id == "da,db"
        assert [(record.source, list(record.target)) for record in result.metadata.label_space] == [
            ("sa", CAT_DOG),
            ("sb", CAT_DOG),
        ]


class TestLoadSource:
    def test_a_viewed_source_loads_the_items_a_run_reads(self, two_datasets: Path) -> None:
        config = _merge_config()
        config.sources = [SourceConfig(name="first", dataset="da", view="first4")]  # type: ignore[assignment]
        config.evaluators = None
        loaded = load_source(config, "first", data_dir=two_datasets)
        assert len(loaded) == 4

    def test_a_loaded_source_matches_what_a_content_digest_task_reads(self, two_datasets: Path) -> None:
        config = _merge_config(
            evaluators=[{"name": "digest", "type": "content-digest"}],
            tasks=[{"name": "t", "evaluator": "digest", "sources": "merged_first4"}],
        )
        result = run_tasks(config, data_dir=two_datasets)["t"]
        assert result.success, result.errors
        loaded = load_source(config, "merged_first4", data_dir=two_datasets)
        assert dataset_digest(loaded).content == result.output.data()["content"]

    def test_a_source_naming_an_unknown_name_lists_the_known_ones(self, two_datasets: Path) -> None:
        with pytest.raises(ValueError, match=r"Unknown source: 'nope'. Available: \['sa', 'sb'"):
            load_source(_merge_config(), "nope", data_dir=two_datasets)

    def test_a_source_naming_an_unknown_view_lists_the_known_ones(self, two_datasets: Path) -> None:
        sources = [{"name": "s", "dataset": "da", "view": "gone"}]
        with pytest.raises(ValueError, match=r"Unknown view: 'gone'. Available: \['va'"):
            load_source(_merge_config(sources=sources), "s", data_dir=two_datasets)

    @pytest.mark.parametrize(
        "operations",
        [
            [{"type": "Shuffle"}, {"type": "Limit", "params": {"size": 3}}],
            [{"type": "Shuffle", "params": {"seed": None}}, {"type": "Limit", "params": {"size": 3}}],
        ],
        ids=["no-seed", "null-seed"],
    )
    def test_a_view_that_would_draw_different_items_on_each_load_is_refused(
        self, two_datasets: Path, operations: list[dict[str, Any]]
    ) -> None:
        config = _merge_config(views=[{"name": "draw", "operations": operations}])
        config.sources = [SourceConfig(name="s", dataset="da", view="draw")]  # type: ignore[assignment]
        with pytest.raises(ValueError, match=r"view 'draw' runs `Shuffle` with no `seed`.*Give it a `seed:`"):
            load_source(config, "s", data_dir=two_datasets)

    def test_a_seeded_shuffle_draws_the_same_items_each_load(self, two_datasets: Path) -> None:
        operations = [{"type": "Shuffle", "params": {"seed": 3}}, {"type": "Limit", "params": {"size": 3}}]
        config = _merge_config(views=[{"name": "draw", "operations": operations}])
        config.sources = [SourceConfig(name="s", dataset="da", view="draw")]  # type: ignore[assignment]
        first = load_source(config, "s", data_dir=two_datasets)
        second = load_source(config, "s", data_dir=two_datasets)
        assert len(first) == 3
        assert dataset_digest(first) == dataset_digest(second)

    def test_an_unseeded_shuffle_as_the_last_operation_only_reorders(self, two_datasets: Path) -> None:
        """Order is not part of a dataset's digest, so a trailing shuffle loads the same items each time."""
        operations = [{"type": "Limit", "params": {"size": 3}}, {"type": "Shuffle"}]
        config = _merge_config(views=[{"name": "draw", "operations": operations}])
        config.sources = [SourceConfig(name="s", dataset="da", view="draw")]  # type: ignore[assignment]
        shuffled = load_source(config, "s", data_dir=two_datasets)
        plain = _merge_config(views=[{"name": "draw", "operations": operations[:1]}])
        plain.sources = config.sources
        assert dataset_digest(shuffled) == dataset_digest(load_source(plain, "s", data_dir=two_datasets))

    def test_an_operand_view_that_shuffles_unseeded_refuses_the_merged_source(self, two_datasets: Path) -> None:
        views = [
            _relabel("va", {"class_0": "cat", "class_1": "dog"}),
            _relabel("vb", {"cat": "cat", "dog": "dog"}),
            {
                "name": "mix",
                "operations": [
                    {"type": "Shuffle"},
                    {
                        "type": "Relabel",
                        "params": {"class_remap": {"class_0": "cat", "class_1": "dog"}, "target": CAT_DOG},
                    },
                    {"type": "Limit", "params": {"size": 2}},
                ],
            },
        ]
        sources = [
            {"name": "sa", "dataset": "da", "view": "mix"},
            {"name": "sb", "dataset": "db", "view": "vb"},
            {"name": "m", "merge": ["sa", "sb"]},
        ]
        config = _merge_config(views=views, sources=sources)
        with pytest.raises(ValueError, match="`Shuffle` with no `seed`"):
            load_source(config, "m", data_dir=two_datasets)
