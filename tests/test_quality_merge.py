"""The pure merge functions ``data-cleaning`` and the quality evaluators share.

Covers the branches a real dataset rarely hits: both sides empty, columns already
aligned, a column missing from one side only, and ``cast_to_int``'s fallback.
"""

import polars as pl
from dataeval.quality import DuplicatesOutput, OutliersOutput

from dataeval_flow.evaluators.quality._merge import cast_to_int, merge_duplicate_outputs, merge_outlier_outputs

_OUTLIER_COLUMNS = ("item_index", "target_index", "metric_name", "metric_value")


def _outliers(rows: dict | None = None) -> OutliersOutput:
    if rows is None:
        return OutliersOutput(pl.DataFrame(schema=dict.fromkeys(_OUTLIER_COLUMNS, pl.Int64)))
    return OutliersOutput(pl.DataFrame(rows))


def _duplicates(rows: dict | None = None) -> DuplicatesOutput:
    schema = {"group_id": pl.Int64, "level": pl.Utf8, "dup_type": pl.Utf8, "item_indices": pl.List(pl.Int64)}
    if rows is None:
        return DuplicatesOutput(pl.DataFrame(schema=schema))
    return DuplicatesOutput(pl.DataFrame(rows))


class TestMergeOutlierOutputs:
    def test_both_sides_empty_returns_an_empty_table_with_the_full_schema(self):
        result = merge_outlier_outputs(_outliers(), _outliers())
        merged = result.data()
        assert merged.shape[0] == 0
        assert set(merged.columns) == set(_OUTLIER_COLUMNS)

    def test_a_present_and_partly_null_target_index_is_kept_not_dropped(self):
        # `.unique()` does not promise row order, so this checks the values, not their order.
        stats = _outliers(
            {"item_index": [0, 1], "target_index": [None, 2], "metric_name": ["a", "a"], "metric_value": [0.1, 0.2]}
        )
        result = merge_outlier_outputs(stats, _outliers())
        merged = result.data()
        assert "target_index" in merged.columns
        assert merged["target_index"].null_count() == 1
        assert merged["target_index"].max() == 2

    def test_cluster_mode_keeps_the_stats_columns(self):
        # Stats mode reports the numbers behind a determination; cluster mode does not.
        # The merge must keep them rather than stripping down to the four shared columns.
        stats = _outliers(
            {
                "item_index": [0],
                "target_index": [None],
                "metric_name": ["brightness"],
                "metric_value": [0.9],
                "direction": ["upper"],
                "bound": [0.5],
                "percentile": [99.0],
                "population_mean": [0.3],
                "population_std": [0.1],
            }
        )
        cluster = _outliers({"item_index": [1], "metric_name": ["cluster_dist"], "metric_value": [1.2]})
        result = merge_outlier_outputs(stats, cluster)
        merged = result.data()

        for col in ("direction", "bound", "percentile", "population_mean", "population_std"):
            assert col in merged.columns
        stats_row = merged.filter(pl.col("item_index") == 0)
        assert stats_row["direction"].to_list() == ["upper"]
        assert stats_row["population_mean"].to_list() == [0.3]
        # The cluster-only row has no statistical context to report.
        cluster_row = merged.filter(pl.col("item_index") == 1)
        assert cluster_row["direction"].to_list() == [None]

    def test_merging_the_same_inputs_twice_gives_identical_frames(self):
        stats = _outliers(
            {"item_index": [0, 1], "target_index": [None, None], "metric_name": ["a", "b"], "metric_value": [0.1, 0.2]}
        )
        cluster = _outliers({"item_index": [2], "metric_name": ["cluster_dist"], "metric_value": [0.7]})
        first = merge_outlier_outputs(stats, cluster).data()
        second = merge_outlier_outputs(stats, cluster).data()
        assert first.equals(second)


class TestMergeDuplicateOutputs:
    def test_an_empty_hash_side_returns_the_cluster_result_unchanged(self):
        cluster_output = _duplicates(
            {"group_id": [0], "level": ["item"], "dup_type": ["near"], "item_indices": [[2, 3]]}
        )
        result = merge_duplicate_outputs(_duplicates(), cluster_output)
        assert result is cluster_output

    def test_a_column_missing_from_only_the_cluster_side_is_filled_with_null(self):
        hash_df = pl.DataFrame(
            {
                "group_id": [0],
                "level": ["item"],
                "dup_type": ["exact"],
                "item_indices": [[0, 1]],
                "orientation": ["same"],
            }
        )
        cluster_df = pl.DataFrame({"group_id": [0], "level": ["item"], "dup_type": ["near"], "item_indices": [[2, 3]]})
        result = merge_duplicate_outputs(DuplicatesOutput(hash_df), DuplicatesOutput(cluster_df))
        merged = result.data()
        assert merged.shape[0] == 2
        assert merged["orientation"].to_list() == ["same", None]


class TestCastToInt:
    def test_a_value_int_cannot_parse_falls_back_to_zero(self):
        assert cast_to_int("not-a-number") == 0
