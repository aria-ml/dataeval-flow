"""Data Analysis Workflow — comprehensive quality analysis across dataset splits.

Assessments are organized by the issue they help diagnose:

- **Image Quality** — anomalous or corrupt images (outliers)
- **Data Redundancy** — duplicate/near-duplicate images, cross-split leakage
- **Label Health** — label completeness, distribution, cross-split parity
- **Metadata Bias** — metadata-factor/label correlations (Balance, Diversity)
- **Distribution Shift** — embedding-space divergence between splits
"""

import contextlib
import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

import numpy as np
import polars as pl
from dataeval import Embeddings, Metadata
from dataeval.bias import Balance, Diversity
from dataeval.core import (
    LabelStatsResult,
    StatsResult,
    divergence_fnn,
    divergence_mst,
    label_parity,
    label_stats,
)
from dataeval.flags import ImageStats
from dataeval.protocols import AnnotatedDataset
from dataeval.quality import Duplicates, Outliers
from pydantic import BaseModel

from dataeval_flow.binning import attach_binning
from dataeval_flow.cache import active_cache, get_or_compute_metadata, get_or_compute_stats
from dataeval_flow.cache import selection_repr as _sel_repr
from dataeval_flow.workflow import WorkflowContext, WorkflowProtocol, WorkflowResult
from dataeval_flow.workflow.base import Reportable
from dataeval_flow.workflows._common import compute_metadata_summary as _compute_metadata_summary
from dataeval_flow.workflows._common import to_serializable as _to_serializable
from dataeval_flow.workflows.analysis.outputs import (
    BiasResult,
    CrossSplitLabelHealth,
    CrossSplitRedundancy,
    CrossSplitResult,
    DataAnalysisMetadata,
    DataAnalysisOutputs,
    DataAnalysisRawOutputs,
    DataAnalysisReport,
    DistributionShiftResult,
    ImageQualityResult,
    LabelHealthResult,
    RedundancyResult,
    SplitResult,
)
from dataeval_flow.workflows.analysis.params import DataAnalysisHealthThresholds, DataAnalysisParameters

__all__ = ["DataAnalysisWorkflow"]

# Note: Dataset analysis runs outlier detection at image level only
# (per_target=False). Target-level outlier analysis (individual bounding
# boxes) is intentionally excluded to focus on image-quality issues.

_logger = logging.getLogger(__name__)

FLAG_MAP: dict[str, ImageStats] = {
    "dimension": ImageStats.DIMENSION,
    "pixel": ImageStats.PIXEL,
    "visual": ImageStats.VISUAL,
}


# ---------------------------------------------------------------------------
# Shared computation layer
# ---------------------------------------------------------------------------


@dataclass
class SplitData:
    """Shared computation results for a single split.

    Produced by ``_compute_split_data`` and consumed by the per-split
    assessment functions.
    """

    metadata: Metadata
    calc_result: "StatsResult"
    img_mask: np.ndarray[Any, Any]
    label_stats: "LabelStatsResult"
    embeddings: np.ndarray[Any, Any] | None
    dataset_len: int


# ---------------------------------------------------------------------------
# Serialization / conversion helpers
# ---------------------------------------------------------------------------


def _inject_image_stats(metadata: Metadata, calc_result: "StatsResult") -> None:
    """Inject computed image/target statistics into *metadata* as factors.

    The stats result labels every value with the entity it describes, so
    ``source_index`` places each one at its own level: a whole-image
    measurement lands on the unit rows and a per-box measurement on the
    instance rows.  Unit-level values propagate down to instance rows, so
    both halves stay visible to the bias evaluators without being
    broadcast by hand.

    Where a statistic is measured at both levels the factor is split in
    two, named for the level it was measured at — ``unit_brightness`` for
    the image and ``instance_brightness`` for the box.  Each is then binned
    over its own population rather than over the replicated copy.

    The only arrays withheld are the hashes.  They travel in the same result
    — one ``compute_stats`` pass serves both outlier and duplicate detection —
    and are near-unique per image, so digitizing them would yield a category
    per item: a factor that correlates with everything and describes nothing.

    Everything else is handed over as it comes, including the vector-valued
    statistics (``histogram``, ``percentiles``, ``center``).  Those have no
    single-column form and cannot become factors either, but dropping them
    *here* would drop them silently; ``add_factors`` records them in
    :attr:`~dataeval.Metadata.dropped_factors`, which is what lets the
    metadata summary report them as measured-but-not-representable rather
    than leaving them missing without explanation.
    """
    # Object, unicode, bytes and void dtypes are the hash columns.  Numeric and
    # boolean arrays are both usable — bool digitizes to a two-value category.
    usable = {name: arr for name, arr in calc_result["stats"].items() if arr.dtype.kind not in "OUSV"}
    if not usable:
        return
    metadata.add_factors(usable, source_index=calc_result["source_index"])


def _labels_from_counts(label_counts: Mapping[int, int]) -> np.ndarray:
    """Reconstruct a flat label array from per-class counts."""
    if not label_counts:
        return np.array([], dtype=int)
    return np.concatenate([np.full(count, cls_id) for cls_id, count in label_counts.items()])


# ---------------------------------------------------------------------------
# Compute shared split data
# ---------------------------------------------------------------------------


def _resolve_outlier_flags(params: DataAnalysisParameters) -> ImageStats:
    """Resolve outlier flags from analysis parameters."""
    flags = ImageStats.NONE
    for name in params.outlier_flags:
        flags |= FLAG_MAP[name]
    return flags


def _compute_split_data(
    dataset: "AnnotatedDataset[Any]",
    params: DataAnalysisParameters,
    extractor: Embeddings | None = None,
    split_name: str = "default",
) -> SplitData:
    """Compute shared data for a single split.

    Builds metadata, runs ``compute_stats`` (via shared cache),
    injects image stats, computes label stats, and extracts embeddings.
    The result is a ``SplitData`` object consumed by the per-split
    assessment functions.
    """
    _logger.info("  Processing metadata for '%s' ...", split_name)
    metadata = get_or_compute_metadata(
        dataset,
        auto_bin_method=params.metadata_auto_bin_method,
        exclude=list(params.metadata_exclude) if params.metadata_exclude else None,
        continuous_factor_bins=params.metadata_continuous_factor_bins,
    )

    # Single compute_stats() call — combines image-stat, outlier-stat,
    # and hash flags into one pass over the dataset.
    is_od = metadata.multi_target
    _logger.info("  Computing image statistics for '%s' ...", split_name)
    outlier_flags = _resolve_outlier_flags(params)
    all_flags = outlier_flags | ImageStats.HASH
    calc_result = get_or_compute_stats(
        desired_flags=all_flags,
        dataset=dataset,
        per_image=True,
        per_target=is_od,
        value_range=params.value_range,
    )
    source_index = calc_result["source_index"]
    img_mask = np.array([si.target is None for si in source_index])

    # Inject image stats as metadata factors for bias analysis
    if params.include_image_stats:
        _inject_image_stats(metadata, calc_result)

    # Label statistics
    _logger.info("  Computing label statistics for '%s' ...", split_name)
    index2label = dataset.metadata.get("index2label")
    ls = label_stats(
        class_labels=metadata.class_labels,
        item_indices=metadata.item_indices,
        index2label=index2label,
        image_count=len(dataset),
    )

    # Embeddings (optional)
    emb = None
    if extractor is not None:
        _logger.info("  Extracting embeddings for '%s' ...", split_name)
        emb = np.asarray(extractor)

    return SplitData(
        metadata=metadata,
        calc_result=calc_result,  # type: ignore[arg-type]
        img_mask=img_mask,
        label_stats=ls,
        embeddings=emb,
        dataset_len=len(dataset),
    )


# ---------------------------------------------------------------------------
# Per-split assessment functions
# ---------------------------------------------------------------------------


def _assess_image_quality(
    data: SplitData,
    outlier_method: Literal["adaptive", "zscore", "modzscore", "iqr"],
    outlier_threshold: float | None = None,
) -> ImageQualityResult:
    """Assess image quality via outlier detection."""
    _logger.info("  Detecting outliers ...")

    # Filter to image-level entries only
    img_calc: StatsResult = {
        "source_index": [si for si, m in zip(data.calc_result["source_index"], data.img_mask, strict=True) if m],
        "object_count": data.calc_result["object_count"],
        "invalid_box_count": data.calc_result["invalid_box_count"],
        "image_count": data.calc_result["image_count"],
        "stats": {k: v[data.img_mask] for k, v in data.calc_result["stats"].items()},
    }

    outliers_eval = Outliers(outlier_threshold=(f"{outlier_method}", outlier_threshold))
    outlier_df = outliers_eval.from_stats(img_calc).data()

    if len(outlier_df) == 0:
        return ImageQualityResult(outlier_count=0, outlier_rate=0.0, outlier_summary={})

    outlier_count: int = outlier_df["item_index"].n_unique()
    metric_agg = outlier_df.group_by("metric_name").agg(pl.col("item_index").n_unique().alias("count"))
    outlier_by_metric: dict[str, int] = dict(metric_agg.iter_rows())

    return ImageQualityResult(
        outlier_count=outlier_count,
        outlier_rate=outlier_count / max(data.dataset_len, 1),
        outlier_summary=_to_serializable(outlier_by_metric),
    )


def _assess_redundancy(data: SplitData) -> RedundancyResult:
    """Assess data redundancy via duplicate detection."""
    _logger.info("  Detecting duplicates ...")
    dup_result = Duplicates().from_stats(data.calc_result)
    exact_groups = dup_result.items.exact or []
    near_groups = dup_result.items.near or []

    near_index_groups = [list(g[0]) if isinstance(g, tuple) else list(g.indices) for g in near_groups]

    return RedundancyResult(
        exact_duplicate_groups=len(exact_groups),
        near_duplicate_groups=len(near_groups),
        exact_duplicates_count=sum(len(g) for g in exact_groups),
        near_duplicates_count=sum(len(g) for g in near_index_groups),
        exact_groups=[list(g) for g in exact_groups],
        near_groups=near_index_groups,
    )


def _assess_label_health(data: SplitData) -> LabelHealthResult:
    """Assess label completeness and distribution."""
    ls = data.label_stats
    class_dist = _to_serializable(
        {ls["index2label"].get(k, str(k)): v for k, v in ls["label_counts_per_class"].items()}
    )

    return LabelHealthResult(
        num_classes=ls["class_count"],
        class_distribution=class_dist,
        empty_images=list(ls["empty_image_indices"]),
    )


def _assess_bias(
    data: SplitData,
    balance: bool,
    diversity_method: Literal["simpson", "shannon"] | None,
) -> BiasResult:
    """Assess metadata bias via Balance and Diversity evaluators."""
    if balance or diversity_method:
        _logger.info("  Running bias analysis ...")

    balance_summary: dict[str, Any] | None = None
    diversity_summary: dict[str, Any] | None = None

    # Suppress sklearn warning about high-cardinality discrete factors
    # being used as y in mutual_info_classif — expected when injected
    # image stats have many unique integer values.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*unique classes.*", module="sklearn")

        if balance and data.metadata.factor_names:
            bal_result = Balance().evaluate(data.metadata)
            balance_summary = _to_serializable(
                {
                    "balance": bal_result.balance.to_dicts(),
                    "factors": bal_result.factors.to_dicts(),
                    "classwise": bal_result.classwise.to_dicts(),
                }
            )

        if diversity_method is not None and data.metadata.factor_names:
            div_result = Diversity(method=diversity_method).evaluate(data.metadata)
            diversity_summary = _to_serializable(
                {
                    "factors": div_result.factors.to_dicts(),
                    "classwise": div_result.classwise.to_dicts(),
                }
            )

    _logger.info("  Summarizing metadata ...")
    meta_summary = _compute_metadata_summary(data.metadata)

    return BiasResult(
        metadata_factors=list(data.metadata.factor_names),
        metadata_summary=meta_summary,
        balance_summary=balance_summary,
        diversity_summary=diversity_summary,
    )


# ---------------------------------------------------------------------------
# Cross-split assessment functions
# ---------------------------------------------------------------------------


def _assess_cross_redundancy(
    calc_a: "StatsResult",
    calc_b: "StatsResult",
    name_a: str,
    name_b: str,
) -> CrossSplitRedundancy:
    """Detect duplicate images across two splits (data leakage).

    When ``Duplicates().from_stats([stats_a, stats_b])`` is called with a
    list of stats, each row in the output DataFrame is a duplicate group.
    We only report groups where both datasets have members (true cross-split
    leakage), ignoring within-split duplicates.
    """
    dup_result = Duplicates().from_stats([calc_a, calc_b])
    ds_names = {0: name_a, 1: name_b}

    exact_groups, near_groups = (
        _extract_cross_groups(dup_result.items.data(), dup_type, ds_names) for dup_type in ("exact", "near")
    )

    exact_count = sum(sum(len(v) for v in g.values()) for g in exact_groups)
    near_count = sum(sum(len(v) for v in g.values()) for g in near_groups)

    return CrossSplitRedundancy(
        duplicate_leakage=_to_serializable(
            {
                "exact_count": exact_count,
                "near_count": near_count,
                "exact_groups": exact_groups,
                "near_groups": near_groups,
            }
        )
    )


def _extract_cross_groups(
    df: pl.DataFrame,
    dup_type: str,
    ds_names: dict[int, str],
) -> list[dict[str, list[int]]]:
    """Extract cross-dataset duplicate groups from a multi-dataset DuplicatesOutput.

    Only includes groups where members span both datasets (true cross-split
    duplicates).  Groups that are entirely within one dataset are skipped.
    """
    filtered = df.filter((pl.col("dup_type") == dup_type) & (pl.col("level") == "item"))
    groups: list[dict[str, list[int]]] = []
    for row in filtered.iter_rows(named=True):
        by_ds: dict[str, list[int]] = {}
        for item, ds_idx in zip(row["item_indices"], row["dataset_indices"], strict=True):
            by_ds.setdefault(ds_names[ds_idx], []).append(item)
        # Only keep groups that span both datasets (true leakage)
        if len(by_ds) >= 2:
            groups.append({k: sorted(v) for k, v in by_ds.items()})
    return groups


def _assess_cross_label_health(
    ls_a: "LabelStatsResult",
    ls_b: "LabelStatsResult",
    name_a: str,
    name_b: str,
) -> CrossSplitLabelHealth:
    """Compare label distributions and test parity between two splits."""
    # Label overlap
    classes_a = set(ls_a["label_counts_per_class"].keys())
    classes_b = set(ls_b["label_counts_per_class"].keys())

    shared = classes_a & classes_b
    only_a = classes_a - classes_b
    only_b = classes_b - classes_a

    i2l: dict[int, str] = {**ls_a["index2label"], **ls_b["index2label"]}

    total_a = max(ls_a["label_count"], 1)
    total_b = max(ls_b["label_count"], 1)

    proportion_diff: dict[str, dict[str, float]] = {}
    for cls in sorted(shared):
        label = i2l.get(cls, str(cls))
        prop_a = ls_a["label_counts_per_class"][cls] / total_a
        prop_b = ls_b["label_counts_per_class"][cls] / total_b
        proportion_diff[label] = {
            name_a: round(prop_a, 4),
            name_b: round(prop_b, 4),
            "difference": round(abs(prop_a - prop_b), 4),
        }

    label_overlap = _to_serializable(
        {
            "shared_classes": [i2l.get(c, str(c)) for c in sorted(shared)],
            f"{name_a}_only": [i2l.get(c, str(c)) for c in sorted(only_a)],
            f"{name_b}_only": [i2l.get(c, str(c)) for c in sorted(only_b)],
            "proportion_comparison": proportion_diff,
        }
    )

    # Label parity — chi-squared test
    num_classes = max(ls_a["class_count"], ls_b["class_count"])
    labels_a = _labels_from_counts(ls_a["label_counts_per_class"])
    labels_b = _labels_from_counts(ls_b["label_counts_per_class"])
    if num_classes > 0 and len(labels_a) > 0 and len(labels_b) > 0:
        lp_result = label_parity(labels_a, labels_b, num_classes=num_classes)
        lp_summary: dict[str, Any] = _to_serializable(
            {
                "chi_squared": lp_result["chi_squared"],
                "p_value": lp_result["p_value"],
                "significant": lp_result["p_value"] < 0.05,
            }
        )
    else:
        lp_summary = {"chi_squared": 0.0, "p_value": 1.0, "significant": False}

    return CrossSplitLabelHealth(
        label_overlap=label_overlap,
        label_parity=lp_summary,
    )


def _assess_distribution_shift(
    emb_a: np.ndarray | None,
    emb_b: np.ndarray | None,
    divergence_method: Literal["mst", "fnn"] | None,
) -> DistributionShiftResult:
    """Compute embedding-space divergence between two splits."""
    if divergence_method is None or emb_a is None or emb_b is None:
        return DistributionShiftResult()

    div_fn = divergence_mst if divergence_method == "mst" else divergence_fnn
    div_result = div_fn(emb_a, emb_b)
    return DistributionShiftResult(
        divergence=float(div_result["divergence"]),
        divergence_method=divergence_method,
    )


# ---------------------------------------------------------------------------
# Findings builders (one per assessment area)
# ---------------------------------------------------------------------------


def _finding_image_quality(
    splits: dict[str, SplitResult],
    thresholds: DataAnalysisHealthThresholds,
) -> Reportable:
    """Cross-split image quality comparison table."""
    rows: list[dict[str, Any]] = []
    total_outliers = 0
    worst_pct = 0.0

    for name, sr in splits.items():
        iq = sr.image_quality
        n = sr.num_samples
        pct = round((iq.outlier_count / max(n, 1)) * 100, 1)
        top = sorted(iq.outlier_summary.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = " ".join(f"{k}({v})" for k, v in top) if top else "-"
        rows.append({"Split": name, "Items": n, "Outliers": iq.outlier_count, "Rate": f"{pct}%", "Top Flags": top_str})
        total_outliers += iq.outlier_count
        worst_pct = max(worst_pct, pct)

    severity: Literal["ok", "info", "warning"] = "warning" if worst_pct > thresholds.image_outliers else "info"
    if total_outliers == 0:
        severity = "ok"

    # Build brief: compact per-split summary
    parts = [f"{sr.image_quality.outlier_count}/{sr.num_samples}" for sr in splits.values()]
    brief = f"{total_outliers} outliers ({', '.join(parts)})"

    return Reportable(
        report_type="pivot_table",
        severity=severity,
        title="Image Quality",
        data={
            "brief": brief,
            "table_data": rows,
            "table_headers": ["Split", "Items", "Outliers", "Rate", "Top Flags"],
        },
        description=f"{total_outliers} images flagged across {len(splits)} split(s).",
    )


def _finding_redundancy(
    splits: dict[str, SplitResult],
    thresholds: DataAnalysisHealthThresholds,
) -> Reportable:
    """Cross-split redundancy comparison table."""
    rows: list[dict[str, Any]] = []
    any_dupes = False
    worst_sev: Literal["ok", "info", "warning"] = "ok"

    for name, sr in splits.items():
        rd = sr.redundancy
        n = sr.num_samples
        exact_pct = round((rd.exact_duplicates_count / max(n, 1)) * 100, 1)
        near_pct = round((rd.near_duplicates_count / max(n, 1)) * 100, 1)
        rows.append(
            {
                "Split": name,
                "Exact": f"{rd.exact_duplicates_count} ({exact_pct}%)",
                "Near": f"{rd.near_duplicates_count} ({near_pct}%)",
            }
        )
        if rd.exact_duplicate_groups or rd.near_duplicate_groups:
            any_dupes = True
        if exact_pct > thresholds.exact_duplicates or near_pct > thresholds.near_duplicates:
            worst_sev = "warning"
        elif any_dupes and worst_sev == "ok":
            worst_sev = "info"

    if not any_dupes:
        return Reportable(
            report_type="key_value",
            severity="ok",
            title="Redundancy",
            data={"brief": "No duplicates in any split"},
            description="No duplicates detected.",
        )

    total_exact = sum(sr.redundancy.exact_duplicates_count for sr in splits.values())
    total_near = sum(sr.redundancy.near_duplicates_count for sr in splits.values())
    brief = f"{total_exact} exact, {total_near} near duplicates"

    return Reportable(
        report_type="pivot_table",
        severity=worst_sev,
        title="Redundancy",
        data={
            "brief": brief,
            "table_data": rows,
            "table_headers": ["Split", "Exact", "Near"],
        },
        description=f"{total_exact} exact + {total_near} near duplicates across {len(splits)} split(s).",
    )


def _finding_label_balance(
    splits: dict[str, SplitResult],
    thresholds: DataAnalysisHealthThresholds,
) -> Reportable:
    """Cross-split label balance comparison table."""
    # Collect all class names across splits
    all_classes: dict[str, dict[str, int]] = {}
    split_names = list(splits.keys())
    imbalance_ratios: dict[str, float] = {}

    split_totals: dict[str, int] = {}
    for name, sr in splits.items():
        lh = sr.label_health
        counts = list(lh.class_distribution.values()) if lh.class_distribution else []
        has_empty = bool(counts) and min(counts) == 0
        ratio = round(max(counts) / min(counts), 1) if counts and not has_empty else 0.0
        imbalance_ratios[name] = ratio
        split_totals[name] = sum(counts)
        for cls, count in lh.class_distribution.items():
            all_classes.setdefault(cls, {})[name] = count

    # Build rows sorted by total count descending, with % of split total
    rows: list[dict[str, Any]] = []
    for cls in sorted(all_classes, key=lambda c: sum(all_classes[c].values()), reverse=True):
        row: dict[str, Any] = {"Class": cls}
        for sn in split_names:
            count = all_classes[cls].get(sn, 0)
            total = split_totals.get(sn, 0)
            pct = round(count / total * 100) if total else 0
            row[sn] = f"{count} ({pct}%)"
        rows.append(row)

    # Determine severity
    num_classes = len(all_classes)
    worst_ratio = max(imbalance_ratios.values()) if imbalance_ratios else 0.0
    any_empty = any(len(sr.label_health.empty_images) > 0 for sr in splits.values())
    severity: Literal["ok", "info", "warning"] = "info"
    if any_empty or worst_ratio > thresholds.class_label_imbalance:
        severity = "warning"

    # Footer with imbalance ratios
    footer_lines: list[str] = []
    footer_lines.append("Imbalance ratio:")
    for n, r in imbalance_ratios.items():
        footer_lines.append(f"  {n}: {r}:1")
    for name, sr in splits.items():
        empty = len(sr.label_health.empty_images)
        if empty:
            footer_lines.append(f"{name}: {empty} images with no labels")

    brief = f"{num_classes} classes, imbalance {'/'.join(f'{r}' for r in imbalance_ratios.values())}:1"

    return Reportable(
        report_type="pivot_table",
        severity=severity,
        title="Label Balance",
        data={
            "brief": brief,
            "table_data": rows,
            "table_headers": ["Class"] + split_names,
            "footer_lines": footer_lines,
        },
        description=f"{num_classes} classes across {len(splits)} split(s).",
    )


def _factor_name(f: dict[str, Any]) -> str:
    """Extract factor name from a factor dict, tolerating key variations."""
    return str(f.get("factor_name", f.get("factor", f.get("name", "?"))))


def _extract_balance_insights(balance_summary: dict[str, Any] | None) -> list[str]:
    """Extract top high-MI factor names from balance results."""
    if balance_summary is None:
        return []
    insights: list[str] = []
    # "balance" holds factor-to-class MI; "factors" holds inter-factor MI
    bal_factors = balance_summary.get("balance", balance_summary.get("factors", []))
    sorted_factors = sorted(bal_factors, key=lambda f: f.get("mi_value", f.get("score", 0)), reverse=True)
    for f in sorted_factors[:3]:
        score = f.get("mi_value", f.get("score", 0))
        if score > 0.1:
            insights.append(f"{_factor_name(f)} (MI={score:.2f})")
    return insights


def _extract_diversity_insights(diversity_summary: dict[str, Any] | None) -> list[str]:
    """Extract top low-diversity factor names from diversity results."""
    if diversity_summary is None:
        return []
    insights: list[str] = []
    div_factors = diversity_summary.get("factors", [])
    sorted_factors = sorted(div_factors, key=lambda f: f.get("diversity_value", f.get("score", 0)))
    for f in sorted_factors[:3]:
        score = f.get("diversity_value", f.get("score", 0))
        if score < 0.5:
            insights.append(f"{_factor_name(f)} ({score:.2f})")
    return insights


def _finding_bias(
    splits: dict[str, SplitResult],
) -> Reportable:
    """Cross-split bias summary."""
    balance_by_split: dict[str, list[str]] = {}
    diversity_by_split: dict[str, list[str]] = {}
    n_factors = 0
    any_warning = False

    for name, sr in splits.items():
        bias = sr.bias
        n_factors = max(n_factors, len(bias.metadata_factors))
        bal = _extract_balance_insights(bias.balance_summary)
        div = _extract_diversity_insights(bias.diversity_summary)
        if bal:
            balance_by_split[name] = bal
            any_warning = True
        if div:
            diversity_by_split[name] = div
            any_warning = True

    brief = f"{n_factors} factors checked"
    if any_warning:
        brief += ", issues found"

    # If no issues at all, return simple key_value
    if not balance_by_split and not diversity_by_split:
        return Reportable(
            report_type="key_value",
            severity="info",
            title="Bias",
            data={
                "brief": brief,
                "detail_lines": ["No high-MI or low-diversity factors in any split."],
            },
            description=f"{n_factors} metadata factors checked across {len(splits)} split(s).",
        )

    # Build table rows: one row per split
    all_split_names = list(splits.keys())
    rows: list[dict[str, str]] = []
    for sn in all_split_names:
        bal_items = balance_by_split.get(sn, [])
        div_items = diversity_by_split.get(sn, [])
        rows.append(
            {
                "Data Split": sn,
                "Top High MI Factors": "\n".join(bal_items) if bal_items else "-",
                "Low Diversity Factors": "\n".join(div_items) if div_items else "-",
            }
        )

    return Reportable(
        report_type="pivot_table",
        severity="warning" if any_warning else "info",
        title="Bias",
        data={
            "brief": brief,
            "table_data": rows,
            "table_headers": ["Data Split", "Top High MI Factors", "Low Diversity Factors"],
        },
        description=f"{n_factors} metadata factors checked across {len(splits)} split(s).",
    )


def _finding_label_overlap(
    cross_split: dict[str, CrossSplitResult],
) -> Reportable:
    """Aggregate label overlap across all split pairs."""
    rows: list[dict[str, Any]] = []
    total_exclusive = 0

    for pair_name, csr in cross_split.items():
        overlap = csr.label_health.label_overlap
        shared = overlap.get("shared_classes", [])
        exclusive = sum(len(v) for k, v in overlap.items() if k.endswith("_only") and isinstance(v, list))
        total_exclusive += exclusive
        status = f"{exclusive} exclusive" if exclusive else "all shared"
        rows.append({"Pair": pair_name, "Shared": len(shared), "Exclusive": exclusive, "Status": status})

    if total_exclusive:
        brief = f"{total_exclusive} exclusive classes across pairs"
        severity: Literal["ok", "info", "warning"] = "warning"
    else:
        # Grab class count from first pair
        first = next(iter(cross_split.values()))
        n = len(first.label_health.label_overlap.get("shared_classes", []))
        brief = f"All {n} classes shared across splits"
        severity = "ok"

    if len(rows) == 1:
        return Reportable(
            report_type="key_value",
            severity=severity,
            title="Label Overlap",
            data={"brief": brief},
            description=brief + ".",
        )

    return Reportable(
        report_type="pivot_table",
        severity=severity,
        title="Label Overlap",
        data={
            "brief": brief,
            "table_data": rows,
            "table_headers": ["Pair", "Shared", "Exclusive", "Status"],
        },
        description=brief + ".",
    )


def _finding_label_parity(
    cross_split: dict[str, CrossSplitResult],
) -> Reportable | None:
    """Aggregate label parity across all split pairs."""
    rows: list[dict[str, Any]] = []
    any_sig = False

    for pair_name, csr in cross_split.items():
        if csr.label_health.label_parity is None:
            continue
        p = csr.label_health.label_parity.get("p_value", 1.0)
        sig = csr.label_health.label_parity.get("significant", False)
        if sig:
            any_sig = True
        rows.append({"Pair": pair_name, "p-value": f"{p:.2g}", "Significant": "yes" if sig else "no"})

    if not rows:
        return None

    n_sig = sum(1 for r in rows if r["Significant"] == "yes")
    brief = f"{n_sig}/{len(rows)} pair(s) significantly different" if any_sig else "No significant differences"

    if len(rows) == 1:
        return Reportable(
            report_type="key_value",
            severity="warning" if any_sig else "ok",
            title="Label Parity",
            data={"brief": brief},
            description=f"Chi-squared test: {brief}.",
        )

    return Reportable(
        report_type="pivot_table",
        severity="warning" if any_sig else "ok",
        title="Label Parity",
        data={
            "brief": brief,
            "table_data": rows,
            "table_headers": ["Pair", "p-value", "Significant"],
        },
        description=f"Chi-squared test: {brief}.",
    )


def _finding_leakage(
    cross_split: dict[str, CrossSplitResult],
) -> Reportable:
    """Aggregate duplicate leakage across all split pairs."""
    rows: list[dict[str, Any]] = []
    total_exact = 0
    total_near = 0

    for pair_name, csr in cross_split.items():
        leakage = csr.redundancy.duplicate_leakage
        exact = leakage.get("exact_count", 0)
        near = leakage.get("near_count", 0)
        total_exact += exact
        total_near += near
        rows.append({"Pair": pair_name, "Exact": exact, "Near": near})

    any_leakage = total_exact > 0 or total_near > 0
    if not any_leakage:
        return Reportable(
            report_type="key_value",
            severity="ok",
            title="Leakage",
            data={"brief": "No cross-split duplicates"},
            description="No cross-split duplicates detected.",
        )

    parts = []
    if total_exact:
        parts.append(f"{total_exact} exact")
    if total_near:
        parts.append(f"{total_near} near")
    brief = f"{' + '.join(parts)} cross-split duplicates"

    if len(rows) == 1:
        return Reportable(
            report_type="key_value",
            severity="warning",
            title="Leakage",
            data={"brief": brief},
            description=f"{brief} (data leakage).",
        )

    return Reportable(
        report_type="pivot_table",
        severity="warning",
        title="Leakage",
        data={
            "brief": brief,
            "table_data": rows,
            "table_headers": ["Pair", "Exact", "Near"],
        },
        description=f"{brief} (data leakage).",
    )


def _divergence_level(div: float, threshold: float) -> tuple[str, Literal["ok", "info", "warning"]]:
    """Classify divergence into (level, severity)."""
    if div > threshold:
        return "high", "warning"
    if div > threshold * 0.4:
        return "moderate", "info"
    return "low", "ok"


def _finding_distribution_shift(
    cross_split: dict[str, CrossSplitResult],
    thresholds: DataAnalysisHealthThresholds,
) -> Reportable | None:
    """Aggregate distribution shift across all split pairs."""
    rows: list[dict[str, Any]] = []
    worst_sev: Literal["ok", "info", "warning"] = "ok"
    sev_rank = {"ok": 0, "info": 1, "warning": 2}

    for pair_name, csr in cross_split.items():
        ds = csr.distribution_shift
        if ds.divergence is None:
            continue
        level, sev = _divergence_level(ds.divergence, thresholds.distribution_shift)
        if sev_rank[sev] > sev_rank[worst_sev]:
            worst_sev = sev
        row = {"Pair": pair_name, "Divergence": f"{ds.divergence:.4f}", "Method": ds.divergence_method, "Level": level}
        rows.append(row)

    if not rows:
        return None

    levels = [r["Level"] for r in rows]
    if "high" in levels:
        brief = f"{levels.count('high')}/{len(rows)} pair(s) high divergence"
    elif "moderate" in levels:
        brief = f"{levels.count('moderate')}/{len(rows)} pair(s) moderate divergence"
    else:
        brief = "Low divergence across all pairs"

    if len(rows) == 1:
        r = rows[0]
        brief = f"{r['Level']} divergence: {r['Divergence']} ({r['Method']})"
        return Reportable(
            report_type="key_value",
            severity=worst_sev,
            title="Distribution Shift",
            data={"brief": brief},
            description=f"{brief.capitalize()}.",
        )

    return Reportable(
        report_type="pivot_table",
        severity=worst_sev,
        title="Distribution Shift",
        data={
            "brief": brief,
            "table_data": rows,
            "table_headers": ["Pair", "Divergence", "Method", "Level"],
        },
        description=brief + ".",
    )


def _build_findings(
    splits: dict[str, SplitResult],
    cross_split: dict[str, CrossSplitResult],
    thresholds: DataAnalysisHealthThresholds,
) -> list[Reportable]:
    """Generate human-readable findings from analysis results."""
    findings: list[Reportable] = []

    # Per-split metrics as cross-split comparison tables
    findings.append(_finding_image_quality(splits, thresholds))
    findings.append(_finding_redundancy(splits, thresholds))
    findings.append(_finding_label_balance(splits, thresholds))
    findings.append(_finding_bias(splits))

    # Cross-split assessments aggregated into one finding each
    if cross_split:
        findings.append(_finding_label_overlap(cross_split))
        parity = _finding_label_parity(cross_split)
        if parity:
            findings.append(parity)
        findings.append(_finding_leakage(cross_split))
        shift = _finding_distribution_shift(cross_split, thresholds)
        if shift:
            findings.append(shift)

    return findings


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class DataAnalysisWorkflow(WorkflowProtocol[DataAnalysisMetadata, DataAnalysisOutputs]):
    """Comprehensive quality analysis across dataset splits.

    Supports two modes of input via ``WorkflowContext``:

    **Multi-split** — ``context.dataset_contexts`` has 2+ entries (e.g. train/test):
        Each entry is analyzed independently, and pairwise cross-split
        comparisons (label overlap, embedding divergence, duplicate leakage)
        are computed.

    **Single omnibus** — single dataset via ``context.dataset`` (e.g. COCO/YOLO):
        The single dataset is analyzed as one split. No cross-split analysis
        is performed.
    """

    @property
    def name(self) -> str:
        """Workflow identifier."""
        return "data-analysis"

    @property
    def description(self) -> str:
        """Human-readable description."""
        return "Comprehensive quality analysis across dataset splits"

    @property
    def params_schema(self) -> type[DataAnalysisParameters]:
        """Pydantic model for workflow parameters."""
        return DataAnalysisParameters

    @property
    def output_schema(self) -> type[DataAnalysisOutputs]:
        """Pydantic model for workflow output."""
        return DataAnalysisOutputs

    def execute(
        self,
        context: WorkflowContext,
        params: BaseModel | None = None,
    ) -> WorkflowResult[DataAnalysisMetadata, DataAnalysisOutputs]:
        """Run data analysis workflow."""
        if not isinstance(context, WorkflowContext):
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                metadata=DataAnalysisMetadata(),
                errors=[f"Expected WorkflowContext, got {type(context).__name__}"],
            )

        if params is None:
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                metadata=DataAnalysisMetadata(),
                errors=["DataAnalysisParameters required (no defaults per CR-4.14-G-1)"],
            )

        if not isinstance(params, DataAnalysisParameters):
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                metadata=DataAnalysisMetadata(),
                errors=[f"Expected DataAnalysisParameters, got {type(params).__name__}"],
            )

        try:
            return self._run(context, params)
        except Exception as e:
            _logger.exception("Workflow '%s' failed", self.name)
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                metadata=DataAnalysisMetadata(),
                errors=[f"Workflow execution failed: {e}"],
            )

    def _run(
        self, context: WorkflowContext, params: DataAnalysisParameters
    ) -> WorkflowResult[DataAnalysisMetadata, DataAnalysisOutputs]:
        """Core execution logic after parameter validation."""
        from dataeval_flow.embeddings import build_embeddings
        from dataeval_flow.view import build_view

        # ── Phase 1: Compute shared data per split ──────────────────
        split_data: dict[str, SplitData] = {}
        # Track resolved datasets per source for the WorkflowResult
        source_datasets: dict[str, AnnotatedDataset[Any]] = {}
        last_dataset: AnnotatedDataset[Any] | None = None

        for split_idx, (split_name, dc) in enumerate(context.dataset_contexts.items(), 1):
            dataset = dc.dataset

            # Apply selection (Limit, Shuffle, ClassFilter, etc.) if configured
            if dc.view_operations:
                dataset = build_view(dataset, dc.view_operations)  # type: ignore[arg-type]

            last_dataset = dataset
            source_datasets[split_name] = dataset
            sel_key = _sel_repr(dataset)

            _logger.info(
                "[data-analysis] Analyzing split %d/%d: '%s' (%d samples)",
                split_idx,
                len(context.dataset_contexts),
                split_name,
                len(dataset),
            )

            embeddings = None
            if dc.extractor:
                embeddings = build_embeddings(
                    dataset,  # type: ignore[arg-type]
                    extractor_config=dc.extractor,
                    transforms=dc.transforms,
                    batch_size=dc.batch_size,
                )

            # Use shared cache infrastructure when available
            with contextlib.ExitStack() as stack:
                if dc.cache is not None:
                    stack.enter_context(active_cache(dc.cache, sel_key))

                split_data[split_name] = _compute_split_data(
                    dataset,  # type: ignore[arg-type]
                    params=params,
                    extractor=embeddings,
                    split_name=split_name,
                )

        # ── Phase 2: Run per-split assessments ──────────────────────
        split_results: dict[str, SplitResult] = {}
        total_samples = 0

        for split_name, data in split_data.items():
            _logger.info("[data-analysis] Assessing '%s' ...", split_name)
            sr = SplitResult(
                num_samples=data.dataset_len,
                image_quality=_assess_image_quality(data, params.outlier_method, params.outlier_threshold),
                redundancy=_assess_redundancy(data),
                label_health=_assess_label_health(data),
                bias=_assess_bias(data, params.balance, params.diversity_method),
            )
            split_results[split_name] = sr
            total_samples += sr.num_samples

        # ── Phase 3: Cross-split assessments ────────────────────────
        cross_split: dict[str, CrossSplitResult] = {}
        if len(context.dataset_contexts) >= 2:
            _logger.info(
                "[data-analysis] Running cross-split analysis (%d splits)",
                len(context.dataset_contexts),
            )
            for name_a, name_b in combinations(split_data.keys(), 2):
                key = f"{name_a}_vs_{name_b}"
                da, db = split_data[name_a], split_data[name_b]

                _logger.info("  Comparing %s vs %s ...", name_a, name_b)
                cross_split[key] = CrossSplitResult(
                    redundancy=_assess_cross_redundancy(
                        da.calc_result,
                        db.calc_result,
                        name_a,
                        name_b,
                    ),
                    label_health=_assess_cross_label_health(
                        da.label_stats,
                        db.label_stats,
                        name_a,
                        name_b,
                    ),
                    distribution_shift=_assess_distribution_shift(
                        da.embeddings,
                        db.embeddings,
                        params.divergence_method,
                    ),
                )

        # ── Phase 4: Assemble outputs & findings ────────────────────
        _logger.info("[data-analysis] Building report (%d total samples)", total_samples)
        findings = _build_findings(split_results, cross_split, params.health_thresholds)

        # Workflow-specific metadata
        result_metadata = DataAnalysisMetadata(
            mode=params.mode,
            split_names=list(context.dataset_contexts.keys()),
        )
        # Splits are binned independently, so each is recorded separately —
        # two splits of one dataset can land on different edges.
        attach_binning(result_metadata, {name: d.metadata for name, d in split_data.items()}, params)
        if params.mode == "preparatory":
            findings.append(
                Reportable(
                    report_type="text",
                    severity="info",
                    title="Preparatory Mode",
                    data="Preparatory mode active.",
                    description="Review per-split outlier and duplicate counts to identify items for removal.",
                )
            )

        raw = DataAnalysisRawOutputs(
            dataset_size=total_samples,
            splits=split_results,
            cross_split=cross_split,
        )

        report = DataAnalysisReport(
            summary=(f"Dataset analysis complete. {len(split_results)} split(s), {total_samples} total items."),
            findings=findings,
        )

        return WorkflowResult(
            name=self.name,
            success=True,
            data=DataAnalysisOutputs(raw=raw, report=report),
            metadata=result_metadata,
            dataset=last_dataset,
            sources=source_datasets,
        )

    def _empty_outputs(self) -> DataAnalysisOutputs:
        return DataAnalysisOutputs(
            raw=DataAnalysisRawOutputs(dataset_size=0),
            report=DataAnalysisReport(summary="Workflow failed", findings=[]),
        )
