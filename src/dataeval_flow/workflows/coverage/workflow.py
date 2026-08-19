"""Data Coverage Workflow — analyze dataset scope and coverage for sufficiency.

Assessments are organized by the dimension they evaluate:

- **Embedding Coverage** — uncovered regions in embedding space
- **Dimensional Completeness** — how effectively data explores embedding dimensions
- **Label Distribution** — class counts, imbalance, empty images
- **Metadata Distribution** — factor balance and diversity
- **Metadata Gaps** — class-factor-value combinations that are under-represented
"""

import contextlib
import logging
import warnings
from typing import Any

import numpy as np
import polars as pl
from dataeval import Metadata
from dataeval.core import label_stats
from dataeval.protocols import AnnotatedDataset, ObjectDetectionTarget
from pydantic import BaseModel

from dataeval_flow.binning import attach_binning
from dataeval_flow.cache import active_cache, get_or_compute_embeddings, get_or_compute_metadata
from dataeval_flow.cache import selection_repr as _sel_repr
from dataeval_flow.config.schemas import FactorSource
from dataeval_flow.workflow import WorkflowContext, WorkflowProtocol, WorkflowResult
from dataeval_flow.workflows._common import compute_metadata_summary as _compute_metadata_summary
from dataeval_flow.workflows._common import normalize_unit_interval
from dataeval_flow.workflows._common import to_serializable as _to_serializable
from dataeval_flow.workflows.coverage.outputs import (
    ClassCoverageRow,
    ClassMetadataGap,
    CompletenessAssessment,
    CoverageAssessment,
    DataCoverageMetadata,
    DataCoverageOutputs,
    DataCoverageRawOutputs,
    DataCoverageReport,
    LabelDistributionResult,
    MetadataDistributionResult,
    MetadataGapResult,
    OntologyAssessment,
)
from dataeval_flow.workflows.coverage.params import DataCoverageParameters
from dataeval_flow.workflows.coverage.report import build_findings

__all__ = ["DataCoverageWorkflow"]

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label distribution
# ---------------------------------------------------------------------------


def _build_label_distribution(ls: Any, index2label: dict[int, str] | None) -> LabelDistributionResult:
    """Assemble label distribution, folding in classes that have no samples.

    ``label_stats`` only reports classes it actually observed, so a class declared in
    the dataset's ``index2label`` but never present would silently vanish — the
    strongest coverage signal there is.  Those are recorded at a count of zero and
    listed in ``missing_classes``.
    """
    observed_counts = ls["label_counts_per_class"]
    class_dist: dict[str, int] = {ls["index2label"].get(k, str(k)): v for k, v in observed_counts.items()}

    declared = sorted((int(k), str(v)) for k, v in (index2label or {}).items())
    missing_classes = [name for idx, name in declared if idx not in observed_counts]
    class_dist.update(dict.fromkeys(missing_classes, 0))

    return LabelDistributionResult(
        num_classes=ls["class_count"],
        class_distribution=_to_serializable(class_dist),
        empty_images=list(ls["empty_image_indices"]),
        missing_classes=missing_classes,
    )


# ---------------------------------------------------------------------------
# Embedding-based assessments
# ---------------------------------------------------------------------------


def _run_coverage(
    metadata: Metadata,
    embeddings: np.ndarray,
    params: DataCoverageParameters,
    unit: str = "image",
) -> CoverageAssessment:
    """Run embedding-space coverage analysis, broken down by class.

    ``unit`` names what one embedding is — an ``"image"`` for image classification,
    a ``"detection crop"`` when the dataset was wrapped by :func:`_crop_view`.
    """
    from dataeval.scope import Coverage

    def _evaluate(method: str) -> Any:
        return Coverage(
            method=method,  # type: ignore[arg-type]
            num_observations=params.num_observations,
            percent=params.coverage_percent,
            min_class_samples=params.min_class_samples,
            isotropy_min_samples=params.isotropy_min_samples,
            near_duplicate_factor=params.near_duplicate_factor,
        ).evaluate(metadata, embeddings=embeddings)

    method = params.coverage_method
    try:
        result = _evaluate(method)
    except OverflowError:
        # The naive radius includes gamma(n_dims / 2 + 1), which overflows float64
        # beyond roughly 340 embedding dimensions. Fall back rather than aborting —
        # the caller may not know the extractor's output width.
        _logger.warning(
            "  Naive coverage is not computable for %d-dimensional embeddings "
            "(its radius overflows); falling back to adaptive coverage.",
            embeddings.shape[1],
        )
        result = _evaluate("adaptive")
        method = "adaptive (naive overflowed)"

    uncovered_count = len(result.uncovered_indices)
    return CoverageAssessment(
        method=method,
        uncovered_count=uncovered_count,
        uncovered_rate=uncovered_count / max(len(embeddings), 1),
        coverage_radius=float(result.coverage_radius),
        observation_count=len(embeddings),
        observation_unit=unit,
        per_class=[
            ClassCoverageRow(
                class_name=str(row["class"]),
                count=int(row["count"]),
                uncovered=int(row["uncovered"]),
                uncovered_fraction=float(row["uncovered_fraction"]),
                dispersion=None if row["dispersion"] is None else float(row["dispersion"]),
                isotropy=None if row["isotropy"] is None else float(row["isotropy"]),
                near_duplicate_fraction=(
                    None if row["near_duplicate_fraction"] is None else float(row["near_duplicate_fraction"])
                ),
                assessable=bool(row["assessable"]),
            )
            for row in result.data().iter_rows(named=True)
        ],
    )


def _is_object_detection(dataset: AnnotatedDataset[Any]) -> bool:
    """Whether the dataset's targets are detections rather than one label per image."""
    if len(dataset) == 0:
        return False
    datum = dataset[0]
    return isinstance(datum, tuple) and len(datum) == 3 and isinstance(datum[1], ObjectDetectionTarget)


def _crop_view(
    dataset: AnnotatedDataset[Any],
    metadata: Metadata,
    sel_key: str,
) -> tuple[AnnotatedDataset[Any], Metadata, str, str]:
    """Present an object-detection dataset's boxes as an image-classification dataset.

    The embedding assessments assume one embedding per label. An object-detection
    image holds many detections of different classes, so whole-image embeddings do
    not line up with ``metadata.class_labels`` — ``Coverage.evaluate`` rejects that
    with a ``ShapeMismatchError``. :class:`~dataeval.data.DetectionCrops` turns each
    ground-truth box into its own classification datum (one crop per detection,
    labeled with that detection's class), restoring the 1:1 alignment.

    Returns ``(dataset, metadata, cache_key, unit)``, all unchanged for an
    image-classification dataset.
    """
    if not _is_object_detection(dataset):
        return dataset, metadata, sel_key, "image"

    from dataeval.data import DetectionCrops

    crops = DetectionCrops(dataset)
    _logger.info(
        "[data-coverage] Object detection dataset — embedding %d detection crops (%d degenerate boxes dropped).",
        len(crops),
        crops.n_dropped,
    )

    # DetectionCrops flattens detections in the same image-then-detection order that
    # Metadata does, so the source labels already align 1:1 with the crops — unless
    # min_size dropped boxes, in which case only a Metadata over the view lines up.
    aligned = crops.n_dropped == 0 and len(crops) == len(metadata.class_labels)
    crop_metadata = metadata if aligned else Metadata(crops)

    # Distinct cache key: crop embeddings must never be served from — or overwrite —
    # the whole-image embeddings cached under the dataset's own selection key.
    return crops, crop_metadata, f"{sel_key}|detection-crops:n={len(crops)}", "detection crop"


def _run_embedding_analysis(
    dc: Any,
    dataset: AnnotatedDataset[Any],
    metadata: Metadata,
    sel_key: str,
    params: DataCoverageParameters,
) -> tuple[CoverageAssessment | None, CompletenessAssessment | None, str | None]:
    """Run the extractor-dependent assessments, or return empties when none is configured.

    Object-detection datasets are embedded as detection crops (see :func:`_crop_view`),
    so both coverage and completeness describe the crop embedding space.

    Returns ``(coverage, completeness, coverage_skipped_reason)``.
    """
    if dc.extractor is None:
        return None, None, None

    emb_dataset, emb_metadata, emb_key, unit = _crop_view(dataset, metadata, sel_key)
    if len(emb_dataset) == 0:
        skipped_reason = f"there are no {unit}s to embed"
        _logger.warning("[data-coverage] Skipping embedding analysis — %s.", skipped_reason)
        return None, None, skipped_reason

    _logger.info("[data-coverage] Computing embeddings (%d %ss) ...", len(emb_dataset), unit)
    # Route through the cache like every other extractor-based workflow — calling
    # build_embeddings directly re-runs extraction on every invocation even when
    # cache_dir is configured.
    with contextlib.ExitStack() as stack:
        if dc.cache is not None:
            stack.enter_context(active_cache(dc.cache, emb_key))
        embeddings_obj = get_or_compute_embeddings(emb_dataset, dc.extractor, dc.transforms, dc.batch_size)
    all_embeddings = normalize_unit_interval(np.array(embeddings_obj))

    # dataeval's coverage functions require strictly more embeddings than
    # num_observations. Skip just this assessment rather than letting the ValueError
    # abort label, metadata and gap analysis too.
    coverage_result: CoverageAssessment | None = None
    skipped_reason: str | None = None
    if len(all_embeddings) > params.num_observations:
        coverage_result = _run_coverage(emb_metadata, all_embeddings, params, unit=unit)
    else:
        skipped_reason = (
            f"needs more than num_observations={params.num_observations} samples, "
            f"but the dataset has {len(all_embeddings)} {unit}s"
        )
        _logger.warning("[data-coverage] Skipping embedding coverage — %s.", skipped_reason)

    completeness_result = _run_completeness(all_embeddings) if params.run_completeness else None
    return coverage_result, completeness_result, skipped_reason


def _run_completeness(embeddings: np.ndarray) -> CompletenessAssessment:
    """Run dimensional completeness analysis."""
    from dataeval.core import completeness

    _logger.info("  Computing dimensional completeness ...")
    result = completeness(embeddings)

    pairs = result.get("nearest_neighbor_pairs", []) if isinstance(result, dict) else []
    serialized_pairs = [[int(a), int(b)] for a, b in pairs]

    return CompletenessAssessment(
        completeness_score=float(result["completeness"]),
        nearest_neighbor_pairs=serialized_pairs,
    )


# ---------------------------------------------------------------------------
# Ontology analysis
# ---------------------------------------------------------------------------


def _run_ontology_analysis(
    class_counts: dict[str, int],
    index2label: dict[int, str] | None,
    params: DataCoverageParameters,
) -> tuple[OntologyAssessment | None, str | None]:
    """Resolve an ontology and assess the dataset's labels against it.

    Returns ``(assessment, skipped_reason)``. An ontology problem never aborts the
    run — label, metadata and gap analysis stay useful without it.
    """
    from dataeval_flow.workflows._ontology import OntologyLoadError, load_ontology, synthesize_ontology
    from dataeval_flow.workflows.coverage.ontology import run_ontology_analysis

    if not class_counts:
        return None, "no labels to project onto the ontology"

    try:
        if params.ontology is not None:
            ontology, source = load_ontology(params.ontology)
            synthesized = False
        else:
            ontology, source = synthesize_ontology(index2label or {})
            synthesized = True
    except OntologyLoadError as exc:
        _logger.warning("[data-coverage] Skipping ontology analysis — %s.", exc)
        return None, str(exc)

    # A resolved ontology can still fail during analysis (e.g. Representation.evaluate,
    # label_reconciliation, ontology_validation). That must not abort label, metadata
    # and gap analysis either — degrade the same way an OntologyLoadError does above.
    try:
        return (
            run_ontology_analysis(
                ontology,
                source=source,
                synthesized=synthesized,
                class_counts=class_counts,
                expected=params.ontology_expected,
                label_pattern=None if synthesized else params.ontology_label_pattern,
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - any analysis failure degrades to a skip reason
        _logger.warning("[data-coverage] Skipping ontology analysis — %s.", exc)
        return None, str(exc)


# ---------------------------------------------------------------------------
# Metadata gap analysis
# ---------------------------------------------------------------------------


def _find_factor_gaps(
    fname: str,
    factor_col: pl.Series,
    class_labels: Any,
    label_map: dict[int, str],
    n_total: int,
    gap_min_representation: int,
) -> list[ClassMetadataGap]:
    """Find under-represented class-factor-value combinations for a single factor."""
    overall_vc = factor_col.value_counts().sort("count", descending=True)
    if len(overall_vc) == 0:
        return []

    overall_dist: dict[Any, float] = {}
    for row in overall_vc.iter_rows(named=True):
        overall_dist[row[fname]] = row["count"] / max(n_total, 1)

    unique_classes = sorted({int(c) for c in class_labels})
    gaps: list[ClassMetadataGap] = []

    for cls_id in unique_classes:
        cls_name = label_map.get(cls_id, str(cls_id))
        cls_mask = np.array(class_labels) == cls_id
        cls_size = int(cls_mask.sum())
        if cls_size == 0:
            continue

        cls_factor_values = factor_col.filter(pl.Series(cls_mask))
        cls_vc = cls_factor_values.value_counts()
        cls_counts: dict[Any, int] = {}
        for row in cls_vc.iter_rows(named=True):
            cls_counts[row[fname]] = row["count"]

        for fval, overall_prop in overall_dist.items():
            expected = overall_prop * cls_size
            actual = cls_counts.get(fval, 0)

            if actual < gap_min_representation and expected > gap_min_representation:
                deficit = 1.0 - (actual / max(expected, 1e-9))
                gaps.append(
                    ClassMetadataGap(
                        class_name=cls_name,
                        factor_name=fname,
                        factor_value=str(fval),
                        class_count=actual,
                        expected_count=round(expected, 1),
                        deficit=round(max(0.0, min(1.0, deficit)), 3),
                    )
                )

    return gaps


def _mi_from_balance(balance_summary: dict[str, Any] | None, factor_names: list[str]) -> dict[str, float] | None:
    """Recover class-to-factor MI from a Balance result already computed, or None if unusable.

    ``Balance.evaluate`` exposes the class-to-factor row as its ``balance`` frame, so a
    coverage run that assessed metadata distribution has already paid for the numbers gap
    analysis needs.  Returns None unless every factor is accounted for, in which case
    :func:`_balance_class_to_factor` computes them.
    """
    if not balance_summary:
        return None
    rows = balance_summary.get("balance")
    if not rows:
        return None

    wanted = set(factor_names)
    mi = {
        str(row["factor_name"]): float(row["mi_value"])
        for row in rows
        if isinstance(row, dict) and row.get("factor_name") in wanted and row.get("mi_value") is not None
    }
    return mi if wanted.issubset(mi) else None


def _balance_class_to_factor(metadata: Metadata, factor_source: FactorSource | None) -> dict[str, float]:
    """Class-to-factor mutual information, computed the way Balance computes it.

    Gap analysis compares these against ``gap_mi_threshold``, and a threshold is only
    meaningful if it always names the same quantity — so this runs Balance rather than
    calling :func:`~dataeval.core.mutual_info` with flags chosen to imitate it.

    Imitating it is no longer possible.  ``factor_source`` decides per factor whether the
    codes or the measured values are read, consulting the encoding record's provenance: a
    cut somebody declared or ratified is honored, and one DataEval derived is read past.
    That is a per-factor decision over two different estimators, and the ``discrete_features``
    flag this used to pass is one boolean per column that no longer selects anything.  A
    coverage run computing its own MI would land on numbers Balance does not report, under a
    threshold documented against Balance's.
    """
    from dataeval.bias import Balance

    with warnings.catch_warnings():
        # sklearn warns about high-cardinality discrete factors from inside the estimator.
        warnings.filterwarnings("ignore", message=".*unique classes.*", module="sklearn")
        output = Balance(factor_source=factor_source).evaluate(metadata)
    return {str(row["factor_name"]): float(row["mi_value"]) for row in output.balance.to_dicts()}


def _run_gap_analysis(
    metadata: Metadata,
    index2label: dict[int, str] | None,
    params: DataCoverageParameters,
    precomputed_mi: dict[str, float] | None = None,
) -> MetadataGapResult:
    """Identify metadata coverage gaps by class using mutual information.

    ``precomputed_mi`` supplies class-to-factor MI a Balance run already produced; when
    absent this runs Balance itself, so the numbers compared against ``gap_mi_threshold``
    are Balance's either way.
    """
    _logger.info("  Running metadata gap analysis ...")

    factor_names = list(metadata.factor_names)
    if not factor_names:
        return MetadataGapResult(mutual_info_class_to_factor={}, gaps=[])

    if precomputed_mi is not None:
        _logger.debug("  Reusing class-to-factor MI computed by Balance")
        computed = precomputed_mi
    else:
        _logger.debug("  Computing class-to-factor MI through Balance")
        computed = _balance_class_to_factor(metadata, params.metadata_factor_source)
    mi_per_factor = {fname: computed[fname] for fname in factor_names if fname in computed}

    # Cross-tabulate per-class metadata distributions for high-MI factors.
    # Label-level rows are the only ones that align with class_labels: they are the
    # item rows themselves for image classification, and one row per detection for
    # object detection (where `dataframe` interleaves every level and is therefore
    # longer).  This mirrors how dataeval builds `Metadata.factor_data` — unit-level
    # factors propagate down onto the label rows.
    df = metadata.rows_at(metadata.label_level)
    if df is None or len(df) == 0:
        return MetadataGapResult(mutual_info_class_to_factor=mi_per_factor, gaps=[])

    class_labels = metadata.class_labels
    label_map = index2label or {}
    n_total = len(class_labels)
    gaps: list[ClassMetadataGap] = []

    for fname, mi_score in mi_per_factor.items():
        if mi_score < params.gap_mi_threshold or fname not in df.columns:
            continue
        gaps.extend(
            _find_factor_gaps(fname, df[fname], class_labels, label_map, n_total, params.gap_min_representation)
        )

    # Sort by deficit descending
    gaps.sort(key=lambda g: g.deficit, reverse=True)

    return MetadataGapResult(
        mutual_info_class_to_factor=mi_per_factor,
        gaps=gaps,
    )


# ---------------------------------------------------------------------------
# Metadata distribution assessment
# ---------------------------------------------------------------------------


def _assess_metadata_distribution(
    metadata: Metadata,
    params: DataCoverageParameters,
) -> MetadataDistributionResult:
    """Assess metadata factor balance and diversity."""
    from dataeval.bias import Balance, Diversity

    _logger.info("[data-coverage] Analyzing metadata distribution ...")
    balance_summary: dict[str, Any] | None = None
    diversity_summary: dict[str, Any] | None = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*unique classes.*", module="sklearn")

        if params.balance and metadata.factor_names:
            bal_result = Balance(factor_source=params.metadata_factor_source).evaluate(metadata)
            balance_summary = _to_serializable(
                {
                    "balance": bal_result.balance.to_dicts(),
                    "factors": bal_result.factors.to_dicts(),
                    "classwise": bal_result.classwise.to_dicts(),
                }
            )

        if params.diversity_method is not None and metadata.factor_names:
            div_result = Diversity(method=params.diversity_method).evaluate(metadata)
            diversity_summary = _to_serializable(
                {
                    "factors": div_result.factors.to_dicts(),
                    "classwise": div_result.classwise.to_dicts(),
                }
            )

    meta_summary = _compute_metadata_summary(metadata)

    return MetadataDistributionResult(
        metadata_factors=list(metadata.factor_names),
        metadata_summary=meta_summary,
        balance_summary=balance_summary,
        diversity_summary=diversity_summary,
    )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class DataCoverageWorkflow(WorkflowProtocol[DataCoverageMetadata, DataCoverageOutputs]):
    """Analyze dataset scope and coverage for sufficiency.

    Evaluates five dimensions of dataset quality:

    1. **Embedding Coverage** — identifies uncovered regions (requires extractor)
    2. **Dimensional Completeness** — measures exploration of embedding dims (requires extractor)
    3. **Label Distribution** — class counts, imbalance, empty images
    4. **Metadata Distribution** — factor balance and diversity
    5. **Metadata Gaps** — class-factor-value under-representations

    The embedding assessments need one embedding per label, so an object-detection
    dataset is embedded as detection crops (:class:`~dataeval.data.DetectionCrops`) —
    one crop per ground-truth box, labeled with that box's class. Their per-class rows
    therefore describe detections, while label, metadata and gap analysis stay on the
    source dataset.
    """

    @property
    def name(self) -> str:
        """Workflow identifier."""
        return "data-coverage"

    @property
    def description(self) -> str:
        """Human-readable description."""
        return "Analyze dataset scope and coverage for sufficiency"

    @property
    def params_schema(self) -> type[DataCoverageParameters]:
        """Pydantic model for workflow parameters."""
        return DataCoverageParameters

    @property
    def output_schema(self) -> type[DataCoverageOutputs]:
        """Pydantic model for workflow output."""
        return DataCoverageOutputs

    def execute(
        self,
        context: WorkflowContext,
        params: BaseModel | None = None,
    ) -> WorkflowResult[DataCoverageMetadata, DataCoverageOutputs]:
        """Run data coverage workflow."""
        if not isinstance(context, WorkflowContext):
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                metadata=DataCoverageMetadata(),
                errors=[f"Expected WorkflowContext, got {type(context).__name__}"],
            )

        if params is None:
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                metadata=DataCoverageMetadata(),
                errors=["DataCoverageParameters required"],
            )

        if not isinstance(params, DataCoverageParameters):
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                metadata=DataCoverageMetadata(),
                errors=[f"Expected DataCoverageParameters, got {type(params).__name__}"],
            )

        try:
            return self._run(context, params)
        except Exception as e:
            _logger.exception("Workflow '%s' failed", self.name)
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                metadata=DataCoverageMetadata(),
                errors=[f"Workflow execution failed: {e}"],
            )

    def _run(
        self, context: WorkflowContext, params: DataCoverageParameters
    ) -> WorkflowResult[DataCoverageMetadata, DataCoverageOutputs]:
        """Core execution logic after parameter validation."""
        from dataeval_flow.view import build_view

        # ── Phase 1: Setup ──────────────────────────────────────────
        if not context.dataset_contexts:
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                metadata=DataCoverageMetadata(),
                errors=["No dataset contexts provided"],
            )

        # Use first dataset context (single-source workflow)
        dc_name, dc = next(iter(context.dataset_contexts.items()))
        dataset: AnnotatedDataset[Any] = dc.dataset

        # Apply view operations if configured
        if dc.view_operations:
            _logger.info("[data-coverage] Applying view (%d operations) ...", len(dc.view_operations))
            dataset = build_view(dataset, list(dc.view_operations))  # type: ignore[arg-type]

        dataset_len = len(dataset)
        has_extractor = dc.extractor is not None
        sel_key = _sel_repr(dataset)

        _logger.info(
            "[data-coverage] Analyzing '%s' (%d samples, extractor=%s)",
            dc_name,
            dataset_len,
            "yes" if has_extractor else "no",
        )

        # ── Phase 2: Metadata + label stats ─────────────────────────
        with contextlib.ExitStack() as stack:
            if dc.cache is not None:
                stack.enter_context(active_cache(dc.cache, sel_key))

            _logger.info("[data-coverage] Computing metadata ...")
            metadata = get_or_compute_metadata(
                dataset,
                auto_bin_method=params.metadata_auto_bin_method,
                exclude=list(params.metadata_exclude) if params.metadata_exclude else None,
                continuous_factor_bins=params.metadata_continuous_factor_bins,
            )

            index2label = dataset.metadata.get("index2label")
            _logger.info("[data-coverage] Computing label statistics ...")
            ls = label_stats(
                class_labels=metadata.class_labels,
                item_indices=metadata.item_indices,
                index2label=index2label,
                image_count=len(dataset),
            )

        label_dist = _build_label_distribution(ls, index2label)

        # ── Phase 3: Embedding-based analysis (conditional) ─────────
        coverage_result, completeness_result, coverage_skipped_reason = _run_embedding_analysis(
            dc, dataset, metadata, sel_key, params
        )

        # ── Phase 3b: Ontology analysis ─────────────────────────────
        ontology_result, ontology_skipped_reason = _run_ontology_analysis(
            {name: count for name, count in label_dist.class_distribution.items()},
            index2label,
            params,
        )

        # ── Phase 4: Metadata distribution ──────────────────────────
        metadata_dist = _assess_metadata_distribution(metadata, params)

        # ── Phase 5: Metadata gap analysis ──────────────────────────
        metadata_gaps = None
        if params.run_gap_analysis and metadata.factor_names:
            metadata_gaps = _run_gap_analysis(
                metadata,
                index2label,
                params,
                precomputed_mi=_mi_from_balance(metadata_dist.balance_summary, list(metadata.factor_names)),
            )

        # ── Phase 6: Assemble outputs & findings ────────────────────
        _logger.info("[data-coverage] Building report ...")
        raw = DataCoverageRawOutputs(
            dataset_size=dataset_len,
            coverage=coverage_result,
            coverage_skipped_reason=coverage_skipped_reason,
            completeness=completeness_result,
            metadata_distribution=metadata_dist,
            metadata_gaps=metadata_gaps,
            label_distribution=label_dist,
            ontology=ontology_result,
            ontology_skipped_reason=ontology_skipped_reason,
        )

        findings = build_findings(raw, params.health_thresholds)

        # Summary line
        parts = [f"{dataset_len} samples"]
        if coverage_result:
            parts.append(f"{coverage_result.uncovered_count} uncovered")
        if completeness_result:
            parts.append(f"completeness={completeness_result.completeness_score:.2f}")
        if metadata_gaps and metadata_gaps.gaps:
            parts.append(f"{len(metadata_gaps.gaps)} gaps")
        if ontology_result and ontology_result.representation.total_deficit:
            parts.append(f"deficit={ontology_result.representation.total_deficit}")
        summary = "Coverage analysis: " + ", ".join(parts) + "."

        report = DataCoverageReport(summary=summary, findings=findings)

        result_metadata = DataCoverageMetadata(
            mode=params.mode,
            has_extractor=has_extractor,
        )
        attach_binning(result_metadata, metadata, params)

        return WorkflowResult(
            name=self.name,
            success=True,
            data=DataCoverageOutputs(raw=raw, report=report),
            metadata=result_metadata,
            dataset=dataset,
        )

    def _empty_outputs(self) -> DataCoverageOutputs:
        return DataCoverageOutputs(
            raw=DataCoverageRawOutputs(
                dataset_size=0,
                metadata_distribution=MetadataDistributionResult(metadata_factors=[], metadata_summary={}),
                label_distribution=LabelDistributionResult(num_classes=0, class_distribution={}),
            ),
            report=DataCoverageReport(summary="Workflow failed", findings=[]),
        )
