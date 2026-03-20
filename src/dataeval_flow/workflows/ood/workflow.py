"""OOD detection workflow."""

from __future__ import annotations

import contextlib
import logging
import time as _time
from collections.abc import Sequence
from typing import Any

import numpy as np
from dataeval.core import factor_deviation, factor_predictors
from dataeval.protocols import AnnotatedDataset
from dataeval.shift import OODDomainClassifier, OODKNeighbors, OODOutput
from numpy.typing import NDArray
from pydantic import BaseModel

from dataeval_flow.cache import (
    active_cache,
    get_or_compute_embeddings,
    get_or_compute_metadata,
    get_or_compute_stats,
    selection_repr,
)
from dataeval_flow.workflow import DatasetContext, WorkflowContext, WorkflowProtocol, WorkflowResult
from dataeval_flow.workflows.ood.outputs import (
    DetectorOODResultDict,
    FactorDeviationDict,
    OODDetectionMetadata,
    OODDetectionOutputs,
    OODDetectionRawOutputs,
    OODDetectionReport,
    OODSampleDict,
)
from dataeval_flow.workflows.ood.params import (
    OODDetectionParameters,
    OODDetectorConfig,
    OODDetectorDomainClassifier,
    OODDetectorKNeighbors,
)
from dataeval_flow.workflows.ood.report import build_findings

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detector factory
# ---------------------------------------------------------------------------

_OODDetector = OODKNeighbors | OODDomainClassifier


def _build_ood_detector(config: OODDetectorConfig) -> _OODDetector:  # type: ignore[type-arg]
    """Instantiate an OOD detector from its discriminated config."""
    if isinstance(config, OODDetectorKNeighbors):
        return OODKNeighbors(
            k=config.k,
            distance_metric=config.distance_metric,
            threshold_perc=config.threshold_perc,
        )
    if isinstance(config, OODDetectorDomainClassifier):
        return OODDomainClassifier(
            n_folds=config.n_folds,
            n_repeats=config.n_repeats,
            n_std=config.n_std,
            threshold_perc=config.threshold_perc,
        )
    raise ValueError(f"Unknown OOD detector config type: {type(config).__name__}")


def _ood_detector_display_name(config: OODDetectorConfig) -> str:  # type: ignore[type-arg]
    """Human-readable name for an OOD detector config, including non-default parameters."""
    names: dict[str, str] = {
        "kneighbors": "K-Neighbors",
        "domain_classifier": "Domain Classifier",
    }
    base = names.get(config.method, config.method)

    # Collect non-default, non-internal parameters as a compact suffix
    parts: list[str] = []
    defaults = {name: field.default for name, field in config.model_fields.items()}
    for name in config.model_fields:
        if name == "method":
            continue
        value = getattr(config, name)
        if value != defaults[name]:
            parts.append(f"{name}={value}")

    if parts:
        base = f"{base} ({', '.join(parts)})"
    return base


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


def _serialize_ood_result(
    output: OODOutput,
    config: OODDetectorConfig,  # type: ignore[type-arg]
    test_size: int,
) -> DetectorOODResultDict:
    """Convert an OODOutput to a serializable dict."""
    is_ood = output.is_ood
    scores = output.instance_score
    ood_count = int(np.sum(is_ood))
    ood_pct = 100.0 * ood_count / test_size if test_size > 0 else 0.0

    # Compute threshold from the predict logic: scores > threshold => OOD
    # We derive it from the first non-OOD sample's boundary or max non-OOD score
    if ood_count < test_size:
        non_ood_scores = scores[~is_ood]
        threshold_score = float(np.max(non_ood_scores)) if len(non_ood_scores) > 0 else 0.0
    else:
        # All samples are OOD
        threshold_score = float(np.min(scores))

    samples: list[OODSampleDict] = [
        OODSampleDict(index=int(i), score=float(scores[i]), is_ood=bool(is_ood[i])) for i in range(test_size)
    ]

    return DetectorOODResultDict(
        method=config.method,
        ood_count=ood_count,
        total_count=test_size,
        ood_percentage=round(ood_pct, 2),
        threshold_score=round(threshold_score, 6),
        samples=samples,
    )


# ---------------------------------------------------------------------------
# Embedding extraction helpers
# ---------------------------------------------------------------------------


def _get_embeddings_for_context(
    dc: DatasetContext,
    dataset: AnnotatedDataset[Any],
) -> NDArray[np.float32]:
    """Extract embeddings for a dataset context, using cache if available."""
    if dc.extractor is None:
        raise ValueError(
            "OOD detection requires a model/extractor to compute embeddings. Configure 'models' in the task config."
        )
    sel_key = selection_repr(dataset)
    with contextlib.ExitStack() as stack:
        if dc.cache is not None:
            stack.enter_context(active_cache(dc.cache, sel_key))
        return get_or_compute_embeddings(
            dataset,
            dc.extractor,
            dc.transforms,
            dc.batch_size,
        )


# ---------------------------------------------------------------------------
# Unique method keys
# ---------------------------------------------------------------------------


def _unique_method_keys(
    detectors: Sequence[OODDetectorConfig],  # type: ignore[type-arg]
) -> list[str]:
    """Return a unique key for each detector, appending a numeric suffix for duplicates."""
    counts: dict[str, int] = {}
    for det in detectors:
        counts[det.method] = counts.get(det.method, 0) + 1
    seen: dict[str, int] = {}
    keys: list[str] = []
    for det in detectors:
        base = det.method
        if counts[base] == 1:
            keys.append(base)
        else:
            idx = seen.get(base, 0) + 1
            seen[base] = idx
            keys.append(f"{base}_{idx}")
    return keys


# ---------------------------------------------------------------------------
# Detector execution
# ---------------------------------------------------------------------------


def _run_all_ood_detectors(
    params: OODDetectionParameters,
    ref_embeddings: NDArray[np.float32],
    test_embeddings: NDArray[np.float32],
) -> tuple[dict[str, DetectorOODResultDict], dict[str, str], list[str], list[NDArray[np.bool_]]]:
    """Run all configured OOD detectors and return results, names, errors, and is_ood arrays."""
    logger.info("[3/5] Running %d OOD detector(s)…", len(params.detectors))
    t0 = _time.monotonic()

    detector_results: dict[str, DetectorOODResultDict] = {}
    detector_names: dict[str, str] = {}
    detector_errors: list[str] = []
    is_ood_arrays: list[NDArray[np.bool_]] = []
    method_keys = _unique_method_keys(params.detectors)
    test_size = len(test_embeddings)

    for det_config, method_key in zip(params.detectors, method_keys, strict=True):
        display = _ood_detector_display_name(det_config)
        detector_names[method_key] = display

        try:
            detector = _build_ood_detector(det_config)
            detector.fit(ref_embeddings)
            output: OODOutput = detector.predict(test_embeddings)

            detector_results[method_key] = _serialize_ood_result(output, det_config, test_size)
            is_ood_arrays.append(output.is_ood)

            ood_count = int(np.sum(output.is_ood))
            logger.info("  %s: %d/%d OOD (%.1f%%)", display, ood_count, test_size, 100.0 * ood_count / test_size)
        except Exception as e:  # noqa: BLE001
            logger.warning("OOD detector %s failed: %s", display, e, exc_info=True)
            detector_errors.append(f"{display}: {e}")

    logger.info("[3/5] OOD detection complete in %.1fs", _time.monotonic() - t0)
    return detector_results, detector_names, detector_errors, is_ood_arrays


# ---------------------------------------------------------------------------
# Metadata insights
# ---------------------------------------------------------------------------


def _extract_metadata_factors(
    dc: DatasetContext,
    dataset: AnnotatedDataset[Any],
) -> dict[str, NDArray[Any]] | None:
    """Extract metadata factor arrays from a dataset, returning None on failure.

    Drops the ``id`` factor (not useful for deviation analysis) and includes
    class labels when available.
    """
    try:
        with contextlib.ExitStack() as stack:
            if dc.cache is not None:
                sel_key = selection_repr(dataset)
                stack.enter_context(active_cache(dc.cache, sel_key))
            metadata = get_or_compute_metadata(dataset)
        factor_names = list(metadata.factor_names)

        # Extract raw continuous values from the dataframe for deviation analysis
        df = metadata.dataframe
        factors: dict[str, NDArray[Any]] = {}
        for name in factor_names:
            if name in df.columns:
                factors[name] = df[name].to_numpy()

        # Drop 'id' — not useful for deviation/predictor analysis
        factors.pop("id", None)

        # Include class labels as a numeric factor when available
        if hasattr(metadata, "class_labels") and metadata.class_labels is not None:
            labels = np.asarray(metadata.class_labels)
            if np.issubdtype(labels.dtype, np.number) and len(labels) == len(df):
                factors["class_label"] = labels

        return factors if factors else None
    except Exception:  # noqa: BLE001
        logger.warning("Failed to extract metadata factors", exc_info=True)
        return None


def _extract_stats_factors(
    dc: DatasetContext,
    dataset: AnnotatedDataset[Any],
) -> dict[str, NDArray[Any]] | None:
    """Compute per-image stats and return as ``{metric_name: array}``."""
    from dataeval.flags import ImageStats

    try:
        with contextlib.ExitStack() as stack:
            if dc.cache is not None:
                sel_key = selection_repr(dataset)
                stack.enter_context(active_cache(dc.cache, sel_key))
            stats_result = get_or_compute_stats(
                desired_flags=ImageStats.ALL,
                dataset=dataset,
                per_image=True,
                per_target=False,
                per_channel=False,
            )
        stats_map = stats_result.get("stats", {})
        if not stats_map:
            return None

        n_images = stats_result.get("image_count", 0)
        factors: dict[str, NDArray[Any]] = {}
        for name, arr in stats_map.items():
            arr = np.asarray(arr)
            # Only keep numeric arrays with exactly one value per image
            if np.issubdtype(arr.dtype, np.number) and len(arr) == n_images:
                factors[f"f_{name}"] = arr
        return factors if factors else None
    except Exception:  # noqa: BLE001
        logger.warning("Failed to extract stats factors", exc_info=True)
        return None


def _merge_factor_parts(
    meta_parts: list[dict[str, NDArray[Any]]],
    stats_parts: list[dict[str, NDArray[Any]]],
) -> list[dict[str, NDArray[Any]]]:
    """Merge per-dataset metadata and stats factor dicts.

    When metadata is available, each metadata dict is augmented with its
    corresponding stats dict.  When metadata is absent, stats-only dicts
    are returned.
    """
    if not meta_parts:
        return list(stats_parts)

    merged: list[dict[str, NDArray[Any]]] = []
    for i, meta_dict in enumerate(meta_parts):
        combined = dict(meta_dict)
        if i < len(stats_parts):
            combined.update(stats_parts[i])
        merged.append(combined)
    # Include any remaining stats-only datasets
    merged.extend(stats_parts[len(meta_parts) :])
    return merged


def _intersect_numeric_factors(
    ref_factors: dict[str, NDArray[Any]],
    test_factor_parts: list[dict[str, NDArray[Any]]],
) -> tuple[dict[str, NDArray[Any]], dict[str, NDArray[Any]]] | None:
    """Intersect factor keys, concatenate test parts, and filter to numeric columns."""
    common_keys = set(ref_factors.keys())
    for t_factors in test_factor_parts:
        common_keys &= set(t_factors.keys())

    if not common_keys:
        logger.warning("Skipping metadata insights: no common factors between reference and test.")
        return None

    sorted_keys = sorted(common_keys)
    ref_common = {k: ref_factors[k] for k in sorted_keys}

    # Concatenate test factors
    test_common: dict[str, NDArray[Any]] = {}
    for key in sorted_keys:
        arrays = [t[key] for t in test_factor_parts if key in t]
        test_common[key] = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]

    # Filter to numeric-only factors (factor_deviation/factor_predictors require numeric data)
    numeric_keys = [k for k in sorted_keys if np.issubdtype(ref_common[k].dtype, np.number)]
    if not numeric_keys:
        logger.info("Skipping metadata insights: no numeric factors.")
        return None

    # Exclude 2D arrays — factor_predictors uses np.column_stack but builds
    # discrete_features from the number of *keys*, causing a dimension mismatch
    # when any array is multi-dimensional.
    numeric_keys = [k for k in numeric_keys if ref_common[k].ndim == 1]
    if not numeric_keys:
        logger.info("Skipping metadata insights: no 1D numeric factors.")
        return None

    # Drop factors that contain NaN/Inf or have zero variance — these cause
    # downstream errors in factor_predictors (sklearn mutual_info_classif).
    clean_keys: list[str] = []
    for k in numeric_keys:
        r, t = ref_common[k], test_common[k]
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(t)):
            logger.debug("Dropping factor %r: contains NaN/Inf values", k)
            continue
        if np.std(t) == 0:
            logger.debug("Dropping factor %r: zero variance in test data", k)
            continue
        clean_keys.append(k)

    if not clean_keys:
        logger.info("Skipping metadata insights: no clean numeric factors after sanitization.")
        return None

    return (
        {k: ref_common[k] for k in clean_keys},
        {k: test_common[k] for k in clean_keys},
    )


def _collect_numeric_factors(
    ref_dc: DatasetContext,
    ref_dataset: AnnotatedDataset[Any],
    test_datasets: list[tuple[str, DatasetContext, AnnotatedDataset[Any]]],
) -> tuple[dict[str, NDArray[Any]], dict[str, NDArray[Any]]] | None:
    """Collect and intersect numeric metadata + stats factors from reference and test datasets.

    Stats are always computed for both reference and test datasets.  Metadata
    factors (from the dataset's own metadata dicts) are included when
    available; if test data lacks metadata, only stats-based factors are used.

    Returns ``(ref_factors, test_factors)`` dicts keyed by factor name, or
    *None* when no usable numeric factors are available.
    """
    # --- Stats factors (always available) ---
    ref_stats = _extract_stats_factors(ref_dc, ref_dataset)

    test_stats_parts: list[dict[str, NDArray[Any]]] = []
    for _, t_dc, t_ds in test_datasets:
        t_stats = _extract_stats_factors(t_dc, t_ds)
        if t_stats is not None:
            test_stats_parts.append(t_stats)

    # --- Metadata factors (may be absent on test data) ---
    ref_meta = _extract_metadata_factors(ref_dc, ref_dataset)

    test_meta_parts: list[dict[str, NDArray[Any]]] = []
    for _, t_dc, t_ds in test_datasets:
        t_meta = _extract_metadata_factors(t_dc, t_ds)
        if t_meta is not None:
            test_meta_parts.append(t_meta)

    # Merge metadata + stats for reference
    ref_factors: dict[str, NDArray[Any]] = {}
    if ref_meta:
        ref_factors.update(ref_meta)
    if ref_stats:
        ref_factors.update(ref_stats)

    if not ref_factors:
        logger.info("Skipping metadata insights: no reference factors.")
        return None

    # Merge metadata + stats for test
    test_factor_parts = _merge_factor_parts(test_meta_parts, test_stats_parts)

    if not test_factor_parts:
        logger.info("Skipping metadata insights: no test factors.")
        return None

    return _intersect_numeric_factors(ref_factors, test_factor_parts)


def _compute_metadata_insights(
    ref_dc: DatasetContext,
    ref_dataset: AnnotatedDataset[Any],
    test_datasets: list[tuple[str, DatasetContext, AnnotatedDataset[Any]]],
    ood_indices: list[int],
    max_insights: int,
) -> tuple[list[FactorDeviationDict] | None, dict[str, float] | None]:
    """Compute factor_deviation and factor_predictors for OOD samples."""
    if not ood_indices:
        return None, None

    logger.info("[5/5] Computing metadata insights for %d OOD samples…", len(ood_indices))
    t0 = _time.monotonic()

    collected = _collect_numeric_factors(ref_dc, ref_dataset, test_datasets)
    if collected is None:
        return None, None
    ref_factors_common, test_factors_common = collected

    # Compute factor_deviation for top OOD samples
    capped_indices = ood_indices[:max_insights]
    deviations_list: list[FactorDeviationDict] | None = None
    try:
        raw_devs = factor_deviation(ref_factors_common, test_factors_common, capped_indices)
        deviations_list = [
            FactorDeviationDict(index=idx, deviations=dict(devs))
            for idx, devs in zip(capped_indices, raw_devs, strict=True)
        ]
    except Exception:  # noqa: BLE001
        logger.warning("factor_deviation failed", exc_info=True)

    # Compute factor_predictors across all OOD samples
    predictors: dict[str, float] | None = None
    try:
        raw_preds = factor_predictors(test_factors_common, ood_indices)
        predictors = {k: round(float(v), 4) for k, v in sorted(raw_preds.items(), key=lambda x: -x[1])}
    except Exception:  # noqa: BLE001
        logger.warning("factor_predictors failed", exc_info=True)

    logger.info("[5/5] Metadata insights complete in %.1fs", _time.monotonic() - t0)
    return deviations_list, predictors


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------


class OODDetectionWorkflow(WorkflowProtocol[OODDetectionMetadata, OODDetectionOutputs]):
    """OOD detection workflow using DataEval OOD detectors."""

    @property
    def name(self) -> str:
        """Name of the workflow, used in configs and task routing."""
        return "ood-detection"

    @property
    def description(self) -> str:
        """Description of the workflow for users."""
        return "Detect out-of-distribution samples in test data against a reference dataset"

    @property
    def params_schema(self) -> type[OODDetectionParameters]:
        """Params schema is the union of all supported OOD detector configs, plus workflow-level settings."""
        return OODDetectionParameters

    @property
    def output_schema(self) -> type[OODDetectionOutputs]:
        """Output schema includes both raw detector outputs and a user-friendly report."""
        return OODDetectionOutputs

    def execute(
        self,
        context: WorkflowContext,
        params: BaseModel | None = None,
    ) -> WorkflowResult[OODDetectionMetadata, OODDetectionOutputs]:
        """Run OOD detection workflow."""
        if not isinstance(context, WorkflowContext):
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=[f"Expected WorkflowContext, got {type(context).__name__}"],
                metadata=OODDetectionMetadata(),
            )

        if params is None:
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=["OODDetectionParameters required"],
                metadata=OODDetectionMetadata(),
            )

        if not isinstance(params, OODDetectionParameters):
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=[f"Expected OODDetectionParameters, got {type(params).__name__}"],
                metadata=OODDetectionMetadata(),
            )

        try:
            return self._run(context, params)
        except Exception as e:
            logger.exception("Workflow '%s' failed", self.name)
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=[f"Workflow execution failed: {e}"],
                metadata=OODDetectionMetadata(),
            )

    def _run(
        self,
        context: WorkflowContext,
        params: OODDetectionParameters,
    ) -> WorkflowResult[OODDetectionMetadata, OODDetectionOutputs]:
        """Core execution logic."""
        # --- 1. Validate: need 2+ datasets ---
        dc_items = list(context.dataset_contexts.items())
        if len(dc_items) < 2:
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=[
                    f"OOD detection requires at least 2 datasets (reference + test), "
                    f"got {len(dc_items)}: {[n for n, _ in dc_items]}"
                ],
                metadata=OODDetectionMetadata(),
            )

        # --- 2. Prepare datasets ---
        ref_dc, ref_dataset, test_datasets = self._prepare_datasets(dc_items)

        # --- 3. Extract embeddings ---
        ref_embeddings, test_embeddings = self._extract_all_embeddings(ref_dc, ref_dataset, test_datasets)

        # --- 4. Run OOD detectors ---
        detector_results, detector_names, detector_errors, is_ood_arrays = _run_all_ood_detectors(
            params, ref_embeddings, test_embeddings
        )

        # --- 5. Compute union OOD indices ---
        ood_indices: list[int] = []
        if is_ood_arrays:
            union_ood = np.zeros(len(test_embeddings), dtype=bool)
            for arr in is_ood_arrays:
                union_ood |= arr
            ood_indices = [int(i) for i in np.where(union_ood)[0]]

        # --- 6. Metadata insights ---
        factor_devs: list[FactorDeviationDict] | None = None
        factor_preds: dict[str, float] | None = None
        if params.metadata_insights and ood_indices:
            factor_devs, factor_preds = _compute_metadata_insights(
                ref_dc, ref_dataset, test_datasets, ood_indices, params.max_ood_insights
            )

        # --- 7. Build outputs ---
        return self._build_workflow_result(
            params,
            ref_embeddings,
            test_embeddings,
            detector_results,
            detector_names,
            detector_errors,
            ood_indices,
            factor_devs,
            factor_preds,
        )

    def _prepare_datasets(
        self,
        dc_items: list[tuple[str, DatasetContext]],
    ) -> tuple[DatasetContext, AnnotatedDataset[Any], list[tuple[str, DatasetContext, AnnotatedDataset[Any]]]]:
        """Identify reference vs test datasets and apply selections."""
        from dataeval_flow.selection import build_selection

        ref_name, ref_dc = dc_items[0]
        test_contexts = dc_items[1:]

        logger.info(
            "[1/5] Preparing datasets: reference=%s, test=%s",
            ref_name,
            [n for n, _ in test_contexts],
        )

        # Apply selection to reference
        ref_dataset: AnnotatedDataset[Any] = ref_dc.dataset
        if ref_dc.selection_steps:
            ref_dataset = build_selection(ref_dataset, ref_dc.selection_steps)  # type: ignore[arg-type]

        # Apply selection to test datasets
        test_datasets: list[tuple[str, DatasetContext, AnnotatedDataset[Any]]] = []
        for t_name, t_dc in test_contexts:
            t_ds = t_dc.dataset
            if t_dc.selection_steps:
                t_ds = build_selection(t_ds, t_dc.selection_steps)  # type: ignore[arg-type]
            test_datasets.append((t_name, t_dc, t_ds))

        return ref_dc, ref_dataset, test_datasets

    def _extract_all_embeddings(
        self,
        ref_dc: DatasetContext,
        ref_dataset: AnnotatedDataset[Any],
        test_datasets: list[tuple[str, DatasetContext, AnnotatedDataset[Any]]],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Extract embeddings for reference and test datasets."""
        logger.info("[2/5] Extracting embeddings…")
        t0 = _time.monotonic()

        ref_embeddings = _get_embeddings_for_context(ref_dc, ref_dataset)
        logger.info("  Reference embeddings: %s", ref_embeddings.shape)

        test_embedding_parts: list[NDArray[np.float32]] = []
        for t_name, t_dc, t_ds in test_datasets:
            emb = _get_embeddings_for_context(t_dc, t_ds)
            test_embedding_parts.append(emb)
            logger.info("  Test embeddings (%s): %s", t_name, emb.shape)

        test_embeddings = (
            np.concatenate(test_embedding_parts, axis=0) if len(test_embedding_parts) > 1 else test_embedding_parts[0]
        )

        logger.info(
            "[2/5] Embeddings ready in %.1fs (ref=%d, test=%d)",
            _time.monotonic() - t0,
            len(ref_embeddings),
            len(test_embeddings),
        )

        return ref_embeddings, test_embeddings

    def _build_workflow_result(
        self,
        params: OODDetectionParameters,
        ref_embeddings: NDArray[np.float32],
        test_embeddings: NDArray[np.float32],
        detector_results: dict[str, DetectorOODResultDict],
        detector_names: dict[str, str],
        detector_errors: list[str],
        ood_indices: list[int],
        factor_deviations: list[FactorDeviationDict] | None,
        factor_predictors_result: dict[str, float] | None,
    ) -> WorkflowResult[OODDetectionMetadata, OODDetectionOutputs]:
        """Build the final workflow result from raw outputs."""
        raw = OODDetectionRawOutputs(
            dataset_size=len(ref_embeddings) + len(test_embeddings),
            reference_size=len(ref_embeddings),
            test_size=len(test_embeddings),
            detectors=detector_results,
            ood_indices=ood_indices,
            factor_deviations=factor_deviations,
            factor_predictors=factor_predictors_result,
        )

        findings = build_findings(raw, params, detector_names)

        summary = f"OOD detection complete. Reference: {raw.reference_size} items, Test: {raw.test_size} items."

        report = OODDetectionReport(summary=summary, findings=findings)

        result_metadata = OODDetectionMetadata(
            mode=params.mode,
            detectors_used=list(detector_results.keys()),
            metadata_insights_enabled=params.metadata_insights and bool(ood_indices),
        )

        return WorkflowResult(
            name=self.name,
            success=True,
            data=OODDetectionOutputs(raw=raw, report=report),
            metadata=result_metadata,
            errors=detector_errors if detector_errors else [],
        )

    def _empty_outputs(self) -> OODDetectionOutputs:
        return OODDetectionOutputs(
            raw=OODDetectionRawOutputs(dataset_size=0),
            report=OODDetectionReport(summary="Workflow failed", findings=[]),
        )
