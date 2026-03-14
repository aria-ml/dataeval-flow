"""Drift monitoring workflow."""

from __future__ import annotations

import contextlib
import logging
import time as _time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import polars as pl
from dataeval.protocols import AnnotatedDataset
from dataeval.shift import (
    DriftDomainClassifier,
    DriftKNeighbors,
    DriftMMD,
    DriftOutput,
    DriftUnivariate,
)
from numpy.typing import NDArray
from pydantic import BaseModel

from dataeval_flow.cache import active_cache, get_or_compute_embeddings, selection_repr
from dataeval_flow.workflow import DatasetContext, WorkflowContext, WorkflowProtocol, WorkflowResult
from dataeval_flow.workflow.base import Reportable
from dataeval_flow.workflows.drift.outputs import (
    ChunkResultDict,
    ClasswiseDriftDict,
    ClasswiseDriftRowDict,
    DetectorResultDict,
    DriftMonitoringMetadata,
    DriftMonitoringOutputs,
    DriftMonitoringRawOutputs,
    DriftMonitoringReport,
)
from dataeval_flow.workflows.drift.params import (
    DriftDetectorConfig,
    DriftDetectorDomainClassifier,
    DriftDetectorKNeighbors,
    DriftDetectorMMD,
    DriftDetectorUnivariate,
    DriftHealthThresholds,
    DriftMonitoringParameters,
)

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detector factory
# ---------------------------------------------------------------------------

# Type alias for the union of supported drift detector instances
_DriftDetector = DriftUnivariate | DriftMMD | DriftDomainClassifier | DriftKNeighbors


def _build_detector(config: DriftDetectorConfig) -> _DriftDetector:  # type: ignore[type-arg]
    """Instantiate a drift detector from its discriminated config."""
    if isinstance(config, DriftDetectorUnivariate):
        return DriftUnivariate(
            method=config.test,
            p_val=config.p_val,
            correction=config.correction,
            alternative=config.alternative,
            n_features=config.n_features,
        )
    if isinstance(config, DriftDetectorMMD):
        return DriftMMD(
            p_val=config.p_val,
            n_permutations=config.n_permutations,
            device=config.device,
        )
    if isinstance(config, DriftDetectorDomainClassifier):
        return DriftDomainClassifier(
            n_folds=config.n_folds,
            threshold=config.threshold,
        )
    if isinstance(config, DriftDetectorKNeighbors):
        return DriftKNeighbors(
            k=config.k,
            distance_metric=config.distance_metric,
            p_val=config.p_val,
        )
    raise ValueError(f"Unknown detector config type: {type(config).__name__}")


def _detector_display_name(config: DriftDetectorConfig) -> str:  # type: ignore[type-arg]
    """Human-readable name for a detector config, including non-default parameters."""
    names: dict[str, str] = {
        "univariate": "Univariate",
        "mmd": "MMD",
        "domain_classifier": "Domain Classifier",
        "kneighbors": "K-Neighbors",
    }
    base = names.get(config.method, config.method)
    if isinstance(config, DriftDetectorUnivariate):
        base = f"{config.test.upper()} {base}"

    # Collect non-default, non-internal parameters as a compact suffix
    parts: list[str] = []
    defaults = {name: field.default for name, field in config.model_fields.items()}
    for name in config.model_fields:
        if name in ("method", "chunking"):
            continue
        # 'test' is already in the base name for Univariate
        if name == "test" and isinstance(config, DriftDetectorUnivariate):
            continue
        value = getattr(config, name)
        if value != defaults[name]:
            parts.append(f"{name}={value}")

    # Add chunking params that differ from ChunkingConfig defaults
    if config.chunking is not None:
        chunking_defaults = {n: f.default for n, f in config.chunking.model_fields.items()}
        for name in config.chunking.model_fields:
            value = getattr(config.chunking, name)
            if value != chunking_defaults[name]:
                label = "z" if name == "threshold_multiplier" else name
                parts.append(f"{label}={value}")

    if parts:
        base = f"{base} ({', '.join(parts)})"
    return base


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


def _serialize_result(output: DriftOutput[Any], config: DriftDetectorConfig) -> DetectorResultDict:  # type: ignore[type-arg]
    """Convert a non-chunked DriftOutput to a serializable dict."""
    output_details = output.details if isinstance(output.details, dict) else {}
    details: dict[str, Any] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in output_details.items()}
    return DetectorResultDict(
        method=config.method,
        drifted=output.drifted,
        distance=float(output.distance),
        threshold=float(output.threshold),
        metric_name=output.metric_name,
        details=details,
    )


def _serialize_chunked_result(output: DriftOutput[pl.DataFrame], config: DriftDetectorConfig) -> DetectorResultDict:  # type: ignore[type-arg]
    """Convert a chunked DriftOutput (with polars DataFrame details) to a serializable dict."""
    output_details = output.details if isinstance(output.details, pl.DataFrame) else pl.DataFrame()
    chunks: list[ChunkResultDict] = [
        ChunkResultDict(
            key=row["key"],
            index=row["index"],
            start_index=row["start_index"],
            end_index=row["end_index"],
            value=float(row["value"]),
            upper_threshold=row.get("upper_threshold"),
            lower_threshold=row.get("lower_threshold"),
            drifted=row["drifted"],
        )
        for row in output_details.iter_rows(named=True)
    ]
    return DetectorResultDict(
        method=config.method,
        drifted=output.drifted,
        distance=float(output.distance),
        threshold=float(output.threshold),
        metric_name=output.metric_name,
        details={},
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------


def _extract_labels(dataset: AnnotatedDataset[Any]) -> NDArray[np.intp] | None:
    """Extract integer class labels from dataset targets.

    Returns None if the dataset has no usable labels.
    """
    try:
        n = len(dataset)
        labels: list[int] = []
        for i in range(n):
            _, target, _ = dataset[i]
            t = np.asarray(target)
            if t.ndim == 0:
                labels.append(int(t))
            elif t.ndim == 1 and t.size > 1:
                # One-hot encoded — take argmax
                labels.append(int(np.argmax(t)))
            elif t.ndim == 1 and t.size == 1:
                labels.append(int(t[0]))
            else:
                return None
        return np.array(labels, dtype=np.intp)
    except Exception:  # noqa: BLE001
        logger.debug("Could not extract labels for classwise drift", exc_info=True)
        return None


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
            "Drift monitoring requires a model/extractor to compute embeddings. Configure 'models' in the task config."
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
# Findings builders
# ---------------------------------------------------------------------------


def _severity_for_detector(
    drifted: bool,
    thresholds: DriftHealthThresholds,
) -> Literal["ok", "info", "warning"]:
    """Determine severity for a non-chunked detector result."""
    if drifted and thresholds.any_drift_is_warning:
        return "warning"
    return "info" if drifted else "ok"


def _severity_for_chunks(
    chunks: list[ChunkResultDict],
    thresholds: DriftHealthThresholds,
) -> Literal["ok", "info", "warning"]:
    """Determine severity for chunked results."""
    if not chunks:
        return "ok"
    n_drifted = sum(1 for c in chunks if c["drifted"])
    pct = 100.0 * n_drifted / len(chunks) if chunks else 0.0

    # Check consecutive drift window
    max_consecutive = _max_consecutive_drifted(chunks)

    if pct >= thresholds.chunk_drift_pct_warning:
        return "warning"
    if max_consecutive >= thresholds.consecutive_chunks_warning:
        return "warning"
    return "info" if n_drifted > 0 else "ok"


def _max_consecutive_drifted(chunks: list[ChunkResultDict]) -> int:
    """Count the longest run of consecutive drifted chunks."""
    max_run = 0
    current_run = 0
    for c in chunks:
        if c["drifted"]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _build_detector_finding(
    name: str,
    result: DetectorResultDict,
    thresholds: DriftHealthThresholds,
    classwise_rows: list[ClasswiseDriftRowDict] | None = None,
) -> Reportable:
    """Build a finding for a single detector (non-chunked)."""
    drifted = result["drifted"]
    severity = _severity_for_detector(drifted, thresholds)
    data: dict[str, Any] = {
        "distance": round(result["distance"], 4),
        "threshold": round(result["threshold"], 4),
        "metric": result["metric_name"],
    }

    # Add p_val from details if available
    details = result.get("details", {})
    if isinstance(details, dict) and "p_val" in details:
        data["p_val"] = round(float(details["p_val"]), 6)

    # Univariate: summarize feature drift
    if isinstance(details, dict) and "feature_drift" in details:
        fd = details["feature_drift"]
        if isinstance(fd, list):
            n_drifted = sum(fd)
            n_total = len(fd)
        else:
            n_drifted = int(np.sum(fd))
            n_total = len(fd)
        data["features_drifted"] = f"{n_drifted} / {n_total}"

    description = f"{name}: distance={data['distance']}, threshold={data['threshold']}"
    if "p_val" in data:
        description += f", p={data['p_val']}"

    # Classwise breakdown → render as a classwise_table instead of key_value
    if classwise_rows:
        drifted_classes = [r["class_name"] for r in classwise_rows if r["drifted"]]
        n_cls_drifted = len(drifted_classes)
        n_total = len(classwise_rows)
        description = f"Classes drifted: {', '.join(drifted_classes)}" if drifted_classes else "No classes drifted"
        if n_cls_drifted > 0 and thresholds.classwise_any_drift_is_warning:
            severity = "warning"

        table_rows = [
            {
                "Class": r["class_name"],
                "Distance": round(r["distance"], 4),
                "PVal": round(r["p_val"], 6) if r["p_val"] is not None else None,
                "Status": "DRIFT" if r["drifted"] else "ok",
            }
            for r in classwise_rows
        ]

        data["table_rows"] = table_rows
        data["brief"] = f"{n_cls_drifted}/{n_total} classes drifted"

        return Reportable(
            report_type="classwise_table",
            severity=severity,
            title=name,
            data=data,
            description=description,
        )

    return Reportable(
        report_type="key_value",
        severity=severity,
        title=name,
        data=data,
        description=description,
    )


def _build_chunked_finding(
    name: str,
    result: DetectorResultDict,
    thresholds: DriftHealthThresholds,
) -> Reportable:
    """Build a table finding for chunked detector results."""
    chunks = result.get("chunks", [])
    if not chunks:
        return _build_detector_finding(name, result, thresholds)

    severity = _severity_for_chunks(chunks, thresholds)
    n_drifted = sum(1 for c in chunks if c["drifted"])
    pct = 100.0 * n_drifted / len(chunks) if chunks else 0.0
    max_consec = _max_consecutive_drifted(chunks)

    rows: list[dict[str, Any]] = [
        {
            "Chunk": c["key"],
            "Distance": round(c["value"], 4),
            "UpperThreshold": round(c["upper_threshold"], 4) if c["upper_threshold"] is not None else None,
            "LowerThreshold": round(c["lower_threshold"], 4) if c["lower_threshold"] is not None else None,
            "Status": "DRIFT" if c["drifted"] else "ok",
        }
        for c in chunks
    ]

    description = f"{n_drifted}/{len(chunks)} chunks drifted ({pct:.0f}%) | max consecutive: {max_consec}"

    return Reportable(
        report_type="chunk_table",
        severity=severity,
        title=name,
        data={
            "table_rows": rows,
            "drift_flags": [c["drifted"] for c in chunks],
        },
        description=description,
    )


def _build_findings(
    raw: DriftMonitoringRawOutputs,
    params: DriftMonitoringParameters,
    detector_names: dict[str, str],
) -> list[Reportable]:
    """Build all report findings from raw results."""
    findings: list[Reportable] = []

    # Index classwise results by detector display name for per-detector lookup
    classwise_by_detector: dict[str, list[ClasswiseDriftRowDict]] = {}
    if raw.classwise:
        for cw in raw.classwise:
            classwise_by_detector[cw["detector"]] = cw["rows"]

    for method_key, result in raw.detectors.items():
        name = detector_names.get(method_key, method_key)
        cw_rows = classwise_by_detector.get(name)
        if result.get("chunks"):
            findings.append(_build_chunked_finding(name, result, params.health_thresholds))
        else:
            findings.append(_build_detector_finding(name, result, params.health_thresholds, cw_rows))

    return findings


# ---------------------------------------------------------------------------
# Classwise drift detection
# ---------------------------------------------------------------------------


def _any_classwise(detectors: Sequence[DriftDetectorConfig]) -> bool:  # type: ignore[type-arg]
    """Return True if any detector has classwise enabled."""
    return any(d.classwise for d in detectors)


def _run_classwise_drift(
    ref_embeddings: NDArray[np.float32],
    test_embeddings: NDArray[np.float32],
    ref_labels: NDArray[np.intp],
    test_labels: NDArray[np.intp],
    params: DriftMonitoringParameters,
    detector_names: dict[str, str],
) -> list[ClasswiseDriftDict]:
    """Run drift detection per class (only for detectors with classwise=True)."""
    unique_classes = np.unique(np.concatenate([ref_labels, test_labels]))
    results: list[ClasswiseDriftDict] = []

    method_keys = _unique_method_keys(params.detectors)

    for det_config, method_key in zip(params.detectors, method_keys, strict=True):
        if not det_config.classwise:
            continue

        name = detector_names.get(method_key, method_key)
        rows: list[ClasswiseDriftRowDict] = []

        for cls in unique_classes:
            ref_mask = ref_labels == cls
            test_mask = test_labels == cls
            ref_cls = ref_embeddings[ref_mask]
            test_cls = test_embeddings[test_mask]

            if len(ref_cls) < 2 or len(test_cls) < 2:
                logger.debug(
                    "Skipping class %s for %s: too few samples (ref=%d, test=%d)",
                    cls,
                    name,
                    len(ref_cls),
                    len(test_cls),
                )
                continue

            try:
                detector = _build_detector(det_config)
                detector.fit(ref_cls)
                output = detector.predict(test_cls)

                p_val: float | None = None
                if isinstance(output.details, dict):
                    p_val = output.details.get("p_val")

                rows.append(
                    ClasswiseDriftRowDict(
                        class_name=str(int(cls)),
                        drifted=output.drifted,
                        distance=float(output.distance),
                        p_val=float(p_val) if p_val is not None else None,
                    )
                )
            except Exception:  # noqa: BLE001
                logger.warning("Classwise detector %s failed for class %s", name, cls, exc_info=True)
                continue

        results.append(ClasswiseDriftDict(detector=name, rows=rows))

    return results


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------


def _unique_method_keys(
    detectors: Sequence[DriftDetectorConfig],  # type: ignore[type-arg]
) -> list[str]:
    """Return a unique key for each detector, appending a numeric suffix for duplicates."""
    counts: dict[str, int] = {}
    keys: list[str] = []
    for det in detectors:
        base = det.method
        counts[base] = counts.get(base, 0) + 1
    # Second pass: assign suffixes only when a method appears more than once
    seen: dict[str, int] = {}
    for det in detectors:
        base = det.method
        if counts[base] == 1:
            keys.append(base)
        else:
            idx = seen.get(base, 0) + 1
            seen[base] = idx
            keys.append(f"{base}_{idx}")
    return keys


def _run_all_detectors(
    params: DriftMonitoringParameters,
    ref_embeddings: NDArray[np.float32],
    test_embeddings: NDArray[np.float32],
) -> tuple[dict[str, DetectorResultDict], dict[str, str], list[str]]:
    """Run all configured drift detectors and return results, names, and errors."""
    logger.info("[3/4] Running %d drift detector(s)…", len(params.detectors))
    t0 = _time.monotonic()

    detector_results: dict[str, DetectorResultDict] = {}
    detector_names: dict[str, str] = {}
    detector_errors: list[str] = []
    method_keys = _unique_method_keys(params.detectors)

    for det_config, method_key in zip(params.detectors, method_keys, strict=True):
        display = _detector_display_name(det_config)
        detector_names[method_key] = display

        try:
            detector = _build_detector(det_config)

            if det_config.chunking is not None:
                from dataeval.utils.thresholds import ZScoreThreshold

                chunk_threshold = ZScoreThreshold(multiplier=det_config.chunking.threshold_multiplier)
                chunked = detector.chunked(
                    chunk_size=det_config.chunking.chunk_size,
                    chunk_count=det_config.chunking.chunk_count,
                    threshold=chunk_threshold,
                )
                chunked.fit(ref_embeddings)
                output = chunked.predict(test_embeddings)
                detector_results[method_key] = _serialize_chunked_result(output, det_config)
            else:
                detector.fit(ref_embeddings)
                output = detector.predict(test_embeddings)
                detector_results[method_key] = _serialize_result(output, det_config)

            status = "DRIFT" if output.drifted else "ok"
            logger.info("  %s: %s (distance=%.4f, threshold=%.4f)", display, status, output.distance, output.threshold)
        except Exception as e:  # noqa: BLE001
            logger.warning("Detector %s failed: %s", display, e, exc_info=True)
            detector_errors.append(f"{display}: {e}")

    logger.info("[3/4] Detection complete in %.1fs", _time.monotonic() - t0)
    return detector_results, detector_names, detector_errors


def _handle_classwise(
    params: DriftMonitoringParameters,
    ref_embeddings: NDArray[np.float32],
    test_embeddings: NDArray[np.float32],
    ref_labels: NDArray[np.intp] | None,
    test_label_parts: list[NDArray[np.intp]],
    detector_names: dict[str, str],
) -> list[ClasswiseDriftDict] | None:
    """Run classwise drift detection if enabled and labels are available."""
    if not _any_classwise(params.detectors):
        logger.info("[4/4] Classwise drift not enabled — skipping.")
        return None

    if ref_labels is None or not test_label_parts:
        logger.warning("Classwise drift requested but labels not available — skipping.")
        return None

    test_labels = np.concatenate(test_label_parts) if len(test_label_parts) > 1 else test_label_parts[0]
    logger.info("[4/4] Running classwise drift detection…")
    t0 = _time.monotonic()
    results = _run_classwise_drift(
        ref_embeddings,
        test_embeddings,
        ref_labels,
        test_labels,
        params,
        detector_names,
    )
    logger.info("[4/4] Classwise detection complete in %.1fs", _time.monotonic() - t0)
    return results


class DriftMonitoringWorkflow(WorkflowProtocol[DriftMonitoringMetadata, DriftMonitoringOutputs]):
    """Drift monitoring workflow using DataEval shift detectors."""

    @property
    def name(self) -> str:
        """Name of the workflow, used in configs and task routing."""
        return "drift-monitoring"

    @property
    def description(self) -> str:
        """Description of the workflow for users."""
        return "Monitor incoming data for distribution drift against a reference dataset"

    @property
    def params_schema(self) -> type[DriftMonitoringParameters]:
        """Params schema is the union of all supported detector configs, plus workflow-level settings."""
        return DriftMonitoringParameters

    @property
    def output_schema(self) -> type[DriftMonitoringOutputs]:
        """Output schema includes both raw detector outputs and a user-friendly report."""
        return DriftMonitoringOutputs

    def execute(
        self,
        context: WorkflowContext,
        params: BaseModel | None = None,
    ) -> WorkflowResult[DriftMonitoringMetadata, DriftMonitoringOutputs]:
        """Run drift monitoring workflow."""
        if not isinstance(context, WorkflowContext):
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=[f"Expected WorkflowContext, got {type(context).__name__}"],
                metadata=DriftMonitoringMetadata(),
            )

        if params is None:
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=["DriftMonitoringParameters required"],
                metadata=DriftMonitoringMetadata(),
            )

        if not isinstance(params, DriftMonitoringParameters):
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=[f"Expected DriftMonitoringParameters, got {type(params).__name__}"],
                metadata=DriftMonitoringMetadata(),
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
                metadata=DriftMonitoringMetadata(),
            )

    def _run(
        self,
        context: WorkflowContext,
        params: DriftMonitoringParameters,
    ) -> WorkflowResult[DriftMonitoringMetadata, DriftMonitoringOutputs]:
        """Core execution logic."""
        # --- 1. Validate: need 2+ datasets ---
        dc_items = list(context.dataset_contexts.items())
        if len(dc_items) < 2:
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=[
                    f"Drift monitoring requires at least 2 datasets (reference + test), "
                    f"got {len(dc_items)}: {[n for n, _ in dc_items]}"
                ],
                metadata=DriftMonitoringMetadata(),
            )

        # Log stubbed update strategy
        if params.update_strategy is not None:
            logger.warning(
                "update_strategy is configured but not yet applied at runtime. "
                "This setting is accepted for forward compatibility."
            )

        # --- 2. Prepare datasets ---
        ref_dc, ref_dataset, test_datasets = self._prepare_datasets(dc_items)

        # --- 3. Extract embeddings ---
        ref_embeddings, test_embeddings, ref_labels, test_label_parts = self._extract_all_embeddings(
            ref_dc, ref_dataset, test_datasets, params
        )

        # --- 4. Run detectors ---
        detector_results, detector_names, detector_errors = _run_all_detectors(params, ref_embeddings, test_embeddings)

        # --- 5. Classwise drift ---
        classwise_results = _handle_classwise(
            params, ref_embeddings, test_embeddings, ref_labels, test_label_parts, detector_names
        )

        # --- 6. Build outputs ---
        return self._build_workflow_result(
            params,
            ref_embeddings,
            test_embeddings,
            detector_results,
            detector_names,
            detector_errors,
            classwise_results,
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
            "[1/4] Preparing datasets: reference=%s, test=%s",
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
        params: DriftMonitoringParameters,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.intp] | None, list[NDArray[np.intp]]]:
        """Extract embeddings for reference and test datasets."""
        logger.info("[2/4] Extracting embeddings…")
        t0 = _time.monotonic()

        ref_embeddings = _get_embeddings_for_context(ref_dc, ref_dataset)
        logger.info("  Reference embeddings: %s", ref_embeddings.shape)

        test_embedding_parts: list[NDArray[np.float32]] = []
        test_label_parts: list[NDArray[np.intp]] = []

        for t_name, t_dc, t_ds in test_datasets:
            emb = _get_embeddings_for_context(t_dc, t_ds)
            test_embedding_parts.append(emb)
            logger.info("  Test embeddings (%s): %s", t_name, emb.shape)

            if _any_classwise(params.detectors):
                t_labels = _extract_labels(t_ds)
                if t_labels is not None:
                    test_label_parts.append(t_labels)

        test_embeddings = (
            np.concatenate(test_embedding_parts, axis=0) if len(test_embedding_parts) > 1 else test_embedding_parts[0]
        )

        ref_labels = _extract_labels(ref_dataset) if _any_classwise(params.detectors) else None

        logger.info(
            "[2/4] Embeddings ready in %.1fs (ref=%d, test=%d)",
            _time.monotonic() - t0,
            len(ref_embeddings),
            len(test_embeddings),
        )

        return ref_embeddings, test_embeddings, ref_labels, test_label_parts

    def _build_workflow_result(
        self,
        params: DriftMonitoringParameters,
        ref_embeddings: NDArray[np.float32],
        test_embeddings: NDArray[np.float32],
        detector_results: dict[str, DetectorResultDict],
        detector_names: dict[str, str],
        detector_errors: list[str],
        classwise_results: list[ClasswiseDriftDict] | None,
    ) -> WorkflowResult[DriftMonitoringMetadata, DriftMonitoringOutputs]:
        """Build the final workflow result from raw outputs."""
        raw = DriftMonitoringRawOutputs(
            dataset_size=len(ref_embeddings) + len(test_embeddings),
            reference_size=len(ref_embeddings),
            test_size=len(test_embeddings),
            detectors=detector_results,
            classwise=classwise_results,
        )

        findings = _build_findings(raw, params, detector_names)

        summary = f"Drift monitoring complete. Reference: {raw.reference_size} items, Test: {raw.test_size} items."

        report = DriftMonitoringReport(summary=summary, findings=findings)

        result_metadata = DriftMonitoringMetadata(
            mode=params.mode,
            detectors_used=list(detector_results.keys()),
            chunking_enabled=any(d.chunking is not None for d in params.detectors),
            classwise_enabled=_any_classwise(params.detectors),
        )

        return WorkflowResult(
            name=self.name,
            success=True,
            data=DriftMonitoringOutputs(raw=raw, report=report),
            metadata=result_metadata,
            errors=detector_errors if detector_errors else [],
        )

    def _empty_outputs(self) -> DriftMonitoringOutputs:
        return DriftMonitoringOutputs(
            raw=DriftMonitoringRawOutputs(dataset_size=0),
            report=DriftMonitoringReport(summary="Workflow failed", findings=[]),
        )
