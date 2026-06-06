"""Parameter Sweep Workflow — efficiently sweep data cleaning parameters."""

__all__ = ["ParameterSweepWorkflow"]

import contextlib
import itertools
import logging
from typing import Any

import polars as pl
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates, Outliers
from pydantic import BaseModel

from dataeval_flow.cache import active_cache, get_or_compute_stats
from dataeval_flow.embeddings import build_extractor
from dataeval_flow.workflow import WorkflowContext, WorkflowProtocol, WorkflowResult
from dataeval_flow.workflow.base import Reportable
from dataeval_flow.workflows.cleaning._internal import (
    _compute_embeddings,
    _merge_duplicate_results,
    _merge_outlier_outputs,
)
from dataeval_flow.workflows.cleaning.workflow import CleaningRunContext
from dataeval_flow.workflows.parameter_sweep.outputs import (
    ParameterSweepMetadata,
    ParameterSweepOutputs,
    ParameterSweepRawOutputs,
    ParameterSweepReport,
    SweepRunResult,
)
from dataeval_flow.workflows.parameter_sweep.params import ParameterSweepParameters

_logger: logging.Logger = logging.getLogger(__name__)

FLAG_MAP: dict[str, ImageStats] = {
    "dimension": ImageStats.DIMENSION,
    "pixel": ImageStats.PIXEL,
    "visual": ImageStats.VISUAL,
}

HASH_FLAG_MAP: dict[str, ImageStats] = {
    "hash_basic": ImageStats.HASH_DUPLICATES_BASIC,
    "hash_d4": ImageStats.HASH_DUPLICATES_D4,
}

# Maps each outcome column to the input parameters that affect it.
# Exact duplicates depend on no swept inputs and are intentionally omitted.
OUTCOME_INPUTS: dict[str, tuple[str, ...]] = {
    "Outliers": (
        "outlier_method",
        "outlier_threshold",
        "outlier_cluster_threshold",
        "outlier_cluster_algorithm",
    ),
    "Near Duplicates": (
        "duplicate_cluster_sensitivity",
        "duplicate_cluster_algorithm",
    ),
}

OUTCOME_FIELD: dict[str, str] = {
    "Outliers": "outlier_count",
    "Near Duplicates": "near_duplicate_groups",
}


def _resolve_flags(params: ParameterSweepParameters) -> tuple[ImageStats, ImageStats]:
    """Resolve outlier and hash flags from parameters."""
    outlier_flags = ImageStats.NONE
    for name in params.outlier_flags:
        outlier_flags |= FLAG_MAP[name]

    hash_flags = ImageStats.NONE
    if params.duplicate_flags is not None:
        for name in params.duplicate_flags:
            hash_flags |= HASH_FLAG_MAP[name]
    else:
        hash_flags = ImageStats.HASH_DUPLICATES_BASIC

    return outlier_flags, hash_flags


class ParameterSweepWorkflow(WorkflowProtocol[ParameterSweepMetadata, ParameterSweepOutputs]):
    """Workflow to sweep parameters for data cleaning."""

    @property
    def name(self) -> str:
        """Workflow identifier."""
        return "parameter-sweep"

    @property
    def description(self) -> str:
        """Human-readable description."""
        return "Sweep data cleaning parameters to analyze result sensitivity"

    @property
    def params_schema(self) -> type[ParameterSweepParameters]:
        """Pydantic model for workflow parameters."""
        return ParameterSweepParameters

    @property
    def output_schema(self) -> type[ParameterSweepOutputs]:
        """Pydantic model for workflow output."""
        return ParameterSweepOutputs

    def execute(
        self,
        context: WorkflowContext,
        params: BaseModel | None = None,
    ) -> WorkflowResult[ParameterSweepMetadata, ParameterSweepOutputs]:
        """Execute the parameter sweep workflow."""
        if not isinstance(context, WorkflowContext):
            return self._fail("Expected WorkflowContext")
        if not isinstance(params, ParameterSweepParameters):
            return self._fail("ParameterSweepParameters required")

        try:
            return self._run_sweep(context, params)
        except Exception as e:
            _logger.exception("Workflow '%s' failed", self.name)
            return self._fail(f"Workflow execution failed: {e}")

    def _run_sweep(
        self, context: WorkflowContext, params: ParameterSweepParameters
    ) -> WorkflowResult[ParameterSweepMetadata, ParameterSweepOutputs]:
        from dataeval_flow.cache import selection_repr as _sel_repr
        from dataeval_flow.selection import build_selection

        # Parameter Sweep is single-dataset
        dc = next(iter(context.dataset_contexts.values()))
        dataset = dc.dataset
        if dc.selection_steps:
            dataset = build_selection(dataset, dc.selection_steps)  # type: ignore[arg-type]

        sel_key = _sel_repr(dataset)
        outlier_flags, hash_flags = _resolve_flags(params)

        # 1. Shared Setup
        extractor = None
        if dc.extractor:
            extractor = build_extractor(dc.extractor, dc.transforms)

        with contextlib.ExitStack() as stack:
            if dc.cache is not None:
                stack.enter_context(active_cache(dc.cache, sel_key))

            # Pre-compute shared stats
            calc_result = get_or_compute_stats(
                desired_flags=outlier_flags | hash_flags,
                dataset=dataset,
            )

            # Pre-compute shared embeddings if needed
            embeddings_array = None
            run_ctx = None
            needs_embeddings = any(
                p is not None
                for p in (
                    *params.outlier_cluster_threshold,
                    *params.outlier_cluster_algorithm,
                    *params.duplicate_cluster_sensitivity,
                    *params.duplicate_cluster_algorithm,
                )
            )
            if needs_embeddings and extractor is not None:
                run_ctx = CleaningRunContext(
                    extractor_config=dc.extractor,
                    transforms=dc.transforms,
                    batch_size=dc.batch_size,
                )
                embeddings_array = _compute_embeddings(dataset, extractor, run_ctx)

            # 2. Sweep over parameters
            sweep_results: list[SweepRunResult] = []

            # Determine which parameters are actually being swept (more than 1 value)
            swept_fields = [
                field
                for field in [
                    "outlier_method",
                    "outlier_threshold",
                    "outlier_cluster_threshold",
                    "outlier_cluster_algorithm",
                    "duplicate_cluster_sensitivity",
                    "duplicate_cluster_algorithm",
                ]
                if len(getattr(params, field)) > 1
            ]

            # Cartesian product of all sweep sequences
            param_combinations = list(
                itertools.product(
                    params.outlier_method,
                    params.outlier_threshold,
                    params.outlier_cluster_threshold,
                    params.outlier_cluster_algorithm,
                    params.duplicate_cluster_sensitivity,
                    params.duplicate_cluster_algorithm,
                )
            )

            _logger.info("Running sweep over %d combinations...", len(param_combinations))

            for combo in param_combinations:
                (
                    m_outlier_method,
                    m_outlier_threshold,
                    m_outlier_cluster_threshold,
                    m_outlier_cluster_algorithm,
                    m_duplicate_cluster_sensitivity,
                    m_duplicate_cluster_algorithm,
                ) = combo

                current_params = {
                    "outlier_method": m_outlier_method,
                    "outlier_threshold": m_outlier_threshold,
                    "outlier_cluster_threshold": m_outlier_cluster_threshold,
                    "outlier_cluster_algorithm": m_outlier_cluster_algorithm,
                    "duplicate_cluster_sensitivity": m_duplicate_cluster_sensitivity,
                    "duplicate_cluster_algorithm": m_duplicate_cluster_algorithm,
                }

                # Outlier detection
                outliers_eval = Outliers(
                    flags=outlier_flags,
                    outlier_threshold=(m_outlier_method, m_outlier_threshold),
                )
                outlier_output = outliers_eval.from_stats(calc_result, per_target=False)

                if m_outlier_cluster_threshold is not None and embeddings_array is not None:
                    # We need a DataCleaningParameters object because _merge_outlier_outputs expects it
                    # but it only uses a few fields. Let's shim it.
                    from dataeval_flow.workflows.cleaning.params import DataCleaningParameters

                    shim_params = DataCleaningParameters(
                        outlier_method=m_outlier_method,
                        outlier_flags=list(params.outlier_flags),
                        outlier_threshold=m_outlier_threshold,
                        outlier_cluster_threshold=m_outlier_cluster_threshold,
                        outlier_cluster_algorithm=m_outlier_cluster_algorithm,
                    )
                    outlier_output = _merge_outlier_outputs(
                        outliers_eval,
                        outlier_output,
                        embeddings_array,
                        shim_params,
                        _run_ctx=run_ctx,
                    )

                outlier_count = outlier_output.data()["item_index"].n_unique() if len(outlier_output.data()) > 0 else 0

                # Duplicate detection
                dup_kwargs: dict[str, Any] = {
                    "merge_near_duplicates": params.duplicate_merge_near,
                    "flags": hash_flags,
                }
                duplicates_eval = Duplicates(**dup_kwargs)
                duplicates_result = duplicates_eval.from_stats(calc_result)

                if m_duplicate_cluster_sensitivity is not None and embeddings_array is not None:
                    from dataeval_flow.workflows.cleaning.params import DataCleaningParameters

                    shim_params = DataCleaningParameters(
                        outlier_method=m_outlier_method,
                        outlier_flags=list(params.outlier_flags),
                        duplicate_cluster_sensitivity=m_duplicate_cluster_sensitivity,
                        duplicate_cluster_algorithm=m_duplicate_cluster_algorithm,
                    )
                    duplicates_result = _merge_duplicate_results(
                        duplicates_result,
                        embeddings_array,
                        shim_params,
                        _run_ctx=run_ctx,
                    )

                exact_groups = (
                    len(duplicates_result.data().filter(pl.col("dup_type") == "exact", pl.col("level") == "item"))
                    if len(duplicates_result.data()) > 0
                    else 0
                )
                near_groups = (
                    len(duplicates_result.data().filter(pl.col("dup_type") == "near", pl.col("level") == "item"))
                    if len(duplicates_result.data()) > 0
                    else 0
                )

                sweep_results.append(
                    SweepRunResult(
                        params=current_params,
                        outlier_count=outlier_count,
                        exact_duplicate_groups=exact_groups,
                        near_duplicate_groups=near_groups,
                    )
                )

            # 3. Assemble report
            findings = self._build_findings(sweep_results, swept_fields)
            raw = ParameterSweepRawOutputs(results=sweep_results)
            report = ParameterSweepReport(
                summary=f"Parameter sweep complete. {len(sweep_results)} combinations evaluated.",
                findings=findings,
            )
            metadata = ParameterSweepMetadata(sweep_parameters=swept_fields)

            return WorkflowResult(
                name=self.name,
                success=True,
                data=ParameterSweepOutputs(dataset_size=len(dataset), raw=raw, report=report),
                metadata=metadata,
                dataset=dataset,
            )

    def _build_findings(self, results: list[SweepRunResult], swept_fields: list[str]) -> list[Reportable]:
        findings: list[Reportable] = []

        for outcome, inputs in OUTCOME_INPUTS.items():
            relevant = [f for f in swept_fields if f in inputs]
            if not relevant:
                continue

            seen: set[tuple[Any, ...]] = set()
            rows: list[dict[str, Any]] = []
            for r in results:
                key = tuple(r.params[f] for f in relevant)
                if key in seen:
                    continue
                seen.add(key)
                row: dict[str, Any] = {f: r.params[f] for f in relevant}
                row[outcome] = getattr(r, OUTCOME_FIELD[outcome])
                rows.append(row)

            findings.append(
                Reportable(
                    report_type="pivot_table",
                    title=f"{outcome} Sweep",
                    data={
                        "brief": f"{len(rows)} unique combinations",
                        "table_data": rows,
                        "table_headers": [*relevant, outcome],
                    },
                    description=f"Effect of {', '.join(relevant)} on {outcome.lower()}.",
                )
            )

        return findings

    def _fail(self, message: str) -> WorkflowResult[ParameterSweepMetadata, ParameterSweepOutputs]:
        return WorkflowResult(
            name=self.name,
            success=False,
            data=ParameterSweepOutputs(
                dataset_size=0,
                raw=ParameterSweepRawOutputs(),
                report=ParameterSweepReport(summary="Workflow failed", findings=[]),
            ),
            metadata=ParameterSweepMetadata(),
            errors=[message],
        )
