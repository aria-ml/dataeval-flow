"""Task orchestration — config → execution bridge."""

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from dataeval_app.config.models import WorkflowConfig
from dataeval_app.config.schemas.task import TaskConfig

# TYPE_CHECKING-only: workflow.__init__ imports from this module at line-level,
# so a regular import would create a circular import.  Quoted annotations in
# run_task / _write_output keep full type-checker coverage with no runtime cost.
if TYPE_CHECKING:
    from dataeval_app.workflow import WorkflowResult

__all__ = ["run_task"]


@runtime_checkable
class _Named(Protocol):
    """Protocol for config objects with a name attribute."""

    name: str


T = TypeVar("T", bound=_Named)


def _resolve_by_name(items: list[T] | None, name: str, kind: str) -> T:
    """Find a config object by name.

    Parameters
    ----------
    items : list[T] | None
        List of config objects with a ``name`` attribute.
    name : str
        Name to look up.
    kind : str
        Human-readable kind for error messages (e.g. "dataset").

    Raises
    ------
    ValueError
        If *items* is ``None`` or *name* is not found.
    """
    if items is None:
        raise ValueError(f"No {kind} configs defined, cannot resolve '{name}'")
    for item in items:
        if item.name == name:
            return item
    available = [item.name for item in items]
    raise ValueError(f"Unknown {kind}: '{name}'. Available: {available}")


def run_task(task: TaskConfig, config: WorkflowConfig, output_dir: Path | None = None) -> "WorkflowResult":
    """Run a single task using config-driven resolution.

    This is the primary entry point for config-driven execution.
    Resolves all ``TaskConfig`` references (dataset, model, preprocessor,
    selection) against ``WorkflowConfig`` lists, builds a
    ``WorkflowContext``, validates params, and runs the workflow.

    Parameters
    ----------
    task : TaskConfig
        Task to execute. References are resolved against *config*.
    config : WorkflowConfig
        Full workflow configuration containing datasets, models, etc.
    output_dir : Path | None
        Root output directory for results. Defaults to ``/output``.

    Returns
    -------
    WorkflowResult
        Result from the workflow execution.

    Raises
    ------
    ValueError
        If a referenced config (dataset, model, etc.) is not found.
    """
    from dataeval_app.config.models import ModelConfig
    from dataeval_app.config.schemas import (
        DatasetConfig,
        PreprocessorConfig,
        SelectionConfig,
    )
    from dataeval_app.dataset import load_dataset
    from dataeval_app.preprocessing import build_preprocessing
    from dataeval_app.workflow import WorkflowContext, get_workflow

    # 1. Resolve dataset
    ds_config: DatasetConfig = _resolve_by_name(config.datasets, task.dataset, "dataset")
    # Future: use ds_config.format to select loader (currently only HuggingFace)
    # Future: use ds_config.splits to select split (currently returns first split)
    dataset = load_dataset(Path(ds_config.path))

    # 2. Resolve preprocessor (optional)
    transforms = None
    if task.preprocessor:
        pre_config: PreprocessorConfig = _resolve_by_name(config.preprocessors, task.preprocessor, "preprocessor")
        transforms = build_preprocessing(pre_config.steps)

    # 3. Resolve model (optional)
    extractor_config = None
    if task.model:
        model_config: ModelConfig = _resolve_by_name(config.models, task.model, "model")
        extractor_config = model_config.extractor

    # 4. Resolve selection (optional)
    selection_steps = None
    if task.selection:
        sel_config: SelectionConfig = _resolve_by_name(config.selections, task.selection, "selection")
        selection_steps = sel_config.steps

    # 5. Build WorkflowContext
    context = WorkflowContext(
        dataset=dataset,
        extractor=extractor_config,
        transforms=transforms,
        selection_steps=selection_steps,
        metadata_auto_bin_method=ds_config.metadata_auto_bin_method,
        metadata_exclude=ds_config.metadata_exclude or [],
        metadata_continuous_factor_bins=ds_config.metadata_continuous_factor_bins,
    )

    # 6. Validate params against workflow's params_schema
    workflow = get_workflow(task.workflow)
    params = None
    if workflow.params_schema is not None:
        params = workflow.params_schema.model_validate(task.params)

    # 7. Run workflow
    result = workflow.execute(context, params)

    # 8. Serialize output based on task.output_format (skip on failure)
    if result.success:
        _write_output(result, task, output_dir=output_dir)

    return result


def _write_output(result: "WorkflowResult", task: TaskConfig, output_dir: Path | None = None) -> None:
    """Serialize WorkflowResult based on task.output_format.

    Writes to *output_dir*/<task.name>/ if the output directory exists,
    otherwise prints to terminal.

    Parameters
    ----------
    result : WorkflowResult
        Successful workflow result to serialize.
    task : TaskConfig
        Task configuration (name, output_format, dataset).
    output_dir : Path | None
        Root output directory.  Defaults to ``/output``.
    """
    import json

    import yaml

    from dataeval_app._jatic_metadata import write_metadata

    if output_dir is None:
        output_dir = Path("/output")
    task_dir = output_dir / task.name

    if task.output_format == "terminal" or not output_dir.exists():
        # Print report to terminal
        if hasattr(result.data, "report"):
            report = result.data.report  # type: ignore[attr-defined]  # guarded by hasattr
            print(f"\n{'=' * 60}")
            print(f"  {task.name}: {report.summary}")
            print(f"{'=' * 60}")
            for finding in report.findings:
                print(f"\n  [{finding.title}]")
                if finding.description:
                    print(f"  {finding.description}")
        return

    task_dir.mkdir(parents=True, exist_ok=True)
    data_dict = result.data.model_dump()

    if task.output_format == "json":
        results_file = task_dir / "results.json"
        results_file.write_text(json.dumps(data_dict, indent=2), encoding="utf-8")
    elif task.output_format == "yaml":
        results_file = task_dir / "results.yaml"
        results_file.write_text(yaml.dump(data_dict, default_flow_style=False), encoding="utf-8")

    # Write JATIC metadata.json alongside results
    write_metadata(task_dir, task.dataset, data_dict)
