"""The extension contract: identity checked at definition, types bound from the class statement."""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import pytest
from dataeval.quality import Duplicates
from pydantic import ValidationError

from dataeval_flow import InputKind, InputSpec, SourceCount, run, run_task
from dataeval_flow._orchestrator import _run_target
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators import Evaluator, EvaluatorConfig, EvaluatorInputs, EvaluatorResult
from dataeval_flow.evaluators.quality import DuplicatesConfig, DuplicatesEvaluator, DuplicatesResult
from dataeval_flow.workflows import Workflow, WorkflowConfig, WorkflowContext, WorkflowResult
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig, DataCleaningResult, DataCleaningWorkflow
from dataeval_flow.workflows.data_splitting import DataSplittingConfig, DataSplittingResult
from tests.evaluator_toys import ToyImages, toy_pipeline
from tests.example_plugin import CountConfig


def test_a_concrete_workflow_must_declare_its_identity() -> None:
    with pytest.raises(TypeError, match="must declare name, description"):

        class Nameless(Workflow[DataCleaningConfig, DataCleaningResult]):
            def run(self, config: DataCleaningConfig, context: WorkflowContext) -> DataCleaningResult:
                raise NotImplementedError


def test_an_abstract_intermediate_base_is_exempt() -> None:
    class SharedBase(Workflow[DataCleaningConfig, DataCleaningResult]):
        """A plugin's own base: it defines no `run`, so it stays abstract."""

    assert SharedBase.config_type is DataCleaningConfig


def test_an_unparameterized_concrete_workflow_is_rejected() -> None:
    with pytest.raises(TypeError, match="config_type"):

        class Bare(Workflow):  # type: ignore[type-arg]
            name: ClassVar[str] = "bare"
            description: ClassVar[str] = "No type arguments."

            def run(self, config: Any, context: WorkflowContext) -> Any:
                raise NotImplementedError


def test_config_type_comes_from_the_type_arguments() -> None:
    assert DataCleaningWorkflow.config_type is DataCleaningConfig
    assert DuplicatesEvaluator.config_type is DuplicatesConfig


def test_a_config_knows_its_result() -> None:
    assert DataCleaningConfig.result_type is DataCleaningResult
    assert DuplicatesConfig.result_type is DuplicatesResult


def test_name_defaults_to_the_type() -> None:
    assert DataSplittingConfig().name == "data-splitting"
    assert DataSplittingConfig(name="split_a").name == "split_a"


class _Untyped(WorkflowConfig[DataCleaningResult]):
    """A config whose class gives ``type`` no default."""


def test_name_defaults_to_the_type_the_entry_gives() -> None:
    assert _Untyped(type="x").name == "x"


def test_a_config_refuses_another_type() -> None:
    with pytest.raises(ValidationError, match="DataSplittingConfig configures type 'data-splitting', not 'x'"):
        DataSplittingConfig(type="x")


def test_a_config_schema_is_titled_by_its_class() -> None:
    assert DataCleaningConfig.model_json_schema()["title"] == "DataCleaningConfig"


def test_a_config_schema_describes_its_type() -> None:
    type_schema = DataCleaningConfig.model_json_schema()["properties"]["type"]
    assert type_schema["const"] == "data-cleaning"
    assert type_schema["description"] == "The workflow type this entry configures: `data-cleaning`."


def test_a_config_that_describes_no_type_gets_the_base_description() -> None:
    type_schema = CountConfig.model_json_schema()["properties"]["type"]
    assert type_schema["const"] == "example.count"
    assert type_schema["description"] == "The workflow or evaluator type this entry configures."


def test_an_evaluator_must_declare_what_it_wraps() -> None:
    """The error names only what is missing, and only the step that supplies it."""
    with pytest.raises(
        TypeError, match=r"^Unwrapped must declare dataeval_class: set `dataeval_class` as a class attribute\.$"
    ):

        class Unwrapped(Evaluator[DuplicatesConfig, Any]):
            name: ClassVar[str] = "x.unwrapped"
            description: ClassVar[str] = "Declares no DataEval class."
            dataeval_methods: ClassVar[Mapping[InputKind, str]] = {}

            def run(self, config: DuplicatesConfig, inputs: Any) -> Any:
                raise NotImplementedError


class _UnresultedWorkflowConfig(WorkflowConfig):  # type: ignore[type-arg]
    """A workflow config whose base names no result class."""

    type: str = "x.unresulted"


class _UnresultedEvaluatorConfig(EvaluatorConfig):  # type: ignore[type-arg]
    """An evaluator config whose base names no result class."""

    type: str = "x.unresulted"


def test_a_workflow_whose_config_names_no_result_class_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"parameterize its base with one, e\.g\. .*WorkflowConfig\[MyResult\]"):

        class Unresulted(Workflow[_UnresultedWorkflowConfig, DataCleaningResult]):
            name: ClassVar[str] = "x.unresulted"
            description: ClassVar[str] = "Its config names no result class."

            def run(self, config: _UnresultedWorkflowConfig, context: WorkflowContext) -> DataCleaningResult:
                raise NotImplementedError


def test_an_evaluator_whose_config_names_no_result_class_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"parameterize its base with one, e\.g\. .*EvaluatorConfig\[MyResult\]"):

        class Unresulted(Evaluator[_UnresultedEvaluatorConfig, Any]):
            name: ClassVar[str] = "x.unresulted"
            description: ClassVar[str] = "Its config names no result class."
            dataeval_class: ClassVar[type] = Duplicates
            dataeval_methods: ClassVar[Mapping[InputKind, str]] = {}

            def run(self, config: _UnresultedEvaluatorConfig, inputs: Sequence[EvaluatorInputs]) -> Any:
                raise NotImplementedError


class _Forgetful(Workflow[DataCleaningConfig, DataCleaningResult]):
    """Returns nothing, as a run that forgets its ``return`` does."""

    name: ClassVar[str] = "x.forgetful"
    description: ClassVar[str] = "Returns None."

    def run(self, config: DataCleaningConfig, context: WorkflowContext) -> DataCleaningResult:
        return None  # type: ignore[return-value]


def test_a_run_that_returns_no_result_becomes_a_failed_result() -> None:
    config = DataCleaningConfig(outlier_method="zscore", outlier_flags=["pixel"])
    result = _run_target(_Forgetful(), config, WorkflowContext())
    assert isinstance(result, DataCleaningResult)
    assert not result.success
    assert result.errors == ["x.forgetful returned NoneType, not a DataCleaningResult"]


@pytest.mark.parametrize("result", [DataSplittingResult, WorkflowResult[Any, Any]], ids=["another", "broader"])
def test_a_workflow_must_produce_the_result_its_config_names(result: Any) -> None:
    """Else `run()`, typed by the config's result class, would type a result the workflow never returns."""
    with pytest.raises(TypeError, match=r"Mismatched returns .*, but its config DataCleaningConfig names DataCleaning"):

        class Mismatched(Workflow[DataCleaningConfig, result]):
            name: ClassVar[str] = "x.mismatched"
            description: ClassVar[str] = "Names another result class than its config does."

            def run(self, config: DataCleaningConfig, context: WorkflowContext) -> Any:
                raise NotImplementedError


class _AnnotatedCleaningResult(DataCleaningResult):
    """A narrower result than ``DataCleaningConfig`` names, which ``run()``'s type still covers."""


def test_a_narrower_result_than_the_config_names_is_accepted() -> None:
    class Narrower(Workflow[DataCleaningConfig, _AnnotatedCleaningResult]):
        name: ClassVar[str] = "x.narrower"
        description: ClassVar[str] = "Returns a subclass of its config's result class."

        def run(self, config: DataCleaningConfig, context: WorkflowContext) -> _AnnotatedCleaningResult:
            raise NotImplementedError

    assert Narrower.config_type is DataCleaningConfig


class _Mislabelled(Workflow[DataCleaningConfig, DataCleaningResult]):
    """Returns another workflow's result, which its annotation cannot stop at run time."""

    name: ClassVar[str] = "x.mislabelled"
    description: ClassVar[str] = "Returns a DataSplittingResult."

    def run(self, config: DataCleaningConfig, context: WorkflowContext) -> DataCleaningResult:
        return DataSplittingResult.failed(type="data-splitting", errors=["not mine"])  # type: ignore[return-value]


def test_a_run_that_returns_another_result_class_becomes_a_failed_result() -> None:
    config = DataCleaningConfig(outlier_method="zscore", outlier_flags=["pixel"])
    result = _run_target(_Mislabelled(), config, WorkflowContext())
    assert type(result) is DataCleaningResult
    assert not result.success
    assert result.errors == ["x.mislabelled returned DataSplittingResult, not a DataCleaningResult"]


class _MetalessOutput:
    """An output that serializes but has no ``meta()``, so Flow cannot record DataEval's execution metadata."""

    def data(self) -> dict[str, int]:
        return {"items": 1}


class _MetalessResult(EvaluatorResult[Any]):
    """The result of an ``x.metaless`` run."""


class _MetalessConfig(EvaluatorConfig[_MetalessResult]):
    """Settings for ``x.metaless``."""

    type: str = "x.metaless"
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset(), sources=SourceCount.ONE)


class _Metaless(Evaluator[_MetalessConfig, Any]):
    """Returns an output whose envelope cannot be recorded."""

    name: ClassVar[str] = "x.metaless"
    description: ClassVar[str] = "Returns an output without meta()."
    dataeval_class: ClassVar[type] = Duplicates
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {}

    def run(self, config: _MetalessConfig, inputs: Sequence[EvaluatorInputs]) -> Any:
        return _MetalessOutput()


def test_an_output_whose_envelope_cannot_be_recorded_becomes_a_failed_result(
    plugins: dict[str, list[tuple[str, str]]],
) -> None:
    plugins["dataeval_flow.evaluators"] = [("x.metaless", f"{__name__}:_Metaless")]
    config = toy_pipeline(evaluators=[_MetalessConfig(name="m")])
    task = TaskConfig(name="t", workflow="m", kind="evaluator", sources="src")
    for result in (run_task(task, config), run(_MetalessConfig(), ToyImages())):
        assert isinstance(result, _MetalessResult)
        assert not result.success
        assert result.errors == ["AttributeError: '_MetalessOutput' object has no attribute 'meta'"]
        assert result.metadata.evaluator == "x.metaless"


def test_evaluator_inputs_is_the_public_name() -> None:
    assert EvaluatorInputs(source="a").source == "a"
