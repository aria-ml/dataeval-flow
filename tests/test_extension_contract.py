"""The extension contract: identity checked at definition, types bound from the class statement."""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import pytest
from dataeval.quality import Duplicates
from pydantic import ValidationError

from dataeval_flow import InputKind, InputSpec, Result, SourceCount, run, run_task
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators import Evaluator, EvaluatorConfig, EvaluatorInputs, EvaluatorResult
from dataeval_flow.evaluators.quality import DuplicatesConfig, DuplicatesEvaluator, DuplicatesResult
from dataeval_flow.steps import ChainResult, InputSlot
from dataeval_flow.workflows import Preset, PresetChain, Workflow, WorkflowConfig
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig, DataCleaningWorkflow
from dataeval_flow.workflows.data_splitting import DataSplittingConfig
from tests.evaluator_toys import ToyImages, toy_pipeline
from tests.example_plugin import CountConfig
from tests.workflow_toys import ToyCountConfig


def test_a_concrete_workflow_must_declare_its_identity() -> None:
    with pytest.raises(TypeError, match="must declare name, description"):

        class Nameless(Preset, Workflow[ToyCountConfig, ChainResult]):
            slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

            @classmethod
            def chain(cls, config: ToyCountConfig) -> PresetChain:
                raise NotImplementedError


def test_a_concrete_workflow_that_is_not_a_preset_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"every workflow type is a preset: mix in `Preset`"):

        class Opaque(Workflow[ToyCountConfig, ChainResult]):
            name: ClassVar[str] = "x.opaque"
            description: ClassVar[str] = "Mixes in no Preset."


def test_an_abstract_intermediate_base_is_exempt() -> None:
    class SharedBase(Preset, Workflow[ToyCountConfig, ChainResult]):
        """A plugin's own base: it defines no `chain`, so it stays abstract."""

    assert SharedBase.config_type is ToyCountConfig


def test_an_unparameterized_concrete_workflow_is_rejected() -> None:
    with pytest.raises(TypeError, match="config_type"):

        class Bare(Preset, Workflow):  # type: ignore[type-arg]
            name: ClassVar[str] = "bare"
            description: ClassVar[str] = "No type arguments."
            slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

            @classmethod
            def chain(cls, config: Any) -> PresetChain:
                raise NotImplementedError


def test_config_type_comes_from_the_type_arguments() -> None:
    assert DataCleaningWorkflow.config_type is DataCleaningConfig
    assert DuplicatesEvaluator.config_type is DuplicatesConfig


def test_a_config_knows_its_result() -> None:
    assert DataCleaningConfig.result_type is ChainResult
    assert DuplicatesConfig.result_type is DuplicatesResult


def test_name_defaults_to_the_type() -> None:
    assert DataSplittingConfig().name == "data-splitting"
    assert DataSplittingConfig(name="split_a").name == "split_a"


class _Untyped(WorkflowConfig[ChainResult]):
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

        class Unresulted(Preset, Workflow[_UnresultedWorkflowConfig, ChainResult]):
            name: ClassVar[str] = "x.unresulted"
            description: ClassVar[str] = "Its config names no result class."
            slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

            @classmethod
            def chain(cls, config: _UnresultedWorkflowConfig) -> PresetChain:
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


@pytest.mark.parametrize("result", [DuplicatesResult, Result[Any, Any]], ids=["another", "broader"])
def test_a_workflow_must_produce_the_result_its_config_names(result: Any) -> None:
    """Else `dataeval_flow.run`, typed by the config's result class, would type a result the workflow never returns."""
    with pytest.raises(TypeError, match=r"Mismatched returns .*, but its config ToyCountConfig names ChainResult"):

        class Mismatched(Preset, Workflow[ToyCountConfig, result]):
            name: ClassVar[str] = "x.mismatched"
            description: ClassVar[str] = "Names another result class than its config does."
            slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

            @classmethod
            def chain(cls, config: ToyCountConfig) -> PresetChain:
                raise NotImplementedError


class _AnnotatedChainResult(ChainResult):
    """A narrower result than ``ToyCountConfig`` names, which ``dataeval_flow.run``'s type still covers."""


def test_a_narrower_result_than_the_config_names_is_accepted() -> None:
    class Narrower(Preset, Workflow[ToyCountConfig, _AnnotatedChainResult]):
        name: ClassVar[str] = "x.narrower"
        description: ClassVar[str] = "Returns a subclass of its config's result class."
        slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

        @classmethod
        def chain(cls, config: ToyCountConfig) -> PresetChain:
            raise NotImplementedError

    assert Narrower.config_type is ToyCountConfig


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
    for result in (run_task(config, task), run(_MetalessConfig(), ToyImages())):
        assert isinstance(result, _MetalessResult)
        assert not result.success
        assert result.errors == ["AttributeError: '_MetalessOutput' object has no attribute 'meta'"]
        assert result.metadata.evaluator == "x.metaless"


def test_evaluator_inputs_is_the_public_name() -> None:
    assert EvaluatorInputs(source="a").source == "a"
