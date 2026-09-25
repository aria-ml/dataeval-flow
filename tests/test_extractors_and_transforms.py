"""Extractors and transforms are registered kinds, like workflows and evaluators."""

import json
import re
from typing import Any, ClassVar

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from torchvision.transforms import v2

from dataeval_flow import PipelineConfig
from dataeval_flow._preprocessing import build_preprocessing
from dataeval_flow.config import PreprocessingStep
from dataeval_flow.config.extractors import (
    BoVWExtractorConfig,
    Extractor,
    FlattenExtractorConfig,
    OnnxExtractorConfig,
    get_extractor,
    list_extractors,
)
from dataeval_flow.config.transforms import ToRGB, Transform, get_transform, list_transforms
from tests.example_plugin import Invert, MeanConfig


def test_every_builtin_extractor_is_listed() -> None:
    assert [cls.name for cls in list_extractors()] == ["bovw", "flatten", "onnx", "torch", "uncertainty"]


def test_only_bovw_is_stateful() -> None:
    assert [cls.name for cls in list_extractors() if cls.stateful] == ["bovw"]


def test_flatten_builds_a_feature_extractor() -> None:
    extractor = get_extractor("flatten")().build(FlattenExtractorConfig(name="flat"), None)
    assert np.asarray(extractor([np.zeros((3, 4, 4))])).shape == (1, 48)


def test_to_rgb_is_a_registered_transform() -> None:
    import torch

    assert "ToRGB" in [cls.name for cls in list_transforms()]
    assert ToRGB()(torch.zeros((1, 2, 2))).shape[0] == 3


def test_a_step_resolves_a_registered_transform() -> None:
    """ToRGB is not a torchvision name, so only the registry can resolve it."""
    assert not hasattr(v2, "ToRGB")
    assert "ToRGB" in repr(build_preprocessing([PreprocessingStep(step="ToRGB")]))


def test_an_unknown_step_lists_what_is_registered() -> None:
    with pytest.raises(ValueError, match="ToRGB"):
        build_preprocessing([PreprocessingStep(step="NoSuchTransform")])


def test_plugin_extractor_and_transform_register(plugins: dict[str, list[tuple[str, str]]]) -> None:
    plugins["dataeval_flow.extractors"] = [("example.mean", "tests.example_plugin:MeanExtractor")]
    plugins["dataeval_flow.transforms"] = [("example.Invert", "tests.example_plugin:Invert")]
    assert "example.mean" in [cls.name for cls in list_extractors()]
    assert "example.Invert" in [cls.name for cls in list_transforms()]


# --- The contract, checked where the class is defined ---


def test_a_concrete_extractor_must_declare_its_identity() -> None:
    with pytest.raises(TypeError, match=r"^Nameless must declare name, description: set `name`, `description`"):

        class Nameless(Extractor[FlattenExtractorConfig]):
            def build(self, config: FlattenExtractorConfig, transforms: Any) -> Any:
                raise NotImplementedError


def test_an_unparameterized_extractor_is_told_how_to_name_its_config() -> None:
    with pytest.raises(TypeError, match=re.escape("e.g. `class Bare(Extractor[MyConfig])`")):

        class Bare(Extractor):  # type: ignore[type-arg]
            name: ClassVar[str] = "bare"
            description: ClassVar[str] = "No type argument."

            def build(self, config: Any, transforms: Any) -> Any:
                raise NotImplementedError


def test_an_extractor_needs_no_result_class() -> None:
    assert get_extractor("flatten").config_type is FlattenExtractorConfig


def test_a_concrete_transform_must_declare_its_identity() -> None:
    with pytest.raises(
        TypeError, match=r"^Nameless must declare description: set `description` as a class attribute\.$"
    ):

        class Nameless(Transform):
            name: ClassVar[str] = "Nameless"

            def __call__(self, data: Any, /) -> Any:
                return data


def test_a_config_refuses_another_model() -> None:
    with pytest.raises(ValidationError, match="FlattenExtractorConfig configures model 'flatten', not 'bovw'"):
        FlattenExtractorConfig(name="flat", model="bovw")


def test_a_config_schema_states_and_describes_its_model() -> None:
    model_schema = FlattenExtractorConfig.model_json_schema()["properties"]["model"]
    assert model_schema["const"] == "flatten"
    assert model_schema["description"] == "The extractor this entry configures: `flatten`."


def test_a_config_that_describes_no_model_gets_the_base_description() -> None:
    model_schema = MeanConfig.model_json_schema()["properties"]["model"]
    assert model_schema["const"] == "example.mean"
    assert model_schema["description"] == "The extractor this entry configures."


def test_a_config_dumps_its_fields_in_the_order_the_cache_keys_on() -> None:
    dumped = BoVWExtractorConfig(name="bovw", vocab_size=512).model_dump_json()
    assert dumped == '{"name":"bovw","preprocessor":null,"batch_size":null,"model":"bovw","vocab_size":512}'
    onnx = OnnxExtractorConfig(name="o", model_path="m.onnx", batch_size=4, image_height=2, image_width=3)
    assert onnx.model_dump_json() == (
        '{"name":"o","preprocessor":null,"batch_size":4,"model":"onnx","model_path":"m.onnx","output_name":null,'
        '"flatten":true,"image_height":2,"image_width":3}'
    )


# --- Configs dispatch by `model` ---


def test_an_entry_is_validated_with_the_config_its_model_names() -> None:
    config = PipelineConfig.model_validate({"extractors": [{"name": "b", "model": "bovw", "vocab_size": 512}]})
    assert config.extractors is not None
    assert isinstance(config.extractors[0], BoVWExtractorConfig)
    assert json.loads(config.model_dump_json())["extractors"][0]["vocab_size"] == 512


def test_an_unknown_model_lists_the_installed_ones() -> None:
    with pytest.raises(ValidationError, match="Unknown extractor: 'resnet'. Installed: .*'bovw'"):
        PipelineConfig.model_validate({"extractors": [{"name": "r", "model": "resnet"}]})


def test_an_entry_without_a_model_says_so() -> None:
    with pytest.raises(ValidationError, match="Each `extractors:` entry needs a `model:`"):
        PipelineConfig.model_validate({"extractors": [{"name": "x"}]})


def test_an_invalid_entry_is_located_by_its_index() -> None:
    entries = [{"name": "f", "model": "flatten"}, {"name": "b", "model": "bovw", "vocab_size": 1}]
    with pytest.raises(ValidationError) as caught:
        PipelineConfig.model_validate({"extractors": entries})
    assert caught.value.errors()[0]["loc"] == ("extractors", 1, "vocab_size")


# --- Plugins ---


class _Mismatched(Extractor[FlattenExtractorConfig]):
    """Registered as `example.mismatched`, but its config configures `flatten`."""

    name: ClassVar[str] = "example.mismatched"
    description: ClassVar[str] = "Its config configures another model."

    def build(self, config: FlattenExtractorConfig, transforms: Any) -> Any:
        raise NotImplementedError


def test_a_plugin_whose_config_configures_another_model_is_refused(plugins: dict[str, list[tuple[str, str]]]) -> None:
    plugins["dataeval_flow.extractors"] = [("example.mismatched", f"{__name__}:_Mismatched")]
    assert "example.mismatched" not in [cls.name for cls in list_extractors()]
    with pytest.raises(ValueError, match="`model` defaults to 'flatten', not 'example.mismatched'"):
        get_extractor("example.mismatched")


def test_a_plugin_extractor_validates_from_yaml_and_embeds(plugins: dict[str, list[tuple[str, str]]]) -> None:
    from dataeval_flow._embeddings import build_extractor

    plugins["dataeval_flow.extractors"] = [("example.mean", "tests.example_plugin:MeanExtractor")]
    config = PipelineConfig.model_validate(yaml.safe_load("extractors:\n  - name: means\n    model: example.mean\n"))
    assert config.extractors is not None
    (entry,) = config.extractors
    assert isinstance(entry, MeanConfig)
    images = [np.full((3, 2, 2), value, dtype=np.float32) for value in (0.0, 1.0)]
    assert np.asarray(build_extractor(entry)(images)).tolist() == [[0.0] * 3, [1.0] * 3]


def test_a_plugin_transform_runs_as_a_step(plugins: dict[str, list[tuple[str, str]]]) -> None:
    plugins["dataeval_flow.transforms"] = [("example.Invert", "tests.example_plugin:Invert")]
    inverted = build_preprocessing([PreprocessingStep(step="example.Invert")])(np.zeros((1, 2, 2), dtype=np.float32))
    assert inverted.tolist() == [[[1.0, 1.0], [1.0, 1.0]]]


def test_a_step_naming_a_broken_transform_plugin_raises_its_failure(
    plugins: dict[str, list[tuple[str, str]]],
) -> None:
    plugins["dataeval_flow.transforms"] = [("example.Gone", "tests.example_plugin_missing:Gone")]
    with pytest.raises(ValueError, match="failed to load tests.example_plugin_missing:Gone") as caught:
        build_preprocessing([PreprocessingStep(step="example.Gone")])
    assert "Unknown transform" not in str(caught.value)


class _Resize(Transform):
    """A transform that tries to take torchvision's `Resize`."""

    name: ClassVar[str] = "Resize"
    description: ClassVar[str] = "Takes a torchvision name."

    def __call__(self, data: Any, /) -> Any:
        return data


def test_a_transform_plugin_cannot_take_a_torchvision_name(plugins: dict[str, list[tuple[str, str]]]) -> None:
    plugins["dataeval_flow.transforms"] = [("Resize", f"{__name__}:_Resize")]
    assert "Resize" not in [cls.name for cls in list_transforms()]
    with pytest.raises(ValueError, match=re.escape("is taken by `torchvision.transforms.v2.Resize`")):
        get_transform("Resize")
    composed = build_preprocessing([PreprocessingStep(step="Resize", params={"size": [4, 4]})]).__wrapped__
    assert type(composed.transforms[0]) is v2.Resize


def test_a_builtin_transform_taking_a_torchvision_name_raises_at_once() -> None:
    from dataeval_flow._registry import Registry
    from dataeval_flow.config.transforms._registry import TRANSFORMS

    registry = Registry(
        kind="transform",
        group="dataeval_flow.tests.no-such-group",
        base=lambda: Transform,
        builtins={"Resize": f"{__name__}:_Resize"},
        check=TRANSFORMS._check,
    )
    with pytest.raises(RuntimeError, match=re.escape("`torchvision.transforms.v2.Resize`")):
        registry.list()


class _Tagged(Transform):
    """Keeps the `dtype` it is given."""

    name: ClassVar[str] = "example.Tagged"
    description: ClassVar[str] = "Records its `dtype` param."

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    def __call__(self, data: Any, /) -> Any:
        return data

    def __repr__(self) -> str:
        return f"Tagged(dtype={self.dtype!r})"


def test_a_registered_transform_takes_its_params_as_written(plugins: dict[str, list[tuple[str, str]]]) -> None:
    """The `dtype`/`interpolation` conversions are torchvision's; a registered transform gets the value as written."""
    plugins["dataeval_flow.transforms"] = [("example.Tagged", f"{__name__}:_Tagged")]
    composed = build_preprocessing([PreprocessingStep(step="example.Tagged", params={"dtype": "float32"})]).__wrapped__
    assert composed.transforms[0].dtype == "float32"


def test_the_example_transform_has_a_stable_repr() -> None:
    assert repr(Invert()) == "Invert()"
