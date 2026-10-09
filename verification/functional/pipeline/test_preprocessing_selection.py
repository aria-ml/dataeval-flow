"""TC-5-1 — preprocessing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow import run
from dataeval_flow._preprocessing import build_preprocessing
from dataeval_flow.config import PipelineConfig, PreprocessingStep, PreprocessorConfig
from dataeval_flow.config.image_transforms import ImageTransform, get_image_transform, list_image_transforms
from dataeval_flow.evaluators.scope import CompletenessConfig
from verification.functional.orchestration.support import (
    MODULE,
    InMemoryImages,
    MeanConfig,
    Means,
)

pytestmark = pytest.mark.required


class _Resize(ImageTransform):
    """A transform that tries to take the name of torchvision's `Resize`."""

    name: ClassVar[str] = "Resize"
    description: ClassVar[str] = "Takes a torchvision name."

    def __call__(self, data: Any, /) -> Any:
        return data


def _chw(channels: int = 3, size: int = 8, fill: int | None = None) -> np.ndarray:
    if fill is not None:
        return np.full((channels, size, size), fill, dtype=np.uint8)
    return np.random.default_rng(0).integers(0, 255, (channels, size, size), dtype=np.uint8)


def _steps(*specs: tuple[str, dict[str, Any]]) -> list[PreprocessingStep]:
    return [PreprocessingStep(step=name, params=params) for name, params in specs]


class TestPreprocessorConfig:
    def test_preprocessor_config_accepts_transforms(self) -> None:
        config = PreprocessorConfig(
            name="resnet",
            steps=[
                PreprocessingStep(step="Resize", params={"size": [224, 224], "antialias": True}),
                PreprocessingStep(
                    step="Normalize", params={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
                ),
            ],
        )
        assert config.name == "resnet"
        assert [step.step for step in config.steps] == ["Resize", "Normalize"]
        assert config.steps[0].params == {"size": [224, 224], "antialias": True}

    def test_a_step_without_params_has_an_empty_set(self) -> None:
        assert PreprocessingStep(step="ToImage").params == {}

    def test_steps_keep_the_order_a_config_file_gives_them(self, tmp_path: Path) -> None:
        text = """
preprocessors:
  - name: p
    steps:
      - {step: ToDtype, params: {dtype: float32, scale: true}}
      - {step: Resize, params: {size: [4, 4]}}
      - {step: Normalize, params: {mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]}}
"""
        config = PipelineConfig.model_validate(yaml.safe_load(text))
        assert config.preprocessors is not None
        assert [step.step for step in config.preprocessors[0].steps] == ["ToDtype", "Resize", "Normalize"]

    def test_a_preprocessor_needs_a_name_and_steps(self) -> None:
        with pytest.raises(ValidationError) as caught:
            PreprocessorConfig()  # type: ignore[call-arg]
        assert {error["loc"][0] for error in caught.value.errors()} == {"name", "steps"}

    def test_an_unknown_key_in_a_step_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            PreprocessingStep(step="Resize", params={}, size=4)  # type: ignore[call-arg]

    def test_a_pipeline_defines_each_preprocessor_name_once(self) -> None:
        entry = {"name": "p", "steps": [{"step": "ToImage"}]}
        with pytest.raises(ValidationError, match="Duplicate name 'p' in preprocessors"):
            PipelineConfig.model_validate({"preprocessors": [entry, entry]})


class TestApplyInOrder:
    def test_the_steps_apply_in_the_order_given(self) -> None:
        resize_then_crop = build_preprocessing(_steps(("Resize", {"size": [4, 4]}), ("CenterCrop", {"size": 2})))
        crop_then_resize = build_preprocessing(_steps(("CenterCrop", {"size": 2}), ("Resize", {"size": [4, 4]})))
        image = _chw()
        assert resize_then_crop(image).shape == (3, 2, 2)
        assert crop_then_resize(image).shape == (3, 4, 4)

    def test_a_normalization_after_a_dtype_conversion_scales_the_values(self) -> None:
        transform = build_preprocessing(
            _steps(
                ("ToDtype", {"dtype": "float32", "scale": True}),
                ("Normalize", {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}),
            )
        )
        black, white = transform(_chw(fill=0)), transform(_chw(fill=255))
        assert black.dtype == np.float32
        assert np.allclose(black, -1.0)
        assert np.allclose(white, 1.0)

    def test_the_result_is_a_numpy_array_in_chw_layout(self) -> None:
        transform = build_preprocessing(_steps(("Resize", {"size": [6, 4], "antialias": True})))
        output = transform(_chw())
        assert isinstance(output, np.ndarray)
        assert output.shape == (3, 6, 4)

    def test_dtype_and_interpolation_are_given_by_name(self) -> None:
        transform = build_preprocessing(
            _steps(("Resize", {"size": [4, 4], "interpolation": "NEAREST"}), ("ToDtype", {"dtype": "float32"}))
        )
        assert transform(_chw()).dtype == np.float32

    @pytest.mark.parametrize(
        ("step", "match"),
        [
            (("ToDtype", {"dtype": "float99"}), "Unknown torch dtype: 'float99'"),
            (("ToDtype", {"dtype": 32}), "Unknown torch dtype"),
            (("Resize", {"size": [4, 4], "interpolation": "SHARPEST"}), "Unknown InterpolationMode: 'SHARPEST'"),
            (("Resize", {"size": "big"}), r"size can (either )?be"),
        ],
    )
    def test_a_step_with_a_parameter_that_cannot_be_built_is_refused(self, step: tuple[str, dict], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            build_preprocessing(_steps(step))

    def test_an_unknown_transform_lists_the_registered_ones(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown image transform: 'Sharpen'.*ToRGB.*torchvision\.transforms\.v2"):
            build_preprocessing(_steps(("Sharpen", {})))

    def test_the_same_steps_give_the_same_output_and_the_same_identity(self) -> None:
        steps = _steps(
            ("Resize", {"size": [4, 4], "antialias": True}), ("ToDtype", {"dtype": "float32", "scale": True})
        )
        first, second = build_preprocessing(steps), build_preprocessing(steps)
        assert np.array_equal(first(_chw()), second(_chw()))
        assert repr(first) == repr(second)  # what the embedding cache keys on
        assert "Resize" in repr(first)
        assert repr(first) != repr(build_preprocessing(_steps(("Resize", {"size": [8, 8], "antialias": True}))))


class TestRegisteredTransforms:
    def test_to_rgb_is_a_registered_transform(self) -> None:
        assert "ToRGB" in [cls.name for cls in list_image_transforms()]
        assert get_image_transform("ToRGB").description

    @pytest.mark.parametrize(("channels", "expected"), [(1, 3), (2, 3), (3, 3), (4, 3)])
    def test_to_rgb_gives_every_image_three_channels(self, channels: int, expected: int) -> None:
        output = build_preprocessing(_steps(("ToRGB", {})))(_chw(channels))
        assert output.shape == (expected, 8, 8)

    def test_to_rgb_repeats_grey_and_drops_alpha(self) -> None:
        transform = build_preprocessing(_steps(("ToRGB", {})))
        grey = _chw(1)
        assert all(np.array_equal(transform(grey)[c], grey[0]) for c in range(3))
        rgba = _chw(4)
        assert np.array_equal(transform(rgba), rgba[:3])
        assert "ToRGB()" in repr(transform)

    def test_a_torchvision_name_means_torchvisions_transform(self) -> None:
        transform = build_preprocessing(_steps(("ToImage", {})))
        assert "ToImage" in repr(transform)
        assert "ToImage" not in [cls.name for cls in list_image_transforms()]


class TestPluginTransforms:
    def test_an_installed_transform_runs_as_a_step_with_its_params_as_written(
        self,
        example_plugin: dict[str, Any],
    ) -> None:
        transform = build_preprocessing(_steps(("example.Invert", {"maximum": 255})))
        assert np.array_equal(transform(_chw(fill=10)), np.full((3, 8, 8), 245, dtype=np.uint8))
        assert "Invert(maximum=255)" in repr(transform)

    def test_an_installed_transform_is_listed_and_composes_with_torchvisions(
        self,
        example_plugin: dict[str, Any],
    ) -> None:
        assert "example.Invert" in [cls.name for cls in list_image_transforms()]
        transform = build_preprocessing(
            _steps(
                ("ToDtype", {"dtype": "float32", "scale": True}), ("example.Invert", {}), ("Resize", {"size": [4, 4]})
            )
        )
        output = transform(_chw(fill=0))
        assert output.shape == (3, 4, 4)
        assert np.allclose(output, 1.0)

    def test_an_extractor_applies_an_installed_transform_from_a_preprocessor_entry(
        self,
        example_plugin: dict[str, Any],
        fresh_caches: None,
    ) -> None:
        Means.firsts.clear()
        invert = PreprocessorConfig(
            name="invert", steps=[PreprocessingStep(step="example.Invert", params={"maximum": 255})]
        )
        result = run(
            CompletenessConfig(),
            InMemoryImages(20),
            extractor=MeanConfig(batch_size=8, preprocessor="invert"),
            definitions=[invert],
        )
        assert result.success, result.errors
        assert np.array_equal(Means.firsts[0], 255 - InMemoryImages(20)[0][0])
        assert result.metadata.preprocessor_id == "invert"

    def test_a_step_naming_a_broken_transform_plugin_raises_the_plugin_s_failure(
        self,
        plugins: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        plugins["dataeval_flow.image_transforms"] = [("example.Gone", "verification.no_such_module:Gone")]
        with caplog.at_level(logging.WARNING):
            assert "example.Gone" not in [cls.name for cls in list_image_transforms()]
        assert "example.Gone" in caplog.text
        with pytest.raises(ValueError, match="failed to load verification.no_such_module:Gone"):
            build_preprocessing(_steps(("example.Gone", {})))

    def test_a_plugin_cannot_take_a_torchvision_transform_s_name(self, plugins: dict[str, Any]) -> None:
        plugins["dataeval_flow.image_transforms"] = [("Resize", f"{__name__}:_Resize")]
        assert "Resize" not in [cls.name for cls in list_image_transforms()]
        with pytest.raises(ValueError, match=r"taken by `torchvision\.transforms\.v2\.Resize`"):
            get_image_transform("Resize")
        # The step still means torchvision's Resize.
        assert build_preprocessing(_steps(("Resize", {"size": [4, 4]})))(_chw()).shape == (3, 4, 4)

    def test_a_plugin_cannot_take_a_built_in_transform_s_name(self, plugins: dict[str, Any]) -> None:
        plugins["dataeval_flow.image_transforms"] = [("ToRGB", f"{MODULE}:Invert")]
        assert get_image_transform("ToRGB").__name__ == "ToRGB"


class TestPreprocessorsInAPipeline:
    def test_an_extractor_names_the_preprocessor_it_applies(
        self,
        example_plugin: dict[str, Any],
        fresh_caches: None,
    ) -> None:
        to_float = PreprocessorConfig(name="to_float", steps=_steps(("ToDtype", {"dtype": "float32", "scale": True})))
        Means.dtypes.clear()
        result = run(
            CompletenessConfig(),
            InMemoryImages(20),
            extractor=MeanConfig(batch_size=8, preprocessor="to_float"),
            definitions=[to_float],
        )
        assert result.success, result.errors
        assert set(Means.dtypes) == {"float32"}
        assert result.metadata.preprocessor_id == "to_float"
        assert result.metadata.resolved_config["extractor"]["preprocessor"] == "to_float"

    def test_naming_a_preprocessor_the_pipeline_does_not_define_is_refused_before_the_task_runs(
        self,
        example_plugin: dict[str, Any],
        fresh_caches: None,
    ) -> None:
        with pytest.raises(ValueError, match="Unknown preprocessor: 'nope'"):
            run(
                CompletenessConfig(),
                InMemoryImages(20),
                extractor=MeanConfig(batch_size=8, preprocessor="nope"),
                definitions=[PreprocessorConfig(name="other", steps=_steps(("ToImage", {})))],
            )
