"""Tests for preprocessing utilities."""

import pytest

from dataeval_app.preprocessing import PreprocessingStep, build_preprocessing


class TestPreprocessingStep:
    """Test PreprocessingStep schema."""

    def test_basic_step(self):
        """PreprocessingStep stores step name and params."""
        step = PreprocessingStep(step="Resize", params={"size": 256})
        assert step.step == "Resize"
        assert step.params == {"size": 256}

    def test_default_params(self):
        """PreprocessingStep defaults to empty params."""
        step = PreprocessingStep(step="ToImage")
        assert step.params == {}


class TestBuildPreprocessing:
    """Test build_preprocessing function."""

    def test_basic_transform(self):
        """build_preprocessing creates Compose from steps."""
        steps = [
            PreprocessingStep(step="Normalize", params={"mean": [0.5], "std": [0.5]}),
        ]
        transform = build_preprocessing(steps)
        assert transform is not None

    def test_dtype_converter(self):
        """build_preprocessing converts dtype string to torch type."""
        steps = [
            PreprocessingStep(step="ToDtype", params={"dtype": "float32", "scale": False}),
        ]
        transform = build_preprocessing(steps)
        assert transform is not None

    def test_invalid_transform_name_raises(self):
        """Unknown transform name raises ValueError."""
        steps = [PreprocessingStep(step="NonExistentTransform", params={})]
        with pytest.raises(ValueError, match="Unknown transform: 'NonExistentTransform'"):
            build_preprocessing(steps)

    def test_invalid_transform_params_raises(self):
        """Invalid param type for transform raises ValueError."""
        steps = [PreprocessingStep(step="Resize", params={"size": "big"})]
        with pytest.raises(ValueError, match="size can be"):
            build_preprocessing(steps)

    def test_interpolation_converter(self):
        """build_preprocessing converts interpolation string to enum."""
        steps = [
            PreprocessingStep(
                step="Resize",
                params={"size": [64, 64], "interpolation": "BILINEAR", "antialias": True},
            ),
        ]
        transform = build_preprocessing(steps)
        assert transform is not None

    def test_invalid_dtype_raises(self):
        """Unknown dtype string raises ValueError."""
        steps = [PreprocessingStep(step="ToDtype", params={"dtype": "nonexistent", "scale": False})]
        with pytest.raises(ValueError, match="Unknown torch dtype: 'nonexistent'"):
            build_preprocessing(steps)

    def test_invalid_interpolation_raises(self):
        """Unknown interpolation mode raises ValueError."""
        steps = [
            PreprocessingStep(
                step="Resize",
                params={"size": [64, 64], "interpolation": "NONEXISTENT"},
            ),
        ]
        with pytest.raises(ValueError, match="Unknown InterpolationMode: 'NONEXISTENT'"):
            build_preprocessing(steps)

    def test_non_string_dtype_raises(self):
        """Non-string dtype (e.g. int from YAML) raises ValueError."""
        steps = [PreprocessingStep(step="ToDtype", params={"dtype": 32, "scale": False})]
        with pytest.raises(ValueError, match="Unknown torch dtype"):
            build_preprocessing(steps)


class TestPreprocessingEndToEnd:
    """End-to-end tests for preprocessing pipeline."""

    def test_preprocessing_from_yaml_config(self, tmp_path):
        """Load preprocessing from YAML config and apply to numpy CHW image."""
        import numpy as np
        import yaml

        # Create YAML config with preprocessing steps
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "preprocessing": [
                        {"step": "ToDtype", "params": {"dtype": "float32", "scale": True}},
                        {
                            "step": "Normalize",
                            "params": {
                                "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225],
                            },
                        },
                    ]
                }
            )
        )

        # Load config
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Parse into PreprocessingStep objects
        steps = [PreprocessingStep(**step) for step in config["preprocessing"]]

        # Build pipeline
        transform = build_preprocessing(steps)

        # Create test image (CHW format, uint8 numpy array)
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (3, 224, 224), dtype=np.uint8)

        # Apply transform
        output = transform(image)

        # Verify output is numpy CHW
        assert isinstance(output, np.ndarray)
        assert output.dtype == np.float32
        assert output.shape == (3, 224, 224)
        assert output.min() < 0  # Normalized values can be negative

    def test_resize_and_normalize_pipeline(self):
        """Full pipeline: resize, convert dtype, normalize."""
        import numpy as np

        steps = [
            PreprocessingStep(step="Resize", params={"size": [128, 128], "antialias": True}),
            PreprocessingStep(step="ToDtype", params={"dtype": "float32", "scale": True}),
            PreprocessingStep(step="Normalize", params={"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}),
        ]
        transform = build_preprocessing(steps)

        # Create test image (CHW format, uint8 numpy array)
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (3, 256, 256), dtype=np.uint8)

        # Apply transform
        output = transform(image)

        # Verify output is numpy CHW with correct size
        assert isinstance(output, np.ndarray)
        assert output.shape == (3, 128, 128)
        assert output.dtype == np.float32
        # After normalize with mean=0.5, std=0.5: values in [-1, 1]
        assert output.min() >= -1.0
        assert output.max() <= 1.0
