"""TC-4-1 — extractor configs and the extractors they build."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow import run, run_tasks
from dataeval_flow._preprocessing import build_preprocessing
from dataeval_flow.config import PipelineConfig, PreprocessingStep, PreprocessorConfig
from dataeval_flow.config.extractors import (
    BoVWExtractorConfig,
    Extractor,
    FlattenExtractorConfig,
    OnnxExtractorConfig,
    TorchExtractorConfig,
    UncertaintyExtractorConfig,
    get_extractor,
    list_extractors,
)
from dataeval_flow.evaluators.scope import CompletenessConfig
from verification.fixtures import write_image_folder
from verification.functional.extractors.models import (
    write_onnx_classifier,
    write_onnx_embedder,
    write_torch_net,
)
from verification.functional.orchestration.support import (
    InMemoryImages,
    MeanConfig,
    Means,
    pipeline_dict,
)

pytestmark = [pytest.mark.required, pytest.mark.usefixtures("fresh_caches")]

BUILT_IN = ["bovw", "flatten", "onnx", "torch", "uncertainty"]
ENTRIES: dict[str, dict[str, Any]] = {
    "flatten": {"model": "flatten"},
    "torch": {"model": "torch", "model_path": "net.pt", "layer_name": "1"},
    "onnx": {"model": "onnx", "model_path": "embed.onnx", "output_name": "flatten0"},
    "bovw": {"model": "bovw", "vocab_size": 512},
    "uncertainty": {
        "model": "uncertainty",
        "model_path": "classifier.onnx",
        "metadata_path": "classifier.json",
        "preds_type": "logits",
    },
}
CONFIG_CLASSES = {
    "flatten": FlattenExtractorConfig,
    "torch": TorchExtractorConfig,
    "onnx": OnnxExtractorConfig,
    "bovw": BoVWExtractorConfig,
    "uncertainty": UncertaintyExtractorConfig,
}


def _images(n: int = 6, shape: tuple[int, int, int] = (3, 8, 8), seed: int = 0) -> list[Any]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, shape, dtype=np.uint8) for _ in range(n)]


class TestExtractorConfigs:
    def test_flatten_extractor_config_constructs(self) -> None:
        config = FlattenExtractorConfig()
        assert (config.model, config.name) == ("flatten", "flatten")
        assert config.batch_size is None
        assert config.preprocessor is None

    def test_torch_extractor_config_constructs(self) -> None:
        config = TorchExtractorConfig(model_path="model.pt", layer_name="layer4", use_output=False)
        assert (config.model, config.model_path, config.layer_name, config.use_output) == (
            "torch",
            "model.pt",
            "layer4",
            False,
        )

    def test_onnx_extractor_config_constructs(self) -> None:
        config = OnnxExtractorConfig(model_path="model.onnx", output_name="flatten0", image_height=224, image_width=224)
        assert (config.model, config.model_path, config.output_name) == ("onnx", "model.onnx", "flatten0")
        assert (config.image_height, config.image_width, config.flatten) == (224, 224, True)

    def test_bovw_extractor_config_constructs(self) -> None:
        assert BoVWExtractorConfig().vocab_size == 2048
        assert BoVWExtractorConfig(vocab_size=512).model == "bovw"

    def test_uncertainty_extractor_config_constructs(self) -> None:
        config = UncertaintyExtractorConfig(
            model_path="yolo.onnx", metadata_path="yolo.json", preds_type="sigmoid", confidence=0.25
        )
        assert (config.model, config.preds_type, config.confidence) == ("uncertainty", "sigmoid", 0.25)

    def test_an_entry_is_named_after_its_model_unless_it_is_given_a_name(self) -> None:
        assert FlattenExtractorConfig().name == "flatten"
        assert FlattenExtractorConfig(name="pixels").name == "pixels"
        assert BoVWExtractorConfig.model_validate({"model": "bovw"}).name == "bovw"

    def test_batch_size_and_preprocessor_are_set_on_any_extractor(self) -> None:
        config = FlattenExtractorConfig(batch_size=32, preprocessor="resize")
        assert (config.batch_size, config.preprocessor) == (32, "resize")

    @pytest.mark.parametrize("model", sorted(CONFIG_CLASSES))
    def test_an_entry_names_its_extractor_by_model(self, model: str) -> None:
        config = PipelineConfig.model_validate({"extractors": [ENTRIES[model]]})
        assert config.extractors is not None
        assert type(config.extractors[0]) is CONFIG_CLASSES[model]
        assert config.extractors[0].name == model

    @pytest.mark.parametrize("model", sorted(CONFIG_CLASSES))
    def test_a_config_round_trips_through_serialization(self, model: str) -> None:
        config = PipelineConfig.model_validate({"extractors": [ENTRIES[model]]})
        reloaded = PipelineConfig.model_validate(yaml.safe_load(yaml.safe_dump(config.model_dump())))
        assert reloaded == config

    def test_a_config_refuses_another_extractors_model(self) -> None:
        with pytest.raises(ValidationError, match="configures model 'flatten', not 'onnx'"):
            FlattenExtractorConfig(model="onnx")

    def test_a_setting_of_another_extractor_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="vocab_size"):
            PipelineConfig.model_validate({"extractors": [{"model": "flatten", "vocab_size": 512}]})

    def test_an_entry_without_a_model_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="needs a `model:`"):
            PipelineConfig.model_validate({"extractors": [{"name": "x", "batch_size": 8}]})

    def test_an_unknown_model_lists_the_installed_extractors(self) -> None:
        with pytest.raises(ValidationError, match=r"Unknown extractor: 'resnet'. Installed: \['bovw', 'flatten'"):
            PipelineConfig.model_validate({"extractors": [{"model": "resnet"}]})

    def test_an_invalid_entry_is_located_by_its_index(self) -> None:
        entries = [ENTRIES["flatten"], {"model": "onnx", "model_path": "m.onnx", "image_height": 0, "image_width": 8}]
        with pytest.raises(ValidationError) as caught:
            PipelineConfig.model_validate({"extractors": entries})
        assert caught.value.errors()[0]["loc"][:3] == ("extractors", 1, "image_height")

    def test_onnx_input_size_needs_both_dimensions(self) -> None:
        with pytest.raises(ValidationError, match="set together"):
            OnnxExtractorConfig(model_path="m.onnx", image_height=224)
        with pytest.raises(ValidationError, match="set together"):
            UncertaintyExtractorConfig(
                model_path="m.onnx", metadata_path="m.json", preds_type="logits", image_width=224
            )

    def test_bovw_vocabulary_size_is_bounded(self) -> None:
        for size in (255, 4097):
            with pytest.raises(ValidationError, match="vocab_size"):
                BoVWExtractorConfig(vocab_size=size)
        assert BoVWExtractorConfig(vocab_size=256).vocab_size == 256
        assert BoVWExtractorConfig(vocab_size=4096).vocab_size == 4096

    def test_uncertainty_needs_its_model_metadata_and_prediction_kind(self) -> None:
        with pytest.raises(ValidationError) as caught:
            UncertaintyExtractorConfig()  # type: ignore[call-arg]
        assert {error["loc"][0] for error in caught.value.errors()} == {"model_path", "metadata_path", "preds_type"}
        with pytest.raises(ValidationError, match="preds_type"):
            UncertaintyExtractorConfig(model_path="m.onnx", metadata_path="m.json", preds_type="softmax")  # type: ignore[arg-type]

    @pytest.mark.parametrize("path", ["/abs/model.onnx", "../model.onnx"])
    def test_model_paths_must_stay_under_the_data_root(self, path: str) -> None:
        with pytest.raises(ValidationError, match="model_path"):
            TorchExtractorConfig(model_path=path)
        with pytest.raises(ValidationError, match="model_path"):
            OnnxExtractorConfig(model_path=path)
        with pytest.raises(ValidationError, match="model_path"):
            UncertaintyExtractorConfig(model_path=path, metadata_path="m.json", preds_type="logits")
        with pytest.raises(ValidationError, match="metadata_path"):
            UncertaintyExtractorConfig(model_path="m.onnx", metadata_path=path, preds_type="logits")

    def test_the_extractor_configs_are_exported_from_the_extractors_package(self) -> None:
        import dataeval_flow.config.extractors as package

        assert {
            "BoVWExtractorConfig",
            "FlattenExtractorConfig",
            "OnnxExtractorConfig",
            "TorchExtractorConfig",
            "UncertaintyExtractorConfig",
            "ExtractorConfig",
            "Extractor",
            "get_extractor",
            "list_extractors",
        } <= set(package.__all__)
        for name in package.__all__:
            assert hasattr(package, name)

    def test_the_built_in_extractors_are_listed_by_name(self) -> None:
        assert [cls.name for cls in list_extractors()] == BUILT_IN
        for cls in list_extractors():
            assert issubclass(cls, Extractor)
            assert cls.description
            assert cls.config_type is CONFIG_CLASSES[cls.name]
        assert [cls.name for cls in list_extractors() if cls.stateful] == ["bovw"]

    def test_an_unknown_extractor_raises_and_lists_the_installed_ones(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown extractor: 'resnet'. Installed: \['bovw'"):
            get_extractor("resnet")


class TestBuildExtractors:
    """An extractor built from its config returns one embedding row per image it is given."""

    def test_flatten_extractor_from_config_embeds_every_sample(self) -> None:
        images = _images(10)
        embeddings = np.asarray(get_extractor("flatten")().build(FlattenExtractorConfig(), None)(images))
        assert embeddings.shape == (10, 3 * 8 * 8)
        assert np.array_equal(embeddings[3], images[3].reshape(-1))

    def test_torch_extractor_embeds_with_the_named_layer(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        name = write_torch_net(tmp_path, features=4)
        to_float = build_preprocessing([PreprocessingStep(step="ToDtype", params={"dtype": "float32", "scale": True})])
        for layer, width in (("1", 4), ("0", 3 * 8 * 8)):
            config = TorchExtractorConfig(model_path=name, layer_name=layer)
            embeddings = np.asarray(get_extractor("torch")().build(config, to_float)(_images(6)))
            assert embeddings.shape == (6, width), layer

    def test_onnx_extractor_embeds_with_the_named_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        monkeypatch.chdir(tmp_path)
        name = write_onnx_embedder(tmp_path)
        config = OnnxExtractorConfig(model_path=name, output_name="flatten0")
        images = _images(6)
        embeddings = np.asarray(get_extractor("onnx")().build(config, None)(images))
        assert embeddings.shape == (6, 3 * 8 * 8)
        assert embeddings.dtype == np.float32
        assert np.array_equal(embeddings[2], images[2].reshape(-1).astype(np.float32))

    def test_onnx_extractor_resizes_images_to_the_configured_input_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The model reads 4 x 4 images; the 8 x 8 images are resized to it because the config says so."""
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        monkeypatch.chdir(tmp_path)
        name = write_onnx_embedder(tmp_path, size=4, name="small.onnx")
        config = OnnxExtractorConfig(model_path=name, output_name="flatten0", image_height=4, image_width=4)
        embeddings = np.asarray(get_extractor("onnx")().build(config, None)(_images(5)))
        assert embeddings.shape == (5, 3 * 4 * 4)

    def test_bovw_extractor_embeds_each_image_as_a_histogram(self) -> None:
        pytest.importorskip("cv2")
        images = _images(8, shape=(3, 64, 64))
        embeddings = np.asarray(get_extractor("bovw")().build(BoVWExtractorConfig(), None)(images))
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == 8
        assert (embeddings >= 0).all()

    def test_uncertainty_extractor_is_read_through_predictions_not_built(self) -> None:
        config = UncertaintyExtractorConfig(model_path="m.onnx", metadata_path="m.json", preds_type="logits")
        with pytest.raises(ValueError, match="only drift and OOD evaluators read it"):
            get_extractor("uncertainty")().build(config, None)


class TestExtractorsInAPipeline:
    def test_a_task_embeds_with_the_extractor_it_names(self) -> None:
        result = run(CompletenessConfig(), InMemoryImages(20), extractor=FlattenExtractorConfig(batch_size=8))
        assert result.success, result.errors
        assert result.metadata.model_id == "flatten (flatten)"

    def test_a_preprocessor_runs_on_each_image_before_the_extractor_sees_it(
        self,
        example_plugin: dict[str, Any],
    ) -> None:
        to_float = PreprocessorConfig(
            name="to_float", steps=[PreprocessingStep(step="ToDtype", params={"dtype": "float32", "scale": True})]
        )
        Means.dtypes.clear()
        plain = run(CompletenessConfig(), InMemoryImages(20), extractor=MeanConfig(batch_size=8))
        assert plain.success, plain.errors
        assert set(Means.dtypes) == {"uint8"}

        Means.dtypes.clear()
        extractor = MeanConfig(batch_size=8, preprocessor="to_float")
        result = run(CompletenessConfig(), InMemoryImages(20), extractor=extractor, definitions=[to_float])
        assert result.success, result.errors
        assert set(Means.dtypes) == {"float32"}
        assert result.metadata.preprocessor_id == "to_float"

    def test_batch_size_sets_how_many_images_the_extractor_gets_at_once(
        self,
        example_plugin: dict[str, Any],
    ) -> None:
        Means.batches.clear()
        result = run(CompletenessConfig(), InMemoryImages(20), extractor=MeanConfig(batch_size=8))
        assert result.success, result.errors
        assert Means.batches == [8, 8, 4]

    def test_an_extractor_with_no_batch_size_fails_the_run_that_needs_embeddings(
        self,
        example_plugin: dict[str, Any],
    ) -> None:
        result = run(CompletenessConfig(), InMemoryImages(20), extractor=MeanConfig())
        assert not result.success
        assert "No batch_size provided" in result.errors[0]

    def test_an_onnx_extractor_finds_its_model_in_the_models_folder_of_the_data_root(self, tmp_path: Path) -> None:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        write_image_folder(tmp_path / "imgs", n_per_class=10, n_classes=2)
        write_onnx_embedder(tmp_path / "models")
        data = pipeline_dict(
            extractors=[
                {"name": "net", "model": "onnx", "model_path": "embed.onnx", "output_name": "flatten0", "batch_size": 8}
            ],
            evaluators=[{"name": "comp", "type": "completeness"}],
            tasks=[{"name": "t", "evaluator": "comp", "sources": "main", "extractor": "net"}],
        )
        result = run_tasks(PipelineConfig.model_validate(data), data_dir=tmp_path)["t"]
        assert result.success, result.errors
        assert result.metadata.model_id == "net (onnx)"

    def test_a_missing_model_file_is_reported_when_the_task_runs(self, tmp_path: Path) -> None:
        pytest.importorskip("onnxruntime")
        write_image_folder(tmp_path / "imgs", n_per_class=4, n_classes=2)
        data = pipeline_dict(
            extractors=[{"name": "net", "model": "onnx", "model_path": "gone.onnx", "batch_size": 8}],
            evaluators=[{"name": "comp", "type": "completeness"}],
            tasks=[{"name": "t", "evaluator": "comp", "sources": "main", "extractor": "net"}],
        )
        result = run_tasks(PipelineConfig.model_validate(data), data_dir=tmp_path)["t"]
        assert not result.success
        assert "gone.onnx" in result.errors[0]


class TestUncertaintyExtractor:
    """The `uncertainty` extractor turns a classifier's predictions into one entropy per image, for drift."""

    @pytest.fixture
    def project(self, tmp_path: Path) -> Path:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        write_image_folder(tmp_path / "ref", n_per_class=15, n_classes=2, seed=0, low=0, high=120)
        write_image_folder(tmp_path / "same", n_per_class=15, n_classes=2, seed=1, low=0, high=120)
        write_image_folder(tmp_path / "bright", n_per_class=15, n_classes=2, seed=2, low=135, high=256)
        write_onnx_classifier(tmp_path / "models")
        return tmp_path

    @staticmethod
    def _pipeline(tasks: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "datasets": [
                {"name": n, "format": "image_folder", "path": n, "infer_labels": True}
                for n in ("ref", "same", "bright")
            ],
            "sources": [{"name": n, "dataset": n} for n in ("ref", "same", "bright")],
            "extractors": [ENTRIES["uncertainty"] | {"name": "unc", "batch_size": 8}],
            "evaluators": [{"name": "ks", "type": "drift-univariate"}, {"name": "comp", "type": "completeness"}],
            "tasks": tasks,
        }

    def test_drift_is_detected_in_the_model_s_uncertainty(self, project: Path) -> None:
        tasks = [
            {"name": "same", "evaluator": "ks", "sources": ["ref", "same"], "extractor": "unc"},
            {"name": "bright", "evaluator": "ks", "sources": ["ref", "bright"], "extractor": "unc"},
        ]
        results = run_tasks(PipelineConfig.model_validate(self._pipeline(tasks)), data_dir=project)
        assert results["same"].success
        assert results["bright"].success
        assert not results["same"].output.drifted
        assert results["bright"].output.drifted
        assert results["bright"].metadata.model_id == "unc (uncertainty)"

    def test_an_evaluator_that_needs_one_row_per_item_is_refused_when_the_config_loads(self, project: Path) -> None:
        tasks = [{"name": "bad", "evaluator": "comp", "sources": "ref", "extractor": "unc"}]
        with pytest.raises(ValidationError, match=r"Task 'bad'.*only drift and OOD evaluators read it"):
            PipelineConfig.model_validate(self._pipeline(tasks))
