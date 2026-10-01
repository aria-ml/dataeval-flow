"""A stub ONNX runtime and model files, so the `uncertainty` extractor runs DataEval's real ONNX predictors in tests
without onnxruntime installed, as DataEval's own tests stub it."""

import json
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.drift_toys import CLASSES

N_CLASSES = 3
N_BOXES = 4


class FakeSession:
    """Stands in for ``onnxruntime.InferenceSession``: each output is a function of the input batch, and `batches`
    records each batch's size."""

    def __init__(self, outputs: Mapping[str, Callable[[np.ndarray], np.ndarray]]) -> None:
        self.outputs = dict(outputs)
        self.batches: list[int] = []

    def get_inputs(self) -> list[types.SimpleNamespace]:
        return [types.SimpleNamespace(name="image")]

    def get_outputs(self) -> list[types.SimpleNamespace]:
        return [types.SimpleNamespace(name=name) for name in self.outputs]

    def run(self, output_names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        tensor = next(iter(feed.values()))
        self.batches.append(int(tensor.shape[0]))
        return [np.asarray(self.outputs[name](tensor), dtype=np.float32) for name in output_names]


def install(monkeypatch: pytest.MonkeyPatch, outputs: Mapping[str, Callable[[np.ndarray], np.ndarray]]) -> FakeSession:
    """Install a stub ``onnxruntime`` whose every session is one `FakeSession` over `outputs`, and return it."""
    session = FakeSession(outputs)
    module = types.ModuleType("onnxruntime")
    module.InferenceSession = lambda *args, **kwargs: session  # type: ignore[attr-defined]  # noqa: ARG005
    module.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    return session


def _brightness(tensor: np.ndarray) -> np.ndarray:
    """Each image's mean pixel, in [0, 1]: what the stub models' scores follow."""
    return tensor.reshape(tensor.shape[0], -1).mean(axis=1)


def _classifier_scores(tensor: np.ndarray) -> np.ndarray:
    """Logits, (B, 3), surer of class 0 as an image brightens."""
    b = _brightness(tensor)
    return np.stack([8 * b, 4 * (1 - b), np.zeros_like(b)], axis=1)


def _detector_scores(tensor: np.ndarray) -> np.ndarray:
    """Per-class sigmoid scores, (B, 4, 3): a box of class 0 and a box of class 1, both scaled by brightness, so a dark
    image's boxes fall below any useful confidence, then two zero-padded boxes."""
    b = _brightness(tensor)[:, None]
    real = np.stack([0.02 + b * [0.9, 0.3, 0.1], 0.02 + b * [0.1, 0.8, 0.4]], axis=1)
    return np.concatenate([real, np.zeros((len(b), N_BOXES - 2, N_CLASSES))], axis=1)


def _detector_boxes(tensor: np.ndarray) -> np.ndarray:
    corners = [[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
    return np.tile(np.asarray(corners, dtype=np.float32), (tensor.shape[0], 1, 1))


CLASSIFIER = {"scores": _classifier_scores}
DETECTOR = {"boxes": _detector_boxes, "scores": _detector_scores}


def model_files(
    root: Path, task: str = "IMAGE_CLASSIFICATION", *, n_classes: int = N_CLASSES, batch_size: int = -1, size: int = 16
) -> tuple[str, str]:
    """Write a stub model file and DataEval's metadata for it under `root`; return their names, relative to `root`."""
    (root / "model.onnx").write_bytes(b"stub")
    output = {"nClasses": n_classes} | ({"nBoxes": N_BOXES} if task == "IMAGE_OBJECT_DETECTION" else {})
    metadata = {
        "interface": {"name": "JATIC_ONNX", "version": "v1"},
        "io": {
            "batchSize": batch_size,
            "interface": task,
            "input": {"channels": "RGB", "height": size, "width": size},
            "output": output,
        },
    }
    (root / "model.json").write_text(json.dumps(metadata), encoding="utf-8")
    return "model.onnx", "model.json"


class _NoBoxes:
    """An unlabelled detection target: no boxes."""

    labels = np.zeros(0, dtype=np.intp)
    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros(0, dtype=np.float32)


class Frames:
    """Unlabelled detection frames, 3x16x16: frame i is a flat grey of `brightness[i]` (0 to 1), plus a little noise.

    `index2label` names the classes, as a reference would; ``None`` names none, as incoming data often does.
    """

    def __init__(
        self, brightness: Sequence[float], seed: int = 0, *, index2label: Mapping[int, str] | None = CLASSES
    ) -> None:
        rng = np.random.default_rng(seed)
        self._images = [np.clip(rng.normal(255 * b, 4, (3, 16, 16)), 0, 255).astype(np.uint8) for b in brightness]
        self.metadata: dict[str, Any] = {"id": f"frames-{seed}-{len(self._images)}"}
        if index2label is not None:
            self.metadata["index2label"] = dict(index2label)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        return self._images[index], _NoBoxes(), {"id": index}
