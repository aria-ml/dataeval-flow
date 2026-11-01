"""A frozen pretrained backbone with a nearest-centroid head: the trainer the sufficiency how-to mounts.

Copy it to `trainers/backbone.py` under the data root, and the ONNX image classifier it reads to `models/`. The
backbone's class scores are the features, embedded once per image; each training size refits only the head.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval_flow.evaluators.performance import Trainer, TrainerKit

# The data root: this file sits in its `trainers/` folder.
DATA_ROOT = Path(__file__).parents[1]
# ImageNet's channel means and deviations, on 0-255 pixels.
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)[:, None, None]
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)[:, None, None]


class Backbone:
    """The pretrained model's outputs for each item, computed once and kept by dataset and index."""

    def __init__(self, weights: str, size: int) -> None:
        self.weights = DATA_ROOT / weights
        self.size = size
        self.session: Any = None
        self.cache: dict[tuple[int, int], NDArray[np.float32]] = {}

    def __call__(self, dataset: Any, indices: Sequence[int]) -> NDArray[np.float32]:
        """The features of the items at `indices`, one row each."""
        missing = sorted({index for index in indices if (id(dataset), index) not in self.cache})
        for start in range(0, len(missing), 64):
            batch = missing[start : start + 64]
            scores = self._run(np.stack([self._prepare(dataset[index][0]) for index in batch]))
            self.cache.update({(id(dataset), index): row for index, row in zip(batch, scores, strict=True)})
        return np.stack([self.cache[id(dataset), index] for index in indices])

    def _prepare(self, image: Any) -> NDArray[np.float32]:
        pixels = np.asarray(image, dtype=np.float32)
        rows = np.linspace(0, pixels.shape[1] - 1, self.size).astype(int)
        cols = np.linspace(0, pixels.shape[2] - 1, self.size).astype(int)
        return (pixels[:, rows][:, :, cols] - MEAN) / STD

    def _run(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.session is None:
            import onnxruntime
            from dataeval.config import get_device

            gpu = get_device().type == "cuda"
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
            self.session = onnxruntime.InferenceSession(str(self.weights), providers=providers)
        return self.session.run(None, {self.session.get_inputs()[0].name: batch})[0]


class Centroids:
    """The head: one mean feature row per class seen in training."""

    def __init__(self) -> None:
        self.labels = np.empty(0, dtype=int)
        self.means = np.empty((0, 0), dtype=np.float32)


class Fit:
    """Refit the head on the items at `indices`; an index may repeat, as DataEval draws with replacement."""

    def __init__(self, backbone: Backbone) -> None:
        self.backbone = backbone

    def train(self, model: Centroids, dataset: Any, indices: Sequence[int]) -> None:
        """Average each class's features."""
        features = self.backbone(dataset, indices)
        labels = np.array([int(np.argmax(dataset[index][1])) for index in indices])
        model.labels = np.unique(labels)
        model.means = np.stack([features[labels == label].mean(axis=0) for label in model.labels])


class Score:
    """The share of the test set whose nearest centroid is its own class."""

    def __init__(self, backbone: Backbone) -> None:
        self.backbone = backbone

    def evaluate(self, model: Centroids, dataset: Any) -> Mapping[str, float]:
        """Accuracy, from 0 to 1."""
        indices = list(range(len(dataset)))
        features = self.backbone(dataset, indices)
        distances = ((features[:, None, :] - model.means[None]) ** 2).sum(axis=2)
        labels = np.array([int(np.argmax(dataset[index][1])) for index in indices])
        return {"accuracy": float(np.mean(model.labels[np.argmin(distances, axis=1)] == labels))}


class BackboneTrainer(Trainer):
    """Nearest centroids over a frozen ONNX classifier's scores; `weights` is relative to the data root."""

    def __init__(self, weights: str, size: int = 224) -> None:
        self.weights = weights
        self.size = size

    def build(self) -> TrainerKit:
        """An empty head, refit at each size; a deep copy resets it. The backbone is loaded on first use."""
        backbone = Backbone(self.weights, self.size)
        return TrainerKit(model=Centroids(), training_strategy=Fit(backbone), evaluation_strategy=Score(backbone))
