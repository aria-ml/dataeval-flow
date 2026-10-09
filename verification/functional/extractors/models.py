"""Tiny real model files for the extractor tests: ONNX graphs written with the `onnx` package, and a torch module.

Not a test module. Each writer returns the file's name, relative to the folder it wrote it in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_CLASSES = 3


def _save_onnx(path: Path, nodes: list, inputs: list, outputs: list, initializers: list | None = None) -> None:
    from onnx import checker, helper, save

    graph = helper.make_graph(nodes, "graph", inputs, outputs, initializers or [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    checker.check_model(model)
    save(model, str(path))


def write_onnx_embedder(folder: Path, *, size: int = 8, name: str = "embed.onnx") -> str:
    """A model whose output ``flatten0`` is its 3 x `size` x `size` input flattened to ``(N, 3 * size * size)``."""
    from onnx import TensorProto, helper

    folder.mkdir(parents=True, exist_ok=True)
    float32 = TensorProto.FLOAT
    _save_onnx(
        folder / name,
        [helper.make_node("Flatten", ["images"], ["flatten0"], axis=1)],
        [helper.make_tensor_value_info("images", float32, [None, 3, size, size])],
        [helper.make_tensor_value_info("flatten0", float32, [None, 3 * size * size])],
    )
    return name


def write_onnx_classifier(folder: Path, *, size: int = 8) -> tuple[str, str]:
    """A linear classifier over `N_CLASSES` classes, with DataEval's metadata file: ``(model, metadata)`` names.

    Its scores are linear in the pixels, so a brighter image gets larger-magnitude logits and a lower entropy.
    """
    from onnx import TensorProto, helper, numpy_helper

    folder.mkdir(parents=True, exist_ok=True)
    pixels = 3 * size * size
    weights = (
        np.stack([np.linspace(-1, 1, pixels), np.linspace(1, -0.5, pixels), np.zeros(pixels)], axis=1) * 0.02
    ).astype(np.float32)
    float32 = TensorProto.FLOAT
    _save_onnx(
        folder / "classifier.onnx",
        [
            helper.make_node("Flatten", ["images"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "weights"], ["scores"]),
        ],
        [helper.make_tensor_value_info("images", float32, [None, 3, size, size])],
        [helper.make_tensor_value_info("scores", float32, [None, N_CLASSES])],
        [numpy_helper.from_array(weights, "weights")],
    )
    metadata = {
        "interface": {"name": "JATIC_ONNX", "version": "v1"},
        "io": {
            "batchSize": -1,
            "interface": "IMAGE_CLASSIFICATION",
            "input": {"channels": "RGB", "height": size, "width": size},
            "output": {"nClasses": N_CLASSES},
        },
    }
    (folder / "classifier.json").write_text(json.dumps(metadata), encoding="utf-8")
    return "classifier.onnx", "classifier.json"


def write_torch_net(folder: Path, *, size: int = 8, features: int = 4, name: str = "net.pt") -> str:
    """A pickled ``Sequential(Flatten, Linear)`` whose layer ``"1"`` outputs `features` values per image."""
    import torch

    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.save(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * size * size, features)), folder / name)
    return name
