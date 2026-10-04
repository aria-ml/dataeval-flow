"""The versions of the libraries a run's numbers depend on, recorded on its result's envelope."""

__all__ = ["library_versions"]

from importlib.metadata import PackageNotFoundError, version
from typing import Any

# Every run reads through these: DataEval and torch compute, NumPy holds the arrays, and Pillow, datamaite and OpenCV
# decode images (datamaite reads COCO, YOLO and HuggingFace images with OpenCV).
_ALWAYS = ("dataeval", "numpy", "pillow", "datamaite", "opencv-python-headless", "opencv-python", "torch")


def library_versions(extractor: Any = None) -> dict[str, str]:
    """Each installed library the run depends on, by distribution name, with its version.

    `extractor` is the task's extractor config, or ``None``. Its ``runtime_distributions`` name the libraries its model
    runs on, such as ``onnxruntime``; those installed are recorded beside those every run reads through. A
    distribution that is not installed is left out.
    """
    found: dict[str, str] = {}
    for name in (*_ALWAYS, *getattr(extractor, "runtime_distributions", ())):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found
