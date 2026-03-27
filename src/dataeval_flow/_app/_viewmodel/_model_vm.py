"""ViewModel for the ModelModal.

Pure logic for model creation/editing.  No UI dependencies.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ModelViewModel"]

MODEL_TYPES = ["onnx", "bovw", "flatten", "torch", "uncertainty"]
_PATH_TYPES = frozenset({"onnx", "torch", "uncertainty"})


class ModelViewModel:
    """ViewModel for creating/editing model entries."""

    def __init__(self, existing: dict[str, Any] | None = None) -> None:
        self.existing = existing
        self.original: dict[str, Any] | None = dict(existing) if existing else None

    @property
    def is_edit_mode(self) -> bool:
        return self.existing is not None

    @staticmethod
    def needs_path(model_type: str) -> bool:
        """Return True if this model type requires a model_path."""
        return model_type in _PATH_TYPES

    @staticmethod
    def needs_vocab(model_type: str) -> bool:
        """Return True if this model type requires a vocab_size."""
        return model_type == "bovw"

    def build_result(
        self,
        name: str,
        model_type: str,
        model_path: str,
        vocab_size_str: str,
    ) -> dict[str, Any] | None:
        """Assemble the result dict. Returns None if invalid."""
        if not name or not model_type:
            return None
        result: dict[str, Any] = {"name": name, "type": model_type}
        if self.needs_path(model_type):
            if not model_path:
                return None
            result["model_path"] = model_path
        if self.needs_vocab(model_type) and vocab_size_str:
            try:
                result["vocab_size"] = int(vocab_size_str)
            except ValueError:
                return None
        return result

    def check_dirty(self, collected: dict[str, Any] | None) -> bool:
        """Return True if collected data differs from original."""
        if not collected:
            return False
        if not self.original:
            return True
        return collected != self.original

    @staticmethod
    def validation_message() -> str:
        """Return the error message for invalid input."""
        return "Name and type are required. Path required for onnx/torch/uncertainty."
