"""The device every tool computes on: the one chosen with `set_device`, else CUDA when torch sees it, else CPU.

The machine decides it, never the config: a pipeline names no device, and each result records the one it ran on.
"""

import pytest
import torch
from dataeval.config import get_device

from dataeval_flow import PipelineConfig, set_device
from dataeval_flow._orchestrator import _apply_device
from dataeval_flow.config.extractors import TorchExtractorConfig


def _gpus(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "Toy GPU")


def test_the_chosen_device_is_dataevals():
    set_device("cpu")
    assert _apply_device() == "cpu"
    assert get_device() == torch.device("cpu")


def test_unset_picks_cuda_when_torch_sees_it(monkeypatch: pytest.MonkeyPatch):
    _gpus(monkeypatch, 1)
    set_device(None)
    assert _apply_device() == "cuda:0 (Toy GPU)"
    assert get_device() == torch.device("cuda")


def test_unset_picks_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch):
    _gpus(monkeypatch, 0)
    set_device(None)
    assert _apply_device() == "cpu"


def test_a_chosen_gpu_is_named_by_its_index(monkeypatch: pytest.MonkeyPatch):
    _gpus(monkeypatch, 2)
    set_device("cuda:1")
    assert _apply_device() == "cuda:1 (Toy GPU)"


def test_an_invalid_device_names_itself():
    with pytest.raises(ValueError, match="gpu0"):
        set_device("gpu0")


@pytest.mark.parametrize(("count", "device"), [(0, "cuda"), (1, "cuda:1")])
def test_a_gpu_torch_cannot_see_is_refused_when_chosen(monkeypatch: pytest.MonkeyPatch, count: int, device: str):
    _gpus(monkeypatch, count)
    with pytest.raises(ValueError, match=f"sees no `{device}`"):
        set_device(device)


def test_the_pipeline_takes_no_device():
    with pytest.raises(ValueError, match="device"):
        PipelineConfig.model_validate({"device": "cpu"})


def test_no_extractor_takes_a_device():
    with pytest.raises(ValueError, match="device"):
        TorchExtractorConfig.model_validate({"name": "t", "model_path": "m.pt", "device": "cpu"})


def test_each_result_records_its_device():
    from dataeval_flow import run_task
    from dataeval_flow.config import TaskConfig
    from dataeval_flow.evaluators.quality import DuplicatesConfig
    from tests.evaluator_toys import toy_pipeline

    pipeline = toy_pipeline(evaluators=[DuplicatesConfig(name="dupes")])
    result = run_task(TaskConfig(name="t", workflow="dupes", kind="evaluator", sources="src"), pipeline)
    assert result.metadata.device == "cpu"
