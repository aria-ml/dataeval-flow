"""The device every tool computes on: the pipeline's `device`, else CUDA when torch sees it, else CPU (decision 25)."""

import pytest
import torch
from dataeval.config import get_device, set_device

from dataeval_flow import PipelineConfig
from dataeval_flow._orchestrator import _apply_device
from dataeval_flow.config.extractors import TorchExtractorConfig


@pytest.fixture(autouse=True)
def _restore_device():
    yield
    set_device(None)


def test_the_configured_device_is_dataevals():
    _apply_device(PipelineConfig(device="cpu"))
    assert get_device() == torch.device("cpu")


def test_unset_picks_cuda_when_torch_sees_it(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _apply_device(PipelineConfig())
    assert get_device() == torch.device("cuda")


def test_unset_picks_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _apply_device(PipelineConfig())
    assert get_device() == torch.device("cpu")


def test_an_invalid_device_names_itself():
    with pytest.raises(ValueError, match="gpu0"):
        _apply_device(PipelineConfig(device="gpu0"))


def test_no_extractor_takes_a_device():
    with pytest.raises(ValueError, match="device"):
        TorchExtractorConfig.model_validate({"name": "t", "model_path": "m.pt", "device": "cpu"})


def test_each_task_applies_the_device(monkeypatch: pytest.MonkeyPatch):
    from dataeval_flow import run_task
    from dataeval_flow.config import TaskConfig
    from dataeval_flow.evaluators.quality import DuplicatesConfig
    from tests.evaluator_toys import toy_pipeline

    seen: list[str | None] = []
    monkeypatch.setattr("dataeval_flow._orchestrator._apply_device", lambda config: seen.append(config.device))
    pipeline = toy_pipeline(evaluators=[DuplicatesConfig(name="dupes")]).model_copy(update={"device": "cpu"})
    run_task(TaskConfig(name="t", workflow="dupes", kind="evaluator", sources="src"), pipeline)
    assert seen == ["cpu"]
