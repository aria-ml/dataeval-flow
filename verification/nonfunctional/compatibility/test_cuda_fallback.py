"""TC-31-1 (NFR-1) — hardware compatibility: a CUDA-less host runs the pipelines on the CPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from verification.helpers import REPO_ROOT

pytestmark = pytest.mark.required

# Runs a torch-extractor pipeline in a fresh interpreter whose CUDA devices are hidden, and prints the outcome.
_SCRIPT = """
import json, sys
from pathlib import Path
import torch
from dataeval_flow import run_tasks, set_device
from dataeval_flow.config import PipelineConfig
from verification.fixtures import write_image_folder

root = Path(sys.argv[1])
write_image_folder(root / "imgs", n_per_class=8, n_classes=2, size=8)
(root / "models").mkdir()
torch.save(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 4)), root / "models" / "tiny.pt")
cfg = PipelineConfig.model_validate({
    "datasets": [{"name": "d", "format": "image_folder", "path": "imgs", "infer_labels": True}],
    "sources": [{"name": "main", "dataset": "d"}],
    "preprocessors": [{"name": "pp", "steps": [{"step": "ToDtype", "params": {"dtype": "float32", "scale": True}}]}],
    # The config names no device: the machine decides, and with CUDA hidden it must be the CPU.
    "extractors": [{"name": "net", "model": "torch", "model_path": "tiny.pt", "preprocessor": "pp", "batch_size": 4}],
    "workflows": [{
        "name": "clean", "type": "quality",
        "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore",
                     "cluster_threshold": 3.0, "cluster_algorithm": "kmeans", "n_clusters": 2},
    }],
    "tasks": [{"name": "t", "workflow": "clean", "sources": "main", "extractor": "net"}],
})
result = run_tasks(cfg, data_dir=root)["t"]
try:
    set_device("cuda")
    refused = None
except ValueError as error:
    refused = str(error)
print(json.dumps({
    "cuda_available": torch.cuda.is_available(),
    "device": result.metadata.device,
    "success": result.success,
    "errors": list(result.errors),
    "refused": refused,
}))
"""


class TestCudaFallback:
    def test_workflow_runs_on_cpu_when_cuda_is_hidden(self, tmp_path: Path) -> None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO_ROOT)}

        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-W", "ignore", "-c", _SCRIPT, str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

        assert proc.returncode == 0, proc.stderr
        outcome = json.loads(proc.stdout.strip().splitlines()[-1])
        assert outcome["refused"] is not None
        assert "PyTorch sees no `cuda`" in outcome["refused"]
        assert {k: v for k, v in outcome.items() if k != "refused"} == {
            "cuda_available": False,
            "device": "cpu",
            "success": True,
            "errors": [],
        }
