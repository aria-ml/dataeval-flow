"""TC-21-1 — hardware compatibility: a CUDA-less host runs the workflows on CPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.required

# Runs a torch-extractor pipeline in a fresh interpreter whose CUDA devices are hidden, and prints the outcome.
_SCRIPT = """
import json, sys
from pathlib import Path
import torch
import dataeval.config
from dataeval_flow import *
from dataeval_flow.preprocessing import PreprocessingStep
from verification.fixtures import write_image_folder

root = Path(sys.argv[1])
write_image_folder(root / "imgs", n_per_class=8, n_classes=2)
(root / "models").mkdir()
torch.save(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 4)), root / "models" / "tiny.pt")
cfg = PipelineConfig(
    datasets=[ImageFolderDatasetConfig(name="d", path="imgs", infer_labels=True)],
    sources=[SourceConfig(name="main", dataset="d")],
    preprocessors=[
        PreprocessorConfig(
            name="pp", steps=[PreprocessingStep(step="ToDtype", params={"dtype": "float32", "scale": True})]
        )
    ],
    # No `device`: the extractor takes DataEval's default, which must resolve to the CPU here.
    extractors=[TorchExtractorConfig(name="net", model_path="tiny.pt", preprocessor="pp", batch_size=4)],
    workflows=[
        DataCleaningWorkflowConfig(
            name="clean", type="data-cleaning", outlier_method="zscore", outlier_flags=["pixel"],
            outlier_cluster_threshold=3.0, outlier_cluster_algorithm="kmeans", outlier_n_clusters=2,
        )
    ],
    tasks=[TaskConfig(name="t", workflow="clean", sources="main", extractor="net")],
)
result = run_tasks(cfg, data_dir=root)[0]
print(json.dumps({
    "cuda_available": torch.cuda.is_available(),
    "device": dataeval.config.get_device().type,
    "success": result.success,
    "errors": list(result.errors),
}))
"""


class TestCudaFallback:
    def test_workflow_runs_on_cpu_when_cuda_is_hidden(self, tmp_path: Path) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(repo_root)}

        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _SCRIPT, str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

        assert proc.returncode == 0, proc.stderr
        outcome = json.loads(proc.stdout.strip().splitlines()[-1])
        assert outcome == {"cuda_available": False, "device": "cpu", "success": True, "errors": []}
