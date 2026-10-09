"""TC-34-1 (NFR-4) — the pipeline seed makes stochastic runs reproducible."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import load_config, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.config import PipelineConfig
from verification.nonfunctional._support import stable, write_project

pytestmark = pytest.mark.required

# A view that shuffles the images and keeps eight: which eight depends on the seed.
SHUFFLED = [{"type": "Shuffle"}, {"type": "Limit", "params": {"size": 8}}]


@pytest.fixture(autouse=True)
def _fresh_caches() -> Iterator[None]:
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


@pytest.fixture(autouse=True)
def _unseeded_dataeval() -> Iterator[None]:
    """Leave DataEval's global seed and PyTorch's deterministic switch as the next test finds them."""
    import torch
    from dataeval.config import set_seed

    set_seed(None)
    yield
    set_seed(None)
    torch.use_deterministic_algorithms(False)


def _run(root: Path, *, seed: int | None, n_tasks: int = 1, deterministic: bool = False) -> dict[str, Any]:
    config = write_project(root, seed=seed, n_tasks=n_tasks, view=SHUFFLED)
    pipeline = load_config(config).model_copy(update={"deterministic": deterministic})
    DatasetCache.clear_instances()
    return run_tasks(pipeline, data_dir=root)


class TestSeedConfiguration:
    def test_seed_defaults_to_none(self) -> None:
        """An unseeded pipeline leaves randomness alone."""
        assert PipelineConfig().seed is None
        assert PipelineConfig().deterministic is False

    def test_seed_is_applied_to_dataeval(self, tmp_path: Path) -> None:
        """Running a seeded pipeline pins DataEval's seed configuration."""
        from dataeval.config import get_seed

        assert get_seed() is None

        results = _run(tmp_path, seed=1234)

        assert results["clean_task"].success
        assert get_seed() == 1234

    def test_deterministic_forces_pytorch_deterministic_algorithms(self, tmp_path: Path) -> None:
        import torch

        assert not torch.are_deterministic_algorithms_enabled()
        _run(tmp_path / "off", seed=3)
        assert not torch.are_deterministic_algorithms_enabled()

        _run(tmp_path / "on", seed=3, deterministic=True)

        assert torch.are_deterministic_algorithms_enabled()

    def test_seed_recorded_in_result_envelope(self, tmp_path: Path) -> None:
        """The seed is part of the provenance, so the envelope alone can repeat the run."""
        result = _run(tmp_path, seed=7)["clean_task"]

        resolved = result.metadata.resolved_config
        assert resolved["seed"] == 7
        assert resolved["deterministic"] is False

    def test_unseeded_run_records_no_seed(self, tmp_path: Path) -> None:
        """An unseeded run must not claim a seed it never applied."""
        result = _run(tmp_path, seed=None)["clean_task"]

        assert "seed" not in result.metadata.resolved_config
        assert "deterministic" not in result.metadata.resolved_config

    def test_same_seed_reproduces_stochastic_output(self, tmp_path: Path) -> None:
        """Two seeded runs agree; different seeds do not all agree, so the seed is what fixes the output."""
        outputs = {
            (seed, attempt): stable(_run(tmp_path / f"{seed}-{attempt}", seed=seed)["clean_task"].to_dict())
            for seed in (1, 2, 3)
            for attempt in (1, 2)
        }

        for seed in (1, 2, 3):
            assert outputs[seed, 1] == outputs[seed, 2], f"seed {seed} gave two different results"
        assert len({repr(outputs[seed, 1]) for seed in (1, 2, 3)}) > 1

    def test_seed_is_applied_per_task_independent_of_task_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A task gives the same result alone as after other tasks, because the seed is reapplied for each one."""
        import dataeval.config

        seeds: list[int | None] = []
        real_set_seed = dataeval.config.set_seed

        def spy(seed: int | None, *args: object, **kwargs: object) -> None:
            seeds.append(seed)
            real_set_seed(seed, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(dataeval.config, "set_seed", spy)
        config = write_project(tmp_path, seed=5, n_tasks=2, view=SHUFFLED)
        pipeline = load_config(config)

        DatasetCache.clear_instances()
        alone = run_tasks(pipeline, "clean_task_2", data_dir=tmp_path)
        assert seeds == [5]

        seeds.clear()
        DatasetCache.clear_instances()
        together = run_tasks(pipeline, data_dir=tmp_path)
        assert seeds == [5, 5]  # once per task, not once per pipeline

        assert alone["clean_task_2"].success
        assert together["clean_task_2"].success
        assert stable(alone["clean_task_2"].to_dict()) == stable(together["clean_task_2"].to_dict())
