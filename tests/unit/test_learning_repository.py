from __future__ import annotations

import json
from pathlib import Path

from adaptive_3dbp.domain.models import ProblemInstance, SolverConfig
from adaptive_3dbp.experiments.runner import solve_instance
from adaptive_3dbp.persistence.sqlite import SQLiteExperimentRepository


ROOT = Path(__file__).resolve().parents[2]


def test_repository_observes_and_trains_online_gpr(tmp_path: Path) -> None:
    repository = SQLiteExperimentRepository(tmp_path / "experiments.db")
    instance = ProblemInstance.model_validate(json.loads((ROOT / "data/raw/example_20.json").read_text()))
    config = SolverConfig(max_evaluations=20, iterations_per_temperature=5, time_limit_seconds=5)

    for seed in (1, 2, 3):
        result = solve_instance(instance, config, seed=seed)
        run_id = repository.save_run(instance, config, result)
        metadata = repository.observe_and_train(run_id, instance, config, result, tmp_path / "models")

    assert metadata["trained"] is True
    assert repository.latest_surrogate_model() is not None
    assert len(repository.list_learning_observations()) == 3
