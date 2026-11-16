from __future__ import annotations

import json
import random
from pathlib import Path

from adaptive_3dbp.constructors.pool import volume_desc_candidate
from adaptive_3dbp.decoders.extreme_points import ExtremePointDecoder
from adaptive_3dbp.domain.models import ProblemInstance, SolverConfig
from adaptive_3dbp.experiments.runner import solve_instance
from adaptive_3dbp.metaheuristics.simulated_annealing.schedule import (
    CALIBRATION_MAX_FACTOR,
    FALLBACK_TEMPERATURE,
    acceptance_probability,
    calibrate_initial_temperature,
)


ROOT = Path(__file__).resolve().parents[2]


def test_sa_returns_feasible_solution() -> None:
    instance = ProblemInstance.model_validate(json.loads((ROOT / "data/raw/example_20.json").read_text()))
    config = SolverConfig(max_evaluations=120, iterations_per_temperature=20, time_limit_seconds=5)
    result = solve_instance(instance, config, seed=7)
    assert result.status == "feasible", result.diagnostics
    assert result.best_found_bins >= result.lower_bound
    assert result.evaluations <= config.max_evaluations
    assert len(result.placements) == len(instance.items)


def test_temperature_calibration_counts_against_evaluation_budget() -> None:
    instance = ProblemInstance.model_validate(json.loads((ROOT / "data/raw/example_20.json").read_text()))
    config = SolverConfig(
        max_evaluations=2,
        temperature_calibration_samples=50,
        iterations_per_temperature=20,
        time_limit_seconds=5,
        multi_start_count=1,
    )
    result = solve_instance(instance, config, seed=7)
    assert result.status == "feasible", result.diagnostics
    assert 1 <= result.evaluations <= config.max_evaluations


def test_sa_parallel_process_pool_returns_feasible_solution() -> None:
    instance = ProblemInstance.model_validate(json.loads((ROOT / "data/raw/example_20.json").read_text()))
    config = SolverConfig(
        max_evaluations=120,
        iterations_per_temperature=20,
        time_limit_seconds=5,
        multi_start_count=4,
        parallel_mode="process_pool",
        parallel_workers=2,
    )
    result = solve_instance(instance, config, seed=13)
    assert result.status == "feasible", result.diagnostics
    assert result.best_found_bins >= result.lower_bound
    assert len(result.placements) == len(instance.items)
    assert any("parallel process pool" in diagnostic for diagnostic in result.diagnostics)


def test_temperature_calibration_can_be_disabled() -> None:
    instance = ProblemInstance.model_validate(json.loads((ROOT / "data/raw/example_20.json").read_text()))
    config = SolverConfig(temperature_calibration_samples=0)
    decoder = ExtremePointDecoder()
    calibration = calibrate_initial_temperature(
        instance,
        decoder,
        decoder.decode(instance, volume_desc_candidate(instance)),
        config,
        random.Random(3),
    )
    assert calibration.temperature == FALLBACK_TEMPERATURE
    assert calibration.fallback
    assert calibration.evaluations == 0


def _multi_bin_instance() -> ProblemInstance:
    rng = random.Random(5)
    items = [
        {
            "id": f"i{index}",
            "length": rng.randint(20, 60),
            "width": rng.randint(20, 60),
            "height": rng.randint(20, 60),
        }
        for index in range(24)
    ]
    return ProblemInstance.model_validate(
        {"instance_id": "multi-bin", "bin": {"length": 100, "width": 100, "height": 100}, "items": items}
    )


def test_temperature_calibration_ignores_bin_count_changes() -> None:
    # Regression: T0 used to average deltas that opened a bin (~1 in E = b + r)
    # with within-level deltas (~1e-3), so one crossing sample made the search
    # near-random. Within-level calibration keeps T0 far below one bin.
    instance = _multi_bin_instance()
    decoder = ExtremePointDecoder()
    start = decoder.decode(instance, volume_desc_candidate(instance))
    for seed in range(5):
        calibration = calibrate_initial_temperature(
            instance, decoder, start, SolverConfig(temperature_calibration_samples=30), random.Random(seed)
        )
        # Collects 30 usable deltas, spending at most 4x that many decodes.
        assert calibration.uphill_samples == 30 or calibration.evaluations == 30 * CALIBRATION_MAX_FACTOR
        assert calibration.evaluations <= 30 * CALIBRATION_MAX_FACTOR
        assert 0 < calibration.temperature < 0.2
        assert acceptance_probability(1.0, calibration.temperature) < 1e-2


def test_stop_temperature_is_relative_to_t0() -> None:
    # Regression: an absolute min_temperature (0.001) was above the calibrated
    # T0 on large instances (T0 shrinks like 1/bins), so starts stopped right
    # after calibration and left most of the budget unused.
    instance = _multi_bin_instance()
    config = SolverConfig(
        initial_temperature=5e-4,
        max_evaluations=400,
        multi_start_count=1,
        stagnation_limit=10_000,
        time_limit_seconds=60,
    )
    result = solve_instance(instance, config, seed=1)
    if result.best_found_bins > result.lower_bound:
        assert result.evaluations == config.max_evaluations


def test_sequential_multi_start_shares_the_budget() -> None:
    instance = _multi_bin_instance()
    config = SolverConfig(
        max_evaluations=300,
        multi_start_count=3,
        stagnation_limit=10_000,
        min_temperature=1e-12,
        time_limit_seconds=60,
    )
    result = solve_instance(instance, config, seed=3)
    assert result.status == "feasible", result.diagnostics
    if result.best_found_bins > result.lower_bound:
        # Starts are evaluation-offset in the convergence trace: with an equal
        # split each start begins at most ~1/3 of the budget after the previous.
        starts = sorted({point.evaluation for point in result.convergence if point.delta == 0.0 and point.improved})
        assert len(starts) == 3
        assert starts[1] <= 101 + config.temperature_calibration_samples
