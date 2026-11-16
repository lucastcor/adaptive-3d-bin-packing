from __future__ import annotations

import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from adaptive_3dbp.domain.bounds import simple_lower_bound
from adaptive_3dbp.domain.models import ConvergencePoint, ProblemInstance, SolverConfig, SolverResult
from adaptive_3dbp.domain.validation import validate_solution
from adaptive_3dbp.experiments.control import SolveCancelled, SolveControl
from adaptive_3dbp.experiments.runner import solve_instance
from adaptive_3dbp.metaheuristics.parallel_tempering.exchange import (
    exchange_adjacent_replicas,
    geometric_ladder,
    swap_acceptance_probability,
)
from adaptive_3dbp.metaheuristics.parallel_tempering.solver import ParallelTemperingSolver


ROOT = Path(__file__).resolve().parents[2]


def _multi_bin_instance() -> ProblemInstance:
    # 24 random boxes 20-60 in a 100^3 bin; with this seed the constructive
    # starts use more bins than the lower bound, so the replica loop runs.
    rng = random.Random(0)
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
        {"instance_id": "pt-multi-bin", "bin": {"length": 100, "width": 100, "height": 100}, "items": items}
    )


def _pt_config(**overrides) -> SolverConfig:
    values = {
        "algorithm": "parallel_tempering",
        "max_evaluations": 160,
        "time_limit_seconds": 120,
        "pt_replicas": 4,
        "pt_swap_interval": 3,
        "temperature_calibration_samples": 12,
    }
    values.update(overrides)
    return SolverConfig(**values)


def _swap_rates(result: SolverResult) -> list[float]:
    line = next(diagnostic for diagnostic in result.diagnostics if diagnostic.startswith("pt swap acceptance: "))
    return [float(value) for value in line.removeprefix("pt swap acceptance: ").split(",")]


@pytest.fixture(scope="module")
def multi_bin_run() -> tuple[ProblemInstance, SolverConfig, SolverResult]:
    instance = _multi_bin_instance()
    config = _pt_config()
    return instance, config, solve_instance(instance, config, seed=11)


def test_pt_returns_feasible_solution_on_example() -> None:
    instance = ProblemInstance.model_validate(json.loads((ROOT / "data/raw/example_20.json").read_text()))
    config = _pt_config(max_evaluations=120)
    result = solve_instance(instance, config, seed=7)
    assert result.solver == "parallel_tempering"
    assert result.status == "feasible", result.diagnostics
    assert result.best_found_bins >= result.lower_bound
    assert result.evaluations <= config.max_evaluations
    assert len(result.placements) == len(instance.items)


def test_pt_returns_feasible_solution_on_multi_bin_instance(multi_bin_run) -> None:
    instance, config, result = multi_bin_run
    assert result.status == "feasible", result.diagnostics
    assert not validate_solution(instance, result.placements)
    assert len(result.placements) == len(instance.items)
    assert result.best_found_bins >= result.lower_bound
    assert result.best_found_bins == len(result.bin_metrics)
    # Best-so-far trace is monotone in the lexicographic score's first key.
    bins_trace = [point.best_bins for point in result.convergence]
    assert bins_trace == sorted(bins_trace, reverse=True)
    assert result.convergence[-1].best_bins == result.best_found_bins


@pytest.mark.parametrize("max_evaluations", [1, 3, 9, 41])
def test_pt_respects_evaluation_budget_exactly(max_evaluations: int) -> None:
    instance = _multi_bin_instance()
    config = _pt_config(max_evaluations=max_evaluations)
    result = solve_instance(instance, config, seed=2)
    assert result.status == "feasible", result.diagnostics
    assert 1 <= result.evaluations <= max_evaluations
    if result.best_found_bins > result.lower_bound:
        # Starts, calibration samples and Metropolis steps all draw from one budget.
        assert result.evaluations == max_evaluations
    assert all(point.evaluation <= max_evaluations for point in result.convergence)


def test_pt_budget_is_used_in_full_on_multi_bin_instance(multi_bin_run) -> None:
    _instance, config, result = multi_bin_run
    assert result.evaluations <= config.max_evaluations
    if result.best_found_bins > result.lower_bound:
        assert result.evaluations == config.max_evaluations


def test_pt_is_deterministic_for_a_seed(multi_bin_run) -> None:
    instance, config, first = multi_bin_run
    second = solve_instance(instance, config, seed=11)

    def comparable(result: SolverResult) -> dict:
        payload = result.model_dump(exclude={"runtime_seconds", "convergence"})
        payload["convergence"] = [point.model_dump(exclude={"elapsed_seconds"}) for point in result.convergence]
        return payload

    assert comparable(first) == comparable(second)


def test_pt_reports_swap_acceptance_per_adjacent_pair(multi_bin_run) -> None:
    _instance, config, result = multi_bin_run
    rates = _swap_rates(result)
    assert len(rates) == config.pt_replicas - 1
    assert all(0.0 <= rate <= 1.0 for rate in rates)
    attempts_line = next(diagnostic for diagnostic in result.diagnostics if diagnostic.startswith("pt swap attempts: "))
    attempts = [int(value) for value in attempts_line.removeprefix("pt swap attempts: ").split(",")]
    if result.best_found_bins > result.lower_bound:
        # Calibration now collects a target number of usable deltas, so its
        # cost varies; even/odd pairs still alternate round by round, hence
        # every pair is tried and the counts differ by at most one.
        assert min(attempts) >= 1
        assert max(attempts) - min(attempts) <= 1


def test_swap_acceptance_matches_closed_form() -> None:
    # Cold replica (T=0.01) holds the worse configuration: always swap.
    assert swap_acceptance_probability(3.30, 3.10, 0.01, 0.1) == 1.0
    # Cold replica holds the better one: exp((E_i - E_j)(1/T_i - 1/T_j)).
    expected = math.exp((3.10 - 3.30) * (1 / 0.01 - 1 / 0.1))
    assert swap_acceptance_probability(3.10, 3.30, 0.01, 0.1) == pytest.approx(expected)
    assert expected == pytest.approx(math.exp(-18.0))
    # Symmetric in the pair labelling; equal energies or temperatures always swap.
    assert swap_acceptance_probability(3.30, 3.10, 0.1, 0.01) == pytest.approx(expected)
    assert swap_acceptance_probability(4.2, 4.2, 0.01, 0.1) == 1.0
    assert swap_acceptance_probability(4.2, 4.5, 0.05, 0.05) == 1.0


class _FixedUniform:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def test_two_replica_exchange_uses_the_closed_form() -> None:
    temperatures = [0.02, 0.04]
    cold, hot = SimpleNamespace(objective_scalar=2.10), SimpleNamespace(objective_scalar=2.12)
    probability = math.exp((2.10 - 2.12) * (1 / 0.02 - 1 / 0.04))  # exp(-0.5) ~ 0.607

    states, attempts, accepts = [cold, hot], [0], [0]
    exchange_adjacent_replicas(states, temperatures, 0, _FixedUniform(probability - 1e-6), attempts, accepts)
    assert states == [hot, cold] and attempts == [1] and accepts == [1]

    states, attempts, accepts = [cold, hot], [0], [0]
    exchange_adjacent_replicas(states, temperatures, 0, _FixedUniform(probability + 1e-6), attempts, accepts)
    assert states == [cold, hot] and attempts == [1] and accepts == [0]

    # With K=2 the only pair is even: odd rounds attempt nothing.
    exchange_adjacent_replicas(states, temperatures, 1, _FixedUniform(0.0), attempts, accepts)
    assert attempts == [1]


def test_geometric_ladder_spans_t_min_to_t_max() -> None:
    ladder = geometric_ladder(0.05, 0.01, 5)
    assert ladder[0] == pytest.approx(0.0005)
    assert ladder[-1] == pytest.approx(0.05)
    ratios = [high / low for low, high in zip(ladder, ladder[1:])]
    assert all(ratio == pytest.approx(ratios[0]) for ratio in ratios)
    assert ratios[0] > 1.0


def test_pt_stops_at_the_lower_bound() -> None:
    # Eight 50^3 cubes fill one 100^3 bin exactly: the first start meets the
    # bound, so neither calibration nor the replica loop spends evaluations.
    instance = ProblemInstance.model_validate(
        {
            "instance_id": "pt-cubes",
            "bin": {"length": 100, "width": 100, "height": 100},
            "items": [{"id": f"c{index}", "length": 50, "width": 50, "height": 50} for index in range(8)],
        }
    )
    config = _pt_config(max_evaluations=500)
    result = solve_instance(instance, config, seed=0)
    assert result.status == "feasible", result.diagnostics
    assert result.best_found_bins == result.lower_bound == 1
    assert result.optimal_proven
    assert result.evaluations == 1


def test_pt_honours_cancel_and_reports_progress() -> None:
    instance = _multi_bin_instance()
    config = _pt_config(max_evaluations=40, initial_temperature=0.05)
    points: list[ConvergencePoint] = []
    ParallelTemperingSolver().solve(instance, config, seed=4, progress_callback=points.append)
    assert points
    assert [point.evaluation for point in points] == sorted(point.evaluation for point in points)
    assert points[-1].evaluation <= config.max_evaluations

    # The checkpoint sits in the replica loop, which only runs above the bound.
    assert points[0].best_bins > simple_lower_bound(instance)
    control = SolveControl()
    control.cancel()
    with pytest.raises(SolveCancelled):
        ParallelTemperingSolver().solve(instance, config, seed=4, control=control)


def test_runner_skips_native_core_for_pt() -> None:
    instance = ProblemInstance.model_validate(json.loads((ROOT / "data/raw/example_20.json").read_text()))
    result = solve_instance(instance, _pt_config(max_evaluations=20, use_native_core=True), seed=1)
    assert result.solver == "parallel_tempering"
    assert "parallel tempering is Python-only; native core skipped" in result.diagnostics


def test_pt_config_fields_validate_and_round_trip() -> None:
    config = _pt_config(pt_replicas=6, pt_swap_interval=7, pt_min_temperature_ratio=0.05)
    restored = SolverConfig.model_validate_json(config.model_dump_json())
    assert restored == config
    assert SolverConfig().algorithm == "simulated_annealing"
    for invalid in ({"pt_replicas": 1}, {"pt_swap_interval": 0}, {"pt_min_temperature_ratio": 1.0}, {"algorithm": "tabu"}):
        with pytest.raises(ValidationError):
            SolverConfig(**invalid)
