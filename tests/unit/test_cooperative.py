from __future__ import annotations

from adaptive_3dbp.domain.models import (
    BinSpec,
    CandidateSolution,
    ItemSpec,
    ProblemInstance,
    SearchBudget,
    SolverConfig,
)
from adaptive_3dbp.metaheuristics.simulated_annealing.cooperative import CooperativeChannel
from adaptive_3dbp.metaheuristics.simulated_annealing.solver import SimulatedAnnealingSolver


class _FakeLock:
    def __enter__(self) -> "_FakeLock":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _candidate() -> CandidateSolution:
    return CandidateSolution(item_order=(0, 1, 2), rotations=(0, 0, 0))


def test_channel_publish_and_fetch_with_logical_version() -> None:
    state: dict = {"version": 0}
    channel_a = CooperativeChannel(state, _FakeLock(), worker_id=0, sync_interval=10)
    channel_b = CooperativeChannel(state, _FakeLock(), worker_id=1, sync_interval=10)

    assert channel_a.publish((3.0, 1.0, 0.0, 0.0), _candidate()) is True
    assert state["version"] == 1
    # Worse score must not overwrite the central memory.
    assert channel_b.publish((4.0, 0.0, 0.0, 0.0), _candidate()) is False
    assert state["version"] == 1

    fetched = channel_b.fetch_better_than((5.0, 0.0, 0.0, 0.0))
    assert fetched is not None
    assert fetched.item_order == (0, 1, 2)
    # Same version seen twice -> no new knowledge.
    assert channel_b.fetch_better_than((5.0, 0.0, 0.0, 0.0)) is None
    # Shared best not better than local -> nothing to adopt.
    channel_a.publish((2.0, 0.0, 0.0, 0.0), _candidate())
    assert channel_b.fetch_better_than((1.0, 0.0, 0.0, 0.0)) is None


def test_cooperative_mode_solves_and_reports_diagnostics() -> None:
    instance = ProblemInstance(
        instance_id="coop-test",
        bin=BinSpec(length=10, width=10, height=10),
        items=[
            ItemSpec(id="a", length=9, width=2, height=2),
            ItemSpec(id="b", length=5, width=5, height=4),
            ItemSpec(id="c", length=4, width=6, height=3),
            ItemSpec(id="d", length=3, width=3, height=8),
            ItemSpec(id="e", length=7, width=2, height=5),
            ItemSpec(id="f", length=2, width=8, height=2),
        ],
    )
    config = SolverConfig(
        parallel_mode="cooperative_process_pool",
        parallel_workers=2,
        multi_start_count=4,
        max_evaluations=240,
        time_limit_seconds=20.0,
        cooperative_sync_interval=10,
    )
    result = SimulatedAnnealingSolver().solve(
        instance=instance,
        config=config,
        budget=SearchBudget(max_evaluations=240, timeout_seconds=20.0),
        seed=11,
    )
    assert result.status == "feasible"
    assert result.best_found_bins >= result.lower_bound
    assert any("cooperative process pool used" in line for line in result.diagnostics)
