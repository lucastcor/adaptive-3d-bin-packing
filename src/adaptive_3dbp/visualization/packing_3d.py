from __future__ import annotations

from adaptive_3dbp.domain.models import ProblemInstance, SolverResult


def solution_scene(instance: ProblemInstance, result: SolverResult) -> dict[str, object]:
    return {
        "bin": instance.bin.model_dump(),
        "placements": [placement.model_dump() for placement in result.placements],
        "metrics": [metric.model_dump() for metric in result.bin_metrics],
    }

