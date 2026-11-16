from __future__ import annotations

from typing import Protocol

from adaptive_3dbp.domain.models import ProblemInstance, SearchBudget, SolverConfig, SolverResult


class Metaheuristic(Protocol):
    name: str

    def solve(
        self,
        instance: ProblemInstance,
        config: SolverConfig,
        budget: SearchBudget,
        seed: int,
    ) -> SolverResult:
        ...

