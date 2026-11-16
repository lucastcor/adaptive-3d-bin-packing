from __future__ import annotations

from typing import Protocol

from adaptive_3dbp.domain.models import CandidateSolution, DecodedSolution, ProblemInstance


class PackingDecoder(Protocol):
    name: str

    def decode(
        self,
        instance: ProblemInstance,
        candidate: CandidateSolution,
        policy_id: str = "min_z_y_x",
    ) -> DecodedSolution:
        ...

