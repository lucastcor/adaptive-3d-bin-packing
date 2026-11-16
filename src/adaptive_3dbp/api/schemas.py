from __future__ import annotations

from pydantic import BaseModel, Field

from adaptive_3dbp.domain.models import ProblemInstance, SolverConfig, SolverResult


class SolveRequest(BaseModel):
    instance: ProblemInstance
    config: SolverConfig = Field(default_factory=SolverConfig)
    seed: int = 123
    persist: bool = True


class SolveResponse(BaseModel):
    run_id: int | None
    result: object


class ResultExportRequest(BaseModel):
    instance: ProblemInstance
    result: SolverResult
    run_id: int | None = None
