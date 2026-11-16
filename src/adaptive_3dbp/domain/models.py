from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, field_validator, model_validator


RotationSpec = Literal["all_orthogonal", "fixed"]


class BinSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    length: PositiveFloat
    width: PositiveFloat
    height: PositiveFloat
    max_weight: PositiveFloat | None = None

    @property
    def volume(self) -> float:
        return float(self.length * self.width * self.height)


class ItemSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    length: PositiveFloat
    width: PositiveFloat
    height: PositiveFloat
    weight: PositiveFloat | None = None
    allowed_rotations: RotationSpec | list[int] = "all_orthogonal"

    @property
    def volume(self) -> float:
        return float(self.length * self.width * self.height)

    @property
    def dimensions(self) -> tuple[float, float, float]:
        return (float(self.length), float(self.width), float(self.height))

    @field_validator("allowed_rotations")
    @classmethod
    def validate_allowed_rotations(cls, value: RotationSpec | list[int]) -> RotationSpec | list[int]:
        if isinstance(value, list):
            invalid = [rotation for rotation in value if rotation < 0 or rotation > 5]
            if invalid:
                raise ValueError(f"rotation ids must be between 0 and 5: {invalid}")
        return value


class MinDistanceConstraint(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id_a: str = Field(min_length=1)
    item_id_b: str = Field(min_length=1)
    distance: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_distinct_items(self) -> "MinDistanceConstraint":
        if self.item_id_a == self.item_id_b:
            raise ValueError("minimum distance constraint must reference two distinct items")
        return self


class PackingConstraints(BaseModel):
    model_config = ConfigDict(frozen=True)

    orthogonal_only: bool = True
    stability_required: bool = False
    support_ratio_min: float | None = Field(default=None, ge=0.0, le=1.0)
    min_distance: list[MinDistanceConstraint] = Field(default_factory=list)


class ProblemInstance(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str = Field(min_length=1)
    problem_type: str = "offline_3d_bpp"
    bin: BinSpec
    items: list[ItemSpec] = Field(min_length=1)
    constraints: PackingConstraints = Field(default_factory=PackingConstraints)
    best_known_bins: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_unique_item_ids(self) -> "ProblemInstance":
        ids = [item.id for item in self.items]
        duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
        if duplicates:
            raise ValueError(f"duplicated item ids: {duplicates}")
        known_ids = set(ids)
        invalid_rules = [
            f"{rule.item_id_a}<->{rule.item_id_b}"
            for rule in self.constraints.min_distance
            if rule.item_id_a not in known_ids or rule.item_id_b not in known_ids
        ]
        if invalid_rules:
            raise ValueError(f"minimum distance constraints reference unknown item ids: {invalid_rules}")
        return self

    @property
    def total_item_volume(self) -> float:
        return float(sum(item.volume for item in self.items))


class Placement(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    item_index: int = Field(ge=0)
    bin_id: int = Field(ge=0)
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    z: float = Field(ge=0)
    rotation_id: int = Field(ge=0, le=5)
    placed_length: PositiveFloat
    placed_width: PositiveFloat
    placed_height: PositiveFloat

    @property
    def volume(self) -> float:
        return float(self.placed_length * self.placed_width * self.placed_height)

    @property
    def min_corner(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def max_corner(self) -> tuple[float, float, float]:
        return (
            self.x + float(self.placed_length),
            self.y + float(self.placed_width),
            self.z + float(self.placed_height),
        )


class BinMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    bin_id: int
    utilization: float
    unused_volume: float
    occupied_volume: float
    compactness: float


class SolutionMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    bins_used: int
    lower_bound: int
    best_known_bins: int | None = None
    gap_to_lower_bound: float
    gap_to_best_known: float | None = None
    unused_volume: float
    utilization_mean: float
    fragmentation: float
    compactness_mean: float
    objective_scalar: float


class ConvergencePoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    evaluation: int
    elapsed_seconds: float
    bins_used: int
    objective_scalar: float
    accepted: bool
    temperature: float
    current_bins: int | None = None
    current_objective_scalar: float | None = None
    best_bins: int | None = None
    best_objective_scalar: float | None = None
    delta: float | None = None
    improved: bool = False


class SolverConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: Literal["simulated_annealing", "parallel_tempering"] = "simulated_annealing"
    decoder: str = "extreme_points"
    configuration_id: str = "sa-default"
    placement_policy_id: str = "min_z_y_x"
    initial_temperature: float | None = Field(default=None, gt=0)
    target_initial_acceptance: float = Field(default=0.82, gt=0, lt=1)
    temperature_calibration_samples: int = Field(default=24, ge=0, le=500)
    cooling_rate: float = Field(default=0.96, gt=0, lt=1)
    iterations_per_temperature: int = Field(default=80, ge=1)
    # Stop temperature = max(min_temperature, min_temperature_ratio * T0).
    # The calibrated T0 shrinks with the bin count (r is a mean over bins), so
    # an absolute floor made large instances stop right after calibration;
    # the relative floor gives every instance the same number of cooling
    # levels. min_temperature is only an absolute safety floor.
    min_temperature: float = Field(default=1e-9, gt=0)
    min_temperature_ratio: float = Field(default=1e-3, gt=0, lt=1)
    max_evaluations: int = Field(default=2500, ge=1)
    time_limit_seconds: float = Field(default=10.0, gt=0)
    stagnation_limit: int = Field(default=600, ge=1)
    reheating_factor: float = Field(default=1.35, ge=1.0)
    multi_start_count: int = Field(default=3, ge=1)
    ruin_size: int = Field(default=4, ge=1)
    use_native_core: bool = False
    parallel_mode: Literal["sequential", "process_pool", "cooperative_process_pool"] = "sequential"
    parallel_workers: int = Field(default=1, ge=1)
    cooperative_sync_interval: int = Field(default=25, ge=1)
    heartbeat_timeout_seconds: float = Field(default=15.0, gt=0)
    convergence_sample_rate: int = Field(default=1, ge=1)
    # Parallel tempering only (algorithm="parallel_tempering"): replica count,
    # Metropolis steps per replica between swap rounds, and T_min / T_max of
    # the geometric ladder (T_max is calibrated like the SA's T0).
    pt_replicas: int = Field(default=8, ge=2)
    pt_swap_interval: int = Field(default=10, ge=1)
    pt_min_temperature_ratio: float = Field(default=0.01, gt=0, lt=1)
    # Optional hottest temperature in "bins" (E = bins + r). The calibrated T0
    # keeps every replica inside its bin-count level; e.g. 0.3 lets the
    # hottest replica open a bin with p = exp(-1/0.3) ~ 3.6%. None = T0.
    pt_max_temperature: float | None = Field(default=None, gt=0)
    operator_probabilities: dict[str, float] = Field(
        default_factory=lambda: {
            "swap": 0.32,
            "relocate": 0.24,
            "reverse": 0.14,
            "rotate": 0.20,
            "ruin_recreate": 0.10,
        }
    )

    @field_validator("operator_probabilities")
    @classmethod
    def validate_operator_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("at least one neighborhood operator must be enabled")
        if any(probability < 0 for probability in value.values()):
            raise ValueError("operator probabilities must be non-negative")
        total = sum(value.values())
        if total <= 0:
            raise ValueError("operator probabilities must have positive total mass")
        return value


class SearchBudget(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_evaluations: int = Field(default=2500, ge=1)
    timeout_seconds: float = Field(default=10.0, gt=0)


class SolverResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    solver: str
    solver_version: str
    configuration_id: str
    seed: int
    status: Literal["feasible", "infeasible", "timeout", "error"]
    best_found_bins: int
    lower_bound: int
    best_known_bins: int | None = None
    optimal_proven: bool = False
    gap_to_best_known: float | None = None
    gap_to_lower_bound: float
    runtime_seconds: float
    evaluations: int
    objective_scalar: float
    placements: list[Placement]
    bin_metrics: list[BinMetrics]
    convergence: list[ConvergencePoint] = Field(default_factory=list)
    features: dict[str, float] = Field(default_factory=dict)
    diagnostics: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)

    def as_experiment_record(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "solver": self.solver,
            "solver_version": self.solver_version,
            "configuration_id": self.configuration_id,
            "seed": self.seed,
            "status": self.status,
            "best_found_bins": self.best_found_bins,
            "lower_bound": self.lower_bound,
            "best_known_bins": self.best_known_bins,
            "gap_to_lower_bound": self.gap_to_lower_bound,
            "gap_to_best_known": self.gap_to_best_known,
            "runtime_seconds": self.runtime_seconds,
            "evaluations": self.evaluations,
            "objective_scalar": self.objective_scalar,
        }


@dataclass(frozen=True, slots=True)
class CandidateSolution:
    item_order: tuple[int, ...]
    rotations: tuple[int, ...]
    placement_policy_id: str = "min_z_y_x"


@dataclass(frozen=True, slots=True)
class DecodedSolution:
    candidate: CandidateSolution
    placements: tuple[Placement, ...]
    bin_metrics: tuple[BinMetrics, ...]
    score: tuple[float, float, float, float]
    objective_scalar: float
    feasible: bool
    diagnostics: tuple[str, ...] = ()
