from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from adaptive_3dbp.decoders.base import PackingDecoder
from adaptive_3dbp.domain.models import DecodedSolution, ProblemInstance, SolverConfig
from adaptive_3dbp.domain.scoring import lexicographic_less
from adaptive_3dbp.metaheuristics.simulated_annealing.operators import propose_neighbor


# Used (divided by the start's bin count) when calibration is disabled or sees
# no uphill move inside a bin-count level. Same order of magnitude as
# calibrated values on the campaign instances (5-10 bins, mean intra-level
# uphill delta ~0.002-0.004, so T0 ~0.01-0.02 at 82% target acceptance). The
# energy is dimensionless (E = bins + r): no dependence on the bin volume.
FALLBACK_TEMPERATURE = 0.05
CALIBRATION_MAX_FACTOR = 4


@dataclass(frozen=True, slots=True)
class TemperatureCalibration:
    temperature: float
    evaluations: int
    best: DecodedSolution
    uphill_samples: int
    fallback: bool


def acceptance_probability(delta: float, temperature: float) -> float:
    if delta <= 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta / temperature)


def calibrate_initial_temperature(
    instance: ProblemInstance,
    decoder: PackingDecoder,
    start: DecodedSolution,
    config: SolverConfig,
    rng: random.Random,
    max_samples: int | None = None,
    deadline: float | None = None,
) -> TemperatureCalibration:
    """Pick T0 so that a typical uphill move is accepted with p0.

    T0 = -mean(delta+) / ln(p0), with delta+ measured on single transitions
    of a short random walk (Ben-Ameur style). Only transitions that keep the
    bin count are used. Mixing in transitions that open a bin (delta ~ 1 in
    the dimensionless energy) made T0 bimodal: whether one of the few samples
    crossed a bin-count level decided between a near-greedy and a near-random
    search. Calibrating on the within-level gradient keeps the bin count
    lexicographically dominant (opening a bin is accepted with ~exp(-1/T0)).

    ``temperature_calibration_samples`` is the number of usable (uphill,
    same-bin-count) deltas to collect. Only ~10-30% of steps qualify, so the
    walk may take up to CALIBRATION_MAX_FACTOR times that many steps; with a
    fixed step count T0 varied 3-4x between seeds. Every step is a decode and
    counts against the evaluation budget (``max_samples`` caps it).
    """
    if config.initial_temperature is not None:
        return TemperatureCalibration(config.initial_temperature, 0, start, 0, fallback=False)
    wanted = config.temperature_calibration_samples
    max_steps = wanted * CALIBRATION_MAX_FACTOR
    if max_samples is not None:
        max_steps = min(max_steps, max(0, max_samples))

    best = start
    previous = start
    uphill: list[float] = []
    evaluations = 0
    while evaluations < max_steps and len(uphill) < wanted:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        neighbor = propose_neighbor(previous.candidate, instance, config, rng)
        decoded = decoder.decode(instance, neighbor, config.placement_policy_id)
        evaluations += 1
        delta = decoded.objective_scalar - previous.objective_scalar
        if delta > 0 and int(decoded.score[0]) == int(previous.score[0]):
            uphill.append(delta)
        if lexicographic_less(decoded.score, best.score):
            best = decoded
        previous = decoded

    if not uphill:
        # Within-level deltas scale like 1/bins (r is a mean over bins).
        fallback = FALLBACK_TEMPERATURE / max(1, int(start.score[0]))
        return TemperatureCalibration(fallback, evaluations, best, 0, fallback=True)
    mean_delta = sum(uphill) / len(uphill)
    temperature = max(1e-9, -mean_delta / math.log(config.target_initial_acceptance))
    return TemperatureCalibration(temperature, evaluations, best, len(uphill), fallback=False)
