"""Temperature ladder and replica-exchange step for Parallel Tempering.

Replica exchange Monte Carlo (Swendsen & Wang, 1986; Geyer, 1991; Hukushima &
Nemoto, 1996): K Metropolis chains run at fixed temperatures and adjacent
temperatures periodically try to exchange configurations. Swapping the pair
(i, j) is accepted with min(1, exp((E_i - E_j)(1/T_i - 1/T_j))), which keeps
the product of Boltzmann distributions exp(-E/T_k) stationary, so every chain
still samples its own temperature while good configurations drift towards
the cold end.

Energies are the SA energy E = bins + r (dimensionless, see domain/scoring.py),
so temperatures are expressed in "bins" exactly as in the SA schedule.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from adaptive_3dbp.domain.models import DecodedSolution


def geometric_ladder(max_temperature: float, min_temperature_ratio: float, count: int) -> list[float]:
    """Geometric ladder T_0 < ... < T_{K-1}, from T_max * ratio up to T_max.

    A geometric spacing keeps the ratio T_{k+1} / T_k constant, which gives
    roughly uniform swap acceptance when the energy variance scales with T.
    Index 0 is the coldest replica.
    """
    if count <= 1:
        return [max_temperature]
    return [max_temperature * min_temperature_ratio ** ((count - 1 - k) / (count - 1)) for k in range(count)]


def swap_acceptance_probability(
    energy_i: float, energy_j: float, temperature_i: float, temperature_j: float
) -> float:
    """min(1, exp((E_i - E_j)(1/T_i - 1/T_j))) for exchanging replicas i and j."""
    exponent = (energy_i - energy_j) * (1.0 / temperature_i - 1.0 / temperature_j)
    if exponent >= 0:
        return 1.0
    return math.exp(exponent)


def exchange_adjacent_replicas(
    states: list[DecodedSolution],
    temperatures: Sequence[float],
    parity: int,
    rng: random.Random,
    attempts: list[int],
    accepts: list[int],
) -> None:
    """Try swaps on the pairs (k, k+1) with k = parity, parity + 2, ...

    Alternating even and odd pairs between rounds (deterministic even-odd
    scheme) lets a configuration travel the whole ladder in K rounds without
    two swaps competing for the same replica. Configurations move between
    slots in place; slot k always runs at temperatures[k]. ``attempts`` and
    ``accepts`` are per-pair counters indexed by k.
    """
    for k in range(parity, len(states) - 1, 2):
        attempts[k] += 1
        probability = swap_acceptance_probability(
            states[k].objective_scalar, states[k + 1].objective_scalar, temperatures[k], temperatures[k + 1]
        )
        if rng.random() < probability:
            states[k], states[k + 1] = states[k + 1], states[k]
            accepts[k] += 1
