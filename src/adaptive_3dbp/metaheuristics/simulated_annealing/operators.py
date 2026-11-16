from __future__ import annotations

import random

from adaptive_3dbp.domain.models import CandidateSolution, ProblemInstance, SolverConfig
from adaptive_3dbp.domain.rotations import fitting_rotation_ids


def _choose_operator(config: SolverConfig, rng: random.Random) -> str:
    total = sum(config.operator_probabilities.values())
    threshold = rng.random() * total
    cumulative = 0.0
    for name, probability in config.operator_probabilities.items():
        cumulative += probability
        if threshold <= cumulative:
            return name
    return next(iter(config.operator_probabilities))


def propose_neighbor(
    candidate: CandidateSolution,
    instance: ProblemInstance,
    config: SolverConfig,
    rng: random.Random,
) -> CandidateSolution:
    operator = _choose_operator(config, rng)
    if operator == "swap":
        return swap(candidate, rng)
    if operator == "relocate":
        return relocate(candidate, rng)
    if operator == "reverse":
        return reverse_subsequence(candidate, rng)
    if operator == "rotate":
        return rotate_one(candidate, instance, rng)
    if operator == "ruin_recreate":
        return ruin_recreate(candidate, config, rng)
    return swap(candidate, rng)


def swap(candidate: CandidateSolution, rng: random.Random) -> CandidateSolution:
    order = list(candidate.item_order)
    if len(order) < 2:
        return candidate
    left, right = rng.sample(range(len(order)), 2)
    order[left], order[right] = order[right], order[left]
    return CandidateSolution(tuple(order), candidate.rotations, candidate.placement_policy_id)


def relocate(candidate: CandidateSolution, rng: random.Random) -> CandidateSolution:
    order = list(candidate.item_order)
    if len(order) < 2:
        return candidate
    source, target = rng.sample(range(len(order)), 2)
    item = order.pop(source)
    order.insert(target, item)
    return CandidateSolution(tuple(order), candidate.rotations, candidate.placement_policy_id)


def reverse_subsequence(candidate: CandidateSolution, rng: random.Random) -> CandidateSolution:
    order = list(candidate.item_order)
    if len(order) < 3:
        return candidate
    left, right = sorted(rng.sample(range(len(order)), 2))
    if left == right:
        return candidate
    order[left : right + 1] = reversed(order[left : right + 1])
    return CandidateSolution(tuple(order), candidate.rotations, candidate.placement_policy_id)


def rotate_one(candidate: CandidateSolution, instance: ProblemInstance, rng: random.Random) -> CandidateSolution:
    rotations = list(candidate.rotations)
    item_index = rng.choice(candidate.item_order)
    fitting = fitting_rotation_ids(instance.items[item_index], instance.bin)
    if len(fitting) <= 1:
        return candidate
    current = rotations[item_index] if item_index < len(rotations) else fitting[0]
    alternatives = [rotation for rotation in fitting if rotation != current]
    rotations[item_index] = rng.choice(alternatives)
    return CandidateSolution(candidate.item_order, tuple(rotations), candidate.placement_policy_id)


def ruin_recreate(candidate: CandidateSolution, config: SolverConfig, rng: random.Random) -> CandidateSolution:
    order = list(candidate.item_order)
    if len(order) < 3:
        return swap(candidate, rng)
    ruin_size = min(config.ruin_size, max(1, len(order) // 2))
    positions = sorted(rng.sample(range(len(order)), ruin_size), reverse=True)
    removed = [order.pop(position) for position in positions]
    rng.shuffle(removed)
    for item in removed:
        order.insert(rng.randrange(len(order) + 1), item)
    return CandidateSolution(tuple(order), candidate.rotations, candidate.placement_policy_id)

