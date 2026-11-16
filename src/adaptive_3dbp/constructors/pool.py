from __future__ import annotations

import random

from adaptive_3dbp.domain.models import CandidateSolution, ProblemInstance
from adaptive_3dbp.domain.rotations import fitting_rotation_ids


def default_rotation_vector(instance: ProblemInstance) -> tuple[int, ...]:
    rotations: list[int] = []
    for item in instance.items:
        fitting = fitting_rotation_ids(item, instance.bin)
        rotations.append(fitting[0] if fitting else 0)
    return tuple(rotations)


def volume_desc_candidate(instance: ProblemInstance) -> CandidateSolution:
    order = tuple(sorted(range(len(instance.items)), key=lambda index: instance.items[index].volume, reverse=True))
    return CandidateSolution(item_order=order, rotations=default_rotation_vector(instance))


def max_dimension_desc_candidate(instance: ProblemInstance) -> CandidateSolution:
    order = tuple(
        sorted(
            range(len(instance.items)),
            key=lambda index: max(instance.items[index].dimensions),
            reverse=True,
        )
    )
    return CandidateSolution(item_order=order, rotations=default_rotation_vector(instance))


def face_area_desc_candidate(instance: ProblemInstance) -> CandidateSolution:
    def largest_face(index: int) -> float:
        length, width, height = instance.items[index].dimensions
        return max(length * width, length * height, width * height)

    order = tuple(sorted(range(len(instance.items)), key=largest_face, reverse=True))
    return CandidateSolution(item_order=order, rotations=default_rotation_vector(instance))


def randomized_candidate(instance: ProblemInstance, rng: random.Random) -> CandidateSolution:
    order = list(range(len(instance.items)))
    rng.shuffle(order)
    rotations: list[int] = []
    for item in instance.items:
        fitting = fitting_rotation_ids(item, instance.bin)
        rotations.append(rng.choice(fitting) if fitting else 0)
    return CandidateSolution(item_order=tuple(order), rotations=tuple(rotations))


def build_initial_pool(instance: ProblemInstance, seed: int, pool_size: int) -> list[CandidateSolution]:
    rng = random.Random(seed)
    pool = [
        volume_desc_candidate(instance),
        max_dimension_desc_candidate(instance),
        face_area_desc_candidate(instance),
    ]
    while len(pool) < pool_size:
        pool.append(randomized_candidate(instance, rng))

    unique: list[CandidateSolution] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for candidate in pool:
        key = (candidate.item_order, candidate.rotations)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    # At most pool_size starts: the three rule-based orders are always built,
    # so multi_start_count=1 used to run up to 3 starts.
    return unique[: max(1, pool_size)]

