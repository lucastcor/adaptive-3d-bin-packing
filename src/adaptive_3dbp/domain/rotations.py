from __future__ import annotations

from collections.abc import Iterable

from adaptive_3dbp.domain.models import BinSpec, ItemSpec


ROTATION_PERMUTATIONS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def rotation_dimensions(item: ItemSpec, rotation_id: int) -> tuple[float, float, float]:
    permutation = ROTATION_PERMUTATIONS[rotation_id]
    dimensions = item.dimensions
    return tuple(float(dimensions[index]) for index in permutation)  # type: ignore[return-value]


def unique_rotation_ids(item: ItemSpec, rotation_ids: Iterable[int]) -> list[int]:
    seen: set[tuple[float, float, float]] = set()
    unique_ids: list[int] = []
    for rotation_id in rotation_ids:
        dims = rotation_dimensions(item, rotation_id)
        if dims not in seen:
            seen.add(dims)
            unique_ids.append(rotation_id)
    return unique_ids


def allowed_rotation_ids(item: ItemSpec) -> list[int]:
    if item.allowed_rotations == "fixed":
        return [0]
    if item.allowed_rotations == "all_orthogonal":
        return unique_rotation_ids(item, range(6))
    return unique_rotation_ids(item, item.allowed_rotations)


# Same tolerance as domain.validation.EPSILON (not imported: validation
# imports this module). Placement and validation already accept 1e-9 of
# slack; without it here an item equal to the bin up to float noise had no
# fitting rotation although it could be placed.
FIT_EPSILON = 1e-9


def fitting_rotation_ids(item: ItemSpec, bin_spec: BinSpec) -> list[int]:
    fitting: list[int] = []
    for rotation_id in allowed_rotation_ids(item):
        length, width, height = rotation_dimensions(item, rotation_id)
        if (
            length <= bin_spec.length + FIT_EPSILON
            and width <= bin_spec.width + FIT_EPSILON
            and height <= bin_spec.height + FIT_EPSILON
        ):
            fitting.append(rotation_id)
    return fitting


def ordered_rotation_ids(item: ItemSpec, preferred_rotation: int, bin_spec: BinSpec) -> list[int]:
    fitting = fitting_rotation_ids(item, bin_spec)
    if preferred_rotation in fitting:
        return [preferred_rotation, *[rotation_id for rotation_id in fitting if rotation_id != preferred_rotation]]
    return fitting

