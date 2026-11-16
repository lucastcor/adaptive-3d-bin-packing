from __future__ import annotations

import math

from adaptive_3dbp.domain.models import Placement, ProblemInstance
from adaptive_3dbp.domain.rotations import fitting_rotation_ids, rotation_dimensions

EPSILON = 1e-9


def aabb_overlap(a: Placement, b: Placement) -> bool:
    ax2, ay2, az2 = a.max_corner
    bx2, by2, bz2 = b.max_corner
    return not (
        ax2 <= b.x + EPSILON
        or bx2 <= a.x + EPSILON
        or ay2 <= b.y + EPSILON
        or by2 <= a.y + EPSILON
        or az2 <= b.z + EPSILON
        or bz2 <= a.z + EPSILON
    )


def aabb_distance(a: Placement, b: Placement) -> float:
    ax2, ay2, az2 = a.max_corner
    bx2, by2, bz2 = b.max_corner
    dx = max(b.x - ax2, a.x - bx2, 0.0)
    dy = max(b.y - ay2, a.y - by2, 0.0)
    dz = max(b.z - az2, a.z - bz2, 0.0)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def minimum_distance_required(instance: ProblemInstance, item_id_a: str, item_id_b: str) -> float:
    for rule in instance.constraints.min_distance:
        if {rule.item_id_a, rule.item_id_b} == {item_id_a, item_id_b}:
            return float(rule.distance)
    return 0.0


def min_distance_satisfied(instance: ProblemInstance, a: Placement, b: Placement) -> bool:
    if a.bin_id != b.bin_id:
        return True
    required = minimum_distance_required(instance, a.item_id, b.item_id)
    if required <= 0:
        return True
    return aabb_distance(a, b) + EPSILON >= required


def required_support_ratio(instance: ProblemInstance) -> float:
    """Minimum fraction of an item's base that must rest on the floor or on tops below.

    ``support_ratio_min`` wins when set; ``stability_required`` alone means full
    base support (1.0). Zero disables the check.
    """
    constraints = instance.constraints
    if constraints.support_ratio_min is not None:
        return float(constraints.support_ratio_min)
    return 1.0 if constraints.stability_required else 0.0


def support_ratio(placement: Placement, bin_placements: list[Placement] | tuple[Placement, ...]) -> float:
    """Share of the base area supported by the bin floor or by tops at the same height."""
    if placement.z <= EPSILON:
        return 1.0
    base_area = float(placement.placed_length * placement.placed_width)
    x2, y2, _z2 = placement.max_corner
    supported = 0.0
    for other in bin_placements:
        if other is placement or other.bin_id != placement.bin_id:
            continue
        ox2, oy2, oz2 = other.max_corner
        if abs(oz2 - placement.z) > EPSILON:
            continue
        x_overlap = max(0.0, min(x2, ox2) - max(placement.x, other.x))
        y_overlap = max(0.0, min(y2, oy2) - max(placement.y, other.y))
        supported += x_overlap * y_overlap
    return min(1.0, supported / base_area) if base_area > 0 else 1.0


def item_weight(instance: ProblemInstance, item_index: int) -> float:
    weight = instance.items[item_index].weight
    return float(weight) if weight is not None else 0.0


def rotation_allowed(instance: ProblemInstance, placement: Placement) -> bool:
    """Placed dimensions must match some allowed rotation of the item.

    Compared by dimensions, not by id: for an item with equal sides several
    ids give the same box, and any of them is the same physical placement.
    """
    item = instance.items[placement.item_index]
    if item.allowed_rotations == "all_orthogonal":
        allowed: list[int] = list(range(6))
    elif item.allowed_rotations == "fixed":
        allowed = [0]
    else:
        allowed = list(item.allowed_rotations)
    got = (float(placement.placed_length), float(placement.placed_width), float(placement.placed_height))
    return any(
        all(abs(a - b) <= EPSILON for a, b in zip(rotation_dimensions(item, rotation_id), got, strict=True))
        for rotation_id in allowed
    )


def point_inside_placement(point: tuple[float, float, float], placement: Placement) -> bool:
    x, y, z = point
    x2, y2, z2 = placement.max_corner
    return (
        placement.x + EPSILON < x < x2 - EPSILON
        and placement.y + EPSILON < y < y2 - EPSILON
        and placement.z + EPSILON < z < z2 - EPSILON
    )


def placement_inside_bin(instance: ProblemInstance, placement: Placement) -> bool:
    x2, y2, z2 = placement.max_corner
    return (
        placement.x >= -EPSILON
        and placement.y >= -EPSILON
        and placement.z >= -EPSILON
        and x2 <= instance.bin.length + EPSILON
        and y2 <= instance.bin.width + EPSILON
        and z2 <= instance.bin.height + EPSILON
    )


def validate_problem_instance(instance: ProblemInstance) -> list[str]:
    diagnostics: list[str] = []
    for index, item in enumerate(instance.items):
        if item.volume <= 0:
            diagnostics.append(f"item {item.id} has non-positive volume")
        if not fitting_rotation_ids(item, instance.bin):
            diagnostics.append(f"item {item.id} at index {index} does not fit into the bin in any allowed rotation")
    if instance.total_item_volume <= 0:
        diagnostics.append("total item volume must be positive")
    if instance.bin.max_weight is not None:
        for item in instance.items:
            if item.weight is not None and item.weight > instance.bin.max_weight:
                diagnostics.append(f"item {item.id} weighs more than the bin max_weight")
    for rule in instance.constraints.min_distance:
        if rule.distance < 0:
            diagnostics.append(
                f"minimum distance for {rule.item_id_a} and {rule.item_id_b} must be non-negative"
            )
    return diagnostics


def validate_solution(instance: ProblemInstance, placements: list[Placement] | tuple[Placement, ...]) -> list[str]:
    diagnostics: list[str] = []
    expected_ids = {item.id for item in instance.items}
    placed_ids = [placement.item_id for placement in placements]

    missing = sorted(expected_ids - set(placed_ids))
    duplicated = sorted({item_id for item_id in placed_ids if placed_ids.count(item_id) > 1})
    unknown = sorted(set(placed_ids) - expected_ids)

    if missing:
        diagnostics.append(f"missing item placements: {missing}")
    if duplicated:
        diagnostics.append(f"duplicated item placements: {duplicated}")
    if unknown:
        diagnostics.append(f"unknown item placements: {unknown}")

    for placement in placements:
        if not placement_inside_bin(instance, placement):
            diagnostics.append(f"placement for {placement.item_id} is outside bin {placement.bin_id}")
        item = instance.items[placement.item_index]
        expected_dims = rotation_dimensions(item, placement.rotation_id)
        got_dims = (float(placement.placed_length), float(placement.placed_width), float(placement.placed_height))
        if any(abs(a - b) > EPSILON for a, b in zip(expected_dims, got_dims, strict=True)):
            diagnostics.append(f"placement for {placement.item_id} does not match rotation {placement.rotation_id}")
        elif not rotation_allowed(instance, placement):
            diagnostics.append(
                f"placement for {placement.item_id} uses rotation {placement.rotation_id}, "
                f"not allowed by {item.allowed_rotations!r}"
            )

    by_bin: dict[int, list[Placement]] = {}
    for placement in placements:
        by_bin.setdefault(placement.bin_id, []).append(placement)

    max_weight = instance.bin.max_weight
    min_support = required_support_ratio(instance)
    for bin_id, bin_placements in by_bin.items():
        if max_weight is not None:
            load = sum(item_weight(instance, placement.item_index) for placement in bin_placements)
            if load > float(max_weight) + EPSILON:
                diagnostics.append(f"bin {bin_id} carries {load:g}, above max_weight {float(max_weight):g}")
        if min_support > 0:
            for placement in bin_placements:
                ratio = support_ratio(placement, bin_placements)
                if ratio + EPSILON < min_support:
                    diagnostics.append(
                        f"placement for {placement.item_id} in bin {bin_id} has support {ratio:.3f}, "
                        f"below required {min_support:.3f}"
                    )
        for left_index, left in enumerate(bin_placements):
            for right in bin_placements[left_index + 1 :]:
                if aabb_overlap(left, right):
                    diagnostics.append(f"placements overlap in bin {bin_id}: {left.item_id} and {right.item_id}")
                required_distance = minimum_distance_required(instance, left.item_id, right.item_id)
                if required_distance > 0 and not min_distance_satisfied(instance, left, right):
                    actual_distance = aabb_distance(left, right)
                    diagnostics.append(
                        "minimum distance violated in bin "
                        f"{bin_id}: {left.item_id} and {right.item_id} require "
                        f"{required_distance}, actual {actual_distance:.6f}"
                    )

    return diagnostics
