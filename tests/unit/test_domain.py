from __future__ import annotations

from adaptive_3dbp.domain.models import BinSpec, ItemSpec, Placement, ProblemInstance
from adaptive_3dbp.domain.rotations import fitting_rotation_ids, rotation_dimensions
from adaptive_3dbp.domain.validation import aabb_distance, validate_solution


def test_rotations_preserve_volume() -> None:
    item = ItemSpec(id="a", length=2, width=3, height=5)
    volumes = []
    for rotation_id in fitting_rotation_ids(item, BinSpec(length=10, width=10, height=10)):
        length, width, height = rotation_dimensions(item, rotation_id)
        volumes.append(length * width * height)
    assert volumes
    assert all(volume == item.volume for volume in volumes)


def test_problem_rejects_duplicate_ids() -> None:
    try:
        ProblemInstance(
            instance_id="duplicate",
            bin=BinSpec(length=10, width=10, height=10),
            items=[
                ItemSpec(id="x", length=1, width=1, height=1),
                ItemSpec(id="x", length=2, width=2, height=2),
            ],
        )
    except ValueError as exc:
        assert "duplicated item ids" in str(exc)
    else:
        raise AssertionError("duplicate ids should be rejected")


def test_min_distance_constraint_is_validated() -> None:
    instance = ProblemInstance(
        instance_id="distance",
        bin=BinSpec(length=20, width=10, height=10),
        items=[
            ItemSpec(id="a", length=2, width=2, height=2),
            ItemSpec(id="b", length=2, width=2, height=2),
        ],
        constraints={
            "orthogonal_only": True,
            "stability_required": False,
            "support_ratio_min": None,
            "min_distance": [{"item_id_a": "a", "item_id_b": "b", "distance": 3}],
        },
    )
    placements = [
        Placement(
            item_id="a",
            item_index=0,
            bin_id=0,
            x=0,
            y=0,
            z=0,
            rotation_id=0,
            placed_length=2,
            placed_width=2,
            placed_height=2,
        ),
        Placement(
            item_id="b",
            item_index=1,
            bin_id=0,
            x=4,
            y=0,
            z=0,
            rotation_id=0,
            placed_length=2,
            placed_width=2,
            placed_height=2,
        ),
    ]
    assert aabb_distance(placements[0], placements[1]) == 2
    assert any("minimum distance violated" in diagnostic for diagnostic in validate_solution(instance, placements))
