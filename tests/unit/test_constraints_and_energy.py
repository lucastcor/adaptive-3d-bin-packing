from __future__ import annotations

from adaptive_3dbp.constructors.pool import volume_desc_candidate
from adaptive_3dbp.decoders.extreme_points import ExtremePointDecoder
from adaptive_3dbp.domain.models import BinMetrics, CandidateSolution, Placement, ProblemInstance
from adaptive_3dbp.domain.scoring import compute_fragmentation, objective_scalar
from adaptive_3dbp.domain.validation import support_ratio, validate_solution


def _bin_metrics(utilizations: list[float]) -> list[BinMetrics]:
    return [
        BinMetrics(bin_id=index, utilization=u, unused_volume=1.0 - u, occupied_volume=u, compactness=1.0)
        for index, u in enumerate(utilizations)
    ]


def test_fragmentation_prefers_concentrated_free_space() -> None:
    # Same bin count and same packed volume: [1.0, 0.5] frees a bin sooner
    # than [0.75, 0.75], so it must score lower (better).
    concentrated = compute_fragmentation(_bin_metrics([1.0, 0.5]))
    even = compute_fragmentation(_bin_metrics([0.75, 0.75]))
    assert concentrated == 0.375
    assert even == 0.4375
    assert concentrated < even


def test_energy_keeps_bin_count_dominant() -> None:
    # r is in [0, 1): the worst 3-bin solution beats the best 4-bin one.
    assert objective_scalar(3, 0.999999) < objective_scalar(4, 0.0)
    assert int(objective_scalar(5, 0.42)) == 5


def _placement(item_id: str, index: int, x: float, y: float, z: float, dims: tuple[float, float, float], rotation: int = 0) -> Placement:
    return Placement(
        item_id=item_id,
        item_index=index,
        bin_id=0,
        x=x,
        y=y,
        z=z,
        rotation_id=rotation,
        placed_length=dims[0],
        placed_width=dims[1],
        placed_height=dims[2],
    )


def test_validation_rejects_disallowed_rotation() -> None:
    instance = ProblemInstance.model_validate(
        {
            "instance_id": "fixed-rotation",
            "bin": {"length": 10, "width": 10, "height": 10},
            "items": [{"id": "a", "length": 2, "width": 3, "height": 5, "allowed_rotations": "fixed"}],
        }
    )
    # rotation 2 = (width, length, height): a real reorientation of a fixed item.
    rotated = _placement("a", 0, 0, 0, 0, (3, 2, 5), rotation=2)
    diagnostics = validate_solution(instance, [rotated])
    assert any("not allowed" in diagnostic for diagnostic in diagnostics)
    assert validate_solution(instance, [_placement("a", 0, 0, 0, 0, (2, 3, 5))]) == []


def test_validation_accepts_equivalent_rotation_of_symmetric_item() -> None:
    # A 2x2x5 fixed item placed with id 2 is the same box as id 0.
    instance = ProblemInstance.model_validate(
        {
            "instance_id": "symmetric",
            "bin": {"length": 10, "width": 10, "height": 10},
            "items": [{"id": "a", "length": 2, "width": 2, "height": 5, "allowed_rotations": "fixed"}],
        }
    )
    assert validate_solution(instance, [_placement("a", 0, 0, 0, 0, (2, 2, 5), rotation=2)]) == []


def test_support_ratio_counts_tops_at_the_same_height() -> None:
    base = _placement("a", 0, 0, 0, 0, (4, 4, 2))
    half_on_top = _placement("b", 1, 2, 0, 2, (4, 4, 2))
    assert support_ratio(base, [base, half_on_top]) == 1.0
    assert support_ratio(half_on_top, [base, half_on_top]) == 0.5


def test_decoder_respects_max_weight() -> None:
    instance = ProblemInstance.model_validate(
        {
            "instance_id": "weight-limited",
            "bin": {"length": 10, "width": 10, "height": 10, "max_weight": 10},
            "items": [{"id": f"i{index}", "length": 2, "width": 2, "height": 2, "weight": 6} for index in range(4)],
        }
    )
    decoded = ExtremePointDecoder().decode(instance, volume_desc_candidate(instance))
    # Geometrically one bin is enough; the weight limit forces one item per bin.
    assert decoded.feasible, decoded.diagnostics
    assert int(decoded.score[0]) == 4


def test_decoder_respects_support_requirement() -> None:
    instance = ProblemInstance.model_validate(
        {
            "instance_id": "needs-support",
            "bin": {"length": 4, "width": 4, "height": 8},
            "constraints": {"support_ratio_min": 1.0},
            "items": [
                {"id": "small", "length": 2, "width": 2, "height": 2, "allowed_rotations": "fixed"},
                {"id": "large", "length": 4, "width": 4, "height": 2, "allowed_rotations": "fixed"},
            ],
        }
    )
    # Order: small first. Without the support rule the large slab would sit on
    # the small cube (z=2) with 25% support.
    decoder = ExtremePointDecoder()
    candidate = CandidateSolution(item_order=(0, 1), rotations=(0, 0))
    decoded = decoder.decode(instance, candidate)
    assert decoded.feasible, decoded.diagnostics
    for placement in decoded.placements:
        same_bin = [other for other in decoded.placements if other.bin_id == placement.bin_id]
        assert support_ratio(placement, same_bin) == 1.0
