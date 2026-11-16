from __future__ import annotations

from adaptive_3dbp.domain.bounds import (
    clique_volume_bound,
    conflict_clique_bound,
    mt_l2_volume_bound,
    simple_lower_bound,
    strong_lower_bound,
    volume_lower_bound,
)
from adaptive_3dbp.domain.models import BinSpec, ItemSpec, ProblemInstance


def _instance(items: list[ItemSpec], bin_spec: BinSpec | None = None) -> ProblemInstance:
    return ProblemInstance(
        instance_id="bounds-test",
        bin=bin_spec or BinSpec(length=10, width=10, height=10),
        items=items,
    )


def _cube(identifier: str, size: float) -> ItemSpec:
    return ItemSpec(id=identifier, length=size, width=size, height=size)


def test_clique_bound_detects_pairwise_incompatible_cubes() -> None:
    # Three 6x6x6 cubes in a 10-cube: any pair sums to 12 > 10 on every axis.
    instance = _instance([_cube("a", 6), _cube("b", 6), _cube("c", 6)])
    assert volume_lower_bound(instance) == 1
    assert conflict_clique_bound(instance) == 3
    assert strong_lower_bound(instance) == 3


def test_rotation_can_remove_conflicts() -> None:
    # 6x6x4 items can stack along z (4+4 <= 10) in some rotation pair.
    items = [
        ItemSpec(id="a", length=6, width=6, height=4),
        ItemSpec(id="b", length=6, width=6, height=4),
    ]
    instance = _instance(items)
    assert conflict_clique_bound(instance) == 1


def test_mt_l2_dominates_continuous_bound() -> None:
    # Volumes just above half capacity cannot share bins in the 1D relaxation.
    items = [_cube(f"i{i}", 8.1) for i in range(3)]  # volume 531.4 > 500
    instance = _instance(items)
    assert volume_lower_bound(instance) == 2
    assert mt_l2_volume_bound(instance) == 3


def test_clique_volume_bound_adds_residual_volume() -> None:
    # Two conflicting big cubes plus many small items that exceed leftover space.
    items = [_cube("big-a", 9), _cube("big-b", 9)]
    items.extend(_cube(f"s{i}", 5) for i in range(8))  # 8 * 125 = 1000
    instance = _instance(items)
    bound = clique_volume_bound(instance)
    # Clique opens 2 bins with 2*1000 - 2*729 = 542 free; rest volume 1000.
    assert bound >= 3


def test_strong_bound_never_below_volume_bound() -> None:
    items = [_cube("a", 5), _cube("b", 4), _cube("c", 3)]
    instance = _instance(items)
    assert strong_lower_bound(instance) >= volume_lower_bound(instance)
    assert simple_lower_bound(instance) == strong_lower_bound(instance)


def test_bounds_sanity_check_against_decoder_upper_bound() -> None:
    """Sanity check / counterexample search (NOT a formal validity certificate).

    Validity of the combined lower bound is established by PROOF (clique +
    residual volume; see docs/decisoes_cientificas.md, sec. 9), not by this
    test. Here, for small randomized instances, we decode every permutation of
    items with the Extreme Points decoder; the best feasible solution found is
    an upper bound on the optimum, so LB <= UB_decoder is a NECESSARY condition.
    It is not sufficient to certify validity: (i) the decoder need not reach the
    true optimum, so UB_decoder can be loose, and (ii) this search fixes
    rotation 0 and does not enumerate the full geometric solution space. A true
    computational certificate would require an exact solver (MIP/CP-SAT) or
    geometrically complete enumeration. This test only searches for
    counterexamples that would expose an obviously invalid bound.
    """
    import itertools
    import random

    from adaptive_3dbp.decoders.extreme_points import ExtremePointDecoder
    from adaptive_3dbp.domain.models import CandidateSolution

    decoder = ExtremePointDecoder()
    rng = random.Random(7)
    for trial in range(8):
        n = rng.randint(3, 5)
        items = [
            ItemSpec(
                id=f"t{trial}-i{idx}",
                length=rng.randint(2, 9),
                width=rng.randint(2, 9),
                height=rng.randint(2, 9),
            )
            for idx in range(n)
        ]
        instance = _instance(items)
        best_bins: int | None = None
        for order in itertools.permutations(range(n)):
            decoded = decoder.decode(
                instance,
                CandidateSolution(item_order=order, rotations=tuple([0] * n)),
            )
            if decoded.feasible:
                bins = len(decoded.bin_metrics)
                best_bins = bins if best_bins is None else min(best_bins, bins)
        assert best_bins is not None, "exhaustive decode found no feasible packing"
        lb = strong_lower_bound(instance)
        assert lb <= best_bins, (
            f"invalid bound: LB={lb} > UB_exhaustive={best_bins} "
            f"(trial {trial}, items={[(i.length, i.width, i.height) for i in items]})"
        )


def test_clique_volume_bound_accounts_for_residual_capacity() -> None:
    """The residual-capacity correction (parecer §5.3, R_C): small items that
    fit in the clique bins' leftover space must NOT raise the bound."""
    # Two conflicting 9-cubes open 2 bins with 2*(1000-729)=542 leftover.
    # Small items totalling < 542 must keep the bound at 2.
    items = [_cube("big-a", 9), _cube("big-b", 9)]
    items.extend(_cube(f"s{i}", 1) for i in range(8))  # 8 * 1 = 8 << 542; 1+9<=10: sem conflito
    instance = _instance(items)
    assert clique_volume_bound(instance) == 2
