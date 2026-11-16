from __future__ import annotations

import json
from pathlib import Path

from adaptive_3dbp.benchmarks.mpv import mpv_instance
from adaptive_3dbp.constructors.pool import volume_desc_candidate
from adaptive_3dbp.decoders.extreme_points import ExtremePointDecoder, _project
from adaptive_3dbp.domain.models import BinSpec, CandidateSolution, ItemSpec, Placement, ProblemInstance
from adaptive_3dbp.domain.validation import validate_solution


ROOT = Path(__file__).resolve().parents[2]


def test_extreme_point_decoder_returns_valid_solution() -> None:
    instance = ProblemInstance.model_validate(json.loads((ROOT / "data/raw/example_20.json").read_text()))
    candidate = volume_desc_candidate(instance)
    decoded = ExtremePointDecoder().decode(instance, candidate)
    assert decoded.feasible, decoded.diagnostics
    assert validate_solution(instance, decoded.placements) == []
    assert len(decoded.placements) == len(instance.items)



def _fixed_instance(sizes: list[tuple[float, float, float]], side: float = 10.0) -> ProblemInstance:
    return ProblemInstance(
        instance_id="tiny",
        bin=BinSpec(length=side, width=side, height=side),
        items=[
            ItemSpec(id=f"item-{index}", length=l, width=w, height=h, allowed_rotations="fixed")
            for index, (l, w, h) in enumerate(sizes)
        ],
    )


def test_decoder_is_first_fit_across_bins() -> None:
    # A fills bin 0 up to z=6, B (z=5) opens bin 1. C fits on top of A
    # (6+4=10) and lower in bin 1 (z=5): first fit keeps it in bin 0. The
    # 0.2.x rule took the lowest point of any bin and spread items out.
    instance = _fixed_instance([(10, 10, 6), (10, 10, 5), (10, 10, 4)])
    decoded = ExtremePointDecoder().decode(instance, CandidateSolution(item_order=(0, 1, 2), rotations=(0, 0, 0)))
    assert decoded.feasible, decoded.diagnostics
    by_item = {placement.item_index: placement for placement in decoded.placements}
    assert (by_item[2].bin_id, by_item[2].z) == (0, 6.0)


def test_extreme_points_are_projected_onto_items_and_walls() -> None:
    slab = Placement(
        item_id="slab", item_index=0, bin_id=0, x=0, y=0, z=0, rotation_id=0,
        placed_length=2, placed_width=10, placed_height=2,
    )
    assert _project((6.0, 0.0, 2.0), 2, [slab]) == (6.0, 0.0, 0.0)  # nothing below: floor
    assert _project((1.0, 0.0, 5.0), 2, [slab]) == (1.0, 0.0, 2.0)  # lands on the slab
    assert _project((2.0, 0.0, 1.0), 2, [slab]) == (2.0, 0.0, 0.0)  # slab's max x edge: half-open
    assert _project((6.0, 3.0, 1.0), 0, [slab]) == (2.0, 3.0, 1.0)  # stops at the slab's side


# CIRRELT-2007-41, Tabela 2 (EP-FFD, n=50, média de 10 instâncias), via
# docs/benchmark_mpv.md. As colunas "sem ordem" e
# "Volume-Altura" não entram: a primeira depende da ordem original dos itens e
# a segunda não bate com nenhuma leitura de "volume, depois altura" (provável
# definição diferente no preprint).
EP_FFD_PUBLISHED = {
    "height_volume": {1: 15.0, 4: 30.1, 5: 9.0, 6: 10.9, 7: 8.2, 8: 10.5},
    "area_height": {1: 14.4, 4: 30.0, 5: 9.2, 6: 10.6, 7: 8.1, 8: 10.1},
    "height_area": {1: 15.0, 4: 30.0, 5: 9.0, 6: 10.9, 7: 8.1, 8: 10.5},
}


def test_decoder_reproduces_published_ep_ffd_on_mpv() -> None:
    def key(rule: str, item) -> tuple[float, float]:
        area = item.length * item.width
        return {
            "height_volume": (item.height, item.volume),
            "area_height": (area, item.height),
            "height_area": (item.height, area),
        }[rule]

    decoder = ExtremePointDecoder()
    totals = dict.fromkeys(EP_FFD_PUBLISHED, 0)
    for instance_class in (1, 4, 5, 6, 7, 8):
        instances = [mpv_instance(instance_class, 50, index) for index in range(1, 11)]
        for rule, published in EP_FFD_PUBLISHED.items():
            bins = []
            for instance in instances:
                order = tuple(sorted(range(50), key=lambda i: key(rule, instance.items[i]), reverse=True))
                decoded = decoder.decode(instance, CandidateSolution(item_order=order, rotations=(0,) * 50))
                assert decoded.feasible, decoded.diagnostics
                bins.append(len({placement.bin_id for placement in decoded.placements}))
            assert abs(sum(bins) / 10 - published[instance_class]) <= 0.25, (instance_class, rule, bins)
            totals[rule] += sum(bins)
    # Exact regression totals of the 0.3.0 decoder (60 instances; the published
    # sums of class means are 83.7, 82.4 and 83.5).
    assert totals == {"height_volume": 833, "area_height": 822, "height_area": 833}
