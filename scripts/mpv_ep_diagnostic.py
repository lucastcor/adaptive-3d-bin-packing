"""Decoder EP contra a Tabela 2 do CIRRELT-2007-41 (EP-FFD, MPV n=50).

Decodifica uma vez cada instância de teste do MPV (classes 1 e 4-8, v=1..10),
com as ordenações da tabela, em quatro variantes do decoder:

- caixa "global": regra da 0.2.x (melhor rank entre TODAS as caixas abertas);
  caixa "first": first fit, a regra da 0.3.0;
- pontos "corner": regra da 0.2.x (6 cantos do item, sem projeção);
  pontos "proj": projeção de Crainic et al. (2008), a regra da 0.3.0.

As regras da 0.2.x estão copiadas aqui (commit b52a009), porque o decoder do
pacote já usa as da 0.3.0. Resultado em docs/status.md.

Uso: .venv/bin/python scripts/mpv_ep_diagnostic.py
"""
from __future__ import annotations

from statistics import mean

from adaptive_3dbp.benchmarks.mpv import mpv_instance
from adaptive_3dbp.decoders.extreme_points import ExtremePointDecoder
from adaptive_3dbp.domain.models import CandidateSolution
from adaptive_3dbp.domain.validation import point_inside_placement

# n=50, colunas: sem ordem, Altura-Volume, Volume-Altura, Área-Altura, Altura-Área
PUBLISHED = {
    1: (14.6, 15.0, 14.4, 14.4, 15.0),
    4: (29.7, 30.1, 29.9, 30.0, 30.0),
    5: (10.1, 9.0, 10.0, 9.2, 9.0),
    6: (11.7, 10.9, 11.7, 10.6, 10.9),
    7: (9.4, 8.2, 9.3, 8.1, 8.1),
    8: (11.6, 10.5, 11.6, 10.1, 10.5),
}
ORDERS = ("none", "height_volume", "volume_height", "area_height", "height_area")
VARIANTS = (("global", "corner"), ("first", "corner"), ("global", "proj"), ("first", "proj"))


class VariantDecoder(ExtremePointDecoder):
    def __init__(self, bin_rule: str, ep_rule: str) -> None:
        self.bin_rule = bin_rule
        self.ep_rule = ep_rule

    def _first_fit_bin_option(self, instance, placements, extreme_points, item_index, rotation_ids, policy_id, excluded_bins=None):
        if self.bin_rule == "first":
            return super()._first_fit_bin_option(
                instance, placements, extreme_points, item_index, rotation_ids, policy_id, excluded_bins
            )
        # 0.2.x: melhor rank entre todas as caixas abertas.
        best = None
        for bin_id, points in enumerate(extreme_points):
            if excluded_bins and bin_id in excluded_bins:
                continue
            option = self._best_option_for_bin(
                instance=instance,
                bin_id=bin_id,
                bin_placements=[placement for placement in placements if placement.bin_id == bin_id],
                points=points,
                item_index=item_index,
                rotation_ids=rotation_ids,
                policy_id=policy_id,
            )
            if option is not None and (best is None or option.rank < best.rank):
                best = option
        return best

    def _update_extreme_points(self, instance, points, placement, bin_placements):
        if self.ep_rule == "proj":
            return super()._update_extreme_points(instance, points, placement, bin_placements)
        # 0.2.x: os 6 cantos do item, sem projeção.
        x2, y2, z2 = placement.max_corner
        candidates = [
            *points,
            (x2, placement.y, placement.z),
            (placement.x, y2, placement.z),
            (placement.x, placement.y, z2),
            (x2, y2, placement.z),
            (x2, placement.y, z2),
            (placement.x, y2, z2),
        ]
        filtered, seen = [], set()
        for point in candidates:
            if not self._point_in_bin_bounds(instance, point):
                continue
            if any(point_inside_placement(point, other) for other in bin_placements):
                continue
            key = tuple(round(coordinate * 1_000_000) for coordinate in point)
            if key in seen:
                continue
            seen.add(key)
            filtered.append(point)
        return sorted(filtered, key=lambda p: (p[2], p[1], p[0]))[: max(1, len(instance.items) * 8)]


def order_for(instance, rule: str) -> tuple[int, ...]:
    """Ordenações da Tabela 2: altura = 3ª dimensão do MPV, área = as outras duas."""
    items = instance.items
    keys = {
        "height_volume": lambda i: (items[i].height, items[i].volume),
        "volume_height": lambda i: (items[i].volume, items[i].height),
        "area_height": lambda i: (items[i].length * items[i].width, items[i].height),
        "height_area": lambda i: (items[i].height, items[i].length * items[i].width),
    }
    indices = range(len(items))
    return tuple(indices) if rule == "none" else tuple(sorted(indices, key=keys[rule], reverse=True))


def main() -> None:
    print("| Cl | Variante | " + " | ".join(ORDERS) + " |")
    print("|---|---|" + "---|" * len(ORDERS))
    totals = {variant: [0.0] * len(ORDERS) for variant in VARIANTS}
    for instance_class, published in PUBLISHED.items():
        print(f"| {instance_class} | publicado EP-FFD | " + " | ".join(f"{x:.1f}" for x in published) + " |")
        instances = [mpv_instance(instance_class, 50, index) for index in range(1, 11)]
        for variant in VARIANTS:
            decoder = VariantDecoder(*variant)
            row = []
            for rule in ORDERS:
                bins = []
                for instance in instances:
                    candidate = CandidateSolution(item_order=order_for(instance, rule), rotations=(0,) * len(instance.items))
                    decoded = decoder.decode(instance, candidate)
                    assert decoded.feasible, decoded.diagnostics
                    bins.append(len({placement.bin_id for placement in decoded.placements}))
                row.append(mean(bins))
            totals[variant] = [total + value for total, value in zip(totals[variant], row)]
            print(f"| {instance_class} | {variant[0]}+{variant[1]} | " + " | ".join(f"{x:.1f}" for x in row) + " |", flush=True)
    published_total = [sum(PUBLISHED[c][k] for c in PUBLISHED) for k in range(len(ORDERS))]
    print("| Σ | publicado | " + " | ".join(f"{x:.1f}" for x in published_total) + " |")
    for variant, row in totals.items():
        print(f"| Σ | {variant[0]}+{variant[1]} | " + " | ".join(f"{x:.1f}" for x in row) + " |")


if __name__ == "__main__":
    main()
