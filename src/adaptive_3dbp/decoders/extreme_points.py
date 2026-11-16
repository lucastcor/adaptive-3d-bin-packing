from __future__ import annotations

from dataclasses import dataclass

from adaptive_3dbp.domain.models import CandidateSolution, DecodedSolution, Placement, ProblemInstance
from adaptive_3dbp.domain.rotations import ordered_rotation_ids, rotation_dimensions
from adaptive_3dbp.domain.scoring import score_solution
from adaptive_3dbp.domain.validation import (
    EPSILON,
    aabb_overlap,
    item_weight,
    min_distance_satisfied,
    point_inside_placement,
    required_support_ratio,
    support_ratio,
    validate_solution,
)


Point3D = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class _PlacementOption:
    rank: tuple[float, ...]
    placement: Placement


class ExtremePointDecoder:
    """Extreme Points, first fit (EP-FFD de Crainic, Perboli e Tadei, 2008).

    Cada item vai para a primeira caixa aberta, em ordem, que o comporta; dentro
    da caixa, a política de posicionamento escolhe o ponto extremo. Os pontos
    novos são os três cantos do item projetados nos eixos negativos.

    Até a 0.2.x o item ia para o melhor rank entre TODAS as caixas (com a
    política padrão, o ponto mais baixo de qualquer caixa), o que espalhava os
    itens, e os cantos não eram projetados. No MPV (n=50, ordenações da Tabela 2
    do CIRRELT-2007-41) a 0.2.x ficava de 3,1 a 6,5 caixas acima da 0.3.0 na soma
    das classes (scripts/mpv_ep_diagnostic.py).
    """

    name = "extreme_points"
    version = "0.3.0"

    def decode(
        self,
        instance: ProblemInstance,
        candidate: CandidateSolution,
        policy_id: str = "min_z_y_x",
    ) -> DecodedSolution:
        placements: list[Placement] = []
        extreme_points: list[list[Point3D]] = [[(0.0, 0.0, 0.0)]]
        bin_loads: list[float] = [0.0]
        diagnostics: list[str] = []
        max_weight = float(instance.bin.max_weight) if instance.bin.max_weight is not None else None

        for item_index in candidate.item_order:
            item = instance.items[item_index]
            preferred_rotation = candidate.rotations[item_index] if item_index < len(candidate.rotations) else 0
            rotation_ids = ordered_rotation_ids(item, preferred_rotation, instance.bin)
            if not rotation_ids:
                diagnostics.append(f"item {item.id} has no fitting rotation")
                continue
            weight = item_weight(instance, item_index)

            option = self._first_fit_bin_option(
                instance=instance,
                placements=placements,
                extreme_points=extreme_points,
                item_index=item_index,
                rotation_ids=rotation_ids,
                policy_id=policy_id,
                excluded_bins={
                    bin_id
                    for bin_id, load in enumerate(bin_loads)
                    if max_weight is not None and load + weight > max_weight + EPSILON
                },
            )

            if option is None:
                bin_id = len(extreme_points)
                extreme_points.append([(0.0, 0.0, 0.0)])
                bin_loads.append(0.0)
                option = self._best_option_for_bin(
                    instance=instance,
                    bin_id=bin_id,
                    bin_placements=[],
                    points=extreme_points[bin_id],
                    item_index=item_index,
                    rotation_ids=rotation_ids,
                    policy_id=policy_id,
                )

            if option is None:
                diagnostics.append(f"item {item.id} could not be placed")
                continue

            placement = option.placement
            placements.append(placement)
            bin_loads[placement.bin_id] += weight
            extreme_points[placement.bin_id] = self._update_extreme_points(
                instance=instance,
                points=extreme_points[placement.bin_id],
                placement=placement,
                bin_placements=[p for p in placements if p.bin_id == placement.bin_id],
            )

        diagnostics.extend(validate_solution(instance, placements))
        feasible = not diagnostics
        metrics, bin_metrics, score = score_solution(instance, placements)
        return DecodedSolution(
            candidate=candidate,
            placements=tuple(placements),
            bin_metrics=tuple(bin_metrics),
            score=score,
            objective_scalar=metrics.objective_scalar,
            feasible=feasible,
            diagnostics=tuple(diagnostics),
        )

    def _first_fit_bin_option(
        self,
        instance: ProblemInstance,
        placements: list[Placement],
        extreme_points: list[list[Point3D]],
        item_index: int,
        rotation_ids: list[int],
        policy_id: str,
        excluded_bins: set[int] | None = None,
    ) -> _PlacementOption | None:
        for bin_id, points in enumerate(extreme_points):
            if excluded_bins and bin_id in excluded_bins:
                continue
            bin_placements = [placement for placement in placements if placement.bin_id == bin_id]
            option = self._best_option_for_bin(
                instance=instance,
                bin_id=bin_id,
                bin_placements=bin_placements,
                points=points,
                item_index=item_index,
                rotation_ids=rotation_ids,
                policy_id=policy_id,
            )
            if option is not None:
                return option
        return None

    def _best_option_for_bin(
        self,
        instance: ProblemInstance,
        bin_id: int,
        bin_placements: list[Placement],
        points: list[Point3D],
        item_index: int,
        rotation_ids: list[int],
        policy_id: str,
    ) -> _PlacementOption | None:
        item = instance.items[item_index]
        min_support = required_support_ratio(instance)
        best: _PlacementOption | None = None
        for point in sorted(points, key=lambda p: (p[2], p[1], p[0])):
            for rotation_id in rotation_ids:
                length, width, height = rotation_dimensions(item, rotation_id)
                if not self._fits_inside_bin(instance, point, (length, width, height)):
                    continue
                placement = Placement(
                    item_id=item.id,
                    item_index=item_index,
                    bin_id=bin_id,
                    x=point[0],
                    y=point[1],
                    z=point[2],
                    rotation_id=rotation_id,
                    placed_length=length,
                    placed_width=width,
                    placed_height=height,
                )
                if any(aabb_overlap(placement, other) for other in bin_placements):
                    continue
                if any(not min_distance_satisfied(instance, placement, other) for other in bin_placements):
                    continue
                if min_support > 0 and support_ratio(placement, bin_placements) + EPSILON < min_support:
                    continue
                rank = self._rank_placement(instance, placement, bin_placements, policy_id)
                option = _PlacementOption(rank=rank, placement=placement)
                if best is None or option.rank < best.rank:
                    best = option
        return best

    def _fits_inside_bin(
        self,
        instance: ProblemInstance,
        point: Point3D,
        dimensions: tuple[float, float, float],
    ) -> bool:
        x, y, z = point
        length, width, height = dimensions
        return (
            x >= -EPSILON
            and y >= -EPSILON
            and z >= -EPSILON
            and x + length <= instance.bin.length + EPSILON
            and y + width <= instance.bin.width + EPSILON
            and z + height <= instance.bin.height + EPSILON
        )

    def _rank_placement(
        self,
        instance: ProblemInstance,
        placement: Placement,
        bin_placements: list[Placement],
        policy_id: str,
    ) -> tuple[float, ...]:
        x2, y2, z2 = placement.max_corner
        max_x = max([x2, *[other.max_corner[0] for other in bin_placements]])
        max_y = max([y2, *[other.max_corner[1] for other in bin_placements]])
        max_z = max([z2, *[other.max_corner[2] for other in bin_placements]])
        bounding_volume = max_x * max_y * max_z
        occupied = placement.volume + sum(other.volume for other in bin_placements)
        local_waste = max(0.0, bounding_volume - occupied)
        contact = self._contact_score(instance, placement, bin_placements)

        if policy_id == "contact_then_z":
            return (-contact, placement.z, placement.y, placement.x, local_waste)
        if policy_id == "min_waste":
            return (local_waste, placement.z, placement.y, placement.x, -contact)
        return (placement.z, placement.y, placement.x, local_waste, -contact)

    def _contact_score(
        self,
        instance: ProblemInstance,
        placement: Placement,
        bin_placements: list[Placement],
    ) -> float:
        score = 0.0
        x2, y2, z2 = placement.max_corner
        if abs(placement.x) <= EPSILON:
            score += float(placement.placed_width * placement.placed_height)
        if abs(placement.y) <= EPSILON:
            score += float(placement.placed_length * placement.placed_height)
        if abs(placement.z) <= EPSILON:
            score += float(placement.placed_length * placement.placed_width)
        if abs(x2 - instance.bin.length) <= EPSILON:
            score += float(placement.placed_width * placement.placed_height)
        if abs(y2 - instance.bin.width) <= EPSILON:
            score += float(placement.placed_length * placement.placed_height)
        if abs(z2 - instance.bin.height) <= EPSILON:
            score += float(placement.placed_length * placement.placed_width)

        for other in bin_placements:
            ox2, oy2, oz2 = other.max_corner
            y_overlap = max(0.0, min(y2, oy2) - max(placement.y, other.y))
            z_overlap = max(0.0, min(z2, oz2) - max(placement.z, other.z))
            x_overlap = max(0.0, min(x2, ox2) - max(placement.x, other.x))
            if abs(x2 - other.x) <= EPSILON or abs(ox2 - placement.x) <= EPSILON:
                score += y_overlap * z_overlap
            if abs(y2 - other.y) <= EPSILON or abs(oy2 - placement.y) <= EPSILON:
                score += x_overlap * z_overlap
            if abs(z2 - other.z) <= EPSILON or abs(oz2 - placement.z) <= EPSILON:
                score += x_overlap * y_overlap
        return score

    def _update_extreme_points(
        self,
        instance: ProblemInstance,
        points: list[Point3D],
        placement: Placement,
        bin_placements: list[Placement],
    ) -> list[Point3D]:
        x2, y2, z2 = placement.max_corner
        # Cada canto é projetado nos dois eixos negativos que não o geraram
        # (Crainic et al., 2008): (x2, y, z) em y e z; (x, y2, z) em x e z;
        # (x, y, z2) em x e y. O próprio item nunca bloqueia os seus cantos,
        # que ficam na face máxima dele (intervalo semiaberto em _project).
        candidates = [
            *points,
            _project((x2, placement.y, placement.z), 1, bin_placements),
            _project((x2, placement.y, placement.z), 2, bin_placements),
            _project((placement.x, y2, placement.z), 0, bin_placements),
            _project((placement.x, y2, placement.z), 2, bin_placements),
            _project((placement.x, placement.y, z2), 0, bin_placements),
            _project((placement.x, placement.y, z2), 1, bin_placements),
        ]

        filtered: list[Point3D] = []
        seen: set[tuple[int, int, int]] = set()
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

    def _point_in_bin_bounds(self, instance: ProblemInstance, point: Point3D) -> bool:
        x, y, z = point
        return (
            -EPSILON <= x <= instance.bin.length + EPSILON
            and -EPSILON <= y <= instance.bin.width + EPSILON
            and -EPSILON <= z <= instance.bin.height + EPSILON
        )


def _project(point: Point3D, axis: int, bin_placements: list[Placement]) -> Point3D:
    """Slide ``point`` along the negative ``axis`` until it touches an item or the wall.

    An item blocks the ray when the point lies over its footprint on the other
    two axes, as a half-open interval [min, max), and the item's max face on
    ``axis`` is not beyond the point.
    """
    others = [other_axis for other_axis in range(3) if other_axis != axis]
    stop = 0.0
    for placement in bin_placements:
        low = (placement.x, placement.y, placement.z)
        high = placement.max_corner
        if (
            all(low[a] - EPSILON <= point[a] < high[a] - EPSILON for a in others)
            and high[axis] <= point[axis] + EPSILON
            and high[axis] > stop
        ):
            stop = high[axis]
    projected = list(point)
    projected[axis] = stop
    return (projected[0], projected[1], projected[2])
