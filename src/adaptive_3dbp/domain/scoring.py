from __future__ import annotations

from adaptive_3dbp.domain.bounds import simple_lower_bound
from adaptive_3dbp.domain.models import BinMetrics, Placement, ProblemInstance, SolutionMetrics


def group_by_bin(placements: list[Placement] | tuple[Placement, ...]) -> dict[int, list[Placement]]:
    grouped: dict[int, list[Placement]] = {}
    for placement in placements:
        grouped.setdefault(placement.bin_id, []).append(placement)
    return grouped


def compute_bin_metrics(instance: ProblemInstance, placements: list[Placement] | tuple[Placement, ...]) -> list[BinMetrics]:
    metrics: list[BinMetrics] = []
    for bin_id, bin_placements in sorted(group_by_bin(placements).items()):
        occupied = sum(placement.volume for placement in bin_placements)
        unused = max(0.0, instance.bin.volume - occupied)
        utilization = occupied / instance.bin.volume if instance.bin.volume else 0.0

        max_x = max((placement.max_corner[0] for placement in bin_placements), default=0.0)
        max_y = max((placement.max_corner[1] for placement in bin_placements), default=0.0)
        max_z = max((placement.max_corner[2] for placement in bin_placements), default=0.0)
        bbox_volume = max(max_x * max_y * max_z, 1e-9)
        compactness = min(1.0, occupied / bbox_volume)

        metrics.append(
            BinMetrics(
                bin_id=bin_id,
                utilization=utilization,
                unused_volume=unused,
                occupied_volume=occupied,
                compactness=compactness,
            )
        )
    return metrics


def compute_fragmentation(bin_metrics: list[BinMetrics]) -> float:
    """Spread of the free space across bins: r = 1 - mean_bins(u^2), in [0, 1).

    With a fixed bin count b and all items packed, mean(u) = sum_i v_i / (b V)
    is constant, so minimising r is maximising mean(u^2): solutions that
    concentrate the free space in one bin (full bins + an emptier one) score
    lower than solutions with evenly filled bins. Example with two bins:
    u = [1.0, 0.5] gives r = 0.375, u = [0.75, 0.75] gives r = 0.4375.
    That is the direction that helps the search empty and then drop a bin
    (Falkenauer-style fitness).

    The previous form mean((1 - u)^2) equals 1 - 2 mean(u) + mean(u^2): at a
    fixed b it pointed the OTHER way (it rewarded evenly filled bins).
    """
    if not bin_metrics:
        return 1.0
    return float(1.0 - sum(metric.utilization**2 for metric in bin_metrics) / len(bin_metrics))


def objective_scalar(bins_used: int, fragmentation: float) -> float:
    """Dimensionless SA energy: E(s) = b(s) + r(s).

    Scalarisation of the lexicographic order (bins, fragmentation); see
    docs/decisoes_cientificas.md, sec. 8. Dominance holds because
    r in [0, 1): b1 < b2 implies E1 < b1 + 1 <= b2 <= E2. The integer part is
    the bin count and the fractional part is the within-bin-count gradient,
    so temperatures are expressed in "bins" and do not depend on the bin
    volume. The total empty volume b V - sum_i v_i is omitted on purpose: it
    is determined by b and adds no gradient at a fixed bin count.
    """
    return float(bins_used) + fragmentation


def score_solution(
    instance: ProblemInstance, placements: list[Placement] | tuple[Placement, ...]
) -> tuple[SolutionMetrics, list[BinMetrics], tuple[float, float, float, float]]:
    bin_metrics = compute_bin_metrics(instance, placements)
    bins_used = len(bin_metrics)
    lower_bound = simple_lower_bound(instance)
    unused_volume = sum(metric.unused_volume for metric in bin_metrics)
    utilization_mean = sum(metric.utilization for metric in bin_metrics) / bins_used if bins_used else 0.0
    compactness_mean = sum(metric.compactness for metric in bin_metrics) / bins_used if bins_used else 0.0
    fragmentation = compute_fragmentation(bin_metrics)
    scalar = objective_scalar(bins_used, fragmentation)
    gap_lb = (bins_used - lower_bound) / lower_bound if lower_bound else 0.0
    gap_bks = None
    if instance.best_known_bins:
        gap_bks = (bins_used - instance.best_known_bins) / instance.best_known_bins

    metrics = SolutionMetrics(
        bins_used=bins_used,
        lower_bound=lower_bound,
        best_known_bins=instance.best_known_bins,
        gap_to_lower_bound=gap_lb,
        gap_to_best_known=gap_bks,
        unused_volume=unused_volume,
        utilization_mean=utilization_mean,
        fragmentation=fragmentation,
        compactness_mean=compactness_mean,
        objective_scalar=scalar,
    )
    # Lexicographic order for incumbent / tie-break / quality reporting.
    # The total unused volume is omitted on purpose: it equals b*V - sum_i v_i,
    # so it is constant within a fixed bin count and never breaks a tie between
    # equal-bin solutions (the same redundancy noted for the scalar energy).
    # Fragmentation and compactness are the genuine within-bin discriminators.
    lexicographic = (float(bins_used), fragmentation, -compactness_mean)
    return metrics, bin_metrics, lexicographic


def lexicographic_less(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return left < right

