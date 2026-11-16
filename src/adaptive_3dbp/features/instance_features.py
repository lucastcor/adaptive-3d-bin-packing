from __future__ import annotations

import math
import statistics

from adaptive_3dbp.domain.bounds import volume_lower_bound
from adaptive_3dbp.domain.models import ProblemInstance
from adaptive_3dbp.domain.rotations import fitting_rotation_ids


def _safe_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def extract_instance_features(instance: ProblemInstance) -> dict[str, float]:
    bin_volume = instance.bin.volume
    volumes = [item.volume for item in instance.items]
    volume_ratios = [volume / bin_volume for volume in volumes]
    norm_dims = [
        (
            item.length / instance.bin.length,
            item.width / instance.bin.width,
            item.height / instance.bin.height,
        )
        for item in instance.items
    ]
    flat_dims = [float(value) for dims in norm_dims for value in dims]
    aspect_ratios = [
        max(item.dimensions) / max(min(item.dimensions), 1e-9)
        for item in instance.items
    ]
    rotations = [len(fitting_rotation_ids(item, instance.bin)) for item in instance.items]
    total_volume = sum(volumes)
    entropy = 0.0
    if total_volume > 0:
        for volume in volumes:
            probability = volume / total_volume
            entropy -= probability * math.log(probability + 1e-12)

    return {
        "n_items": float(len(instance.items)),
        "bin_volume": float(bin_volume),
        "total_item_volume": float(total_volume),
        "volume_ratio": float(total_volume / bin_volume),
        "volume_lower_bound": float(volume_lower_bound(instance)),
        "mean_item_volume_ratio": _safe_mean(volume_ratios),
        "std_item_volume_ratio": _safe_std(volume_ratios),
        "min_item_volume_ratio": min(volume_ratios, default=0.0),
        "max_item_volume_ratio": max(volume_ratios, default=0.0),
        "mean_normalized_dimension": _safe_mean(flat_dims),
        "std_normalized_dimension": _safe_std(flat_dims),
        "mean_aspect_ratio": _safe_mean(aspect_ratios),
        "max_aspect_ratio": max(aspect_ratios, default=0.0),
        "fraction_large_items": sum(1 for ratio in volume_ratios if ratio >= 0.08) / max(len(volume_ratios), 1),
        "fraction_elongated_items": sum(1 for ratio in aspect_ratios if ratio >= 2.5) / max(len(aspect_ratios), 1),
        "fraction_nearly_cubic_items": sum(1 for ratio in aspect_ratios if ratio <= 1.25) / max(len(aspect_ratios), 1),
        "mean_feasible_rotations": _safe_mean([float(value) for value in rotations]),
        "size_entropy": entropy,
        "dimensional_cv": _safe_std(flat_dims) / max(_safe_mean(flat_dims), 1e-9),
    }

