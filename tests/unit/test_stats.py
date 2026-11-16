from __future__ import annotations

import math

from adaptive_3dbp.benchmarks.mpv import mpv_unit_key
from adaptive_3dbp.experiments.stats import (
    holm_correction,
    per_instance_mean_losses,
    predefined_comparisons,
    wilcoxon_signed_rank,
)


def test_exact_wilcoxon_matches_known_value() -> None:
    # 6 positive differences, distinct ranks: two-sided exact p = 2/64.
    result = wilcoxon_signed_rank([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert result["method"].startswith("exact")
    assert math.isclose(result["p_value"], 2 / 64)
    assert result["rank_biserial"] == 1.0


def test_zeros_are_dropped_and_reported() -> None:
    result = wilcoxon_signed_rank([0.0, 0.0, 0.0, 0.2, -0.1])
    assert result["n"] == 2
    assert result["zeros"] == 3
    degenerate = wilcoxon_signed_rank([0.0, 0.0])
    assert degenerate["p_value"] is None and degenerate["zeros"] == 2


def test_normal_approximation_applies_tie_correction() -> None:
    # 20 differences in two tied magnitude groups. Tie-corrected variance is
    # smaller, so |z| grows and p shrinks relative to the uncorrected formula.
    diffs = [1.0] * 12 + [-1.0] * 2 + [2.0] * 6
    result = wilcoxon_signed_rank(diffs)
    assert "tie-corrected" in result["method"]
    n = 20
    w = result["statistic"]
    mean_w = n * (n + 1) / 4
    uncorrected = math.erfc(abs(w - mean_w) / math.sqrt(n * (n + 1) * (2 * n + 1) / 24) / math.sqrt(2))
    assert result["p_value"] < uncorrected


def test_unit_key_averages_dependent_instances_into_one_observation() -> None:
    records = [
        {"label": "arm:test", "instance_id": "mpv-c1-n50-v01", "loss": 0.2},
        {"label": "arm:test", "instance_id": "mpv-c1-n50-v01", "loss": 0.4},
        {"label": "arm:test", "instance_id": "mpv-c4-n50-v01", "loss": 0.6},
        {"label": "arm:test", "instance_id": "mpv-c1-n50-v02", "loss": 0.1},
        {"label": "arm:train", "instance_id": "mpv-c1-n50-v01", "loss": 9.0},
    ]
    by_instance = per_instance_mean_losses(records)
    assert by_instance["arm"] == {"mpv-c1-n50-v01": 0.30000000000000004, "mpv-c4-n50-v01": 0.6, "mpv-c1-n50-v02": 0.1}
    by_unit = per_instance_mean_losses(records, unit_key=mpv_unit_key)
    assert set(by_unit["arm"]) == {"mpv-n50-v01", "mpv-n50-v02"}
    assert math.isclose(by_unit["arm"]["mpv-n50-v01"], 0.45)


def test_holm_is_monotone_and_preserves_none() -> None:
    adjusted = holm_correction([0.01, None, 0.04, 0.03])
    assert adjusted[1] is None
    assert adjusted == [0.03, None, 0.06, 0.06]


def test_families_get_separate_holm_corrections() -> None:
    instances = {f"i{index}": 0.1 for index in range(6)}
    better = {key: value - 0.05 * (index + 1) for index, (key, value) in enumerate(instances.items())}
    losses = {
        "noisy_contextual_ego": better,
        "fixed_default": dict(instances),
        "random_search": dict(instances),
        "ablation_min_y_incumbent": dict(instances),
    }
    primary = predefined_comparisons(losses)
    ablation = predefined_comparisons(
        losses, baselines=("ablation_min_y_incumbent",), family="ablation"
    )
    assert {c["baseline"] for c in primary} == {"fixed_default", "random_search"}
    assert all(c["family"] == "primary" for c in primary)
    # A single-member family is not penalised by the primary family's size.
    assert ablation[0]["p_value_holm"] == ablation[0]["wilcoxon_p_value"]
    assert primary[0]["p_value_holm"] == min(1.0, 2 * primary[0]["wilcoxon_p_value"])
    assert primary[0]["wins"] == 6 and primary[0]["ties"] == 0
