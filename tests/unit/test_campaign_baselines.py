"""Braços de baseline por instância (per_class / nearest_neighbor) na campanha."""
from __future__ import annotations

from adaptive_3dbp.experiments import campaign as C


def test_baseline_arms_registered_with_labels() -> None:
    assert C.PER_INSTANCE_BASELINE_ARMS == ("per_class", "nearest_neighbor")
    for arm in C.PER_INSTANCE_BASELINE_ARMS:
        assert arm in C.ARM_LABELS and C.ARM_LABELS[arm]


def test_instance_class_key_strips_split_suffix() -> None:
    assert (
        C._instance_class_key("small-high-elongated-homogeneous-nodist-test-0-slot0")
        == "small-high-elongated-homogeneous-nodist"
    )
    assert (
        C._instance_class_key("medium-low-mixed-heterogeneous-dist-train-12-slot3")
        == "medium-low-mixed-heterogeneous-dist"
    )
    # id fora do padrão do gerador: devolve o próprio id (fallback global)
    assert C._instance_class_key("custom-instance") == "custom-instance"


def test_per_instance_baseline_flag_adds_arms() -> None:
    # o spec com per_instance_baselines deve acrescentar exatamente os 2 braços
    base = list(C.BASE_ARMS)
    arms = list(base)
    arms.extend(arm for arm in C.PER_INSTANCE_BASELINE_ARMS if arm not in arms)
    assert "per_class" in arms and "nearest_neighbor" in arms
    assert len(arms) == len(base) + 2
