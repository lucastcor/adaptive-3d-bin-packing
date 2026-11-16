from __future__ import annotations

from adaptive_3dbp.tuning.per_instance_baselines import (
    Observation,
    nearest_neighbor_policy,
    per_class_policy,
    virtual_best_configuration,
)


def _obs(inst, cls, feats, cfg_id, loss):
    return Observation(
        instance_id=inst,
        instance_class=cls,
        features=tuple(feats),
        config={"id": cfg_id},
        loss=loss,
        config_key=cfg_id,
    )


def test_virtual_best_picks_min_loss_per_instance() -> None:
    obs = [
        _obs("i1", "A", [0.0], "cfg-a", 0.5),
        _obs("i1", "A", [0.0], "cfg-b", 0.2),  # melhor para i1
        _obs("i2", "B", [1.0], "cfg-a", 0.1),  # melhor para i2
        _obs("i2", "B", [1.0], "cfg-b", 0.9),
    ]
    vbc = virtual_best_configuration(obs)
    assert vbc["i1"]["id"] == "cfg-b"
    assert vbc["i2"]["id"] == "cfg-a"


def test_per_class_policy_uses_best_config_in_class() -> None:
    train = [
        _obs("t1", "A", [0.0], "cfg-a", 0.4),
        _obs("t1", "A", [0.0], "cfg-b", 0.1),
        _obs("t2", "A", [0.1], "cfg-a", 0.5),
        _obs("t2", "A", [0.1], "cfg-b", 0.2),  # cfg-b melhor em média na classe A
        _obs("t3", "B", [9.0], "cfg-a", 0.05),  # cfg-a melhor na classe B
        _obs("t3", "B", [9.0], "cfg-b", 0.9),
    ]
    targets = [_obs("test-A", "A", [0.05], "?", 0.0), _obs("test-B", "B", [9.0], "?", 0.0)]
    pol = per_class_policy(train, targets)
    assert pol["test-A"]["id"] == "cfg-b"
    assert pol["test-B"]["id"] == "cfg-a"


def test_nearest_neighbor_copies_closest_train_instance() -> None:
    train = [
        _obs("near", "A", [0.0, 0.0], "cfg-near", 0.1),
        _obs("far", "A", [10.0, 10.0], "cfg-far", 0.1),
    ]
    targets = [_obs("q", "A", [0.2, 0.1], "?", 0.0)]
    pol = nearest_neighbor_policy(train, targets, k=1)
    assert pol["q"]["id"] == "cfg-near"
