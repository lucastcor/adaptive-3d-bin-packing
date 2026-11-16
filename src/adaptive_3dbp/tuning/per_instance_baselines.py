"""Baselines de configuração POR INSTÂNCIA (resposta ao parecer §3.6).

Estes baselines completam a comparação: além dos configuradores GLOBAIS
(Random Search, EGO clássico, SMAC3, irace), que retornam uma única
configuração, e da proposta (EGO contextual, que retorna uma política), o
parecer pede baselines por instância *simples* para verificar se o GP contextual
agrega valor além de um mapeamento ingênuo características -> configuração:

* ``virtual_best_configuration`` — oráculo retrospectivo (teto de ganho): a
  melhor configuração observada para CADA instância. Não é implementável em
  produção (usa o resultado da instância-alvo), mas limita por cima o ganho de
  qualquer método por instância.
* ``per_class_policy`` — escolhe, para cada classe de instância, a configuração
  que minimiza a perda média de treino naquela classe (config por classe).
* ``nearest_neighbor_policy`` — para uma instância nova, copia a melhor
  configuração da instância de treino mais próxima no espaço de características
  (kNN com k=1; estilo ISAC sem clustering).

As políticas operam sobre observações já persistidas
(``(features, configuração, perda)``) e devolvem uma configuração ``dict`` para
cada instância-alvo, sem reexecutar o solver na instância-alvo (zero-shot, exceto
o VBC, que é oráculo por definição). A avaliação efetiva (rodar a config
escolhida em validação/teste) é feita pela campanha; ver
``experiments/campaign.py`` e ``docs/experiment_protocol.md``.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class Observation:
    """Uma observação de desempenho persistida."""

    instance_id: str
    instance_class: str
    features: tuple[float, ...]
    config: dict[str, Any]
    loss: float
    config_key: str = field(default="")

    def key(self) -> str:
        return self.config_key or repr(sorted(self.config.items()))


def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else math.inf


def virtual_best_configuration(observations: Sequence[Observation]) -> dict[str, dict[str, Any]]:
    """Oráculo: melhor configuração observada para cada instância (teto de ganho)."""
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for obs in observations:
        cur = best.get(obs.instance_id)
        if cur is None or obs.loss < cur[0]:
            best[obs.instance_id] = (obs.loss, obs.config)
    return {inst: cfg for inst, (_, cfg) in best.items()}


def per_class_policy(
    train: Sequence[Observation], targets: Sequence[Observation]
) -> dict[str, dict[str, Any]]:
    """Config por classe: melhor config média de treino na classe da instância-alvo."""
    by_class: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cfg_of_key: dict[str, dict[str, Any]] = {}
    for obs in train:
        by_class[obs.instance_class][obs.key()].append(obs.loss)
        cfg_of_key.setdefault(obs.key(), obs.config)
    best_key_per_class: dict[str, str] = {}
    for cls, configs in by_class.items():
        best_key_per_class[cls] = min(configs, key=lambda k: _mean(configs[k]))
    # fallback global se a classe não existir no treino
    global_best = None
    if train:
        all_keys: dict[str, list[float]] = defaultdict(list)
        for obs in train:
            all_keys[obs.key()].append(obs.loss)
        global_best = min(all_keys, key=lambda k: _mean(all_keys[k]))
    out: dict[str, dict[str, Any]] = {}
    for tgt in targets:
        key = best_key_per_class.get(tgt.instance_class, global_best)
        if key is not None:
            out[tgt.instance_id] = cfg_of_key[key]
    return out


def _euclidean(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _standardize(observations: Sequence[Observation]) -> tuple[list[float], list[float]]:
    if not observations:
        return [], []
    dim = len(observations[0].features)
    means = [0.0] * dim
    for obs in observations:
        for d in range(dim):
            means[d] += obs.features[d]
    means = [m / len(observations) for m in means]
    stds = [0.0] * dim
    for obs in observations:
        for d in range(dim):
            stds[d] += (obs.features[d] - means[d]) ** 2
    stds = [math.sqrt(s / len(observations)) or 1.0 for s in stds]
    return means, stds


def nearest_neighbor_policy(
    train: Sequence[Observation], targets: Sequence[Observation], k: int = 1
) -> dict[str, dict[str, Any]]:
    """kNN no espaço de características: copia a melhor config da(s) instância(s) de treino mais próxima(s).

    Para k=1, escolhe a melhor configuração (menor perda) da instância de treino
    mais próxima. Para k>1, escolhe a config com menor perda média entre as k
    instâncias de treino mais próximas. Características são padronizadas (z-score)
    com base no treino.
    """
    if not train:
        return {}
    means, stds = _standardize(train)

    def norm(feats: tuple[float, ...]) -> tuple[float, ...]:
        return tuple((feats[d] - means[d]) / stds[d] for d in range(len(feats)))

    # melhor config por instância de treino
    best_cfg_per_instance: dict[str, dict[str, Any]] = {}
    best_loss_per_instance: dict[str, float] = {}
    feats_per_instance: dict[str, tuple[float, ...]] = {}
    losses_per_instance_key: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cfg_of_key: dict[str, dict[str, Any]] = {}
    for obs in train:
        feats_per_instance[obs.instance_id] = norm(obs.features)
        losses_per_instance_key[obs.instance_id][obs.key()].append(obs.loss)
        cfg_of_key.setdefault(obs.key(), obs.config)
        if obs.instance_id not in best_loss_per_instance or obs.loss < best_loss_per_instance[obs.instance_id]:
            best_loss_per_instance[obs.instance_id] = obs.loss
            best_cfg_per_instance[obs.instance_id] = obs.config

    out: dict[str, dict[str, Any]] = {}
    train_instances = list(feats_per_instance)
    for tgt in targets:
        tf = norm(tgt.features)
        neighbors = sorted(train_instances, key=lambda i: _euclidean(tf, feats_per_instance[i]))[:k]
        if k == 1:
            out[tgt.instance_id] = best_cfg_per_instance[neighbors[0]]
        else:
            # config com menor perda média entre os vizinhos
            agg: dict[str, list[float]] = defaultdict(list)
            for nb in neighbors:
                for key, losses in losses_per_instance_key[nb].items():
                    agg[key].append(_mean(losses))
            best_key = min(agg, key=lambda key: _mean(agg[key]))
            out[tgt.instance_id] = cfg_of_key[best_key]
    return out
