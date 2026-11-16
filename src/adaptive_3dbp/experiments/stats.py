"""Protocolo estatístico da campanha (parecer §11).

Unidade experimental: a INSTÂNCIA (seeds são replicações internas e são
agregadas por média antes de qualquer teste). Comparações primárias são
pré-definidas (proposta vs. cada baseline), testadas com Wilcoxon signed-rank
pareado por instância, corrigidas por Holm para multiplicidade, e acompanhadas
de intervalo bootstrap percentil (n_boot=10000, seed=0) da diferença média e
da TAXA DE VITÓRIA empírica (fração de instâncias vencidas — significância
prática; NÃO é uma estimativa probabilística de P(vitória)).

Referências: Demšar (JMLR 2006); García & Herrera (JMLR 2008);
Eggensperger et al. (JAIR 2019).
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections.abc import Callable
from typing import Any

_EXACT_N_LIMIT = 12  # até aqui a distribuição exata do W é enumerável (2^n)


def wilcoxon_signed_rank(differences: list[float]) -> dict[str, Any]:
    """Wilcoxon signed-rank test bicaudal, pareado.

    Zeros são descartados (método de Wilcoxon); empates recebem rank médio.
    Para n <= 12 usa a distribuição exata (enumeração dos 2^n sinais);
    acima disso, aproximação normal com correção de empates
    (var -= Σ(t³ − t)/48 por grupo de |d| empatados).

    Tamanho de efeito: correlação rank-biserial pareada
    r = (W+ − W−)/(W+ + W−) em [−1, 1] (Kerby, 2014); positivo = diferenças
    positivas dominam.
    """
    nonzero = [d for d in differences if abs(d) > 1e-12]
    n = len(nonzero)
    zeros = len(differences) - n
    if n == 0:
        return {
            "n": 0,
            "zeros": zeros,
            "statistic": None,
            "p_value": None,
            "rank_biserial": None,
            "method": "degenerate (todas as diferenças nulas)",
        }

    ranked = sorted((abs(d), i) for i, d in enumerate(nonzero))
    ranks = [0.0] * n
    tie_term = 0.0
    pos = 0
    while pos < n:
        end = pos
        while end + 1 < n and abs(ranked[end + 1][0] - ranked[pos][0]) <= 1e-12:
            end += 1
        mean_rank = (pos + end) / 2.0 + 1.0
        for k in range(pos, end + 1):
            ranks[ranked[k][1]] = mean_rank
        group = end - pos + 1
        tie_term += group**3 - group
        pos = end + 1

    w_plus = sum(rank for rank, d in zip(ranks, nonzero, strict=True) if d > 0)
    w_minus = sum(rank for rank, d in zip(ranks, nonzero, strict=True) if d < 0)
    w = min(w_plus, w_minus)

    if n <= _EXACT_N_LIMIT:
        # Distribuição exata sob H0: cada |d| recebe sinal +/- com prob. 1/2.
        count = 0
        total = 2**n
        for signs in itertools.product((0, 1), repeat=n):
            w_p = sum(rank for rank, sign in zip(ranks, signs, strict=True) if sign)
            if min(w_p, sum(ranks) - w_p) <= w + 1e-12:
                count += 1
        p_value = count / total
        method = f"exact (n={n})"
    else:
        mean_w = n * (n + 1) / 4.0
        var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
        z = (w - mean_w) / math.sqrt(max(var_w, 1e-12))
        p_value = 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))
        method = f"normal approximation, tie-corrected (n={n})"
    total_rank = w_plus + w_minus
    return {
        "n": n,
        "zeros": zeros,
        "statistic": float(w),
        "w_plus": float(w_plus),
        "w_minus": float(w_minus),
        "p_value": min(float(p_value), 1.0),
        "rank_biserial": float((w_plus - w_minus) / total_rank) if total_rank > 0 else None,
        "method": method,
    }


def holm_correction(p_values: list[float | None]) -> list[float | None]:
    """Correção de Holm (step-down) preservando posições None."""
    indexed = [(p, i) for i, p in enumerate(p_values) if p is not None]
    indexed.sort()
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    running_max = 0.0
    for rank, (p, original_index) in enumerate(indexed):
        value = min((m - rank) * p, 1.0)
        running_max = max(running_max, value)  # monotonicidade
        adjusted[original_index] = running_max
    return adjusted


def bootstrap_ci_mean(differences: list[float], n_boot: int = 10_000, seed: int = 0) -> tuple[float, float] | None:
    if not differences:
        return None
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(
        statistics.fmean(rng.choice(differences) for _ in range(n)) for _ in range(n_boot)
    )
    lo = means[max(0, int(0.025 * n_boot) - 1)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return (float(lo), float(hi))


def per_instance_mean_losses(
    records: list[dict[str, Any]],
    label_suffix: str = ":test",
    unit_key: Callable[[str], str] | None = None,
) -> dict[str, dict[str, float]]:
    """Agrega perdas por (braço, unidade) — média sobre seeds, depois sobre instâncias.

    A unidade experimental é a instância; seeds são replicações internas e
    NÃO entram nos testes como amostras independentes (parecer §11.1).
    ``unit_key`` agrupa instâncias que não são independentes (no MPV, as
    classes com o mesmo (n, v) compartilham a seed do gerador): cada unidade
    vira uma única observação, a média das suas instâncias.
    """
    per_instance = _per_instance_means(records, label_suffix)
    if unit_key is None:
        return per_instance
    units: dict[str, dict[str, list[float]]] = {}
    for arm, instances in per_instance.items():
        for instance_id, value in instances.items():
            units.setdefault(arm, {}).setdefault(unit_key(instance_id), []).append(value)
    return {arm: {unit: statistics.fmean(values) for unit, values in grouped.items()} for arm, grouped in units.items()}


def _per_instance_means(records: list[dict[str, Any]], label_suffix: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for record in records:
        label = str(record.get("label") or "")
        if not label.endswith(label_suffix):
            continue
        arm = label[: -len(label_suffix)]
        instance_id = str(record.get("instance_id"))
        loss = record.get("loss")
        if loss is None:
            continue
        grouped.setdefault(arm, {}).setdefault(instance_id, []).append(float(loss))
    return {
        arm: {instance: statistics.fmean(values) for instance, values in instances.items()}
        for arm, instances in grouped.items()
    }


PRIMARY_BASELINES = (
    "fixed_default",
    "random_search",
    "classic_ego",
    "smac3",
    "smac3_ac",
    "irace",
    "per_class",
    "nearest_neighbor",
)
ABLATION_BASELINES = ("ablation_min_y_incumbent", "ablation_gp_fixed_hyperparams")


def predefined_comparisons(
    losses_by_arm: dict[str, dict[str, float]],
    proposal: str = "noisy_contextual_ego",
    baselines: tuple[str, ...] = PRIMARY_BASELINES,
    bootstrap_seed: int = 0,
    family: str = "primary",
) -> list[dict[str, Any]]:
    """Comparações pré-definidas: proposta vs. cada baseline presente.

    diferença = baseline − proposta (positivo favorece a proposta). Holm é
    aplicado dentro da família (``primary``: baselines; ``ablation``:
    variantes da proposta), como pré-registrado.
    """
    proposal_losses = losses_by_arm.get(proposal)
    if not proposal_losses:
        return []
    comparisons: list[dict[str, Any]] = []
    p_values: list[float | None] = []
    for baseline in baselines:
        baseline_losses = losses_by_arm.get(baseline)
        if not baseline_losses:
            continue
        shared = sorted(set(proposal_losses) & set(baseline_losses))
        if not shared:
            continue
        diffs = [baseline_losses[i] - proposal_losses[i] for i in shared]
        test = wilcoxon_signed_rank(diffs)
        wins = sum(1 for d in diffs if d > 1e-12)
        losses_count = sum(1 for d in diffs if d < -1e-12)
        comparisons.append(
            {
                "comparison": f"{proposal} vs {baseline}",
                "family": family,
                "baseline": baseline,
                "n_instances": len(shared),
                "mean_difference": statistics.fmean(diffs),
                "median_difference": statistics.median(diffs),
                "bootstrap_ci_95": bootstrap_ci_mean(diffs, seed=bootstrap_seed),
                # Empirical fraction of instances on which the proposal wins.
                # This is a WIN RATE (descriptive), not a probabilistic estimate
                # of P(win) — naming kept precise per technical review (sec. 9).
                "win_rate": wins / len(diffs) if diffs else None,
                "wins": wins,
                "ties": len(diffs) - wins - losses_count,
                "losses": losses_count,
                **{f"wilcoxon_{k}": v for k, v in test.items()},
            }
        )
        p_values.append(test.get("p_value"))
    adjusted = holm_correction(p_values)
    for comparison, p_holm in zip(comparisons, adjusted, strict=True):
        comparison["p_value_holm"] = p_holm
    return comparisons
