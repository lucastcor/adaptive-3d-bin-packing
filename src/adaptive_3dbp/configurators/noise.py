from __future__ import annotations

import numpy as np


NOISE_FLOOR_RATIO = 0.05  # piso: fração da variância global (evita colapso a ~0)
SHRINKAGE_PSEUDO_REPLICATIONS = 2.0  # força do prior global no shrinkage


def shrunk_noise_variances(variances: list[float], replications: list[int]) -> list[float]:
    """Shrinkage das variâncias locais para a variância global + piso.

    Com poucas replicações (ex.: 2 seeds) a variância amostral tem ~1 grau de
    liberdade e pode colapsar para ~0 ou explodir. Aplicamos o estimador
    ``σ̃²_i = λ_i s²_i + (1 − λ_i) σ²_global`` com ``λ_i = (r_i − 1)/(r_i − 1 + r₀)``
    (λ cresce com o número de replicações; r₀ = pseudo-replicações do prior) e
    um piso proporcional à variância global. A variância usada pelo GP é a do
    estimador da média: ``σ̃²_i / r_i``.
    """
    global_variance = max(float(np.mean(variances)) if variances else 1e-6, 1e-6)
    floor = NOISE_FLOOR_RATIO * global_variance
    shrunk: list[float] = []
    for variance, reps in zip(variances, replications, strict=True):
        degrees = max(reps - 1, 0)
        lam = degrees / (degrees + SHRINKAGE_PSEUDO_REPLICATIONS)
        blended = lam * float(variance) + (1.0 - lam) * global_variance
        shrunk.append(max(blended, floor) / max(reps, 1))
    return shrunk


def mean_noise_variance(grouped_losses: list[list[float]]) -> float:
    """Variance of the mean loss over instances x seeds, from seed noise only.

    For a fixed instance set the between-instance spread is the same for every
    configuration, so it is not observation noise; only the seed variance of
    each instance is. Var(mean) = sum_i (s_i^2 / S_i) / I^2.
    """
    groups = [group for group in grouped_losses if group]
    if not groups:
        return 1e-6
    total = 0.0
    for group in groups:
        values = np.asarray(group, dtype=np.float64)
        variance = float(values.var(ddof=1)) if len(values) > 1 else 0.0
        total += variance / len(values)
    return max(total / (len(groups) ** 2), 1e-8)
