from __future__ import annotations

import math
import random

import numpy as np

from adaptive_3dbp.configurators.noise import NOISE_FLOOR_RATIO
from adaptive_3dbp.configurators.search_space import SearchSpace
from adaptive_3dbp.domain.models import SolverConfig
from adaptive_3dbp.surrogates.exact_gp import ExactGaussianProcess


def expected_improvement(mean: float, std: float, best: float) -> float:
    if std <= 1e-12:
        return max(0.0, best - mean)
    improvement = best - mean
    z = improvement / std
    normal_pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    normal_cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return improvement * normal_cdf + std * normal_pdf


def plugin_incumbent(gp: ExactGaussianProcess, observed_x: np.ndarray) -> float:
    """Plug-in incumbent for noisy observations: min of the GP posterior mean.

    Using min(y) as incumbent under noise is biased low ("winner's curse"):
    lucky replications make the target look better than the latent function,
    shrinking EI and stalling the search. The plug-in strategy evaluates the
    posterior mean at the observed inputs and takes its minimum (Picheny,
    Wagner & Ginsbourger, 2013; Vazquez et al., 2008; same motivation as the
    Noisy EI of Letham et al., 2019, at a fraction of the cost).
    """
    prediction = gp.predict(observed_x)
    return float(np.min(prediction.mean))


class ClassicEGOConfigurator:
    name = "classic_ego"

    def __init__(self, search_space: SearchSpace, seed: int = 0, candidate_pool_size: int = 512) -> None:
        self.search_space = search_space
        self.rng = random.Random(seed)
        self.candidate_pool_size = candidate_pool_size
        self.configs: list[SolverConfig] = []
        self.targets: list[float] = []
        self.noise_variances: list[float | None] = []

    def observe(self, config: SolverConfig, target: float, noise_variance: float | None = None) -> None:
        """``noise_variance`` is the variance of ``target`` as an estimate (e.g.
        seed noise of a mean over instances x seeds). Without it the GP
        interpolates the targets (noiseless, jitter only)."""
        self.configs.append(config)
        self.targets.append(target)
        self.noise_variances.append(noise_variance)

    def _noise(self) -> np.ndarray | None:
        known = [value for value in self.noise_variances if value is not None]
        if not known:
            return None
        fallback = float(np.median(known))
        # Same relative floor as the contextual arm: tied seeds give a zero
        # sample variance that would make the GP interpolate that point.
        floor = NOISE_FLOOR_RATIO * max(float(np.mean(known)), 1e-12)
        values = [fallback if value is None else value for value in self.noise_variances]
        return np.asarray([max(value, floor) for value in values], dtype=np.float64)

    def suggest(self, n_suggestions: int = 1) -> list[SolverConfig]:
        if len(self.configs) < 3:
            return [self.search_space.sample(self.rng) for _ in range(n_suggestions)]

        x = np.asarray([self.search_space.encode(config) for config in self.configs], dtype=np.float64)
        y = np.asarray(self.targets, dtype=np.float64)
        gp = ExactGaussianProcess()
        gp.fit(x, y, self._noise())
        pool = [self.search_space.sample(self.rng) for _ in range(self.candidate_pool_size)]
        pool_x = np.asarray([self.search_space.encode(config) for config in pool], dtype=np.float64)
        prediction = gp.predict(pool_x)
        best = plugin_incumbent(gp, x)
        scores = [
            expected_improvement(float(mean), float(std), best)
            for mean, std in zip(prediction.mean, prediction.std, strict=True)
        ]
        ranked = sorted(zip(scores, pool, strict=True), key=lambda item: item[0], reverse=True)
        return [config for _score, config in ranked[:n_suggestions]]

