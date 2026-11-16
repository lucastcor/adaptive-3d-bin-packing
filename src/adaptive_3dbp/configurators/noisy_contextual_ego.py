from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from adaptive_3dbp.configurators.classic_ego import expected_improvement
from adaptive_3dbp.configurators.noise import shrunk_noise_variances as _shrunk_noise_variances
from adaptive_3dbp.configurators.search_space import SearchSpace
from adaptive_3dbp.domain.models import SolverConfig
from adaptive_3dbp.surrogates.exact_gp import ExactGaussianProcess


@dataclass(slots=True)
class ContextualObservation:
    context: dict[str, float]
    config: SolverConfig
    target_mean: float
    target_variance: float
    replications: int


@dataclass(slots=True)
class NoiseAwareContextualEGO:
    search_space: SearchSpace
    seed: int = 0
    candidate_pool_size: int = 512
    observations: list[ContextualObservation] = field(default_factory=list)
    rng: random.Random = field(default=None, repr=False)  # type: ignore[assignment]
    # Flags de ablation (estudo de quais componentes geram o ganho):
    use_plugin_incumbent: bool = True   # False -> incumbente min(y) ingênuo (viesado sob ruído)
    optimize_gp_hyperparameters: bool = True  # False -> lengthscales fixos (sem MLE/MAP)
    # Features de contexto usadas pelo GP. None = todas as chaves do contexto
    # (ordenadas). A campanha passa FEATURE_NAMES (16 features adimensionais).
    feature_names: tuple[str, ...] | None = None

    name: str = "noisy_contextual_ego"
    # GP ajustado para o histórico atual. O ajuste (MLE com sementes fixas) é
    # determinístico, e a política zero-shot consulta o mesmo histórico
    # congelado para cada instância de validação/teste: sem cache, eram 90
    # ajustes O(n³) idênticos por braço com n ≈ 1200 observações.
    # A chave guarda a última observação (comparada por identidade): manter a
    # referência impede que outro objeto reaproveite o mesmo id.
    _gp_cache: tuple[tuple[int, tuple[str, ...]], ContextualObservation, ExactGaussianProcess, list[float]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def observe(
        self,
        context: dict[str, float],
        config: SolverConfig,
        targets: list[float],
    ) -> None:
        target_array = np.asarray(targets, dtype=np.float64)
        variance = float(target_array.var(ddof=1)) if len(target_array) > 1 else 1e-6
        self.observations.append(
            ContextualObservation(
                context=context,
                config=config,
                target_mean=float(target_array.mean()),
                target_variance=max(variance, 1e-6),
                replications=len(targets),
            )
        )

    def _feature_names(self, context: dict[str, float]) -> list[str]:
        return list(self.feature_names) if self.feature_names is not None else sorted(context)

    def _rows(self, context: dict[str, float], feature_names: list[str], configs: list[SolverConfig]) -> np.ndarray:
        prefix = [float(context.get(name, 0.0)) for name in feature_names]
        return np.asarray([[*prefix, *self.search_space.encode(config)] for config in configs], dtype=np.float64)

    def _fit_gp(self, feature_names: list[str]) -> tuple[ExactGaussianProcess, list[float]]:
        key = (len(self.observations), tuple(feature_names))
        last = self.observations[-1]
        cache = self._gp_cache
        if cache is not None and cache[0] == key and cache[1] is last:
            return cache[2], cache[3]
        x_rows: list[list[float]] = []
        y: list[float] = []
        raw_variances: list[float] = []
        replication_counts: list[int] = []
        for observation in self.observations:
            row = [float(observation.context.get(name, 0.0)) for name in feature_names]
            row.extend(self.search_space.encode(observation.config))
            x_rows.append(row)
            y.append(observation.target_mean)
            raw_variances.append(observation.target_variance)
            replication_counts.append(max(observation.replications, 1))
        noise = _shrunk_noise_variances(raw_variances, replication_counts)

        gp = ExactGaussianProcess()
        gp.fit(
            np.asarray(x_rows, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(noise, dtype=np.float64),
            optimize_hyperparameters=self.optimize_gp_hyperparameters,
        )
        self._gp_cache = (key, last, gp, y)
        return gp, y

    def recommend(self, context: dict[str, float], pool_seed: int = 0) -> SolverConfig:
        """Zero-shot recommendation for a new instance: argmin of the posterior mean.

        EI is an acquisition criterion (it rewards uncertainty to explore);
        a configuration we commit to without further runs should minimise the
        predicted loss instead. Candidates are every configuration observed in
        training plus a pool drawn from ``pool_seed``, which the caller derives
        from the instance, so the recommendation does not depend on the
        tuner's RNG state or on the order in which instances are queried.
        """
        if not self.observations:
            return self.search_space.sample(random.Random(pool_seed))
        if len(self.observations) < 4:
            return min(self.observations, key=lambda observation: observation.target_mean).config
        feature_names = self._feature_names(context)
        gp, _y = self._fit_gp(feature_names)
        pool_rng = random.Random(pool_seed)
        candidates = [observation.config for observation in self.observations]
        candidates.extend(self.search_space.sample(pool_rng) for _ in range(self.candidate_pool_size))
        prediction = gp.predict(self._rows(context, feature_names, candidates))
        return candidates[int(np.argmin(prediction.mean))]

    def suggest(self, context: dict[str, float], n_suggestions: int = 1) -> list[SolverConfig]:
        if len(self.observations) < 4:
            return [self.search_space.sample(self.rng) for _ in range(n_suggestions)]

        feature_names = self._feature_names(context)
        gp, y = self._fit_gp(feature_names)

        pool = [self.search_space.sample(self.rng) for _ in range(self.candidate_pool_size)]
        prediction = gp.predict(self._rows(context, feature_names, pool))
        # Contextual plug-in incumbent: evaluate every *known* configuration
        # under the *query* context and take the minimum posterior mean. Using
        # min of empirical means is biased low under noise (winner's curse;
        # Picheny et al., 2013; Letham et al., 2019) and, in the contextual
        # setting, would also mix incomparable contexts.
        if self.use_plugin_incumbent:
            observed = [observation.config for observation in self.observations]
            incumbent_prediction = gp.predict(self._rows(context, feature_names, observed))
            incumbent = float(np.min(incumbent_prediction.mean))
        else:
            # Braço de ablation: incumbente ingênuo min(y).
            incumbent = float(min(y))
        scores = [
            expected_improvement(float(mean), float(std), incumbent)
            for mean, std in zip(prediction.mean, prediction.std, strict=True)
        ]
        ranked = sorted(zip(scores, pool, strict=True), key=lambda item: item[0], reverse=True)
        return [config for _score, config in ranked[:n_suggestions]]

