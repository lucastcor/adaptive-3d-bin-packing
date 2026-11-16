"""Sugestão de configuração por instância a partir das observações persistidas.

Fecha o ciclo do aprendizado online: as observações gravadas pelo
``observe_and_train`` alimentam um GP contextual e o Expected Improvement
(com incumbente plug-in pela média posterior, avaliado sob o contexto da
instância consultada) ranqueia configurações candidatas amostradas do espaço
de busca. Com poucas observações (< ``min_observations``) o fallback é
amostragem aleatória, sinalizada na resposta.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

import numpy as np

from adaptive_3dbp.configurators.classic_ego import expected_improvement
from adaptive_3dbp.domain.models import ProblemInstance, SolverConfig
from adaptive_3dbp.features.instance_features import extract_instance_features
from adaptive_3dbp.learning.online_gpr import FEATURE_NAMES, config_vector
from adaptive_3dbp.surrogates.exact_gp import ExactGaussianProcess


DEFAULT_SEARCH_SPACE: dict[str, Any] = {
    "target_initial_acceptance": {"type": "float", "low": 0.55, "high": 0.95},
    "cooling_rate": {"type": "float", "low": 0.88, "high": 0.995},
    "iterations_per_temperature": {"type": "int", "low": 20, "high": 500, "log": True},
    "stagnation_limit": {"type": "int", "low": 100, "high": 5000, "log": True},
    "ruin_size": {"type": "int", "low": 1, "high": 12},
    "multi_start_count": {"type": "int", "low": 1, "high": 8},
    "operator_probabilities": {
        "type": "simplex",
        "operators": ["swap", "relocate", "reverse", "rotate", "ruin_recreate"],
    },
}


def _sample_config(definition: dict[str, Any], rng: random.Random) -> SolverConfig:
    from adaptive_3dbp.configurators.search_space import SearchSpace

    return SearchSpace(definition).sample(rng)


def suggest_configs(
    instance: ProblemInstance,
    observations: list[dict[str, Any]],
    n_suggestions: int = 3,
    seed: int = 0,
    pool_size: int = 256,
    min_observations: int = 4,
    search_space_definition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    definition = search_space_definition or DEFAULT_SEARCH_SPACE
    rng = random.Random(seed)
    features = extract_instance_features(instance)
    context = [float(features.get(name, 0.0)) for name in FEATURE_NAMES]

    pool = [_sample_config(definition, rng) for _ in range(pool_size)]

    if len(observations) < min_observations:
        return {
            "mode": "random_fallback",
            "observation_count": len(observations),
            "reason": f"menos de {min_observations} observações: sugerindo amostras aleatórias do espaço de busca",
            "suggestions": [
                {"config": config.model_dump(), "predicted_loss": None, "uncertainty": None, "expected_improvement": None}
                for config in pool[:n_suggestions]
            ],
        }

    x_rows: list[list[float]] = []
    y: list[float] = []
    noise: list[float] = []
    stored_configs: list[SolverConfig] = []
    for row in observations:
        try:
            x_rows.append(json.loads(str(row["x_json"])))
            y.append(float(row["target"]))
            noise.append(max(float(row["noise_variance"]), 1e-9))
            stored_configs.append(SolverConfig.model_validate_json(str(row["config_json"])))
        except (KeyError, ValueError):
            continue

    gp = ExactGaussianProcess()
    n = len(y)
    gp.fit(
        np.asarray(x_rows, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(noise, dtype=np.float64),
        optimize_hyperparameters=True,
        max_iterations=120 if n <= 150 else 50,
        n_restarts=1 if n <= 150 else 0,
    )

    # Incumbente plug-in contextual: configurações conhecidas sob o contexto consultado.
    incumbent_rows = [context + config_vector(config) for config in stored_configs]
    incumbent_prediction = gp.predict(np.asarray(incumbent_rows, dtype=np.float64))
    incumbent = float(np.min(incumbent_prediction.mean))

    pool_rows = [context + config_vector(config) for config in pool]
    prediction = gp.predict(np.asarray(pool_rows, dtype=np.float64))
    scored = []
    for config, mean, std in zip(pool, prediction.mean, prediction.std, strict=True):
        mean_f, std_f = float(mean), float(std)
        if not (math.isfinite(mean_f) and math.isfinite(std_f)):
            continue
        scored.append(
            {
                "config": config,
                "predicted_loss": mean_f,
                "uncertainty": std_f,
                "expected_improvement": expected_improvement(mean_f, std_f, incumbent),
            }
        )
    scored.sort(key=lambda item: item["expected_improvement"], reverse=True)

    return {
        "mode": "contextual_ego",
        "observation_count": len(observations),
        "incumbent_predicted_loss": incumbent,
        "suggestions": [
            {
                "config": item["config"].model_dump(),
                "predicted_loss": item["predicted_loss"],
                "uncertainty": item["uncertainty"],
                "expected_improvement": item["expected_improvement"],
            }
            for item in scored[:n_suggestions]
        ],
    }
