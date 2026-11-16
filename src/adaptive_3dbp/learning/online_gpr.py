from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adaptive_3dbp.domain.models import ProblemInstance, SolverConfig, SolverResult
from adaptive_3dbp.features.instance_features import extract_instance_features
from adaptive_3dbp.surrogates.exact_gp import ExactGaussianProcess


FEATURE_NAMES = (
    "n_items",
    "volume_ratio",
    "volume_lower_bound",
    "mean_item_volume_ratio",
    "std_item_volume_ratio",
    "max_item_volume_ratio",
    "mean_normalized_dimension",
    "std_normalized_dimension",
    "mean_aspect_ratio",
    "max_aspect_ratio",
    "fraction_large_items",
    "fraction_elongated_items",
    "fraction_nearly_cubic_items",
    "mean_feasible_rotations",
    "size_entropy",
    "dimensional_cv",
)

CONFIG_FEATURE_NAMES = (
    "target_initial_acceptance",
    "cooling_rate",
    "iterations_per_temperature",
    "min_temperature",
    "max_evaluations",
    "time_limit_seconds",
    "stagnation_limit",
    "reheating_factor",
    "multi_start_count",
    "ruin_size",
    "use_native_core",
    "parallel_workers",
    "operator_swap",
    "operator_relocate",
    "operator_reverse",
    "operator_rotate",
    "operator_ruin_recreate",
)


@dataclass(frozen=True, slots=True)
class LearningObservation:
    run_id: int
    instance_id: str
    x: list[float]
    target: float
    noise_variance: float
    context_json: str
    config_json: str
    result_json: str


# Perda — pré-definida ANTES das campanhas (parecer §8).
# Métrica primária: gap ao lower bound (o objetivo do 3D-BPP é minimizar
# caixas); falha entra como penalidade dominante. O tempo NÃO entra por padrão:
# na campanha 632591ae o termo 0.01·log1p(runtime) decidia sozinho 6 de 7
# "vitórias" (7/10 instâncias de teste tinham gap idêntico em todos os
# métodos) e media contenção de CPU, não qualidade. Tempo segue reportado à
# parte (runtime_seconds) e pode ser reativado explicitamente com
# runtime_weight > 0 para estudos de custo. Avaliações não são penalizadas:
# o orçamento por run já é fixo em avaliações.
LOSS_WEIGHT_RUNTIME = 0.0
LOSS_FAILURE_PENALTY = 10.0
ONLINE_NOISE_VARIANCE = 1e-3


def performance_loss(result: SolverResult, runtime_weight: float = LOSS_WEIGHT_RUNTIME) -> float:
    failure_penalty = LOSS_FAILURE_PENALTY if result.status != "feasible" else 0.0
    runtime_penalty = runtime_weight * math.log1p(max(result.runtime_seconds, 0.0)) if runtime_weight else 0.0
    return float(result.gap_to_lower_bound + runtime_penalty + failure_penalty)


def config_vector(config: SolverConfig) -> list[float]:
    probabilities = config.operator_probabilities
    return [
        float(config.target_initial_acceptance),
        float(config.cooling_rate),
        float(config.iterations_per_temperature),
        float(config.min_temperature),
        float(config.max_evaluations),
        float(config.time_limit_seconds),
        float(config.stagnation_limit),
        float(config.reheating_factor),
        float(config.multi_start_count),
        float(config.ruin_size),
        1.0 if config.use_native_core else 0.0,
        float(config.parallel_workers),
        float(probabilities.get("swap", 0.0)),
        float(probabilities.get("relocate", 0.0)),
        float(probabilities.get("reverse", 0.0)),
        float(probabilities.get("rotate", 0.0)),
        float(probabilities.get("ruin_recreate", 0.0)),
    ]


def observation_vector(instance: ProblemInstance, config: SolverConfig) -> tuple[list[float], dict[str, float]]:
    features = extract_instance_features(instance)
    x = [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
    x.extend(config_vector(config))
    return x, features


def build_learning_observation(
    run_id: int,
    instance: ProblemInstance,
    config: SolverConfig,
    result: SolverResult,
) -> LearningObservation:
    x, features = observation_vector(instance, config)
    # Single-run observations are noisy. Constant prior for the seed noise of
    # the gap (std ~0.03, about one bin in 30 on the campaign instances). The
    # previous 1e-4 + 1e-6*evaluations made the noise GROW with the budget,
    # the opposite of what more search does. Replications can later replace
    # this with the empirical variance per configuration.
    noise_variance = ONLINE_NOISE_VARIANCE
    return LearningObservation(
        run_id=run_id,
        instance_id=instance.instance_id,
        x=x,
        target=performance_loss(result),
        noise_variance=noise_variance,
        context_json=json.dumps(features, sort_keys=True),
        config_json=config.model_dump_json(),
        result_json=result.model_dump_json(),
    )


def train_online_gpr(
    observations: list[dict[str, object]],
    artifact_root: Path,
    min_observations: int = 3,
) -> dict[str, object]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    if len(observations) < min_observations:
        return {
            "trained": False,
            "observation_count": len(observations),
            "reason": f"waiting for at least {min_observations} observations",
        }

    x = np.asarray([json.loads(str(row["x_json"])) for row in observations], dtype=np.float64)
    y = np.asarray([float(row["target"]) for row in observations], dtype=np.float64)
    noise = np.asarray([float(row["noise_variance"]) for row in observations], dtype=np.float64)
    # Piso de ruído: observações de run único carregam variâncias declarativas
    # que podem colapsar para ~0; um piso proporcional à variância média
    # estabiliza a Cholesky e evita sobreconfiança (parecer §6.3).
    noise = np.maximum(noise, 0.05 * max(float(np.mean(noise)), 1e-8))
    gp = ExactGaussianProcess()
    # MAP hyperparameter optimisation is O(n^3) per step; keep the online path
    # responsive by shrinking the budget as the dataset grows. Full-budget
    # refits belong to offline campaigns.
    n = x.shape[0]
    gp.fit(
        x,
        y,
        noise,
        optimize_hyperparameters=True,
        max_iterations=150 if n <= 150 else 60,
        n_restarts=2 if n <= 150 else 0,
    )

    last_run_id = max(int(row["run_id"]) for row in observations)
    artifact_path = artifact_root / f"online_gpr_run_{last_run_id}.npz"
    latest_artifact_path = artifact_root / "online_gpr_latest.npz"
    gp.save(artifact_path)
    gp.save(latest_artifact_path)
    metadata = {
        "trained": True,
        "model_type": "exact_gp_matern_5_2",
        "observation_count": len(observations),
        "feature_names": [*FEATURE_NAMES, *CONFIG_FEATURE_NAMES],
        "target": "gap_to_lower_bound + failure_penalty",
        "artifact_path": str(artifact_path),
        "latest_artifact_path": str(latest_artifact_path),
        "target_min": float(np.min(y)),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
    }
    (artifact_root / f"online_gpr_run_{last_run_id}.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (artifact_root / "online_gpr_latest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
