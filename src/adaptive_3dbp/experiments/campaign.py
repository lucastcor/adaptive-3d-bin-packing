"""Runner de campanha experimental reprodutível.

Executa o protocolo completo do plano experimental: tuning no split *train*,
reavaliação estática no *validation* (seeds novas; não escolhe incumbente) e
avaliação única no *test*; braços incluem o baseline
de configuração fixa padrão, Random Search, EGO clássico (sem contexto), EGO
contextual ciente de ruído, SMAC3/irace quando disponíveis e os braços de
ablation (incumbente min(y) e GP sem MLE). Tudo é gravado com metadados
completos (versão do código, hardware, timestamp, spec) e exportado em
JSON + CSV (+ Parquet quando pyarrow está instalado) com relatório Markdown.
"""

from __future__ import annotations

import csv
import json
import platform
import re
import statistics
import subprocess
import sys
import time
import uuid
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from adaptive_3dbp import __version__
from adaptive_3dbp.configurators.classic_ego import ClassicEGOConfigurator
from adaptive_3dbp.configurators.noise import mean_noise_variance
from adaptive_3dbp.configurators.noisy_contextual_ego import NoiseAwareContextualEGO
from adaptive_3dbp.configurators.random_search import RandomSearchConfigurator
from adaptive_3dbp.configurators.search_space import SearchSpace
from adaptive_3dbp.domain.models import ProblemInstance, SolverConfig
from adaptive_3dbp.experiments.instances import InstanceClass, generate_dataset
from adaptive_3dbp.features.instance_features import extract_instance_features
from adaptive_3dbp.learning.suggest import DEFAULT_SEARCH_SPACE
from adaptive_3dbp.learning.online_gpr import FEATURE_NAMES, performance_loss
from adaptive_3dbp.tuning.per_instance_baselines import (
    Observation,
    nearest_neighbor_policy,
    per_class_policy,
)
from adaptive_3dbp.system.profile import system_info
from adaptive_3dbp.tuning.objective import TuningObjective, _solve_with_hard_timeout
from adaptive_3dbp.tuning.smac_runner import run_smac, run_smac_ac, smac_available
from adaptive_3dbp.tuning.irace_runner import (
    best_values_from_campaign,
    generate_irace_campaign,
    irace_available,
    run_irace,
)
from adaptive_3dbp.tuning.objective import values_to_config

ProgressCallback = Callable[[dict[str, Any]], None]

BASE_ARMS = ("fixed_default", "random_search", "classic_ego", "noisy_contextual_ego", "smac3", "smac3_ac", "irace")
ABLATION_ARMS = ("ablation_min_y_incumbent", "ablation_gp_fixed_hyperparams")
# Baselines de configuração POR INSTÂNCIA simples (parecer §3.6/§3.8): escolhem
# a configuração de cada instância-alvo a partir do histórico de treino, sem
# nenhum run da instância-alvo (zero-shot). São concorrentes diretos do EGO
# contextual e respondem se o GP agrega valor além de um mapa classe→config.
PER_INSTANCE_BASELINE_ARMS = ("per_class", "nearest_neighbor")

# Rótulos declaram o PROBLEMA que cada braço resolve (parecer §5.1): SMAC,
# irace, random search e EGO clássico produzem UMA configuração global (single
# best configuration); o EGO contextual produz uma configuração POR instância
# em regime zero-shot (só features da instância nova; nenhum run dela é usado
# para escolher a configuração). São baselines, não concorrentes do mesmo
# problema — a comparação responde: "a política por instância supera a melhor
# configuração global?".
ARM_LABELS = {
    "fixed_default": "Configuração fixa padrão (baseline)",
    "random_search": "Random Search (config. global)",
    "classic_ego": "EGO clássico (config. global, sem contexto)",
    "noisy_contextual_ego": "EGO contextual (zero-shot por instância)",
    "smac3": "SMAC3 HPO (config. global)",
    "smac3_ac": "SMAC3 AC com instâncias e features (config. global)",
    "irace": "irace (config. global)",
    # Afeta só a aquisição (EI) durante o tuning: a recomendação zero-shot
    # usa o mínimo da média posterior nos dois casos.
    "ablation_min_y_incumbent": "Ablation: incumbente min(y) no EI",
    "ablation_gp_fixed_hyperparams": "Ablation: GP sem MLE",
    "per_class": "Config. por classe (baseline por instância)",
    "nearest_neighbor": "Vizinho mais próximo / kNN (baseline por instância)",
}

# Braços que aprendem uma política por instância π(z) e são avaliados em
# regime zero-shot (Protocolo A do parecer).
PER_INSTANCE_ARMS = ("noisy_contextual_ego", "ablation_min_y_incumbent", "ablation_gp_fixed_hyperparams")

_CLASS_SUFFIX_RE = re.compile(r"-(train|validation|test)-\d+-slot\d+$")
# MPV benchmark ids: "mpv-c<class>-n<n>-v<index>" -> class key "mpv-c<class>-n<n>".
_MPV_ID_RE = re.compile(r"^(mpv-c\d+-n\d+)-v\d+$")


def _instance_class_key(instance_id: str) -> str:
    """Classe da instância a partir do id gerado (ex.: 'small-high-elongated-
    homogeneous-nodist-test-0-slot0' -> 'small-high-elongated-homogeneous-nodist').

    Se o id não segue o formato do gerador, devolve o próprio id (a política
    por classe cai no fallback global para instâncias de classe desconhecida).
    """
    mpv = _MPV_ID_RE.match(instance_id)
    if mpv:
        return mpv.group(1)
    stripped = _CLASS_SUFFIX_RE.sub("", instance_id)
    return stripped or instance_id


def _feature_vector(instance: ProblemInstance) -> tuple[float, ...]:
    feats = extract_instance_features(instance)
    return tuple(float(feats.get(name, 0.0)) for name in FEATURE_NAMES)


def _tune_per_instance_baseline(
    arm: str,
    objective: TuningObjective,
    space_definition: dict[str, Any],
    trials: int,
    tuner_seed: int,
) -> tuple[SolverConfig | None, float | None, str | None, Callable[[ProblemInstance], SolverConfig] | None]:
    """Baseline de configuração por instância (per_class / nearest_neighbor).

    Orçamento idêntico ao Random Search (trials configs × instâncias de treino ×
    seeds). Cada config candidata é avaliada em TODAS as instâncias de treino,
    registrando a perda média por (instância, config). A política então escolhe,
    para cada instância-alvo nova, uma config do histórico de treino congelado
    usando apenas a classe (per_class) ou as features (nearest_neighbor) da
    instância-alvo — zero-shot, sem nenhum run da instância-alvo.
    """
    space = SearchSpace(space_definition)
    configs = list(RandomSearchConfigurator(space, seed=tuner_seed).suggest(trials))
    if not configs:
        return None, None, f"{arm} sem trials", None

    registry: dict[str, SolverConfig] = {}
    observations: list[Observation] = []
    train_instances = objective.instances
    best_loss, best_config = float("inf"), None
    for idx, config in enumerate(configs):
        key = str(idx)
        registry[key] = config
        grouped = objective.replicate_losses_grouped(config)  # por instância -> [perdas por seed]
        config_mean_losses: list[float] = []
        for instance, losses in zip(train_instances, grouped, strict=True):
            mean_loss = statistics.fmean(losses) if losses else float("inf")
            config_mean_losses.append(mean_loss)
            observations.append(
                Observation(
                    instance_id=instance.instance_id,
                    instance_class=_instance_class_key(instance.instance_id),
                    features=_feature_vector(instance),
                    config={"key": key},
                    loss=mean_loss,
                    config_key=key,
                )
            )
        config_loss = statistics.fmean(config_mean_losses) if config_mean_losses else float("inf")
        if config_loss < best_loss:
            best_loss, best_config = config_loss, config

    policy_fn = per_class_policy if arm == "per_class" else nearest_neighbor_policy

    def recommend(instance: ProblemInstance) -> SolverConfig:
        target = Observation(
            instance_id=instance.instance_id,
            instance_class=_instance_class_key(instance.instance_id),
            features=_feature_vector(instance),
            config={},
            loss=0.0,
        )
        chosen = policy_fn(observations, [target])
        picked = chosen.get(instance.instance_id)
        if picked and "key" in picked:
            return registry[picked["key"]]
        return best_config or SolverConfig()

    return best_config, best_loss, None, recommend


def _select_campaign_classes(available: list[InstanceClass], limit: int) -> list[InstanceClass]:
    """Select a tractable, varied pilot subset instead of only the hardest classes.

    A campaign with all classes at medium/high occupancy saturates the decoder
    before the tuners can be compared. The ordered seed below mixes occupancy,
    size, shape and heterogeneity while keeping the first few classes usable on
    a laptop; remaining slots are filled greedily for feature coverage.
    """
    if limit <= 0:
        return []
    by_key = {
        (spec.size, spec.occupancy, spec.shape, spec.heterogeneity, spec.min_distance): spec for spec in available
    }
    preferred = [
        ("small", "low", "elongated", "homogeneous", False),
        ("small", "high", "elongated", "homogeneous", False),
        ("small", "low", "mixed", "homogeneous", False),
        ("small", "low", "cubic", "homogeneous", False),
        ("small", "high", "mixed", "heterogeneous", False),
        ("medium", "low", "cubic", "homogeneous", False),
        ("small", "low", "elongated", "heterogeneous", False),
        ("medium", "low", "mixed", "homogeneous", False),
        ("medium", "low", "cubic", "heterogeneous", False),
        ("medium", "low", "elongated", "heterogeneous", False),
        ("medium", "low", "mixed", "heterogeneous", False),
        ("small", "high", "cubic", "heterogeneous", False),
        ("medium", "high", "cubic", "homogeneous", False),
        ("medium", "high", "mixed", "homogeneous", False),
        ("large", "low", "cubic", "homogeneous", False),
        ("large", "low", "mixed", "heterogeneous", False),
        ("medium", "high", "elongated", "homogeneous", False),
    ]
    selected: list[InstanceClass] = []
    seen: set[InstanceClass] = set()
    for key in preferred:
        spec = by_key.get(key)
        if spec is not None and spec not in seen:
            selected.append(spec)
            seen.add(spec)
            if len(selected) >= limit:
                return selected

    def score(spec: InstanceClass) -> tuple[int, int, int, int, int]:
        covered_size = {picked.size for picked in selected}
        covered_occupancy = {picked.occupancy for picked in selected}
        covered_shape = {picked.shape for picked in selected}
        covered_heterogeneity = {picked.heterogeneity for picked in selected}
        coverage_gain = sum(
            (
                spec.size not in covered_size,
                spec.occupancy not in covered_occupancy,
                spec.shape not in covered_shape,
                spec.heterogeneity not in covered_heterogeneity,
            )
        )
        size_cost = {"small": 0, "medium": 1, "large": 2}.get(spec.size, 3)
        occupancy_cost = {"low": 0, "high": 1}.get(spec.occupancy, 2)
        shape_cost = {"mixed": 0, "cubic": 1, "elongated": 2}.get(spec.shape, 3)
        distance_cost = 1 if spec.min_distance else 0
        return (-coverage_gain, size_cost, occupancy_cost, shape_cost, distance_cost)

    for spec in sorted((spec for spec in available if spec not in seen), key=score):
        selected.append(spec)
        if len(selected) >= limit:
            return selected
    return selected


def _estimated_arm_solver_runs(
    arm: str,
    train_count: int,
    validation_count: int,
    test_count: int,
    seed_count: int,
    trials: int,
) -> int:
    evaluation_runs = (validation_count + test_count) * seed_count
    tuning_runs = train_count * seed_count * trials
    if arm == "fixed_default":
        return evaluation_runs
    if arm in ("smac3", "smac3_ac"):
        return tuning_runs + evaluation_runs if smac_available() else 0
    if arm == "irace":
        # irace tuning runs outside this Python objective; validation/test are
        # counted only when a local irace executable can produce an incumbent.
        return evaluation_runs if irace_available() else 0
    return tuning_runs + evaluation_runs


@dataclass(slots=True)
class ArmOutcome:
    arm: str
    status: str
    # "global_config" (single best configuration) ou "per_instance_zero_shot"
    # (política π(z) avaliada sem nenhum run da instância-alvo). Parecer §5.1.
    protocol: str = "global_config"
    tuning_seconds: float = 0.0
    tuning_solver_runs: int = 0
    training_best_loss: float | None = None
    validation_loss: float | None = None
    test_loss: float | None = None
    validation_metrics: dict[str, Any] | None = None
    test_metrics: dict[str, Any] | None = None
    best_config: dict[str, Any] | None = None
    reason: str | None = None


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_revision(root: Path) -> str | None:
    return _git(root, "rev-parse", "--short", "HEAD") or None


def _git_state(root: Path) -> dict[str, Any]:
    """Revision plus whether the code differed from it when the campaign ran.

    Campaigns 632591ae/6f670fb1 recorded 19506b5 while running code that only
    landed in the next commit. ``dirty`` and the hash of ``git diff HEAD -- src``
    make that visible: a confirmatory campaign must have ``dirty == False``.
    ``cpp/`` is included because the native core is the solver when
    ``budget.native_core`` is on.
    """
    revision = _git_revision(root)
    # Untracked files count: a new module that the code imports but that is
    # not committed is just as much "code outside HEAD".
    # cpp/ too: with budget.native_core the C++ source is part of the solver.
    status = _git(root, "status", "--porcelain", "--untracked-files=normal", "--", "src", "configs", "cpp")
    diff = _git(root, "diff", "HEAD", "--", "src", "configs", "cpp")
    return {
        "git_revision": revision,
        "git_dirty": bool(status) if status is not None else None,
        "git_diff_crc32": f"{zlib.crc32(diff.encode('utf-8')):08x}" if diff else None,
    }


def _load_split(manifest: dict[str, Any], split: str) -> list[ProblemInstance]:
    instances = []
    for entry in manifest["splits"].get(split, []):
        payload = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        instances.append(ProblemInstance.model_validate(payload))
    return instances


def _mean_context(instances: list[ProblemInstance]) -> dict[str, float]:
    rows = [extract_instance_features(instance) for instance in instances]
    keys = sorted(rows[0])
    return {key: statistics.fmean(float(row[key]) for row in rows) for key in keys}


def _mean_optional(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return statistics.fmean(values) if values else None


def _quality_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [record for record in records if record.get("status") == "feasible"]
    optimal_runs = sum(1 for record in records if bool(record.get("optimal_proven")))
    return {
        "runs": len(records),
        "feasible_runs": len(feasible),
        "timeout_runs": sum(1 for record in records if record.get("status") == "timeout"),
        "error_runs": sum(1 for record in records if record.get("status") == "error"),
        "optimal_runs": optimal_runs,
        "optimal_rate": optimal_runs / len(records) if records else None,
        "mean_loss": _mean_optional(records, "loss"),
        "mean_bins_used": _mean_optional(feasible, "bins_used"),
        "best_bins_used": min((int(record["bins_used"]) for record in feasible), default=None),
        # Gap apenas sobre runs viáveis: misturar gap sentinela de runs falhos
        # com runs reais produz médias sem significado físico (ex.: "1 bin e
        # gap 0.667", impossível em runs viáveis).
        "mean_gap_to_lower_bound": _mean_optional(feasible, "gap_to_lower_bound"),
        "mean_runtime_seconds": _mean_optional(records, "runtime_seconds"),
        "mean_evaluations": _mean_optional(records, "evaluations"),
        "mean_bin_utilization": _mean_optional(feasible, "bin_utilization_mean"),
    }


def _phase_complete(metrics: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    runs = int(metrics.get("runs") or 0)
    feasible_runs = int(metrics.get("feasible_runs") or 0)
    timeout_runs = int(metrics.get("timeout_runs") or 0)
    error_runs = int(metrics.get("error_runs") or 0)
    return runs > 0 and feasible_runs == runs and timeout_runs == 0 and error_runs == 0


def _has_complete_validation(outcome: ArmOutcome) -> bool:
    return outcome.status == "ok" and outcome.validation_loss is not None and _phase_complete(outcome.validation_metrics)


def _comparison_warning(outcomes: list[ArmOutcome]) -> str | None:
    ok_outcomes = [outcome for outcome in outcomes if outcome.status == "ok" and outcome.validation_metrics]
    if not ok_outcomes:
        return None
    total_runs = sum(int((outcome.validation_metrics or {}).get("runs") or 0) for outcome in ok_outcomes)
    non_feasible = sum(
        max(
            int((outcome.validation_metrics or {}).get("runs") or 0)
            - int((outcome.validation_metrics or {}).get("feasible_runs") or 0),
            0,
        )
        for outcome in ok_outcomes
    )
    incomplete_arms = [outcome for outcome in ok_outcomes if not _has_complete_validation(outcome)]
    if not incomplete_arms:
        return None
    if total_runs <= 0:
        return "Comparação principal indisponível: nenhum run de validação foi registrado."
    return (
        f"Comparação principal apenas diagnóstica: {non_feasible}/{total_runs} runs de validação "
        "não foram viáveis ou terminaram em timeout. Não interprete empates como equivalência entre "
        "estratégias; ajuste o orçamento/dataset e rode novamente antes de usar a tabela no artigo."
    )


def _curation_probe_configs(base: SolverConfig, panel_size: int) -> list[tuple[str, SolverConfig]]:
    """Small deterministic panel used only to measure whether an instance discriminates quality.

    This is not a tuner: the panel is fixed before the campaign and is only used
    to reject/penalize synthetic candidates where SA parameters cannot change
    bins/gap at all. Final validation/test still use the chosen campaign arms.
    """
    variants = [
        ("default", base),
        (
            "fast-local",
            base.model_copy(
                update={
                    "configuration_id": "curation-fast-local",
                    "cooling_rate": 0.90,
                    "iterations_per_temperature": max(20, base.iterations_per_temperature // 2),
                    "multi_start_count": 1,
                    "ruin_size": 2,
                    "operator_probabilities": {
                        "swap": 0.45,
                        "relocate": 0.30,
                        "reverse": 0.10,
                        "rotate": 0.10,
                        "ruin_recreate": 0.05,
                    },
                }
            ),
        ),
        (
            "exploratory",
            base.model_copy(
                update={
                    "configuration_id": "curation-exploratory",
                    "cooling_rate": 0.985,
                    "iterations_per_temperature": max(80, base.iterations_per_temperature),
                    "multi_start_count": 3,
                    "ruin_size": 8,
                    "reheating_factor": 1.65,
                    "operator_probabilities": {
                        "swap": 0.15,
                        "relocate": 0.20,
                        "reverse": 0.10,
                        "rotate": 0.20,
                        "ruin_recreate": 0.35,
                    },
                }
            ),
        ),
    ]
    return variants[: max(1, min(int(panel_size), len(variants)))]


def _make_dataset_candidate_evaluator(
    *,
    max_evaluations: int,
    time_limit_seconds: float,
    temperature_calibration_samples: int,
    generate_args: dict[str, Any],
    split_seeds: dict[str, list[int]],
) -> tuple[Callable[[ProblemInstance, InstanceClass, str, int, int], dict[str, Any]], dict[str, Any]]:
    """Build a deterministic preflight evaluator for generated campaign data.

    The goal is not to tune on generated instances before the campaign starts.
    The preflight only protects the benchmark: discard candidates that the
    default solver cannot even complete, and prefer candidates that are not
    solved at the lower bound immediately. This keeps the campaign from
    reporting artificial ties caused by trivial or pathological instances.

    Determinism: decisions depend only on evaluation-budgeted solver runs with
    dedicated curation seeds (never the experiment's train/validation/test
    seeds). The wall-clock limit is a safety cap; projected runtime is
    recorded for information but no longer accepts, rejects or ranks a
    candidate (it made the same spec + seed produce different datasets on
    different runs, e.g. campaigns 632591ae vs 6f670fb1). ``generate_dataset``
    only calls this evaluator on training instances.
    """
    preflight_evaluations = int(
        generate_args.get("curation_evaluations", min(160, max(80, max_evaluations // 8)))
    )
    preflight_time = float(generate_args.get("curation_time_limit_seconds", 120.0))
    trivial_evaluation_threshold = int(
        generate_args.get("trivial_evaluation_threshold", max(12, min(30, preflight_evaluations // 5)))
    )
    seed_count = max(1, int(generate_args.get("curation_seed_count", 2)))
    slow_projection_factor = float(generate_args.get("slow_projection_factor", 0.75))
    prefer_nontrivial = bool(generate_args.get("prefer_nontrivial", True))
    prefer_discriminative = bool(generate_args.get("prefer_discriminative", True))
    panel_size = int(generate_args.get("curation_panel_size", 3))
    discriminative_bins_threshold = float(generate_args.get("discriminative_bins_threshold", 0.25))
    discriminative_gap_threshold = float(generate_args.get("discriminative_gap_threshold", 0.03))
    target_min_lower_bound = float(generate_args.get("target_min_lower_bound", 4.0))
    config = SolverConfig(
        max_evaluations=preflight_evaluations,
        time_limit_seconds=preflight_time,
        # Same calibration as the experiment runs (it used to be capped at 6,
        # which usually fell back to the default T0: a different regime).
        temperature_calibration_samples=temperature_calibration_samples,
        multi_start_count=2,
        use_native_core=False,
        parallel_mode="sequential",
        parallel_workers=1,
    )
    probe_configs = _curation_probe_configs(config, panel_size)

    def stable_seed(spec: InstanceClass, split: str, replica: int, attempt: int, index: int) -> int:
        token = f"{spec.label}:{split}:{replica}:{attempt}"
        return 90_000 + sum((char_index + 1) * ord(char) for char_index, char in enumerate(token)) + index * 9_973

    experiment_seeds = {int(seed) for seeds in split_seeds.values() for seed in seeds}

    def seeds_for(spec: InstanceClass, split: str, replica: int, attempt: int) -> list[int]:
        # Curation-only seeds: disjoint from every experiment seed.
        seeds: list[int] = []
        index = 0
        while len(seeds) < seed_count:
            candidate = stable_seed(spec, split, replica, attempt, index)
            if candidate not in experiment_seeds:
                seeds.append(candidate)
            index += 1
        return seeds

    def evaluate(
        instance: ProblemInstance,
        spec: InstanceClass,
        split: str,
        replica: int,
        attempt: int,
    ) -> dict[str, Any]:
        panel_rows: list[dict[str, Any]] = []
        for seed in seeds_for(spec, split, replica, attempt):
            for config_name, probe_config in probe_configs:
                result = _solve_with_hard_timeout(instance, probe_config, seed=seed, timeout_seconds=preflight_time)
                loss = performance_loss(result)
                evaluations = max(int(result.evaluations), 0)
                projected_runtime = (
                    (float(result.runtime_seconds) / max(evaluations, 1)) * max_evaluations
                    if result.status == "feasible"
                    else float("inf")
                )
                panel_rows.append(
                    {
                        "config": config_name,
                        "seed": seed,
                        "status": result.status,
                        "loss": loss,
                        "bins_used": result.best_found_bins,
                        "lower_bound": result.lower_bound,
                        "gap_to_lower_bound": result.gap_to_lower_bound,
                        "optimal_proven": result.optimal_proven,
                        "evaluations": evaluations,
                        "runtime_seconds": result.runtime_seconds,
                        "projected_runtime_seconds": projected_runtime,
                    }
                )
        probe_rows = [row for row in panel_rows if row["config"] == "default"]
        feasible = bool(probe_rows) and all(row["status"] == "feasible" for row in probe_rows)
        complete_panel: dict[str, list[dict[str, Any]]] = {}
        for config_name, _ in probe_configs:
            rows = [row for row in panel_rows if row["config"] == config_name and row["status"] == "feasible"]
            if len(rows) == len(probe_rows) and rows:
                complete_panel[config_name] = rows
        panel_quality = [
            (
                statistics.fmean(float(row["bins_used"]) for row in rows),
                statistics.fmean(float(row["gap_to_lower_bound"]) for row in rows),
            )
            for rows in complete_panel.values()
        ]
        bins_spread = (
            max(bins for bins, _gap in panel_quality) - min(bins for bins, _gap in panel_quality)
            if len(panel_quality) >= 2
            else 0.0
        )
        gap_spread = (
            max(gap for _bins, gap in panel_quality) - min(gap for _bins, gap in panel_quality)
            if len(panel_quality) >= 2
            else 0.0
        )
        quality_discriminative = bool(
            bins_spread >= discriminative_bins_threshold or gap_spread >= discriminative_gap_threshold
        )
        too_easy = bool(
            feasible
            and all(bool(row["optimal_proven"]) and int(row["evaluations"]) <= trivial_evaluation_threshold for row in probe_rows)
        )
        too_slow = bool(
            any(
                row["status"] == "feasible"
                and float(row["projected_runtime_seconds"]) > time_limit_seconds * slow_projection_factor
                for row in probe_rows
            )
        )
        # too_slow is informational only (machine-dependent); see docstring.
        hard_reject = not feasible
        accepted = (
            feasible
            and (not prefer_nontrivial or not too_easy)
            and (not prefer_discriminative or quality_discriminative)
        )
        losses = [float(row["loss"]) for row in probe_rows]
        mean_loss = statistics.fmean(losses) if losses else 1_000.0
        mean_lower_bound = statistics.fmean(float(row["lower_bound"]) for row in probe_rows) if probe_rows else 0.0
        mean_projected_runtime = statistics.fmean(
            float(row["projected_runtime_seconds"])
            for row in probe_rows
            if row["status"] == "feasible" and row["projected_runtime_seconds"] != float("inf")
        ) if any(row["status"] == "feasible" for row in probe_rows) else float("inf")
        lower_bound_shortfall = max(0.0, target_min_lower_bound - mean_lower_bound)
        if feasible:
            # Prefer informative, still-tractable candidates. Easy-but-safe
            # candidates remain possible fallbacks; slow candidates are hard
            # rejects because they would reproduce timeout-heavy campaigns.
            rank = (
                (100.0 if too_easy else 0.0)
                + (50.0 if prefer_discriminative and not quality_discriminative else 0.0)
                + (20.0 * lower_bound_shortfall)
                + mean_loss
                - 2.0 * min(bins_spread, 3.0)
                - 1.0 * min(gap_spread, 2.0)
                - 0.01 * mean_lower_bound
            )
        else:
            rank = 10_000.0 + mean_loss
        reason = "aceita"
        if not feasible:
            reason = "preflight falhou em pelo menos uma seed de curadoria"
        elif too_easy:
            reason = (
                "baseline atingiu o lower bound muito cedo; aceita só se não houver "
                "candidato mais informativo"
            )
        elif prefer_discriminative and not quality_discriminative:
            reason = (
                "preflight tratável, mas o painel fixo de configurações não gerou diferença "
                "de caixas/gap; aceita só como fallback"
            )
        statuses = {str(row["status"]) for row in probe_rows}
        status = "feasible" if statuses == {"feasible"} else ("timeout" if "timeout" in statuses else "error")
        return {
            "accepted": accepted,
            "hard_reject": hard_reject,
            "rank": rank,
            "reason": reason,
            "preflight_status": status,
            "preflight_loss": mean_loss,
            "preflight_bins_used": statistics.fmean(float(row["bins_used"]) for row in probe_rows),
            "preflight_lower_bound": mean_lower_bound,
            "preflight_gap_to_lower_bound": statistics.fmean(float(row["gap_to_lower_bound"]) for row in probe_rows),
            "preflight_optimal_proven": all(bool(row["optimal_proven"]) for row in probe_rows),
            "preflight_evaluations": statistics.fmean(float(row["evaluations"]) for row in probe_rows),
            "preflight_runtime_seconds": statistics.fmean(float(row["runtime_seconds"]) for row in probe_rows),
            "preflight_projected_runtime_seconds": mean_projected_runtime,
            "preflight_too_easy": too_easy,
            "preflight_too_slow": too_slow,
            "preflight_quality_discriminative": quality_discriminative,
            "preflight_bins_spread": bins_spread,
            "preflight_gap_spread": gap_spread,
            "preflight_lower_bound_shortfall": lower_bound_shortfall,
            "preflight_seed_count": len(probe_rows),
            "preflight_runs": probe_rows,
            "preflight_panel_runs": panel_rows,
        }

    metadata = {
        "preflight_evaluations": preflight_evaluations,
        "preflight_time_limit_seconds": preflight_time,
        "trivial_evaluation_threshold": trivial_evaluation_threshold,
        "curation_seed_count": seed_count,
        "slow_projection_factor": slow_projection_factor,
        "prefer_nontrivial": prefer_nontrivial,
        "prefer_discriminative": prefer_discriminative,
        "curation_panel_size": len(probe_configs),
        "discriminative_bins_threshold": discriminative_bins_threshold,
        "discriminative_gap_threshold": discriminative_gap_threshold,
        "target_min_lower_bound": target_min_lower_bound,
    }
    return evaluate, metadata


def _curation_summary(manifest: dict[str, Any]) -> dict[str, Any] | None:
    entries = [
        entry.get("curation")
        for split_entries in manifest.get("splits", {}).values()
        for entry in split_entries
        if isinstance(entry.get("curation"), dict)
    ]
    if not entries:
        return None
    class_slots = manifest.get("class_slots") or []
    mapped_slots = [slot for slot in class_slots if isinstance(slot, dict)]

    def mean_entry(key: str) -> float | None:
        values = [
            float(entry[key])
            for entry in entries
            if entry.get(key) is not None and isinstance(entry.get(key), (int, float))
        ]
        return statistics.fmean(values) if values else None

    return {
        "instances": len(entries),
        "accepted": sum(1 for entry in entries if not bool(entry.get("fallback"))),
        "fallback": sum(1 for entry in entries if bool(entry.get("fallback"))),
        "class_replacements": sum(1 for entry in entries if bool(entry.get("class_replaced"))),
        "class_slots": len(mapped_slots) or len(manifest.get("classes", [])),
        "class_slots_replaced": sum(1 for slot in mapped_slots if bool(slot.get("class_replaced"))),
        "discriminative": sum(1 for entry in entries if bool(entry.get("preflight_quality_discriminative"))),
        "too_easy": sum(1 for entry in entries if bool(entry.get("preflight_too_easy"))),
        "too_slow": sum(1 for entry in entries if bool(entry.get("preflight_too_slow"))),
        "timeouts": sum(1 for entry in entries if entry.get("preflight_status") == "timeout"),
        "errors": sum(1 for entry in entries if entry.get("preflight_status") == "error"),
        "mean_preflight_evaluations": mean_entry("preflight_evaluations"),
        "mean_preflight_lower_bound": mean_entry("preflight_lower_bound"),
        "mean_preflight_bins_spread": mean_entry("preflight_bins_spread"),
        "mean_preflight_gap_spread": mean_entry("preflight_gap_spread"),
    }


def _discrimination_warning(outcomes: list[ArmOutcome]) -> str | None:
    def near_equal(values: list[float], tolerance: float = 1e-6) -> bool:
        return bool(values) and max(values) - min(values) <= tolerance

    for phase, attr in (("validação", "validation_metrics"), ("teste", "test_metrics")):
        metrics = [
            getattr(outcome, attr)
            for outcome in outcomes
            if outcome.status == "ok" and getattr(outcome, attr)
        ]
        complete = [item for item in metrics if _phase_complete(item)]
        if len(complete) < 2:
            continue
        bins = [float(item["mean_bins_used"]) for item in complete if item.get("mean_bins_used") is not None]
        gaps = [float(item["mean_gap_to_lower_bound"]) for item in complete if item.get("mean_gap_to_lower_bound") is not None]
        evaluations = [float(item["mean_evaluations"]) for item in complete if item.get("mean_evaluations") is not None]
        optimal_rates = [float(item["optimal_rate"]) for item in complete if item.get("optimal_rate") is not None]
        comparable_quality = len(bins) == len(complete) and len(gaps) == len(complete)
        tied_primary_quality = comparable_quality and near_equal(bins) and near_equal(gaps)
        if (
            tied_primary_quality
            and len(evaluations) == len(complete)
            and max(gaps, default=0.0) <= 1e-9
            and max(evaluations, default=0.0) <= 15.0
            and min(optimal_rates, default=0.0) >= 0.95
        ):
            return (
                f"Dataset pouco discriminativo no split de {phase}: todos os braços atingiram o lower bound "
                "quase imediatamente (≈15 avaliações ou menos). O empate é esperado e não demonstra "
                "equivalência entre estratégias. Use mais classes/instâncias ou um manifest curado antes "
                "de usar a tabela no artigo."
            )
        if tied_primary_quality:
            return (
                f"Qualidade primária não discriminou no split de {phase}: todos os braços tiveram a mesma "
                "média de caixas e o mesmo gap ao lower bound. Qualquer diferença restante de perda vem de "
                "penalidade de falha ou de um termo de tempo explicitamente ativado; não interprete o braço "
                "destacado como superior em qualidade. Use mais "
                "instâncias/classes, outro manifest curado ou um orçamento que produza diferenças de caixas/gap "
                "antes de usar a tabela como resultado principal."
            )
    return None


def _tune_arm(
    arm: str,
    objective: TuningObjective,
    space_definition: dict[str, Any],
    trials: int,
    tuner_seed: int,
    output_dir: Path,
    src_path: Path,
) -> tuple[SolverConfig | None, float | None, str | None, Callable[[ProblemInstance], SolverConfig] | None]:
    """Retorna (incumbente, melhor perda de treino, motivo-se-pulado, política π(z) ou None).

    Política None = braço de configuração global (single best configuration);
    política presente = braço por instância, avaliado em zero-shot.
    """
    space = SearchSpace(space_definition)
    if arm == "fixed_default":
        return SolverConfig(), None, None, None
    if arm == "random_search":
        # Trials do Random Search são independentes (todas as configs são
        # sorteadas de uma vez): avalia o lote inteiro num único batch paralelo.
        # Seleção e perdas são idênticas ao laço serial; só alarga a frente
        # paralela (trials × instâncias × seeds runs independentes).
        configs = list(RandomSearchConfigurator(space, seed=tuner_seed).suggest(trials))
        if not configs:
            return None, None, "random_search sem trials", None
        losses = objective.evaluate_configs(configs)
        best_index = min(range(len(losses)), key=losses.__getitem__)
        return configs[best_index], losses[best_index], None, None
    if arm == "classic_ego":
        # Modelo de ruído análogo ao do braço contextual: variância de seed da
        # média, com o mesmo piso relativo (NOISE_FLOOR_RATIO). Antes o GP
        # clássico interpolava as perdas sem ruído. Os dois braços diferem no
        # contexto e na unidade de observação (média global vs. por instância).
        ego = ClassicEGOConfigurator(space, seed=tuner_seed)
        best_loss, best_config = float("inf"), None
        for _ in range(trials):
            config = ego.suggest(1)[0]
            grouped = objective.replicate_losses_grouped(config)
            loss = statistics.fmean(value for group in grouped for value in group)
            ego.observe(config, loss, noise_variance=mean_noise_variance(grouped))
            if loss < best_loss:
                best_loss, best_config = loss, config
        return best_config, best_loss, None, None
    if arm in PER_INSTANCE_ARMS:
        tuner = NoiseAwareContextualEGO(
            space,
            seed=tuner_seed,
            use_plugin_incumbent=arm != "ablation_min_y_incumbent",
            optimize_gp_hyperparameters=arm != "ablation_gp_fixed_hyperparams",
            feature_names=FEATURE_NAMES,
        )
        # Tuning contextual POR INSTÂNCIA (parecer §5.1): cada instância de
        # treino gera uma observação sob o SEU vetor de features (a média de
        # contextos apagaria exatamente a interação contexto-configuração que
        # a tese investiga). A sugestão de cada trial usa o contexto de uma
        # instância de treino em rodízio. Orçamento idêntico aos demais braços
        # (trials × instâncias × seeds runs do solver).
        contexts = [extract_instance_features(instance) for instance in objective.instances]
        best_loss, best_config = float("inf"), None
        for trial in range(trials):
            query_context = contexts[trial % len(contexts)]
            config = tuner.suggest(query_context, 1)[0]
            # Os runs de um trial usam a MESMA config e são independentes:
            # dispara todas as instâncias × seeds em paralelo e só então observa,
            # uma observação por instância na ordem de instância. O estado do
            # tuner fica idêntico ao laço serial (suggest é único por trial e o
            # observe não altera runs já concluídos).
            grouped = objective.replicate_losses_grouped(config)
            trial_losses: list[float] = []
            for context, losses in zip(contexts, grouped, strict=True):
                tuner.observe(context, config, losses)
                trial_losses.extend(losses)
            loss = statistics.fmean(trial_losses)
            if loss < best_loss:
                best_loss, best_config = loss, config

        def recommend(instance: ProblemInstance) -> SolverConfig:
            # Recomendação zero-shot: apenas features da instância nova +
            # histórico de TREINO congelado (recommend não altera observations;
            # nenhum run da instância avaliada participa da escolha). Usa o
            # mínimo da média posterior (não o EI, que é critério de
            # exploração) e um pool semeado pela instância, independente da
            # ordem em que validação e teste consultam a política.
            pool_seed = zlib.crc32(f"{tuner_seed}:{instance.instance_id}".encode("utf-8"))
            return tuner.recommend(extract_instance_features(instance), pool_seed=pool_seed)

        return best_config, best_loss, None, recommend
    if arm in PER_INSTANCE_BASELINE_ARMS:
        return _tune_per_instance_baseline(arm, objective, space_definition, trials, tuner_seed)
    if arm == "smac3":
        if not smac_available():
            return None, None, "smac não instalado (pip install '.[tuners]')", None
        payload = run_smac(
            objective, space_definition, n_trials=trials, seed=tuner_seed, output_directory=output_dir / "smac3_output"
        )
        return values_to_config(payload["incumbent_values"]), float(payload["incumbent_loss"]), None, None
    if arm == "smac3_ac":
        if not smac_available():
            return None, None, "smac não instalado (pip install '.[tuners]')", None
        features = {instance.instance_id: list(_feature_vector(instance)) for instance in objective.instances}
        payload = run_smac_ac(
            objective,
            space_definition,
            features,
            n_trials=trials,
            seed=tuner_seed,
            output_directory=output_dir / "smac3_ac_output",
        )
        return values_to_config(payload["incumbent_values"]), float(payload["incumbent_loss"]), None, None
    if arm == "irace":
        campaign_dir = output_dir / "irace_campaign"
        # Mesmo orçamento de runs do solver que os demais braços. O piso de 96
        # dava ao irace mais orçamento que aos outros em campanhas pequenas;
        # se o orçamento for pequeno demais para o irace, o braço falha e o
        # relatório registra o motivo (não se compensa com orçamento extra).
        max_experiments = trials * objective.runs_per_trial
        generate_irace_campaign(
            output_dir=campaign_dir,
            search_space_definition=space_definition,
            instances=objective.instances,
            src_path=src_path,
            max_experiments=max_experiments,
            max_evaluations=objective.max_evaluations,
            time_limit_seconds=objective.time_limit_seconds,
            seed=tuner_seed,
            temperature_calibration_samples=objective.temperature_calibration_samples,
            use_native_core=objective.use_native_core,
        )
        if not irace_available():
            return None, None, f"irace não encontrado; campanha gerada em {campaign_dir}", None
        irace_run = run_irace(campaign_dir)
        if irace_run.get("returncode") != 0:
            # Ex.: orçamento abaixo do mínimo do irace para o espaço de busca.
            return None, None, f"irace falhou: {irace_run.get('error') or 'erro desconhecido'}", None
        values, stats = best_values_from_campaign(campaign_dir)
        # irace's runs happen in its target runner, outside this objective:
        # count them so the arm's tuning budget is visible in the report.
        objective.solver_runs += int(stats.get("evaluations") or 0)
        if values is None:
            return None, None, "irace não produziu avaliações", None
        best_mean_cost = stats.get("best_mean_cost")
        # Sem custo reportado, a perda de treino fica desconhecida (None), não 0.
        return values_to_config(values), float(best_mean_cost) if best_mean_cost is not None else None, None, None
    return None, None, f"braço desconhecido: {arm}", None


def run_campaign(
    spec: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
    repository: Any = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    root = project_root or Path(__file__).resolve().parents[3]
    # Captured before anything runs: edits made while the campaign runs must
    # not be attributed to it.
    git_state = _git_state(root)
    campaign_id = spec.get("campaign_id") or f"campaign-{uuid.uuid4().hex[:8]}"
    output_dir = Path(spec.get("output_dir") or root / "artifacts" / "campaigns" / campaign_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    def notify(stage: str, detail: str, fraction: float, **extra: Any) -> None:
        if progress_callback is not None:
            progress_callback({"stage": stage, "detail": detail, "fraction": fraction, **extra})

    # ---------------- dataset ----------------
    notify("dataset", "preparando instâncias", 0.02)
    budget = spec.get("budget") or {}
    max_evaluations = int(budget.get("max_evaluations", 1500))
    time_limit = float(budget.get("time_limit_seconds", 5.0))
    # Same default as SolverConfig. Calibration now only uses uphill moves that
    # keep the bin count, so 6 samples often yielded none (fallback T0).
    temperature_calibration_samples = int(
        budget.get("temperature_calibration_samples", spec.get("temperature_calibration_samples", 24))
    )
    # Same solver core for every arm (and for irace's target runner).
    use_native_core = bool(budget.get("native_core", False))
    native_fingerprint: dict[str, Any] | None = None
    if use_native_core:
        from adaptive_3dbp.native.cpp_bridge import ensure_native_core, native_core_fingerprint

        # Build once before any run, then record exactly which binary ran.
        ensure_native_core()
        native_fingerprint = native_core_fingerprint()
    seeds = [int(seed) for seed in spec.get("seeds", [11, 23, 47])]
    validation_seeds = [seed + 10_000 for seed in seeds]
    test_seeds = [seed + 20_000 for seed in seeds]
    dataset_spec = spec.get("dataset") or {}
    if "manifest" in dataset_spec:
        dataset_manifest_path = Path(dataset_spec["manifest"]).expanduser()
        manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        dataset_source = "manifest"
    elif "benchmark" in dataset_spec:
        # Standard benchmark: deterministic, no curation, no solver run.
        benchmark = dict(dataset_spec["benchmark"])
        if benchmark.get("name", "mpv") != "mpv":
            raise ValueError(f"benchmark desconhecido: {benchmark.get('name')}")
        from adaptive_3dbp.benchmarks.mpv import DEFAULT_SPLITS, REPORTED_CLASSES, generate_mpv_dataset

        dataset_dir = output_dir / "dataset"
        splits = benchmark.get("splits")
        manifest = generate_mpv_dataset(
            dataset_dir,
            classes=tuple(int(value) for value in benchmark.get("classes", REPORTED_CLASSES)),
            sizes=tuple(int(value) for value in benchmark.get("sizes", (50,))),
            splits={name: tuple(int(v) for v in values) for name, values in splits.items()} if splits else DEFAULT_SPLITS,
            rotations=benchmark.get("rotations", "fixed"),
        )
        dataset_manifest_path = dataset_dir / "manifest.json"
        dataset_source = "benchmark:mpv"
    else:
        generate_args = dataset_spec.get("generate") or {}
        from adaptive_3dbp.experiments.instances import all_classes

        selected_classes = None
        replacement_classes = None
        classes_limit = generate_args.get("classes_limit")
        if classes_limit:
            # Subconjunto balanceado para pilotos: precisa ter variedade de
            # contexto, mas não pode começar só pelas classes mais caras
            # (medium/high), senão a validação vira timeout e o ranking deixa
            # de medir os tuners.
            available = all_classes(bool(generate_args.get("include_min_distance", True)))
            selected_classes = _select_campaign_classes(available, int(classes_limit))
            replacement_pool_size = min(
                len(available),
                int(classes_limit) + int(generate_args.get("max_replacement_classes", 10)),
            )
            replacement_classes = _select_campaign_classes(available, replacement_pool_size)
        dataset_dir = output_dir / "dataset"
        curate_generated = bool(generate_args.get("curate", True))
        occupancy_scale = float(generate_args.get("occupancy_scale", 1.6 if curate_generated else 1.0))
        candidate_evaluator = None
        curation_metadata = None
        if curate_generated:
            notify("dataset", "triando instâncias sintéticas", 0.03)
            candidate_evaluator, curation_metadata = _make_dataset_candidate_evaluator(
                max_evaluations=max_evaluations,
                time_limit_seconds=time_limit,
                temperature_calibration_samples=temperature_calibration_samples,
                generate_args=generate_args,
                split_seeds={"train": seeds, "validation": validation_seeds, "test": test_seeds},
            )
        manifest = generate_dataset(
            dataset_dir,
            per_class=generate_args.get("per_class"),
            seed=int(generate_args.get("seed", 0)),
            include_min_distance=bool(generate_args.get("include_min_distance", True)),
            classes=selected_classes,
            replacement_classes=replacement_classes,
            candidate_evaluator=candidate_evaluator,
            candidate_attempts=int(generate_args.get("candidate_attempts", 8 if curate_generated else 1)),
            max_replacement_classes=int(generate_args.get("max_replacement_classes", 10)),
            occupancy_scale=occupancy_scale,
            curation_metadata=curation_metadata,
            require_accepted_candidate=bool(generate_args.get("require_accepted_candidate", curate_generated)),
            replace_rejected_classes=bool(generate_args.get("replace_rejected_classes", curate_generated)),
        )
        dataset_manifest_path = dataset_dir / "manifest.json"
        dataset_source = "generated"
    train = _load_split(manifest, "train")
    validation = _load_split(manifest, "validation")
    test = _load_split(manifest, "test")
    if not train or not validation or not test:
        raise ValueError("dataset precisa de splits train/validation/test não vazios")

    trials = int(spec.get("trials", 20))
    tuner_seed = int(spec.get("tuner_seed", 0))
    arms = list(spec.get("tuners") or BASE_ARMS)
    if spec.get("ablations"):
        arms.extend(arm for arm in ABLATION_ARMS if arm not in arms)
    if spec.get("per_instance_baselines"):
        arms.extend(arm for arm in PER_INSTANCE_BASELINE_ARMS if arm not in arms)
    space_definition = spec.get("search_space") or DEFAULT_SEARCH_SPACE
    dataset_curation = _curation_summary(manifest)

    # Paralelismo dos runs do solver. Os runs de uma campanha são independentes
    # e determinísticos por (instância, config, seed); rodá-los em paralelo só
    # muda o tempo de parede, nunca os resultados (ver TuningObjective._run_batch).
    # Default = workers recomendados da máquina (núcleos físicos − 1, deixando 1
    # livre para SO/backend); pode ser sobrescrito por spec["max_parallel_runs"].
    # Teto = núcleos físicos (sem sobre-assinatura, que estreitaria a folga do
    # limite de tempo e poderia cortar runs antes do orçamento de avaliações).
    hardware = system_info()
    workers_info = hardware.get("workers", {}) if isinstance(hardware, dict) else {}
    recommended_workers = max(1, int(workers_info.get("recommended", 1) or 1))
    maximum_workers = max(recommended_workers, int(workers_info.get("maximum", recommended_workers) or recommended_workers))
    requested_workers = spec.get("max_parallel_runs")
    if requested_workers is None:
        max_parallel_runs = recommended_workers
    else:
        max_parallel_runs = max(1, min(int(requested_workers), maximum_workers))

    persist_repository = repository if spec.get("persist") else None
    estimated_solver_runs = sum(
        _estimated_arm_solver_runs(
            arm,
            train_count=len(train),
            validation_count=len(validation),
            test_count=len(test),
            seed_count=len(seeds),
            trials=trials,
        )
        for arm in arms
    )
    completed_solver_runs = 0

    def on_solver_run(event: dict[str, Any]) -> None:
        nonlocal completed_solver_runs
        label = str(event.get("label") or "")
        arm_id, _, phase = label.partition(":")
        arm_label = ARM_LABELS.get(arm_id, arm_id or "campanha")
        phase_label = phase or "execução"
        total = max(estimated_solver_runs, 1)
        if event.get("event") == "finished":
            completed_solver_runs += 1
            detail = (
                f"{arm_label} · {phase_label} · run {completed_solver_runs}/{estimated_solver_runs} "
                f"· loss {float(event.get('loss', 0.0)):.4f}"
            )
        elif event.get("event") == "error":
            detail = f"{arm_label} · {phase_label} · erro em {event.get('instance_id')} seed {event.get('seed')}"
        else:
            next_run = min(completed_solver_runs + 1, total)
            detail = (
                f"{arm_label} · {phase_label} · run {next_run}/{estimated_solver_runs} "
                f"· {event.get('instance_id')} seed {event.get('seed')}"
            )
        notify(
            "solver",
            detail,
            0.05 + 0.84 * min(completed_solver_runs / total, 1.0),
            completed_solver_runs=completed_solver_runs,
            estimated_solver_runs=estimated_solver_runs,
            current_arm=arm_id or None,
            current_phase=phase or None,
        )

    # ---------------- metadados ----------------
    metadata = {
        "campaign_id": campaign_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "package_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        **git_state,
        "hardware": hardware,
        "spec": spec,
        "seeds": seeds,
        "validation_seeds": validation_seeds,
        "test_seeds": test_seeds,
        "estimated_solver_runs": estimated_solver_runs,
        # Workers de run em paralelo usados nesta execução. Registrado por
        # transparência/auditoria: NÃO afeta os resultados (apenas o tempo de
        # parede), então campanhas com contagens diferentes continuam idênticas.
        "max_parallel_runs": max_parallel_runs,
        "temperature_calibration_samples": temperature_calibration_samples,
        "native_core": use_native_core,
        "native_core_fingerprint": native_fingerprint,
        "dataset_source": dataset_source,
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_classes": manifest.get("classes", []),
        "dataset_per_class": manifest.get("per_class", {}),
        "dataset_summary": {split: len(entries) for split, entries in manifest["splits"].items()},
        "dataset_curation": dataset_curation,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    # ---------------- execução por braço ----------------
    all_records: list[dict[str, Any]] = []
    all_solutions: list[dict[str, Any]] = []
    outcomes: list[ArmOutcome] = []
    total_arms = max(len(arms), 1)
    for index, arm in enumerate(arms):
        notify("tuning", ARM_LABELS.get(arm, arm), 0.05 + 0.75 * index / total_arms)
        train_objective = TuningObjective(
            instances=train,
            seeds=seeds,
            max_evaluations=max_evaluations,
            time_limit_seconds=time_limit,
            temperature_calibration_samples=temperature_calibration_samples,
            use_native_core=use_native_core,
            repository=persist_repository,
            campaign_id=campaign_id,
            run_callback=on_solver_run,
            max_parallel_runs=max_parallel_runs,
        )
        train_objective.label = f"{arm}:train"
        tuning_started = time.perf_counter()
        try:
            incumbent, training_best, skip_reason, recommend_policy = _tune_arm(
                arm, train_objective, space_definition, trials, tuner_seed, output_dir, root
            )
        except Exception as exc:  # braço defeituoso não derruba a campanha
            outcomes.append(ArmOutcome(arm=arm, status="error", reason=str(exc)))
            all_records.extend(train_objective.records)
            all_solutions.extend(train_objective.solutions)
            continue
        tuning_seconds = time.perf_counter() - tuning_started
        if incumbent is None:
            outcomes.append(ArmOutcome(arm=arm, status="skipped", reason=skip_reason))
            all_records.extend(train_objective.records)
            all_solutions.extend(train_objective.solutions)
            continue

        evaluation_objective = TuningObjective(
            instances=validation,
            seeds=validation_seeds,
            max_evaluations=max_evaluations,
            time_limit_seconds=time_limit,
            temperature_calibration_samples=temperature_calibration_samples,
            use_native_core=use_native_core,
            repository=persist_repository,
            campaign_id=campaign_id,
            run_callback=on_solver_run,
            max_parallel_runs=max_parallel_runs,
        )
        evaluation_objective.label = f"{arm}:validation"
        # Avaliação ESTÁTICA (parecer §5.5): a política/configuração é
        # congelada após o treino; runs de validação/teste NÃO alimentam o
        # tuner (mede generalização, não desempenho prequential).
        if recommend_policy is not None:
            validation_loss = evaluation_objective.evaluate_config_map(recommend_policy)
        else:
            validation_loss = evaluation_objective.evaluate_config(incumbent)

        test_objective = TuningObjective(
            instances=test,
            seeds=test_seeds,
            max_evaluations=max_evaluations,
            time_limit_seconds=time_limit,
            temperature_calibration_samples=temperature_calibration_samples,
            use_native_core=use_native_core,
            repository=persist_repository,
            campaign_id=campaign_id,
            run_callback=on_solver_run,
            max_parallel_runs=max_parallel_runs,
        )
        test_objective.label = f"{arm}:test"
        if recommend_policy is not None:
            test_loss = test_objective.evaluate_config_map(recommend_policy)
        else:
            test_loss = test_objective.evaluate_config(incumbent)
        validation_metrics = _quality_metrics(evaluation_objective.records)
        test_metrics = _quality_metrics(test_objective.records)

        outcomes.append(
            ArmOutcome(
                arm=arm,
                status="ok",
                protocol="per_instance_zero_shot" if recommend_policy is not None else "global_config",
                tuning_seconds=tuning_seconds,
                tuning_solver_runs=train_objective.solver_runs,
                training_best_loss=training_best,
                validation_loss=validation_loss,
                test_loss=test_loss,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                best_config=incumbent.model_dump(),
            )
        )
        all_records.extend(train_objective.records)
        all_records.extend(evaluation_objective.records)
        all_records.extend(test_objective.records)
        all_solutions.extend(train_objective.solutions)
        all_solutions.extend(evaluation_objective.solutions)
        all_solutions.extend(test_objective.solutions)

    # ---------------- exportações ----------------
    notify("export", "gravando CSV/relatório", 0.9)
    runs_csv = output_dir / "runs.csv"
    if all_records:
        with runs_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)
    solutions_jsonl = output_dir / "solutions.jsonl"
    if all_solutions:
        with solutions_jsonl.open("w", encoding="utf-8") as handle:
            for solution in all_solutions:
                handle.write(json.dumps(solution, ensure_ascii=False, default=str) + "\n")
    parquet_written = False
    try:  # Parquet opcional
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        if all_records:
            table = pa.Table.from_pylist(all_records)
            pq.write_table(table, output_dir / "runs.parquet")
            parquet_written = True
    except ImportError:
        pass

    summary = {
        "campaign_id": campaign_id,
        "elapsed_seconds": time.time() - started,
        "arms": [asdict(outcome) for outcome in outcomes],
        "records": len(all_records),
        "estimated_solver_runs": estimated_solver_runs,
        "dataset": {
            "source": dataset_source,
            "manifest_path": str(dataset_manifest_path),
            "summary": {split: len(entries) for split, entries in manifest["splits"].items()},
            "per_class": manifest.get("per_class", {}),
            "classes": manifest.get("classes", []),
            "curation": dataset_curation,
        },
        "artifacts": {
            "metadata": str(output_dir / "metadata.json"),
            "runs_csv": str(runs_csv) if all_records else None,
            "runs_parquet": str(output_dir / "runs.parquet") if parquet_written else None,
            "solutions_jsonl": str(solutions_jsonl) if all_solutions else None,
            "report": str(output_dir / "report.md"),
            "dataset_manifest": str(dataset_manifest_path),
            "summary": str(output_dir / "summary.json"),
            "output_dir": str(output_dir),
        },
    }
    # Artefatos prontos para o artigo: tabela booktabs e gráfico de comparação.
    results_table = export_results_table_tex(output_dir, outcomes)
    summary["artifacts"]["results_table_tex"] = str(results_table)
    comparison_png = export_comparison_png(output_dir, outcomes)
    summary["artifacts"]["comparison_png"] = str(comparison_png) if comparison_png else None
    # Detector de saturação de tempo: se o tempo médio por run encosta no
    # limite, o relógio (e não a configuração) vira o orçamento efetivo — toda
    # configuração roda "o que der no tempo" e os braços empatam por construção.
    saturated = [
        outcome.arm
        for outcome in outcomes
        if outcome.status == "ok"
        and (outcome.validation_metrics or {}).get("mean_runtime_seconds") is not None
        and float(outcome.validation_metrics["mean_runtime_seconds"]) >= 0.9 * time_limit
    ]
    stop_reasons: dict[str, int] = {}
    for record in all_records:
        reason = str(record.get("stop_reason") or "unknown")
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
    summary["stop_reasons"] = stop_reasons
    core_counts: dict[str, int] = {}
    for record in all_records:
        core = str(record.get("core") or "unknown")
        core_counts[core] = core_counts.get(core, 0) + 1
    summary["core_counts"] = core_counts
    if native_fingerprint is not None:
        from adaptive_3dbp.native.cpp_bridge import native_core_fingerprint

        final_fingerprint = native_core_fingerprint()
        changed = [
            key
            for key in ("source_sha256", "binary_sha256")
            if final_fingerprint.get(key) != native_fingerprint.get(key)
        ]
        if changed:
            summary["native_core_changed_warning"] = (
                f"O core nativo mudou durante a campanha ({', '.join(changed)} diferente do início): "
                "runs podem ter usado binários diferentes. Não use como campanha confirmatória."
            )
    if core_counts.get("python_fallback"):
        summary["native_fallback_warning"] = (
            f"{core_counts['python_fallback']} de {len(all_records)} runs pediram o core nativo e rodaram em "
            "Python (falha do nativo; ver diagnostics em runs.csv). Mesma semântica, mas ~250x mais lento: "
            "runs limitados por tempo podem divergir entre braços."
        )
    time_stops = stop_reasons.get("time", 0)
    summary["time_budget_saturated"] = bool(saturated) or time_stops > 0
    if saturated or time_stops:
        summary["time_budget_warning"] = (
            f"{time_stops} de {len(all_records)} runs pararam no limite de tempo ({time_limit:.0f}s) antes do "
            "orçamento de avaliações"
            + (f"; tempo médio/run ≥ 90% do limite em {len(saturated)} braço(s)" if saturated else "")
            + ". Nesses runs o relógio, não as avaliações, decidiu o resultado (depende de máquina e carga). "
            "Aumente Tempo/run (folga) e use Aval./run como orçamento real, ou reduza Aval./run."
        )
    comparison_warning = _comparison_warning(outcomes)
    discrimination_warning = _discrimination_warning(outcomes)
    summary["comparison_valid"] = (
        comparison_warning is None
        and discrimination_warning is None
        and any(_has_complete_validation(outcome) for outcome in outcomes)
    )
    if comparison_warning:
        summary["comparison_warning"] = comparison_warning
    if discrimination_warning:
        summary["dataset_discrimination_warning"] = discrimination_warning

    # Protocolo estatístico (parecer §11): comparações primárias pré-definidas
    # pareadas por instância (unidade experimental; seeds agregadas por média),
    # Wilcoxon signed-rank, correção de Holm e IC bootstrap da diferença.
    from adaptive_3dbp.experiments.stats import (
        ABLATION_BASELINES,
        per_instance_mean_losses,
        predefined_comparisons,
    )

    unit_key = None
    summary["statistical_unit"] = "instance"
    if str(manifest.get("benchmark", "")).startswith("martello_pisinger_vigo"):
        from adaptive_3dbp.benchmarks.mpv import mpv_unit_key

        # Classes sharing (n, v) share the generator seed: one unit per (n, v).
        unit_key = mpv_unit_key
        summary["statistical_unit"] = manifest.get("statistical_unit", "mpv (n, v)")
    test_losses_by_arm = per_instance_mean_losses(all_records, ":test", unit_key=unit_key)
    statistics_payload = predefined_comparisons(test_losses_by_arm, bootstrap_seed=tuner_seed)
    statistics_payload += predefined_comparisons(
        test_losses_by_arm, baselines=ABLATION_BASELINES, bootstrap_seed=tuner_seed, family="ablation"
    )
    summary["statistical_comparisons"] = statistics_payload

    report = render_campaign_report(metadata, outcomes)
    report += _render_statistics_section(statistics_payload, unit=str(summary["statistical_unit"]))
    if summary.get("time_budget_warning"):
        report += (
            "\n> **Aviso de saturação de tempo**: " + summary["time_budget_warning"] + "\n"
        )
    if summary.get("native_fallback_warning"):
        report += "\n> **Aviso de core**: " + summary["native_fallback_warning"] + "\n"
    if summary.get("native_core_changed_warning"):
        report += "\n> **Aviso de core**: " + summary["native_core_changed_warning"] + "\n"
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["report_markdown"] = report
    notify("done", "campanha concluída", 1.0)
    return summary


def export_results_table_tex(output_dir: Path, outcomes: list[ArmOutcome]) -> Path:
    """Tabela booktabs pronta para \\input{} no artigo (perda + qualidade)."""
    discrimination_warning = _discrimination_warning(outcomes)
    valid = [] if discrimination_warning else [o for o in outcomes if _has_complete_validation(o)]
    best_validation = min((o.validation_loss for o in valid), default=None)
    warning = _comparison_warning(outcomes)
    caption = "Perda em valida\\c{c}\\~ao e teste por bra\\c{c}o, com m\\'etricas de qualidade."
    if warning:
        caption += " Campanha diagn\\'ostica: valida\\c{c}\\~ao incompleta."
    if discrimination_warning:
        caption += " Campanha diagn\\'ostica: dataset pouco discriminativo."

    def num(value: float | None, digits: int = 4) -> str:
        return f"{value:.{digits}f}" if value is not None else "---"

    def metric(metrics: dict[str, Any] | None, key: str, digits: int = 2, scale: float = 1.0, suffix: str = "") -> str:
        if not metrics or metrics.get(key) is None:
            return "---"
        return f"{float(metrics[key]) * scale:.{digits}f}{suffix}"

    lines = [
        "% Tabela gerada automaticamente pela campanha — pronta para \\input{} no artigo.",
        "% Requer \\usepackage{booktabs}. Perda menor é melhor; treino é viesado (winner's curse).",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:campanha}",
        "\\small",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Bra\\c{c}o & Valida\\c{c}\\~ao & Teste & Caixas (m\\'edia) & Gap m\\'edio & \\'Otimo (\\%) & Tempo/run (s) & Tuning (s) \\\\",
        "\\midrule",
    ]
    for outcome in outcomes:
        label = ARM_LABELS.get(outcome.arm, outcome.arm).replace("&", "\\&")
        if outcome.status != "ok":
            reason = (outcome.reason or outcome.status).replace("&", "\\&")
            lines.append(f"{label} & \\multicolumn{{7}}{{c}}{{{reason}}} \\\\")
            continue
        validation = num(outcome.validation_loss)
        if best_validation is not None and _has_complete_validation(outcome) and outcome.validation_loss == best_validation:
            validation = f"\\textbf{{{validation}}}"
        tuning_cell = f"{outcome.tuning_seconds:.1f}" if outcome.arm != "fixed_default" else "--- (baseline)"
        lines.append(
            f"{label} & {validation} & {num(outcome.test_loss)} & "
            f"{metric(outcome.validation_metrics, 'mean_bins_used')} & "
            f"{metric(outcome.validation_metrics, 'mean_gap_to_lower_bound', 3)} & "
            f"{metric(outcome.validation_metrics, 'optimal_rate', 0, 100.0)} & "
            f"{metric(outcome.validation_metrics, 'mean_runtime_seconds', 2)} & "
            f"{tuning_cell} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path = output_dir / "results_table.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_comparison_png(output_dir: Path, outcomes: list[ArmOutcome]) -> Path | None:
    """Gráfico de barras validação/teste em qualidade de publicação (matplotlib opcional)."""
    comparable = [o for o in outcomes if o.status == "ok" and o.validation_loss is not None]
    if not comparable:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    labels = [ARM_LABELS.get(o.arm, o.arm) for o in comparable]
    validation = [o.validation_loss or 0.0 for o in comparable]
    test = [o.test_loss or 0.0 for o in comparable]
    best_candidates = [] if _discrimination_warning(outcomes) else [
        index for index, outcome in enumerate(comparable) if _has_complete_validation(outcome)
    ]
    best_index = min(best_candidates, key=lambda index: validation[index]) if best_candidates else None
    max_value = max([*validation, *test, 1e-9])

    positions = np.arange(len(labels))
    height = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 1.3 + 0.62 * len(labels)), dpi=200)
    bars_val = ax.barh(positions - height / 2, validation, height, label="validação", color="#2563eb")
    ax.barh(positions + height / 2, test, height, label="teste (uma única vez)", color="#0f766e", alpha=0.85)
    if best_index is not None:
        bars_val[best_index].set_color("#b45309")
    ax.set_yticks(positions, labels, fontsize=9)
    ax.invert_yaxis()
    # Folga à direita para os rótulos de valor não colidirem com a moldura.
    ax.set_xlim(0, max_value * 1.18)
    ax.set_xlabel("perda composta (menor é melhor)", fontsize=9)
    ax.set_title("Comparação de configuradores — incumbentes reavaliados em seeds frescas", fontsize=10, pad=12)
    # Legenda FORA da área de plotagem (abaixo do eixo x): nunca sobrepõe barras.
    ax.legend(
        frameon=False,
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16 - 0.5 / max(len(labels), 1)),
        ncol=2,
    )
    ax.grid(axis="x", alpha=0.3)
    for index in range(len(labels)):
        ax.text(validation[index], index - height / 2, f" {validation[index]:.4f}", va="center", fontsize=7.5, color="#172033")
        ax.text(test[index], index + height / 2, f" {test[index]:.4f}", va="center", fontsize=7.5, color="#3f5852")
    fig.tight_layout()
    path = output_dir / "comparison.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _render_statistics_section(comparisons: list[dict[str, Any]], unit: str = "instance") -> str:
    if not comparisons:
        return ""
    unit_text = (
        "por unidade (n, v) do MPV — média sobre seeds e sobre as classes que compartilham a seed do gerador"
        if unit.startswith("mpv")
        else "por instância (média sobre seeds)"
    )
    lines = [
        "",
        "## Comparações pré-definidas",
        "",
        f"Pareadas {unit_text}; n = número de unidades; diferença = baseline − proposta",
        "(positivo favorece a proposta). Wilcoxon signed-rank bicaudal; Holm dentro de",
        "cada família (primária: baselines; ablation: variantes da proposta); IC 95%",
        "bootstrap percentil (n_boot=10000, seed=0) da diferença média. 'Taxa de vitória'",
        "= fração de instâncias vencidas (descritiva, não probabilística); V/E/D =",
        "vitórias/empates/derrotas; r = rank-biserial pareado (tamanho de efeito).",
        "O IC do bootstrap (diferença média) e o p de Wilcoxon (postos) medem coisas",
        "distintas e podem divergir, sobretudo com muitos empates.",
        "",
        "| Família | Comparação | n | V/E/D | Δ média | Δ mediana | IC 95% | Taxa vit. | r | p (Wilcoxon) | p (Holm) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for comparison in comparisons:
        ci = comparison.get("bootstrap_ci_95")
        ci_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "—"
        win = comparison.get("win_rate", comparison.get("win_probability"))
        win_text = f"{win:.2f}" if win is not None else "—"
        p_raw = comparison.get("wilcoxon_p_value")
        p_raw_text = f"{p_raw:.4f}" if p_raw is not None else "—"
        p_holm = comparison.get("p_value_holm")
        p_holm_text = f"{p_holm:.4f}" if p_holm is not None else "—"
        effect = comparison.get("wilcoxon_rank_biserial")
        effect_text = f"{effect:+.2f}" if effect is not None else "—"
        record_text = f"{comparison.get('wins', 0)}/{comparison.get('ties', 0)}/{comparison.get('losses', 0)}"
        lines.append(
            f"| {comparison.get('family', 'primary')} | {comparison['comparison']} | {comparison['n_instances']} | "
            f"{record_text} | {comparison['mean_difference']:.4f} | {comparison['median_difference']:.4f} | "
            f"{ci_text} | {win_text} | {effect_text} | {p_raw_text} | {p_holm_text} |"
        )
    lines.append("")
    lines.append(
        "> Com poucas instâncias o poder estatístico é baixo: trate p-valores como"
        " descritivos no piloto e dimensione a campanha principal pela análise de poder."
    )
    lines.append("")
    return "\n".join(lines)


def render_campaign_report(metadata: dict[str, Any], outcomes: list[ArmOutcome]) -> str:
    hardware = metadata.get("hardware", {})
    curation = metadata.get("dataset_curation") or {}

    def fmt_meta(value: Any, digits: int = 2) -> str:
        return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else "—"

    lines = [
        f"# Campanha {metadata['campaign_id']}",
        "",
        f"- Data: {metadata['created_at']} | Código: v{metadata['package_version']}"
        + (f" ({metadata['git_revision']})" if metadata.get("git_revision") else "")
        + (
            f" — **árvore com alterações não commitadas** (diff {metadata.get('git_diff_crc32')}); "
            "não use como campanha confirmatória"
            if metadata.get("git_dirty")
            else ""
        ),
        f"- Hardware: {hardware.get('os', '?')} {hardware.get('architecture', '?')}, "
        f"{hardware.get('cpu_physical', '?')} núcleos físicos",
        f"- Seeds de tuning: {metadata['seeds']} | validação: {metadata['validation_seeds']} | "
        f"teste: {metadata.get('test_seeds', [])}",
        f"- Execuções estimadas do solver: {metadata.get('estimated_solver_runs', '—')}",
        f"- Amostras de calibração de temperatura por run: {metadata.get('temperature_calibration_samples', '—')}",
        f"- Dataset: {metadata['dataset_summary']}",
        (
            f"- Curadoria do dataset: {curation.get('accepted', 0)}/{curation.get('instances', 0)} instâncias aceitas "
            f"no preflight; {curation.get('discriminative', 0)} com spread de caixas/gap; "
            f"{curation.get('class_slots_replaced', curation.get('class_replacements', 0))}/"
            f"{curation.get('class_slots', '—')} slot(s) de classe substituído(s); "
            f"{curation.get('fallback', 0)} fallback(s); {curation.get('too_easy', 0)} fácil(eis) demais; "
            f"{curation.get('too_slow', 0)} lenta(s) demais; "
            f"LB médio {fmt_meta(curation.get('mean_preflight_lower_bound'))}."
            if curation
            else "- Curadoria do dataset: não aplicada (manifest local ou geração explícita sem curadoria)."
        ),
        "- Artefatos: runs.csv para análise tabular; solutions.jsonl para placements, bins e convergência por run.",
        "",
        "**Protocolo: avaliação estática zero-shot.** Após o tuning no split de treino, a",
        "configuração (ou política por instância) de cada braço é CONGELADA; validação e",
        "teste não alimentam nenhum tuner (mede-se generalização, não desempenho prequential).",
        "Braços de configuração global (Random Search, EGO clássico, SMAC3, irace) respondem",
        "ao problema *single best configuration*; o EGO contextual responde à configuração",
        "por instância em zero-shot — apenas as features da instância nova são usadas para",
        "recomendar, nunca execuções dela. A comparação entre os dois grupos responde:",
        "\"a política por instância supera a melhor configuração global?\".",
        "",
        "A métrica primária é o gap ao lower bound (qualidade); a perda soma apenas uma",
        "penalidade de falha (tempo é reportado à parte, sem entrar na perda). O incumbente de cada braço é",
        "reavaliado de forma idêntica nas mesmas seeds frescas. A perda de treino é viesada",
        "por seleção (winner's curse) e fica apenas como referência.",
        "",
    ]
    warning = _comparison_warning(outcomes)
    if warning:
        lines.extend([f"> **Aviso de validação**: {warning}", ""])
    discrimination_warning = _discrimination_warning(outcomes)
    if discrimination_warning:
        lines.extend([f"> **Aviso de dataset**: {discrimination_warning}", ""])

    lines.extend(
        [
            "| Braço | Validação | Teste | Qualidade val. | Qualidade teste | Treino (viesada) | Tuning | Execuções | Status |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    valid = [] if discrimination_warning else [outcome for outcome in outcomes if _has_complete_validation(outcome)]
    best_validation = min((outcome.validation_loss for outcome in valid), default=None)

    def fmt(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "—"

    def fmt_quality(metrics: dict[str, Any] | None) -> str:
        if not metrics:
            return "—"
        bins = metrics.get("mean_bins_used")
        gap = metrics.get("mean_gap_to_lower_bound")
        optimal = metrics.get("optimal_rate")
        timeouts = int(metrics.get("timeout_runs") or 0)
        pieces = []
        if bins is not None:
            pieces.append(f"{float(bins):.2f} bins")
        if gap is not None:
            pieces.append(f"gap {float(gap):.3f}")
        if optimal is not None:
            pieces.append(f"ótimo {float(optimal) * 100:.0f}%")
        if timeouts:
            pieces.append(f"{timeouts} timeout(s)")
        return ", ".join(pieces) if pieces else "—"

    for outcome in outcomes:
        label = ARM_LABELS.get(outcome.arm, outcome.arm)
        if outcome.status == "ok":
            marker = (
                " **(melhor)**"
                if best_validation is not None
                and _has_complete_validation(outcome)
                and outcome.validation_loss == best_validation
                else ""
            )
            lines.append(
                f"| {label}{marker} | {fmt(outcome.validation_loss)} | {fmt(outcome.test_loss)} | "
                f"{fmt_quality(outcome.validation_metrics)} | {fmt_quality(outcome.test_metrics)} | "
                f"{fmt(outcome.training_best_loss)} | {outcome.tuning_seconds:.1f}s | "
                f"{outcome.tuning_solver_runs} | ok |"
            )
        else:
            lines.append(f"| {label} | — | — | — | — | — | — | — | {outcome.reason or outcome.status} |")
    lines.append("")
    return "\n".join(lines)
