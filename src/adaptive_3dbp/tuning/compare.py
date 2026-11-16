"""Equal-budget comparison of configurators: Random Search, EGO variants, SMAC3, irace.

Protocol (docs/experiment_protocol.md):
- every tuner optimises the SAME objective (mean performance_loss over the same
  instances and tuning seeds) with the SAME budget of solver runs;
- irace's budget unit is one (config, instance, seed) run, so its
  maxExperiments is set to n_trials * |instances| * |seeds| (parity);
- the comparable performance number is the *validation loss*: every tuner's
  incumbent is re-evaluated on the same fresh seeds, identically. Training
  "best_loss" values are biased by selection (winner's curse) and measured
  differently across tuners — they are reported, but must not be compared.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from adaptive_3dbp.configurators.classic_ego import ClassicEGOConfigurator
from adaptive_3dbp.configurators.noisy_contextual_ego import NoiseAwareContextualEGO
from adaptive_3dbp.configurators.random_search import RandomSearchConfigurator
from adaptive_3dbp.configurators.search_space import SearchSpace
from adaptive_3dbp.domain.models import SolverConfig
from adaptive_3dbp.features.instance_features import extract_instance_features
from adaptive_3dbp.tuning.irace_runner import (
    best_values_from_campaign,
    generate_irace_campaign,
    irace_available,
    run_irace,
)
from adaptive_3dbp.tuning.objective import TuningObjective, values_to_config
from adaptive_3dbp.tuning.smac_runner import run_smac, smac_available


def _mean_context(objective: TuningObjective) -> dict[str, float]:
    rows = [extract_instance_features(instance) for instance in objective.instances]
    keys = sorted(rows[0])
    return {key: statistics.fmean(float(row[key]) for row in rows) for key in keys}


def _run_tracked(
    objective: TuningObjective,
    tune: Callable[[], tuple[float, SolverConfig | None]],
) -> dict[str, Any]:
    runs_before = objective.solver_runs
    started = time.perf_counter()
    best_loss, best_config = tune()
    elapsed = time.perf_counter() - started
    entry: dict[str, Any] = {
        "status": "ok",
        "training_best_loss": best_loss,
        "tuning_seconds": elapsed,
        "solver_runs": objective.solver_runs - runs_before,
        "best_config": best_config.model_dump() if best_config else None,
    }
    if best_config is not None:
        entry["validation_loss"] = objective.validation_loss(best_config)
    return entry


def run_comparison(
    objective: TuningObjective,
    search_space_definition: dict[str, Any],
    n_trials: int = 30,
    seed: int = 0,
    output_dir: Path | None = None,
    include_smac: bool = True,
    include_irace: bool = True,
    src_path: Path | None = None,
) -> dict[str, Any]:
    space = SearchSpace(search_space_definition)
    results: dict[str, Any] = {
        "n_trials": n_trials,
        "seed": seed,
        "tuning_seeds": objective.seeds,
        "validation_seeds": [s + 10_000 for s in objective.seeds],
        "solver_runs_per_trial": objective.runs_per_trial,
        "budget_solver_runs": n_trials * objective.runs_per_trial,
        "tuners": {},
    }

    # ---------------- Random Search (Bergstra & Bengio, 2012) ----------------
    def tune_random() -> tuple[float, SolverConfig | None]:
        tuner = RandomSearchConfigurator(space, seed=seed)
        best_loss, best_config = float("inf"), None
        for config in tuner.suggest(n_trials):
            loss = objective.evaluate_config(config)
            if loss < best_loss:
                best_loss, best_config = loss, config
        return best_loss, best_config

    results["tuners"]["random_search"] = _run_tracked(objective, tune_random)

    # ---------------- Classic EGO (Jones et al., 1998) ----------------
    def tune_classic() -> tuple[float, SolverConfig | None]:
        ego = ClassicEGOConfigurator(space, seed=seed)
        best_loss, best_config = float("inf"), None
        for _ in range(n_trials):
            config = ego.suggest(1)[0]
            loss = objective.evaluate_config(config)
            ego.observe(config, loss)
            if loss < best_loss:
                best_loss, best_config = loss, config
        return best_loss, best_config

    results["tuners"]["classic_ego"] = _run_tracked(objective, tune_classic)

    # ---------------- Noise-aware contextual EGO ----------------
    def tune_noisy() -> tuple[float, SolverConfig | None]:
        context = _mean_context(objective)
        noisy = NoiseAwareContextualEGO(space, seed=seed)
        best_loss, best_config = float("inf"), None
        for _ in range(n_trials):
            config = noisy.suggest(context, 1)[0]
            losses = objective.replicate_losses(config)
            noisy.observe(context, config, losses)
            loss = statistics.fmean(losses)
            if loss < best_loss:
                best_loss, best_config = loss, config
        return best_loss, best_config

    results["tuners"]["noisy_contextual_ego"] = _run_tracked(objective, tune_noisy)

    # ---------------- SMAC3 (Lindauer et al., 2022) ----------------
    if include_smac and smac_available():

        def tune_smac() -> tuple[float, SolverConfig | None]:
            payload = run_smac(objective, search_space_definition, n_trials=n_trials, seed=seed)
            config = values_to_config(payload["incumbent_values"])
            return float(payload["incumbent_loss"]), config

        results["tuners"]["smac3"] = _run_tracked(objective, tune_smac)
    else:
        results["tuners"]["smac3"] = {
            "status": "skipped",
            "reason": "smac não instalado — pip install '.[tuners]' (requer compilador C++ para pyrfr)",
        }

    # ---------------- irace (López-Ibáñez et al., 2016) ----------------
    if include_irace:
        campaign_dir = (output_dir or Path("artifacts/tuning")) / "irace_campaign"
        max_experiments = max(n_trials * objective.runs_per_trial, 96)  # irace exige orçamento mínimo
        resolved_src = src_path or Path(__file__).resolve().parents[2]
        generate_irace_campaign(
            output_dir=campaign_dir,
            search_space_definition=search_space_definition,
            instances=objective.instances,
            src_path=resolved_src,
            max_experiments=max_experiments,
            max_evaluations=objective.max_evaluations,
            time_limit_seconds=objective.time_limit_seconds,
            seed=seed,
        )
        if irace_available():
            runs_before = objective.solver_runs
            started = time.perf_counter()
            run_payload = run_irace(campaign_dir)
            elapsed = time.perf_counter() - started
            best_values, stats = best_values_from_campaign(campaign_dir)
            entry: dict[str, Any] = {
                "status": "ok" if run_payload["returncode"] == 0 and best_values else "error",
                "tuning_seconds": elapsed,
                "solver_runs": stats.get("evaluations", 0),
                "max_experiments": max_experiments,
                "campaign_dir": str(campaign_dir),
                "campaign_stats": stats,
                "returncode": run_payload["returncode"],
            }
            if best_values is not None:
                config = values_to_config(best_values)
                entry["best_config"] = config.model_dump()
                entry["training_best_loss"] = stats.get("best_mean_cost")
                entry["validation_loss"] = objective.validation_loss(config)
            results["tuners"]["irace"] = entry
        else:
            results["tuners"]["irace"] = {
                "status": "skipped",
                "reason": (
                    "executável 'irace' não encontrado — instale R e o pacote irace "
                    "(install.packages('irace')); a campanha pronta foi gerada em "
                    f"{campaign_dir} (rode 'irace --scenario scenario.txt' lá e re-execute a comparação)"
                ),
                "campaign_dir": str(campaign_dir),
                "max_experiments": max_experiments,
            }
    else:
        results["tuners"]["irace"] = {"status": "skipped", "reason": "desabilitado por flag"}

    results["report_markdown"] = render_markdown_report(results)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (output_dir / "comparison.md").write_text(results["report_markdown"], encoding="utf-8")
    return results


_TUNER_LABELS = {
    "random_search": "Random Search",
    "classic_ego": "EGO clássico",
    "noisy_contextual_ego": "EGO contextual noise-aware",
    "smac3": "SMAC3",
    "irace": "irace",
}


def render_markdown_report(results: dict[str, Any]) -> str:
    lines = [
        "# Comparação de configuradores",
        "",
        f"- Trials por tuner: {results['n_trials']} (= {results['budget_solver_runs']} execuções do solver)",
        f"- Seeds de tuning: {results['tuning_seeds']} | Seeds de validação: {results['validation_seeds']}",
        "",
        "A coluna comparável é **Perda (validação)**: o incumbente de cada tuner reavaliado",
        "nas mesmas seeds de validação. `best_loss` de treino tem viés de seleção e não deve",
        "ser comparado entre tuners.",
        "",
        "| Tuner | Perda (validação) | Perda treino (viesada) | Tempo de tuning | Execuções do solver | Status |",
        "|---|---|---|---|---|---|",
    ]
    entries = results["tuners"]
    valid = {name: entry for name, entry in entries.items() if entry.get("validation_loss") is not None}
    best_validation = min((entry["validation_loss"] for entry in valid.values()), default=None)
    for name, entry in entries.items():
        label = _TUNER_LABELS.get(name, name)
        if entry.get("status") == "ok" and entry.get("validation_loss") is not None:
            marker = " **(melhor)**" if best_validation is not None and entry["validation_loss"] == best_validation else ""
            lines.append(
                f"| {label}{marker} | {entry['validation_loss']:.4f} | "
                f"{entry.get('training_best_loss', float('nan')):.4f} | "
                f"{entry.get('tuning_seconds', 0.0):.1f}s | {entry.get('solver_runs', '—')} | ok |"
            )
        else:
            reason = entry.get("reason", entry.get("status", "—"))
            lines.append(f"| {label} | — | — | — | — | {reason} |")
    lines.append("")
    return "\n".join(lines)
