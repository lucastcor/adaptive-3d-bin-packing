"""SMAC3 baseline runner (optional dependency).

SMAC3 (Lindauer et al., JMLR 2022) is the reference model-based algorithm
configurator (random-forest surrogate + racing/intensification). Install with
``pip install '.[tuners]'``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from adaptive_3dbp.tuning.objective import TuningObjective, values_to_config
from adaptive_3dbp.tuning.space_conversion import to_configspace

# Seeds de solver derivadas ficam acima deste piso: nunca colidem com as seeds
# canônicas de treino nem com as de validação/teste da campanha (seed + 10_000
# e seed + 20_000), então o SMAC não treina em seeds reservadas.
_DERIVED_SEED_FLOOR = 1_000_000
_DERIVED_SEED_SPAN = 2**31 - 1 - _DERIVED_SEED_FLOOR


def smac_available() -> bool:
    try:
        import smac  # noqa: F401

        return True
    except ImportError:
        return False


def derive_solver_seeds(trial_seed: int, base_seeds: Sequence[int]) -> list[int]:
    """Map the seed SMAC assigns to a trial onto one solver seed per objective seed.

    Deterministic (hash of ``(trial_seed, base_seed)``), so SMAC's pairing is
    preserved: its intensifier evaluates every configuration on the SAME
    sequence of trial seeds, hence on the same solver seeds. Distinct trial
    seeds give distinct solver seeds, so re-evaluating a configuration during
    intensification measures fresh solver randomness instead of replaying
    identical runs.
    """
    seeds: list[int] = []
    for base in base_seeds:
        digest = hashlib.blake2b(f"smac3:{int(trial_seed)}:{int(base)}".encode(), digest_size=8).digest()
        seeds.append(_DERIVED_SEED_FLOOR + int.from_bytes(digest, "big") % _DERIVED_SEED_SPAN)
    return seeds


def make_smac_target(objective: TuningObjective) -> Callable[..., float]:
    """SMAC target function that honours SMAC's ``seed``.

    With ``deterministic=False`` SMAC re-evaluates promising configurations
    under new seeds (up to ``max_config_calls``). Each call runs the objective
    on its instances with ``len(objective.seeds)`` solver seeds derived from
    SMAC's seed, so one call still costs ``objective.runs_per_trial`` solver
    runs and ``n_trials`` calls cost ``n_trials * runs_per_trial`` — the same
    budget as the other tuners. ``objective.seeds`` is restored after every
    call (validation_loss derives its held-out seeds from it).
    """
    base_seeds = list(objective.seeds)

    def target(config, seed: int = 0) -> float:
        objective.seeds = derive_solver_seeds(seed, base_seeds)
        try:
            return objective.evaluate_values(dict(config))
        finally:
            objective.seeds = list(base_seeds)

    return target


def make_smac_instance_target(objective: TuningObjective) -> Callable[..., float]:
    """Target for SMAC's algorithm-configuration mode: ONE solver run per call.

    SMAC picks the (instance, seed) pair; the solver seed is derived from
    SMAC's seed exactly like :func:`make_smac_target`, so it never collides
    with the campaign's train/validation/test seeds.
    """
    by_id = {instance.instance_id: instance for instance in objective.instances}
    base_seeds = list(objective.seeds)

    def target(config, instance: str, seed: int = 0) -> float:
        objective.seeds = derive_solver_seeds(seed, base_seeds[:1])
        try:
            losses = objective.replicate_losses_on(by_id[instance], values_to_config(dict(config)))
        finally:
            objective.seeds = list(base_seeds)
        return float(losses[0])

    return target


def run_smac_ac(
    objective: TuningObjective,
    search_space_definition: dict[str, Any],
    instance_features: dict[str, list[float]],
    n_trials: int = 50,
    seed: int = 0,
    output_directory: Path | None = None,
) -> dict[str, Any]:  # pragma: no cover - requires optional smac dependency
    """SMAC3 in algorithm-configuration mode (AlgorithmConfigurationFacade).

    Instances and their features are given to SMAC, which races configurations
    over instances (intensification) with a random forest over
    (configuration, instance features) — SMAC's intended use for algorithm
    configuration (Hutter et al., 2011). The result is still ONE global
    configuration. ``n_trials`` is in tuner trials as for the other arms; SMAC
    gets ``n_trials * runs_per_trial`` single (config, instance, seed) runs,
    the same solver-run budget.
    """
    from smac import AlgorithmConfigurationFacade, Scenario

    space = to_configspace(search_space_definition)
    budget_runs = n_trials * objective.runs_per_trial
    scenario = Scenario(
        space,
        deterministic=False,
        n_trials=budget_runs,
        seed=seed,
        instances=[instance.instance_id for instance in objective.instances],
        instance_features=instance_features,
        **({"output_directory": output_directory} if output_directory is not None else {}),
    )
    # pca_components=None: with more instance features than PCA components
    # SMAC 2.4 trains the forest on PCA-reduced features but marginalises over
    # instances with the raw ones (AssertionError in
    # predict_marginalized_over_instances_batch). 16 features are few enough
    # to use directly.
    model = AlgorithmConfigurationFacade.get_model(scenario, pca_components=None)
    facade = AlgorithmConfigurationFacade(
        scenario, make_smac_instance_target(objective), model=model, overwrite=True
    )
    incumbent = facade.optimize()
    incumbent_values = dict(incumbent)
    try:
        incumbent_loss = float(facade.runhistory.get_cost(incumbent))
    except Exception:  # pragma: no cover - API differences across smac versions
        incumbent_loss = float("nan")
    return {
        "tuner": "smac3_ac",
        "n_trials": n_trials,
        "solver_run_budget": budget_runs,
        "incumbent_values": incumbent_values,
        "incumbent_config": values_to_config(incumbent_values).model_dump(),
        "incumbent_loss": incumbent_loss,
    }


def run_smac(
    objective: TuningObjective,
    search_space_definition: dict[str, Any],
    n_trials: int = 50,
    seed: int = 0,
    output_directory: Path | None = None,
) -> dict[str, Any]:  # pragma: no cover - requires optional smac dependency
    from smac import HyperparameterOptimizationFacade, Scenario

    space = to_configspace(search_space_definition)

    scenario = Scenario(
        space,
        deterministic=False,
        n_trials=n_trials,
        seed=seed,
        **({"output_directory": output_directory} if output_directory is not None else {}),
    )
    facade = HyperparameterOptimizationFacade(scenario, make_smac_target(objective), overwrite=True)
    incumbent = facade.optimize()
    incumbent_values = dict(incumbent)
    # Budget fairness: read the incumbent cost from SMAC's own runhistory
    # instead of re-evaluating (which would consume extra solver runs that the
    # other tuners do not get).
    try:
        incumbent_loss = float(facade.runhistory.get_cost(incumbent))
    except Exception:  # pragma: no cover - API differences across smac versions
        incumbent_loss = float("nan")
    return {
        "tuner": "smac3",
        "n_trials": n_trials,
        "incumbent_values": incumbent_values,
        "incumbent_config": values_to_config(incumbent_values).model_dump(),
        "incumbent_loss": incumbent_loss,
    }
