from __future__ import annotations

from adaptive_3dbp.domain.models import ProblemInstance, SearchBudget, SolverConfig, SolverResult
from adaptive_3dbp.experiments.control import SolveControl
from adaptive_3dbp.metaheuristics.parallel_tempering.solver import ParallelTemperingSolver
from adaptive_3dbp.metaheuristics.simulated_annealing.solver import SimulatedAnnealingSolver
from adaptive_3dbp.native.cpp_bridge import NativeCoreError, solve_with_native_core, solve_with_native_core_parallel


def solve_instance(
    instance: ProblemInstance,
    config: SolverConfig,
    seed: int = 0,
    control: SolveControl | None = None,
    progress_callback=None,
) -> SolverResult:
    if config.algorithm == "parallel_tempering":
        # Replica exchange is implemented in the Python core only.
        result = ParallelTemperingSolver().solve(
            instance=instance,
            config=config,
            budget=SearchBudget(max_evaluations=config.max_evaluations, timeout_seconds=config.time_limit_seconds),
            seed=seed,
            control=control,
            progress_callback=progress_callback,
        )
        if config.use_native_core:
            return result.model_copy(
                update={"diagnostics": [*result.diagnostics, "parallel tempering is Python-only; native core skipped"]}
            )
        return result
    if config.use_native_core and config.parallel_mode == "cooperative_process_pool":
        # Cooperative central-memory exchange is implemented in the Python core only.
        python_config = config.model_copy(update={"use_native_core": False})
        result = SimulatedAnnealingSolver().solve(
            instance=instance,
            config=python_config,
            budget=SearchBudget(
                max_evaluations=python_config.max_evaluations,
                timeout_seconds=python_config.time_limit_seconds,
            ),
            seed=seed,
            control=control,
            progress_callback=progress_callback,
        )
        return result.model_copy(
            update={"diagnostics": [*result.diagnostics, "cooperative mode is Python-only; native core skipped"]}
        )
    if config.use_native_core:
        try:
            if config.parallel_mode == "process_pool" and config.parallel_workers > 1 and config.multi_start_count > 1:
                return solve_with_native_core_parallel(instance, config, seed, control=control)
            return solve_with_native_core(instance, config, seed, control=control)
        except NativeCoreError:
            fallback_config = config.model_copy(update={"use_native_core": False})
            result = SimulatedAnnealingSolver().solve(
                instance=instance,
                config=fallback_config,
                budget=SearchBudget(
                    max_evaluations=fallback_config.max_evaluations,
                    timeout_seconds=fallback_config.time_limit_seconds,
                ),
                seed=seed,
                control=control,
                progress_callback=progress_callback,
            )
            return result.model_copy(update={"diagnostics": [*result.diagnostics, "native core unavailable; used Python fallback"]})

    return SimulatedAnnealingSolver().solve(
        instance=instance,
        config=config,
        budget=SearchBudget(max_evaluations=config.max_evaluations, timeout_seconds=config.time_limit_seconds),
        seed=seed,
        control=control,
        progress_callback=progress_callback,
    )
