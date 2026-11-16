from __future__ import annotations

import random
import time

from adaptive_3dbp import __version__
from adaptive_3dbp.constructors.pool import build_initial_pool
from adaptive_3dbp.decoders.extreme_points import ExtremePointDecoder
from adaptive_3dbp.domain.bounds import simple_lower_bound
from adaptive_3dbp.domain.models import (
    ConvergencePoint,
    DecodedSolution,
    ProblemInstance,
    SearchBudget,
    SolverConfig,
    SolverResult,
)
from adaptive_3dbp.domain.scoring import lexicographic_less, score_solution
from adaptive_3dbp.domain.validation import validate_problem_instance, validate_solution
from adaptive_3dbp.experiments.control import SolveControl
from adaptive_3dbp.features.instance_features import extract_instance_features
from adaptive_3dbp.metaheuristics.parallel_tempering.exchange import exchange_adjacent_replicas, geometric_ladder
from adaptive_3dbp.metaheuristics.simulated_annealing.operators import propose_neighbor
from adaptive_3dbp.metaheuristics.simulated_annealing.schedule import (
    acceptance_probability,
    calibrate_initial_temperature,
)


class ParallelTemperingSolver:
    """Replica exchange over the SA neighbourhood, decoder and energy.

    K = ``pt_replicas`` Metropolis chains run at fixed temperatures on a
    geometric ladder from T_max (calibrated like the SA's T0 on the best
    start) down to T_max * ``pt_min_temperature_ratio``. Each round, every
    replica makes ``pt_swap_interval`` steps, then adjacent temperatures try
    to exchange configurations (even/odd pairs on alternate rounds). Replicas
    are interleaved step by step in one process, so a run is a deterministic
    function of the seed and the evaluation budget is shared by all replicas
    (start decodes and calibration samples included).

    Cooling, reheating and stagnation (SA-only fields) do not apply: the hot
    end of the ladder provides the diversification. ``multi_start_count`` is
    replaced by ``pt_replicas``.
    """

    name = "parallel_tempering"

    def __init__(self, decoder: ExtremePointDecoder | None = None) -> None:
        self.decoder = decoder or ExtremePointDecoder()

    def solve(
        self,
        instance: ProblemInstance,
        config: SolverConfig | None = None,
        budget: SearchBudget | None = None,
        seed: int = 0,
        control: SolveControl | None = None,
        progress_callback=None,
    ) -> SolverResult:
        config = config or SolverConfig(algorithm="parallel_tempering")
        budget = budget or SearchBudget(
            max_evaluations=config.max_evaluations,
            timeout_seconds=config.time_limit_seconds,
        )
        diagnostics = validate_problem_instance(instance)
        lower_bound = simple_lower_bound(instance)
        if diagnostics:
            return SolverResult(
                instance_id=instance.instance_id,
                solver=self.name,
                solver_version=__version__,
                configuration_id=config.configuration_id,
                seed=seed,
                status="infeasible",
                best_found_bins=0,
                lower_bound=lower_bound,
                best_known_bins=instance.best_known_bins,
                gap_to_lower_bound=0.0,
                runtime_seconds=0.0,
                evaluations=0,
                objective_scalar=float("inf"),
                placements=[],
                bin_metrics=[],
                diagnostics=diagnostics,
                features=extract_instance_features(instance),
            )

        started = time.perf_counter()
        rng = random.Random(seed)
        replica_count = config.pt_replicas
        diagnostics_out: list[str] = []
        if config.parallel_mode != "sequential":
            diagnostics_out.append("parallel tempering interleaves replicas in one process; parallel_mode ignored")

        # Each unique start is decoded once (the decoder is deterministic);
        # when the pool is smaller than K, replicas cycle through the starts.
        evaluations = 0
        starts: list[DecodedSolution] = []
        for candidate in build_initial_pool(instance, seed, replica_count)[:replica_count]:
            starts.append(self.decoder.decode(instance, candidate, config.placement_policy_id))
            evaluations += 1
            if evaluations >= budget.max_evaluations or int(starts[-1].score[0]) <= lower_bound:
                break
        # Lexicographic order (same as lexicographic_less): the best start seeds
        # the calibration walk and the coldest replica.
        starts.sort(key=lambda decoded: decoded.score)
        best = starts[0]

        calibration = calibrate_initial_temperature(
            instance,
            self.decoder,
            best,
            config,
            rng,
            max_samples=0 if int(best.score[0]) <= lower_bound else budget.max_evaluations - evaluations,
            deadline=started + budget.timeout_seconds,
        )
        evaluations += calibration.evaluations
        if lexicographic_less(calibration.best.score, best.score):
            best = calibration.best
        hottest = max(calibration.temperature, config.pt_max_temperature or 0.0)
        temperatures = geometric_ladder(hottest, config.pt_min_temperature_ratio, replica_count)

        # Slot k always runs at temperatures[k] (index 0 = coldest); swaps move
        # configurations between slots. Better starts go to colder slots.
        current = [starts[k % len(starts)] for k in range(replica_count)]
        pair_attempts = [0] * (replica_count - 1)
        pair_accepts = [0] * (replica_count - 1)
        rounds = 0
        sweeps = 0
        replica = 0

        convergence = [
            ConvergencePoint(
                evaluation=evaluations,
                elapsed_seconds=time.perf_counter() - started,
                bins_used=int(best.score[0]),
                objective_scalar=best.objective_scalar,
                accepted=True,
                temperature=temperatures[0],
                current_bins=int(current[0].score[0]),
                current_objective_scalar=current[0].objective_scalar,
                best_bins=int(best.score[0]),
                best_objective_scalar=best.objective_scalar,
                delta=0.0,
                improved=True,
            )
        ]
        if progress_callback is not None:
            progress_callback(convergence[-1])

        while evaluations < budget.max_evaluations and int(best.score[0]) > lower_bound:
            if control is not None:
                paused_seconds = control.checkpoint()
                if paused_seconds > 0:
                    started += paused_seconds
            if time.perf_counter() - started >= budget.timeout_seconds:
                break

            state = current[replica]
            temperature = temperatures[replica]
            neighbor = propose_neighbor(state.candidate, instance, config, rng)
            decoded_neighbor = self.decoder.decode(instance, neighbor, config.placement_policy_id)
            evaluations += 1

            delta = decoded_neighbor.objective_scalar - state.objective_scalar
            accepted = rng.random() < acceptance_probability(delta, temperature)
            if accepted:
                current[replica] = decoded_neighbor

            improved = lexicographic_less(decoded_neighbor.score, best.score)
            if improved:
                best = decoded_neighbor
            if evaluations % config.convergence_sample_rate == 0 or improved:
                convergence.append(
                    ConvergencePoint(
                        evaluation=evaluations,
                        elapsed_seconds=time.perf_counter() - started,
                        bins_used=int(best.score[0]),
                        objective_scalar=best.objective_scalar,
                        accepted=accepted,
                        temperature=temperature,
                        current_bins=int(current[replica].score[0]),
                        current_objective_scalar=current[replica].objective_scalar,
                        best_bins=int(best.score[0]),
                        best_objective_scalar=best.objective_scalar,
                        delta=delta,
                        improved=improved,
                    )
                )
                if progress_callback is not None:
                    progress_callback(convergence[-1])

            # One sweep = one step per replica; a round = pt_swap_interval sweeps.
            replica += 1
            if replica == replica_count:
                replica = 0
                sweeps += 1
                if sweeps == config.pt_swap_interval:
                    sweeps = 0
                    exchange_adjacent_replicas(current, temperatures, rounds % 2, rng, pair_attempts, pair_accepts)
                    rounds += 1

        convergence.append(
            ConvergencePoint(
                evaluation=evaluations,
                elapsed_seconds=time.perf_counter() - started,
                bins_used=int(best.score[0]),
                objective_scalar=best.objective_scalar,
                accepted=True,
                temperature=temperatures[0],
                current_bins=int(current[0].score[0]),
                current_objective_scalar=current[0].objective_scalar,
                best_bins=int(best.score[0]),
                best_objective_scalar=best.objective_scalar,
                delta=0.0,
                improved=False,
            )
        )
        if progress_callback is not None:
            progress_callback(convergence[-1])

        runtime = time.perf_counter() - started
        diagnostics_out.append(
            f"parallel tempering: replicas={replica_count}, starts={len(starts)}, "
            f"t_max={temperatures[-1]:.6g}, t_min={temperatures[0]:.6g}, swap_interval={config.pt_swap_interval}, "
            f"rounds={rounds}{', calibration fallback' if calibration.fallback else ''}"
        )
        # Pairs (k, k+1) from cold to hot; a pair never attempted reports 0.00
        # (see the attempt counts to tell it apart from a pair that always rejects).
        diagnostics_out.append(
            "pt swap acceptance: "
            + ",".join(f"{accepts / attempts if attempts else 0.0:.2f}" for accepts, attempts in zip(pair_accepts, pair_attempts))
        )
        diagnostics_out.append("pt swap attempts: " + ",".join(str(attempts) for attempts in pair_attempts))
        diagnostics_out.extend(best.diagnostics)

        solution_diagnostics = validate_solution(instance, best.placements)
        metrics, bin_metrics, _score = score_solution(instance, best.placements)
        status = "feasible" if not solution_diagnostics else "error"

        return SolverResult(
            instance_id=instance.instance_id,
            solver=self.name,
            solver_version=__version__,
            configuration_id=config.configuration_id,
            seed=seed,
            status=status,
            best_found_bins=metrics.bins_used,
            lower_bound=metrics.lower_bound,
            best_known_bins=metrics.best_known_bins,
            optimal_proven=metrics.bins_used == metrics.lower_bound,
            gap_to_best_known=metrics.gap_to_best_known,
            gap_to_lower_bound=metrics.gap_to_lower_bound,
            runtime_seconds=runtime,
            evaluations=evaluations,
            objective_scalar=metrics.objective_scalar,
            placements=list(best.placements),
            bin_metrics=bin_metrics,
            convergence=convergence,
            diagnostics=[*dict.fromkeys(diagnostics_out + solution_diagnostics)],
            features=extract_instance_features(instance),
        )
