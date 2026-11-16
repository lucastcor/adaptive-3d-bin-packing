from __future__ import annotations

import multiprocessing
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait

from adaptive_3dbp import __version__
from adaptive_3dbp.constructors.pool import build_initial_pool
from adaptive_3dbp.decoders.extreme_points import ExtremePointDecoder
from adaptive_3dbp.domain.bounds import simple_lower_bound
from adaptive_3dbp.domain.models import (
    CandidateSolution,
    ConvergencePoint,
    DecodedSolution,
    ProblemInstance,
    SearchBudget,
    SolverConfig,
    SolverResult,
)
from adaptive_3dbp.domain.scoring import lexicographic_less, score_solution
from adaptive_3dbp.domain.validation import validate_problem_instance, validate_solution
from adaptive_3dbp.features.instance_features import extract_instance_features
from adaptive_3dbp.experiments.control import SolveControl
from adaptive_3dbp.metaheuristics.simulated_annealing.cooperative import CooperativeChannel
from adaptive_3dbp.metaheuristics.simulated_annealing.operators import propose_neighbor
from adaptive_3dbp.metaheuristics.simulated_annealing.schedule import (
    acceptance_probability,
    calibrate_initial_temperature,
)


class SimulatedAnnealingSolver:
    name = "simulated_annealing"

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
        config = config or SolverConfig()
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
        initial_pool = build_initial_pool(instance, seed, config.multi_start_count)

        best_decoded: DecodedSolution | None = None
        best_convergence: list[ConvergencePoint] = []
        evaluations = 0
        diagnostics_out: list[str] = []
        worker_count = min(config.parallel_workers, len(initial_pool), os.cpu_count() or 1)
        use_parallel = config.parallel_mode == "process_pool" and worker_count > 1 and len(initial_pool) > 1
        use_cooperative = (
            config.parallel_mode == "cooperative_process_pool" and worker_count > 1 and len(initial_pool) > 1
        )
        if control is not None and use_parallel:
            diagnostics_out.append("job control requested; Python process_pool runs without cooperative pause")

        if use_cooperative:
            best_decoded, evaluations = self._solve_cooperative(
                instance=instance,
                config=config,
                budget=budget,
                seed=seed,
                initial_pool=initial_pool,
                worker_count=worker_count,
                best_convergence=best_convergence,
                diagnostics_out=diagnostics_out,
                control=control,
                progress_callback=progress_callback,
            )
        elif use_parallel:
            per_start_evaluations = max(1, budget.max_evaluations // len(initial_pool))
            worker_config = config.model_copy(update={"parallel_mode": "sequential", "parallel_workers": 1})
            diagnostics_out.append(
                f"parallel process pool used: workers={worker_count}, starts={len(initial_pool)}, "
                f"evaluations_per_start={per_start_evaluations}"
            )
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _run_single_start_task,
                        instance,
                        worker_config,
                        per_start_evaluations,
                        budget.timeout_seconds,
                        seed + start_index * 9973,
                        initial_candidate,
                        start_index * per_start_evaluations,
                    )
                    for start_index, initial_candidate in enumerate(initial_pool)
                ]
                for future in as_completed(futures):
                    try:
                        decoded, convergence, used_evaluations = future.result()
                    except Exception as exc:  # pragma: no cover - defensive diagnostics for worker crashes.
                        diagnostics_out.append(f"parallel worker failed: {exc}")
                        continue
                    evaluations += used_evaluations
                    best_convergence.extend(convergence)
                    if decoded.diagnostics:
                        diagnostics_out.extend(decoded.diagnostics)
                    if best_decoded is None or lexicographic_less(decoded.score, best_decoded.score):
                        best_decoded = decoded
        else:
            for start_index, initial_candidate in enumerate(initial_pool):
                remaining_evaluations = budget.max_evaluations - evaluations
                remaining_time = budget.timeout_seconds - (time.perf_counter() - started)
                if remaining_evaluations <= 0 or remaining_time <= 0:
                    break
                # Equal share of what is left, so every start gets a real run
                # (the first start used to receive the whole remaining budget,
                # which made multi_start_count inert). Evaluations a start
                # leaves unused (stagnation, lower bound) flow to later starts.
                remaining_starts = len(initial_pool) - start_index
                start_evaluations = max(1, remaining_evaluations // remaining_starts)

                decoded, convergence, used_evaluations = self._run_single_start(
                    instance=instance,
                    config=config,
                    budget=SearchBudget(
                        max_evaluations=start_evaluations,
                        timeout_seconds=max(0.001, remaining_time),
                    ),
                    seed=seed + start_index * 9973,
                    initial_candidate=initial_candidate,
                    elapsed_offset=time.perf_counter() - started,
                    evaluation_offset=evaluations,
                    control=control,
                    progress_callback=progress_callback,
                )
                evaluations += used_evaluations
                best_convergence.extend(convergence)
                if decoded.diagnostics:
                    diagnostics_out.extend(decoded.diagnostics)
                if best_decoded is None or lexicographic_less(decoded.score, best_decoded.score):
                    best_decoded = decoded
                if best_decoded.score[0] <= lower_bound:
                    break

        runtime = time.perf_counter() - started
        if best_decoded is None:
            raise RuntimeError("SA did not evaluate any candidate")

        solution_diagnostics = validate_solution(instance, best_decoded.placements)
        metrics, bin_metrics, _score = score_solution(instance, best_decoded.placements)
        status = "feasible" if not solution_diagnostics else "error"
        best_convergence.sort(key=lambda point: (point.evaluation, point.elapsed_seconds))

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
            placements=list(best_decoded.placements),
            bin_metrics=bin_metrics,
            convergence=best_convergence,
            diagnostics=[*dict.fromkeys(diagnostics_out + solution_diagnostics)],
            features=extract_instance_features(instance),
        )

    def _solve_cooperative(
        self,
        instance: ProblemInstance,
        config: SolverConfig,
        budget: SearchBudget,
        seed: int,
        initial_pool: list[CandidateSolution],
        worker_count: int,
        best_convergence: list[ConvergencePoint],
        diagnostics_out: list[str],
        control: SolveControl | None = None,
        progress_callback=None,
    ) -> tuple[DecodedSolution | None, int]:
        """Cooperative multi-start: workers share the global best via central memory.

        Coordination uses a logical version counter (no physical clock sync is
        needed on a single host) and heartbeats for failure detection; the
        parent process is the fixed coordinator. See cooperative.py docstring.

        Pause/cancel travel through the same central memory: the coordinator
        mirrors the SolveControl flags into the shared dict every tick and
        workers honour them at their sync points.
        """
        per_start_evaluations = max(1, budget.max_evaluations // len(initial_pool))
        worker_config = config.model_copy(update={"parallel_mode": "sequential", "parallel_workers": 1})
        diagnostics_out.append(
            f"cooperative process pool used: workers={worker_count}, starts={len(initial_pool)}, "
            f"evaluations_per_start={per_start_evaluations}, sync_interval={config.cooperative_sync_interval}"
        )
        best_decoded: DecodedSolution | None = None
        evaluations = 0
        deadline = time.perf_counter() + budget.timeout_seconds * 3 + 30.0
        with multiprocessing.Manager() as manager:
            state = manager.dict({"version": 0})
            lock = manager.Lock()
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_worker = {
                    executor.submit(
                        _run_cooperative_start_task,
                        instance,
                        worker_config,
                        per_start_evaluations,
                        budget.timeout_seconds,
                        seed + start_index * 9973,
                        initial_candidate,
                        start_index * per_start_evaluations,
                        state,
                        lock,
                        start_index,
                        config.cooperative_sync_interval,
                    ): start_index
                    for start_index, initial_candidate in enumerate(initial_pool)
                }
                pending = set(future_to_worker)
                finished_workers: set[int] = set()
                stalled_workers: set[int] = set()
                solve_started = time.perf_counter()
                while pending:
                    done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                    now = time.time()
                    if control is not None:
                        # Mirror pause/cancel into central memory; workers honour
                        # the flags at their sync points (granularidade grossa).
                        state["control_paused"] = control.is_paused
                        state["control_cancelled"] = control.is_cancelled
                    if progress_callback is not None:
                        total_evaluations = sum(
                            int(state.get(f"evaluations_{worker_id}", 0)) for worker_id in range(len(initial_pool))
                        )
                        best_score = state.get("best_score")
                        if best_score is not None and total_evaluations > 0:
                            progress_callback(
                                ConvergencePoint(
                                    evaluation=total_evaluations,
                                    elapsed_seconds=time.perf_counter() - solve_started,
                                    bins_used=int(best_score[0]),
                                    # score = (bins, fragmentation, -compactness); E = bins + r.
                                    objective_scalar=float(best_score[0]) + float(best_score[1]),
                                    accepted=True,
                                    temperature=0.0,
                                    best_bins=int(best_score[0]),
                                    improved=False,
                                )
                            )
                    for future, worker_id in future_to_worker.items():
                        if worker_id in finished_workers or worker_id in stalled_workers:
                            continue
                        last_heartbeat = state.get(f"heartbeat_{worker_id}")
                        if last_heartbeat is not None and now - last_heartbeat > config.heartbeat_timeout_seconds:
                            stalled_workers.add(worker_id)
                            diagnostics_out.append(
                                f"cooperative worker {worker_id} missed heartbeat for more than "
                                f"{config.heartbeat_timeout_seconds}s (continuing without it)"
                            )
                    for future in done:
                        worker_id = future_to_worker[future]
                        finished_workers.add(worker_id)
                        try:
                            decoded, convergence, used_evaluations = future.result()
                        except Exception as exc:  # pragma: no cover - defensive diagnostics.
                            diagnostics_out.append(f"cooperative worker {worker_id} failed: {exc}")
                            continue
                        evaluations += used_evaluations
                        best_convergence.extend(convergence)
                        if decoded.diagnostics:
                            diagnostics_out.extend(decoded.diagnostics)
                        if best_decoded is None or lexicographic_less(decoded.score, best_decoded.score):
                            best_decoded = decoded
                    if control is not None and control.is_paused:
                        # Paused time must not consume the coordinator deadline.
                        deadline += 1.0
                    if time.perf_counter() > deadline:
                        diagnostics_out.append("cooperative coordinator deadline exceeded; abandoning pending workers")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
        return best_decoded, evaluations

    def _run_single_start(
        self,
        instance: ProblemInstance,
        config: SolverConfig,
        budget: SearchBudget,
        seed: int,
        initial_candidate: CandidateSolution,
        elapsed_offset: float,
        evaluation_offset: int,
        channel: CooperativeChannel | None = None,
        control: SolveControl | None = None,
        progress_callback=None,
    ) -> tuple[DecodedSolution, list[ConvergencePoint], int]:
        rng = random.Random(seed)
        started = time.perf_counter()
        lower_bound = simple_lower_bound(instance)
        current = self.decoder.decode(instance, initial_candidate, config.placement_policy_id)
        evaluations = 1
        # A start already at the lower bound is optimal: skip calibration (it
        # used to spend up to 4x temperature_calibration_samples decodes there)
        # and let the lower-bound stop below end the start right away.
        at_lower_bound = int(current.score[0]) <= lower_bound
        calibration = calibrate_initial_temperature(
            instance,
            self.decoder,
            current,
            config,
            rng,
            max_samples=0 if at_lower_bound else budget.max_evaluations - evaluations,
            deadline=started + budget.timeout_seconds,
        )
        evaluations += calibration.evaluations
        best = calibration.best
        temperature = calibration.temperature
        accepted_since_improvement = 0
        reheats = 0
        convergence = [
            ConvergencePoint(
                evaluation=evaluation_offset + evaluations,
                elapsed_seconds=elapsed_offset,
                bins_used=int(best.score[0]),
                objective_scalar=best.objective_scalar,
                accepted=True,
                temperature=temperature,
                current_bins=int(current.score[0]),
                current_objective_scalar=current.objective_scalar,
                best_bins=int(best.score[0]),
                best_objective_scalar=best.objective_scalar,
                delta=0.0,
                improved=True,
            )
        ]
        if progress_callback is not None:
            progress_callback(convergence[-1])
        if int(best.score[0]) <= lower_bound:
            # Optimal already (start or calibration walk reached the bound).
            if channel is not None:
                channel.publish(best.score, best.candidate)
                channel.report_evaluations(evaluation_offset + evaluations)
            return best, convergence, evaluations

        stop_temperature = max(config.min_temperature, config.min_temperature_ratio * temperature)
        while (
            evaluations < budget.max_evaluations
            and temperature >= stop_temperature
            and time.perf_counter() - started < budget.timeout_seconds
        ):
            level_iterations = min(config.iterations_per_temperature, budget.max_evaluations - evaluations)
            for _ in range(level_iterations):
                if control is not None:
                    paused_seconds = control.checkpoint()
                    if paused_seconds > 0:
                        started += paused_seconds
                if time.perf_counter() - started >= budget.timeout_seconds:
                    break
                neighbor = propose_neighbor(current.candidate, instance, config, rng)
                decoded_neighbor = self.decoder.decode(instance, neighbor, config.placement_policy_id)
                evaluations += 1

                delta = decoded_neighbor.objective_scalar - current.objective_scalar
                accepted = rng.random() <= acceptance_probability(delta, temperature)
                if accepted:
                    current = decoded_neighbor
                    accepted_since_improvement += 1

                improved = lexicographic_less(decoded_neighbor.score, best.score)
                if improved:
                    best = decoded_neighbor
                    accepted_since_improvement = 0
                if evaluations % config.convergence_sample_rate == 0 or improved:
                    convergence.append(
                        ConvergencePoint(
                            evaluation=evaluation_offset + evaluations,
                            elapsed_seconds=elapsed_offset + time.perf_counter() - started,
                            bins_used=int(best.score[0]),
                            objective_scalar=best.objective_scalar,
                            accepted=accepted,
                            temperature=temperature,
                            current_bins=int(current.score[0]),
                            current_objective_scalar=current.objective_scalar,
                            best_bins=int(best.score[0]),
                            best_objective_scalar=best.objective_scalar,
                            delta=delta,
                            improved=improved,
                        )
                    )
                    if progress_callback is not None:
                        progress_callback(convergence[-1])

                if channel is not None and evaluations % channel.sync_interval == 0:
                    channel.heartbeat()
                    channel.report_evaluations(evaluation_offset + evaluations)
                    if channel.is_cancelled():
                        channel.publish(best.score, best.candidate)
                        return best, convergence, evaluations
                    paused_seconds = channel.wait_if_paused()
                    if paused_seconds > 0:
                        started += paused_seconds
                    channel.publish(best.score, best.candidate)
                    fetched = channel.fetch_better_than(best.score)
                    if fetched is not None:
                        adopted = self.decoder.decode(instance, fetched, config.placement_policy_id)
                        evaluations += 1
                        current = adopted
                        if lexicographic_less(adopted.score, best.score):
                            best = adopted
                            accepted_since_improvement = 0

                if accepted_since_improvement >= config.stagnation_limit:
                    if config.reheating_factor > 1.0 and reheats < 2:
                        temperature *= config.reheating_factor
                        accepted_since_improvement = 0
                        reheats += 1
                    else:
                        if channel is not None:
                            channel.publish(best.score, best.candidate)
                            channel.report_evaluations(evaluation_offset + evaluations)
                        return best, convergence, evaluations

                if int(best.score[0]) <= lower_bound:
                    if channel is not None:
                        channel.publish(best.score, best.candidate)
                        channel.report_evaluations(evaluation_offset + evaluations)
                    return best, convergence, evaluations

            temperature *= config.cooling_rate

        convergence.append(
            ConvergencePoint(
                evaluation=evaluation_offset + evaluations,
                elapsed_seconds=elapsed_offset + time.perf_counter() - started,
                bins_used=int(best.score[0]),
                objective_scalar=best.objective_scalar,
                accepted=True,
                temperature=temperature,
                current_bins=int(current.score[0]),
                current_objective_scalar=current.objective_scalar,
                best_bins=int(best.score[0]),
                best_objective_scalar=best.objective_scalar,
                delta=0.0,
                improved=False,
            )
        )
        if progress_callback is not None:
            progress_callback(convergence[-1])
        if channel is not None:
            channel.publish(best.score, best.candidate)
            channel.report_evaluations(evaluation_offset + evaluations)
        return best, convergence, evaluations


def _run_single_start_task(
    instance: ProblemInstance,
    config: SolverConfig,
    max_evaluations: int,
    timeout_seconds: float,
    seed: int,
    initial_candidate: CandidateSolution,
    evaluation_offset: int,
) -> tuple[DecodedSolution, list[ConvergencePoint], int]:
    solver = SimulatedAnnealingSolver()
    return solver._run_single_start(
        instance=instance,
        config=config,
        budget=SearchBudget(max_evaluations=max_evaluations, timeout_seconds=timeout_seconds),
        seed=seed,
        initial_candidate=initial_candidate,
        elapsed_offset=0.0,
        evaluation_offset=evaluation_offset,
    )


def _run_cooperative_start_task(
    instance: ProblemInstance,
    config: SolverConfig,
    max_evaluations: int,
    timeout_seconds: float,
    seed: int,
    initial_candidate: CandidateSolution,
    evaluation_offset: int,
    state: object,
    lock: object,
    worker_id: int,
    sync_interval: int,
) -> tuple[DecodedSolution, list[ConvergencePoint], int]:
    channel = CooperativeChannel(state, lock, worker_id, sync_interval)
    channel.heartbeat()
    solver = SimulatedAnnealingSolver()
    return solver._run_single_start(
        instance=instance,
        config=config,
        budget=SearchBudget(max_evaluations=max_evaluations, timeout_seconds=timeout_seconds),
        seed=seed,
        initial_candidate=initial_candidate,
        elapsed_offset=0.0,
        evaluation_offset=evaluation_offset,
        channel=channel,
    )
