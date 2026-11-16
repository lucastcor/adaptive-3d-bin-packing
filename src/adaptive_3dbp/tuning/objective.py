"""Shared tuning objective so every tuner optimises exactly the same function.

The objective is the mean ``performance_loss`` (gap to the strong lower bound
plus a failure penalty; runtime is reported but not penalised by default) over
a fixed set of training instances and seeds. Fixing instances, seeds and budget across tuners
is required for a fair comparison (Lopez-Ibanez et al., 2016; Lindauer et al.,
2022).
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from adaptive_3dbp import __version__
from adaptive_3dbp.domain.bounds import simple_lower_bound
from adaptive_3dbp.domain.models import ProblemInstance, SolverConfig, SolverResult
from adaptive_3dbp.experiments.runner import solve_instance
from adaptive_3dbp.features.instance_features import extract_instance_features
from adaptive_3dbp.learning.online_gpr import performance_loss

OPERATOR_NAMES = ("swap", "relocate", "reverse", "rotate", "ruin_recreate")
# Pontos de convergência amostrados por run de campanha (além das melhorias).
CONVERGENCE_POINTS_PER_RUN = 100
RunCallback = Callable[[dict[str, Any]], None]


def _failed_result(
    instance: ProblemInstance,
    config: SolverConfig,
    seed: int,
    status: str,
    elapsed_seconds: float,
    reason: str,
) -> SolverResult:
    lower_bound = simple_lower_bound(instance)
    return SolverResult(
        instance_id=instance.instance_id,
        solver="simulated_annealing",
        solver_version=__version__,
        configuration_id=config.configuration_id,
        seed=seed,
        status=status,  # type: ignore[arg-type]
        best_found_bins=0,
        lower_bound=lower_bound,
        best_known_bins=instance.best_known_bins,
        optimal_proven=False,
        gap_to_best_known=None,
        gap_to_lower_bound=1.0,
        runtime_seconds=max(0.0, elapsed_seconds),
        evaluations=0,
        objective_scalar=1_000_000_000.0,
        placements=[],
        bin_metrics=[],
        convergence=[],
        features=extract_instance_features(instance),
        diagnostics=[reason],
    )


def _solve_instance_worker(instance_json: str, config_json: str, seed: int, result_queue: Any) -> None:
    try:
        instance = ProblemInstance.model_validate_json(instance_json)
        config = SolverConfig.model_validate_json(config_json)
        result = solve_instance(instance, config, seed)
        result_queue.put(("ok", result.model_dump_json()))
    except BaseException as exc:  # pragma: no cover - defensive worker boundary.
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _solve_with_hard_timeout(
    instance: ProblemInstance,
    config: SolverConfig,
    seed: int,
    timeout_seconds: float,
) -> SolverResult:
    """Run one solver call behind a wall-clock guard.

    The SA loop already receives ``time_limit_seconds``, but expensive
    preparation steps such as temperature calibration can happen before the
    evaluation loop checks the budget. Campaigns need a process-level guard so
    one difficult instance cannot stall a whole tuner comparison.
    """
    started = time.perf_counter()
    # O limite interno do SA (time_limit_seconds) é quem governa o orçamento:
    # ele para a busca e devolve a melhor solução anytime. Este guard externo
    # existe APENAS para travamentos patológicos, então precisa de margem para
    # o overhead legítimo fora do loop (spawn do subprocesso ~0.5-1s, imports,
    # calibração de temperatura, decode final). Sem margem, runs válidos que
    # usam o orçamento inteiro são mortos no fio da navalha e viram "timeout"
    # com penalidade 10.0 — degenerando a comparação inteira.
    limit = max(float(timeout_seconds) * 2.0 + 5.0, 1.0)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_solve_instance_worker,
        args=(instance.model_dump_json(), config.model_dump_json(), seed, result_queue),
    )
    try:
        process.start()
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return _failed_result(instance, config, seed, "error", elapsed, f"falha ao iniciar worker do run: {exc}")

    message: tuple[str, str] | None = None
    deadline = started + limit
    while time.perf_counter() < deadline:
        remaining = max(0.0, deadline - time.perf_counter())
        try:
            # Read before join: SolverResult JSON can be large (placements +
            # convergence). If the parent joins first, the child may block
            # waiting for the queue feeder to flush, turning a valid run into a
            # hard timeout.
            message = result_queue.get(timeout=min(0.1, remaining))
            break
        except queue.Empty:
            if not process.is_alive():
                process.join(0.1)
                try:
                    message = result_queue.get(timeout=0.1)
                except queue.Empty:
                    message = None
                break

    elapsed = time.perf_counter() - started
    if message is None and process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():  # pragma: no cover - rare hard-kill path.
            process.kill()
            process.join(1.0)
        result_queue.close()
        return _failed_result(
            instance,
            config,
            seed,
            "timeout",
            limit,
            f"run interrompido pelo limite externo de {limit:.2f}s",
        )

    process.join(1.0)
    if message is None:
        result_queue.close()
        return _failed_result(
            instance,
            config,
            seed,
            "error",
            elapsed,
            f"worker terminou sem devolver resultado (exitcode={process.exitcode})",
        )
    try:
        result_queue.close()
    except Exception:
        pass

    kind, payload = message
    if kind == "ok":
        return SolverResult.model_validate_json(payload)
    return _failed_result(instance, config, seed, "error", elapsed, str(payload))


def _run_record(
    label: str,
    instance: ProblemInstance,
    seed: int,
    result: SolverResult,
    loss: float,
    config: SolverConfig,
) -> dict[str, Any]:
    utilizations = [metric.utilization for metric in result.bin_metrics]
    return {
        "label": label,
        "instance_id": instance.instance_id,
        "seed": seed,
        "loss": loss,
        "bins_used": result.best_found_bins,
        "lower_bound": result.lower_bound,
        "best_known_bins": result.best_known_bins,
        "gap_to_lower_bound": result.gap_to_lower_bound,
        "gap_to_best_known": result.gap_to_best_known,
        "optimal_proven": result.optimal_proven,
        "runtime_seconds": result.runtime_seconds,
        "evaluations": result.evaluations,
        "status": result.status,
        "objective_scalar": result.objective_scalar,
        "placements_count": len(result.placements),
        "bin_utilization_mean": statistics.fmean(utilizations) if utilizations else None,
        "bin_utilization_min": min(utilizations) if utilizations else None,
        "bin_utilization_max": max(utilizations) if utilizations else None,
        "stop_reason": stop_reason(result, config),
        "core": solver_core(result, config),
        "diagnostics": " | ".join(result.diagnostics),
        "config_json": config.model_dump_json(),
    }


def solver_core(result: SolverResult, config: SolverConfig) -> str:
    """Which core produced the run: ``native``, ``python`` or ``python_fallback``
    (native requested but it failed and the Python core ran instead; same
    semantics, ~250x slower, so time-limited runs could differ)."""
    if not config.use_native_core:
        return "python"
    if any("used Python fallback" in diagnostic for diagnostic in result.diagnostics):
        return "python_fallback"
    return "native"


def stop_reason(result: SolverResult, config: SolverConfig) -> str:
    """Why the run ended. ``time`` means the wall-clock cap, not the evaluation
    budget, decided the run: its result depends on the machine and its load,
    so campaigns report how many runs ended this way."""
    if result.status in ("error", "infeasible"):
        return "failure"
    if result.evaluations >= config.max_evaluations:
        return "evaluations"
    if result.optimal_proven:
        return "lower_bound"
    if result.runtime_seconds >= 0.98 * config.time_limit_seconds:
        return "time"
    return "converged"


def _solution_record(label: str, instance: ProblemInstance, seed: int, result: SolverResult) -> dict[str, Any]:
    return {
        "label": label,
        "instance_id": instance.instance_id,
        "seed": seed,
        "status": result.status,
        "bins_used": result.best_found_bins,
        "lower_bound": result.lower_bound,
        "best_known_bins": result.best_known_bins,
        "gap_to_lower_bound": result.gap_to_lower_bound,
        "gap_to_best_known": result.gap_to_best_known,
        "optimal_proven": result.optimal_proven,
        "runtime_seconds": result.runtime_seconds,
        "evaluations": result.evaluations,
        "objective_scalar": result.objective_scalar,
        "bin_metrics": [metric.model_dump(mode="json") for metric in result.bin_metrics],
        "placements": [placement.model_dump(mode="json") for placement in result.placements],
        "convergence": [point.model_dump(mode="json") for point in result.convergence],
        "diagnostics": result.diagnostics,
    }


def values_to_config(values: dict[str, Any], base: SolverConfig | None = None) -> SolverConfig:
    """Build a SolverConfig from flat tuner values.

    Operator probabilities may arrive flattened as ``op_<name>`` weights (the
    representation used for SMAC/irace, which have no simplex parameter type);
    they are renormalised here, which preserves the original simplex semantics.
    """
    base = base or SolverConfig()
    payload: dict[str, Any] = {}
    weights: dict[str, float] = {}
    for key, value in values.items():
        if key.startswith("op_"):
            weights[key[3:]] = max(float(value), 1e-6)
        else:
            payload[key] = value
    if weights:
        total = sum(weights.values())
        payload["operator_probabilities"] = {name: weight / total for name, weight in weights.items()}
    return base.model_copy(update=payload)


def config_to_flat_values(config: SolverConfig, parameter_names: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in parameter_names:
        if name.startswith("op_"):
            values[name] = float(config.operator_probabilities.get(name[3:], 0.0))
        else:
            values[name] = getattr(config, name)
    return values


@dataclass(slots=True)
class TuningObjective:
    instances: list[ProblemInstance]
    seeds: list[int] = field(default_factory=lambda: [11, 23, 47])
    max_evaluations: int = 1500
    time_limit_seconds: float = 5.0
    temperature_calibration_samples: int = 24
    # Same core for every arm of a campaign. The native core is only allowed
    # once Python/C++ decoder parity is tested (tests/unit/test_native_parity.py).
    use_native_core: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    solver_runs: int = 0
    # Instrumentação de campanha: cada execução do solver vira uma linha em
    # `records` (exportável em CSV/Parquet); `label` identifica tuner/fase.
    records: list[dict[str, Any]] = field(default_factory=list)
    solutions: list[dict[str, Any]] = field(default_factory=list)
    label: str = ""
    repository: Any = None
    campaign_id: str | None = None
    run_callback: RunCallback | None = None
    # Quantos runs INDEPENDENTES do solver podem executar em paralelo. Cada run
    # roda no seu próprio subprocesso (já isolado, com guarda de timeout); este
    # número só controla quantos ficam em voo ao mesmo tempo. 1 = serial.
    # IMPORTANTE: o RESULTADO de cada run depende apenas de (instância, config,
    # seed) e é determinístico; o bookkeeping (records/CSV/persistência) é
    # sempre montado em ordem determinística de tarefa. Logo, o número de
    # workers muda só o tempo de parede — nunca os resultados — e a campanha é
    # reproduzível bit-a-bit entre máquinas com contagens de núcleos diferentes.
    max_parallel_runs: int = 1

    @property
    def runs_per_trial(self) -> int:
        """Solver executions consumed by one trial (needed for budget parity
        with tuners whose unit is a single (config, instance, seed) run, e.g.
        irace's maxExperiments)."""
        return len(self.instances) * len(self.seeds)

    def _fixed(self, config: SolverConfig) -> SolverConfig:
        return config.model_copy(
            update={
                "max_evaluations": self.max_evaluations,
                "time_limit_seconds": self.time_limit_seconds,
                "temperature_calibration_samples": self.temperature_calibration_samples,
                "use_native_core": self.use_native_core,
                "parallel_mode": "sequential",
                "parallel_workers": 1,
                # Só muda o que é gravado (melhorias são sempre gravadas), nunca
                # a busca. Com taxa 1, o piloto MPV gerou 27 GB de solutions.jsonl.
                "convergence_sample_rate": max(1, self.max_evaluations // CONVERGENCE_POINTS_PER_RUN),
            }
        )

    def _emit(
        self,
        event: str,
        instance: ProblemInstance,
        seed: int,
        ordinal: int,
        result: SolverResult | None = None,
        loss: float | None = None,
    ) -> None:
        """Fire a progress event. Always called from the parent thread (never a
        worker), so the campaign callback can mutate its counters race-free."""
        if self.run_callback is None:
            return
        payload: dict[str, Any] = {
            "event": event,
            "label": self.label,
            "instance_id": instance.instance_id,
            "seed": seed,
            "objective_run": ordinal,
        }
        if event == "finished" and result is not None:
            payload.update(
                loss=loss,
                runtime_seconds=result.runtime_seconds,
                evaluations=result.evaluations,
                status=result.status,
            )
        self.run_callback(payload)

    def _commit_run(
        self,
        instance: ProblemInstance,
        config: SolverConfig,
        seed: int,
        result: SolverResult,
        loss: float,
    ) -> None:
        """Deterministic bookkeeping for one finished run. Called in task order
        from the parent thread so records, CSV order and persistence are
        identical regardless of how many runs executed in parallel."""
        self.solver_runs += 1
        self.records.append(_run_record(self.label, instance, seed, result, loss, config))
        self.solutions.append(_solution_record(self.label, instance, seed, result))
        if self.repository is not None:
            self.repository.save_run(
                instance,
                config,
                result,
                campaign_id=self.campaign_id,
                experiment_id=self.label or None,
            )

    def _run_batch(self, tasks: list[tuple[ProblemInstance, SolverConfig, int]]) -> list[float]:
        """Execute a batch of INDEPENDENT solver runs and return their losses in
        task order.

        Each task is ``(instance, fixed_config, seed)`` where ``fixed_config``
        already went through :meth:`_fixed`. Up to ``max_parallel_runs`` run
        concurrently, each in its own hardened subprocess (own timeout/kill via
        :func:`_solve_with_hard_timeout`). The worker threads only wait on those
        subprocesses (they release the GIL on ``join``), so the real work runs
        truly in parallel across cores.

        Reproducibility guarantee: results are collected and committed in
        deterministic task order, so losses, records, CSV rows and persisted
        rows are identical to the serial path — only wall-clock time changes.
        """
        if not tasks:
            return []
        workers = max(1, min(self.max_parallel_runs, len(tasks)))
        losses: list[float] = [0.0] * len(tasks)

        if workers == 1:
            # Caminho serial: comportamento histórico (started+finished por run,
            # bookkeeping inline, na ordem natural). Mantido para reprodutibilidade
            # exata e como referência do teste serial-vs-paralelo.
            for index, (instance, config, seed) in enumerate(tasks):
                self._emit("started", instance, seed, self.solver_runs + 1)
                result = _solve_with_hard_timeout(instance, config, seed, self.time_limit_seconds)
                loss = performance_loss(result)
                losses[index] = loss
                self._commit_run(instance, config, seed, result, loss)
                self._emit("finished", instance, seed, self.solver_runs, result=result, loss=loss)
            return losses

        results: list[SolverResult | None] = [None] * len(tasks)
        for instance, _config, seed in tasks:
            self._emit("started", instance, seed, self.solver_runs + 1)
        base = self.solver_runs
        done = 0
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="campaign-run") as pool:
            future_to_index = {
                pool.submit(_solve_with_hard_timeout, instance, config, seed, self.time_limit_seconds): index
                for index, (instance, config, seed) in enumerate(tasks)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                instance, config, seed = tasks[index]
                try:
                    result = future.result()
                except BaseException as exc:  # pragma: no cover - _solve_with_hard_timeout não deveria levantar
                    result = _failed_result(
                        instance, config, seed, "error", 0.0, f"worker paralelo falhou: {type(exc).__name__}: {exc}"
                    )
                loss = performance_loss(result)
                results[index] = result
                losses[index] = loss
                done += 1
                # Progresso AO VIVO em ordem de conclusão (apenas display: move a
                # barra). O bookkeeping autoritativo vem depois, em ordem de tarefa.
                self._emit("finished", instance, seed, base + done, result=result, loss=loss)
        for index, (instance, config, seed) in enumerate(tasks):
            self._commit_run(instance, config, seed, results[index], losses[index])
        return losses

    def _losses(
        self,
        config: SolverConfig,
        seeds: list[int],
        instances: list[ProblemInstance] | None = None,
    ) -> list[float]:
        fixed = self._fixed(config)
        target = instances if instances is not None else self.instances
        tasks = [(instance, fixed, seed) for instance in target for seed in seeds]
        return self._run_batch(tasks)

    def evaluate_config(self, config: SolverConfig) -> float:
        losses = self._losses(config, self.seeds)
        loss = statistics.fmean(losses)
        self.history.append({"config": self._fixed(config).model_dump(), "loss": loss, "losses": losses})
        return loss

    def evaluate_values(self, values: dict[str, Any]) -> float:
        return self.evaluate_config(values_to_config(values))

    def evaluate_configs(self, configs: list[SolverConfig]) -> list[float]:
        """Mean loss for each config, running ALL (config × instance × seed)
        solver calls in ONE parallel batch.

        Used by tuners whose trials are mutually independent (e.g. Random
        Search, where every configuration is drawn upfront). Per-config means
        and the recorded history are identical to calling :meth:`evaluate_config`
        in a loop; batching only widens the parallel front (trials × instances ×
        seeds independent runs instead of instances × seeds per trial), which
        keeps every core busy. Results are unchanged — only wall-clock time.
        """
        configs = list(configs)
        if not configs:
            return []
        tasks: list[tuple[ProblemInstance, SolverConfig, int]] = []
        for config in configs:
            fixed = self._fixed(config)
            for instance in self.instances:
                for seed in self.seeds:
                    tasks.append((instance, fixed, seed))
        flat = self._run_batch(tasks)
        per_config = self.runs_per_trial
        means: list[float] = []
        for index, config in enumerate(configs):
            chunk = flat[index * per_config : (index + 1) * per_config]
            mean = statistics.fmean(chunk)
            means.append(mean)
            self.history.append({"config": self._fixed(config).model_dump(), "loss": mean, "losses": chunk})
        return means

    def replicate_losses(self, config: SolverConfig) -> list[float]:
        """Per-seed losses (used by noise-aware tuners as replications)."""
        return self._losses(config, self.seeds)

    def replicate_losses_on(self, instance: ProblemInstance, config: SolverConfig) -> list[float]:
        """Per-seed losses on a single instance.

        Used by per-instance (contextual) tuners: each training instance
        yields one observation under its OWN feature vector, instead of
        averaging away the context (parecer §5.1). Total solver runs per trial
        are identical to ``replicate_losses`` (instances × seeds).
        """
        return self._losses(config, self.seeds, instances=[instance])

    def replicate_losses_grouped(self, config: SolverConfig) -> list[list[float]]:
        """Run ``config`` on every instance × seed in ONE parallel batch and
        return per-instance seed-loss lists, in instance order.

        Equivalent to calling :meth:`replicate_losses_on` once per instance, but
        all instances × seeds runs go out together (one config, so they are
        independent) and are observed afterwards. Used by per-instance
        (contextual) tuners during training: a single trial fans out across all
        cores instead of evaluating one instance at a time. The grouping and
        order match ``replicate_losses_on`` per instance, so the tuner state is
        identical to the serial loop.
        """
        fixed = self._fixed(config)
        tasks = [(instance, fixed, seed) for instance in self.instances for seed in self.seeds]
        flat = self._run_batch(tasks)
        width = len(self.seeds)
        return [flat[index * width : (index + 1) * width] for index in range(len(self.instances))]

    def evaluate_config_map(self, config_for: Callable[[ProblemInstance], SolverConfig]) -> float:
        """Evaluate a per-instance configuration policy π(z) (zero-shot).

        ``config_for`` must use only frozen historical knowledge — no run of
        the evaluated instance may inform its own configuration (protocol A,
        zero-shot; guarantees no test leakage by construction). Consumes the
        same instances × seeds budget as ``evaluate_config``.

        All instance × seed runs are dispatched in ONE parallel batch (each
        instance keeps its own policy-chosen config); losses are then regrouped
        per instance for the mean. ``config_for`` is still called exactly once
        per instance, in instance order, so the policy and result are unchanged.
        """
        configs = [self._fixed(config_for(instance)) for instance in self.instances]
        tasks = [
            (instance, configs[index], seed)
            for index, instance in enumerate(self.instances)
            for seed in self.seeds
        ]
        losses = self._run_batch(tasks)
        loss = statistics.fmean(losses)
        self.history.append({"config": "per_instance_policy", "loss": loss, "losses": losses})
        return loss

    def validation_loss(self, config: SolverConfig, offset: int = 10_000) -> float:
        """Re-evaluate an incumbent on *fresh* seeds, identically for every tuner.

        The best training loss of each tuner is biased by selection (winner's
        curse) and is computed differently across tuners (single evaluation,
        replication mean, racing). Re-evaluating every incumbent on the same
        held-out seeds is the only apples-to-apples performance number.
        """
        validation_seeds = [seed + offset for seed in self.seeds]
        return statistics.fmean(self._losses(config, validation_seeds))
