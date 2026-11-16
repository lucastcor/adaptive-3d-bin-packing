from __future__ import annotations

import functools
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

from adaptive_3dbp import __version__
from adaptive_3dbp.constructors.pool import build_initial_pool
from adaptive_3dbp.domain.bounds import simple_lower_bound
from adaptive_3dbp.domain.models import (
    CandidateSolution,
    ConvergencePoint,
    DecodedSolution,
    ItemSpec,
    Placement,
    ProblemInstance,
    SolverConfig,
    SolverResult,
)
from adaptive_3dbp.domain.scoring import compute_bin_metrics, score_solution
from adaptive_3dbp.domain.validation import validate_problem_instance, validate_solution
from adaptive_3dbp.experiments.control import SolveCancelled, SolveControl
from adaptive_3dbp.features.instance_features import extract_instance_features

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows: only the in-process lock applies.
    fcntl = None  # type: ignore[assignment]


class NativeCoreError(RuntimeError):
    pass


ROOT_DIR = Path(__file__).resolve().parents[3]
CPP_SOURCE = ROOT_DIR / "cpp" / "sa_core.cpp"
NATIVE_BINARY = ROOT_DIR / "artifacts" / "bin" / "sa_core"
# Build stamp (also written by scripts/build_native.sh): the sha256 of the
# exact source bytes compiled, the flags and the compiler. A content hash,
# unlike mtimes, survives checkouts, copies and clock skew, and never runs a
# binary built from other source or by another compiler.
NATIVE_HASH_FILE = NATIVE_BINARY.with_name(NATIVE_BINARY.name + ".sha256")
# flock()ed around check + build, so concurrent processes compile once.
NATIVE_LOCK_FILE = NATIVE_BINARY.with_name(NATIVE_BINARY.name + ".lock")
# -ffp-contract=off: no FMA contraction, so the native decoder reproduces the
# Python reference bit for bit. Keep in sync with scripts/build_native.sh.
COMPILE_FLAGS = ("-O3", "-std=c++17", "-ffp-contract=off")
PROTOCOL_VERSION = "v2"
START_SEED_STRIDE = 9973
DECODE_TIMEOUT_SECONDS = 60.0

_KNOWN_POLICIES = frozenset({"min_z_y_x", "contact_then_z", "min_waste"})
_KNOWN_OPERATORS = frozenset({"swap", "relocate", "reverse", "rotate", "ruin_recreate"})
# CPython >= 3.12 sums floats with Neumaier compensation; the native core
# emulates whichever sum() the running interpreter uses.
_SUM_MODE = "neumaier" if sys.version_info >= (3, 12) else "naive"
_UNPLACED_MESSAGES = {
    "no_fitting_rotation": "has no fitting rotation",
    "could_not_be_placed": "could not be placed",
}
# What a corrupted or truncated payload raises while it is interpreted
# (pydantic's ValidationError and JSONDecodeError are ValueErrors).
_MALFORMED_PAYLOAD_ERRORS = (KeyError, IndexError, TypeError, ValueError)
_build_lock = threading.Lock()


def native_core_available() -> bool:
    return NATIVE_BINARY.exists() or shutil.which("g++") is not None


@contextmanager
def _native_build_lock() -> Iterator[None]:
    """Thread lock plus an inter-process flock on NATIVE_LOCK_FILE."""
    with _build_lock:
        if fcntl is None:
            yield
            return
        NATIVE_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(NATIVE_LOCK_FILE, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@functools.lru_cache(maxsize=8)
def _compiler_identity(compiler: str) -> str:
    """First line of ``<compiler> --version`` (same as build_native.sh)."""
    completed = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=False)
    lines = (completed.stdout or completed.stderr or "").strip().splitlines()
    return lines[0].strip() if lines else "unknown"


def _stamp_text(source_sha256: str, compiler_identity: str) -> str:
    return f"source_sha256 {source_sha256}\nflags {' '.join(COMPILE_FLAGS)}\ncompiler {compiler_identity}\n"


def _read_stamp() -> str | None:
    try:
        return NATIVE_HASH_FILE.read_text(encoding="utf-8")
    except OSError:
        return None


def _parse_stamp(text: str | None) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in (text or "").splitlines():
        key, _, value = line.strip().partition(" ")
        if key:
            fields[key] = value.strip()
    return fields


def _stamp_matches(stamp: dict[str, str], source_sha256: str, compiler_identity: str | None) -> bool:
    if stamp.get("source_sha256") != source_sha256 or stamp.get("flags") != " ".join(COMPILE_FLAGS):
        return False
    # Without a compiler the existing binary is the only option; with one, a
    # binary built by another compiler (version) is rebuilt.
    return compiler_identity is None or stamp.get("compiler") == compiler_identity


def ensure_native_core() -> Path:
    """Return the native binary, rebuilding it when its stamp is stale."""
    with _native_build_lock():
        if not CPP_SOURCE.exists():
            if NATIVE_BINARY.exists():
                return NATIVE_BINARY
            raise NativeCoreError(f"native core source not found: {CPP_SOURCE}")
        source = CPP_SOURCE.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        compiler = shutil.which("g++")
        identity = _compiler_identity(compiler) if compiler is not None else None
        if NATIVE_BINARY.exists() and _stamp_matches(_parse_stamp(_read_stamp()), digest, identity):
            return NATIVE_BINARY
        if compiler is None or identity is None:
            raise NativeCoreError("g++ is not available")
        NATIVE_BINARY.parent.mkdir(parents=True, exist_ok=True)
        # Compile a private copy of the bytes that were hashed (the stamp
        # always describes the binary, even if the source changes meanwhile)
        # and swap binary + stamp in atomically.
        suffix = f".{os.getpid()}.{threading.get_ident()}.tmp"
        tmp_binary = NATIVE_BINARY.with_name(NATIVE_BINARY.name + suffix)
        tmp_stamp = NATIVE_HASH_FILE.with_name(NATIVE_HASH_FILE.name + suffix)
        try:
            with tempfile.TemporaryDirectory(prefix="sa_core_build_") as build_dir:
                build_source = Path(build_dir) / CPP_SOURCE.name
                build_source.write_bytes(source)
                command = [compiler, *COMPILE_FLAGS, str(build_source), "-o", str(tmp_binary)]
                completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise NativeCoreError(completed.stderr.strip() or "native core compilation failed")
            tmp_stamp.write_text(_stamp_text(digest, identity), encoding="utf-8")
            os.replace(tmp_binary, NATIVE_BINARY)
            os.replace(tmp_stamp, NATIVE_HASH_FILE)
        finally:
            tmp_binary.unlink(missing_ok=True)
            tmp_stamp.unlink(missing_ok=True)
        return NATIVE_BINARY


def native_core_fingerprint() -> dict[str, Any]:
    """Provenance of the native core for experiment metadata (never builds).

    ``up_to_date`` says whether the stamp matches the current source and
    flags, i.e. whether ``ensure_native_core()`` would run this binary as is.
    """
    with _native_build_lock():
        source = CPP_SOURCE.read_bytes() if CPP_SOURCE.exists() else None
        stamp = _read_stamp()
        binary = NATIVE_BINARY.read_bytes() if NATIVE_BINARY.exists() else None
    source_sha256 = hashlib.sha256(source).hexdigest() if source is not None else None
    fields = _parse_stamp(stamp)
    return {
        "source_sha256": source_sha256,
        "stamp": stamp,
        "stamp_fields": fields,
        "binary": str(NATIVE_BINARY),
        "binary_sha256": hashlib.sha256(binary).hexdigest() if binary is not None else None,
        "compile_flags": list(COMPILE_FLAGS),
        "protocol_version": PROTOCOL_VERSION,
        "up_to_date": (
            binary is not None and source_sha256 is not None and _stamp_matches(fields, source_sha256, None)
        ),
    }


def _drain(stream: IO[str] | None, sink: list[str]) -> None:
    if stream is not None:
        sink.append(stream.read())
        stream.close()


def _run_native_process(
    binary: Path, protocol: str, timeout_seconds: float, control: SolveControl | None
) -> subprocess.CompletedProcess[str]:
    if control is None:
        return subprocess.run(
            [str(binary)],
            input=protocol,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )

    process = subprocess.Popen(
        [str(binary)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Drain both pipes while polling pause/cancel: a payload larger than the
    # pipe buffer would otherwise block the core until the timeout.
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    readers = [
        threading.Thread(target=_drain, args=(process.stdout, stdout_chunks), daemon=True),
        threading.Thread(target=_drain, args=(process.stderr, stderr_chunks), daemon=True),
    ]
    for reader in readers:
        reader.start()
    assert process.stdin is not None
    try:
        process.stdin.write(protocol)
        process.stdin.close()
    except BrokenPipeError:
        pass  # the core exited early; its exit code and stderr tell why

    paused = False
    pause_started = 0.0
    paused_total = 0.0
    started = time.perf_counter()
    try:
        while process.poll() is None:
            if control.is_cancelled:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise SolveCancelled("native solve cancelled")

            if control.is_paused and not paused:
                # POSIX only: without SIGSTOP (Windows) the subprocess cannot be
                # suspended, so we must NOT mark it as paused (the timeout would
                # be frozen while the process keeps running unbounded).
                if hasattr(signal, "SIGSTOP"):
                    process.send_signal(signal.SIGSTOP)
                    paused = True
                    pause_started = time.perf_counter()
            elif not control.is_paused and paused:
                if hasattr(signal, "SIGCONT"):
                    process.send_signal(signal.SIGCONT)
                paused = False
                paused_total += time.perf_counter() - pause_started

            effective_elapsed = time.perf_counter() - started - paused_total
            if not paused and effective_elapsed > timeout_seconds:
                process.kill()
                raise subprocess.TimeoutExpired(str(binary), timeout_seconds)
            time.sleep(0.05)
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        if paused and process.poll() is None and hasattr(signal, "SIGCONT"):
            process.send_signal(signal.SIGCONT)
        for reader in readers:
            reader.join(timeout=5.0)

    return subprocess.CompletedProcess([str(binary)], process.returncode, "".join(stdout_chunks), "".join(stderr_chunks))


def _parse_payload(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if completed.returncode != 0 or not isinstance(payload, dict) or payload.get("status") == "error":
        detail = ""
        if isinstance(payload, dict):
            detail = "; ".join(str(message) for message in payload.get("diagnostics", []) or [])
        message = detail or (completed.stderr or "").strip() or (completed.stdout or "").strip()
        raise NativeCoreError((message or "native core failed")[:500])
    return payload


def _item_id(instance: ProblemInstance, item_index: Any) -> str:
    # Explicit range check: a negative index would silently wrap in Python.
    if not isinstance(item_index, int) or isinstance(item_index, bool) or not 0 <= item_index < len(instance.items):
        raise IndexError(f"item_index out of range: {item_index!r}")
    return instance.items[item_index].id


def _placements_from_payload(instance: ProblemInstance, payload: dict[str, Any]) -> list[Placement]:
    return [
        Placement(
            item_id=_item_id(instance, row["item_index"]),
            item_index=row["item_index"],
            bin_id=row["bin_id"],
            x=row["x"],
            y=row["y"],
            z=row["z"],
            rotation_id=row["rotation_id"],
            placed_length=row["placed_length"],
            placed_width=row["placed_width"],
            placed_height=row["placed_height"],
        )
        for row in payload["placements"]
    ]


def _unplaced_diagnostics(instance: ProblemInstance, payload: dict[str, Any]) -> list[str]:
    """The Python decoder's messages for items it had to skip."""
    return [
        f"item {_item_id(instance, row['item_index'])} {_UNPLACED_MESSAGES.get(row['reason'], row['reason'])}"
        for row in payload.get("unplaced", [])
    ]


def _malformed(exc: Exception) -> NativeCoreError:
    return NativeCoreError(f"malformed native payload: {type(exc).__name__}: {str(exc)[:200]}")


def decode_with_native_core(
    instance: ProblemInstance,
    candidate: CandidateSolution,
    policy_id: str = "min_z_y_x",
) -> DecodedSolution:
    """Decode one candidate in C++; drop-in for ExtremePointDecoder.decode.

    ``score`` and ``objective_scalar`` are the native values (so parity tests
    compare both engines); the placements are re-validated in Python.
    """
    binary = ensure_native_core()
    protocol = _build_decode_protocol(instance, candidate, policy_id)
    try:
        completed = _run_native_process(binary, protocol, DECODE_TIMEOUT_SECONDS, None)
    except (OSError, subprocess.SubprocessError) as exc:
        raise NativeCoreError(f"native core could not be executed: {exc}") from exc
    payload = _parse_payload(completed)
    try:
        placements = _placements_from_payload(instance, payload)
        diagnostics = _unplaced_diagnostics(instance, payload) + validate_solution(instance, placements)
        score = (float(payload["best_found_bins"]), float(payload["fragmentation"]), -float(payload["compactness_mean"]))
        objective_scalar = float(payload["objective_scalar"])
    except _MALFORMED_PAYLOAD_ERRORS as exc:
        raise _malformed(exc) from exc
    return DecodedSolution(
        candidate=candidate,
        placements=tuple(placements),
        bin_metrics=tuple(compute_bin_metrics(instance, placements)),
        score=score,
        objective_scalar=objective_scalar,
        feasible=not diagnostics,
        diagnostics=tuple(diagnostics),
    )


def solve_with_native_core(
    instance: ProblemInstance,
    config: SolverConfig,
    seed: int,
    control: SolveControl | None = None,
) -> SolverResult:
    # Same initial pool as the Python solver (same seed, same random starts).
    return _solve_native(instance, config, seed, control, build_initial_pool(instance, seed, config.multi_start_count))


def _solve_native(
    instance: ProblemInstance,
    config: SolverConfig,
    seed: int,
    control: SolveControl | None,
    initial_pool: list[CandidateSolution],
) -> SolverResult:
    algorithm = getattr(config, "algorithm", "simulated_annealing")
    if algorithm != "simulated_annealing":
        raise NativeCoreError(f"native core implements simulated_annealing only, not {algorithm!r}")
    lower_bound = simple_lower_bound(instance)
    instance_diagnostics = validate_problem_instance(instance)
    if instance_diagnostics:
        # Same early exit as SimulatedAnnealingSolver.solve.
        return SolverResult(
            instance_id=instance.instance_id,
            solver="simulated_annealing_cpp_core",
            solver_version=__version__,
            configuration_id=f"{config.configuration_id}:native",
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
            diagnostics=instance_diagnostics,
            features=extract_instance_features(instance),
        )

    binary = ensure_native_core()
    protocol = _build_solve_protocol(instance, config, seed, lower_bound, initial_pool)
    try:
        completed = _run_native_process(binary, protocol, config.time_limit_seconds + 5, control)
    except (OSError, subprocess.SubprocessError) as exc:
        # e.g. binary built for another platform ("Exec format error"), missing
        # interpreter or timeout: degrade gracefully to the Python core.
        raise NativeCoreError(f"native core could not be executed: {exc}") from exc
    payload = _parse_payload(completed)
    try:
        return _result_from_payload(instance, config, seed, lower_bound, payload)
    except _MALFORMED_PAYLOAD_ERRORS as exc:
        # Corrupted output must reach the runner's Python fallback.
        raise _malformed(exc) from exc


def _result_from_payload(
    instance: ProblemInstance,
    config: SolverConfig,
    seed: int,
    lower_bound: int,
    payload: dict[str, Any],
) -> SolverResult:
    evaluations = int(payload["evaluations"])
    if not 0 <= evaluations <= config.max_evaluations:
        raise NativeCoreError(
            f"native core reported {evaluations} evaluations for a budget of {config.max_evaluations}"
        )
    runtime_seconds = float(payload["runtime_seconds"])
    placements = _placements_from_payload(instance, payload)
    # Re-validate in Python (rotations, weights, support, overlap, distance):
    # any violation turns the result into status "error" with diagnostics.
    diagnostics = [
        *dict.fromkeys(_unplaced_diagnostics(instance, payload) + validate_solution(instance, placements))
    ]
    metrics, bin_metrics, _score = score_solution(instance, placements)
    convergence = [
        ConvergencePoint(
            evaluation=int(row["evaluation"]),
            elapsed_seconds=float(row["elapsed_seconds"]),
            bins_used=int(row["bins_used"]),
            objective_scalar=float(row["objective_scalar"]),
            accepted=bool(row["accepted"]),
            temperature=float(row["temperature"]),
            current_bins=int(row["current_bins"]),
            current_objective_scalar=float(row["current_objective_scalar"]),
            best_bins=int(row["best_bins"]),
            best_objective_scalar=float(row["best_objective_scalar"]),
            delta=float(row["delta"]),
            improved=bool(row["improved"]),
        )
        for row in payload.get("convergence", [])
    ]
    return SolverResult(
        instance_id=instance.instance_id,
        solver="simulated_annealing_cpp_core",
        solver_version=__version__,
        configuration_id=f"{config.configuration_id}:native",
        seed=seed,
        status="feasible" if not diagnostics else "error",
        best_found_bins=metrics.bins_used,
        lower_bound=lower_bound,
        best_known_bins=instance.best_known_bins,
        optimal_proven=metrics.bins_used == lower_bound,
        gap_to_best_known=metrics.gap_to_best_known,
        gap_to_lower_bound=metrics.gap_to_lower_bound,
        runtime_seconds=runtime_seconds,
        evaluations=evaluations,
        objective_scalar=metrics.objective_scalar,
        placements=placements,
        bin_metrics=bin_metrics,
        convergence=convergence,
        features=extract_instance_features(instance),
        diagnostics=diagnostics,
    )


def solve_with_native_core_parallel(
    instance: ProblemInstance,
    config: SolverConfig,
    seed: int,
    control: SolveControl | None = None,
) -> SolverResult:
    """One native subprocess per initial candidate, like the Python process_pool."""
    ensure_native_core()
    started = time.perf_counter()
    initial_pool = build_initial_pool(instance, seed, config.multi_start_count)
    starts = len(initial_pool)
    worker_count = min(config.parallel_workers, starts, os.cpu_count() or 1)
    per_start_evaluations = max(1, config.max_evaluations // starts)
    worker_config = config.model_copy(
        update={
            "max_evaluations": per_start_evaluations,
            "parallel_mode": "sequential",
            "parallel_workers": 1,
        }
    )

    results: list[SolverResult] = []
    diagnostics: list[str] = [
        f"parallel native subprocess pool used: workers={worker_count}, starts={starts}, "
        f"evaluations_per_start={per_start_evaluations}"
    ]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _solve_native,
                instance,
                worker_config,
                seed + start_index * START_SEED_STRIDE,
                control,
                [candidate],
            )
            for start_index, candidate in enumerate(initial_pool)
        ]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except SolveCancelled:
                # Propagate cancellation: the runner must NOT fall back to the
                # Python core and silently re-run a cancelled job.
                for pending in futures:
                    pending.cancel()
                raise
            except Exception as exc:  # pragma: no cover - defensive diagnostics for native worker crashes.
                diagnostics.append(f"native subprocess worker failed: {exc}")

    if control is not None and control.is_cancelled:
        raise SolveCancelled("native parallel solve cancelled")
    if not results:
        raise NativeCoreError("; ".join(diagnostics))

    best = min(results, key=lambda result: (result.best_found_bins, result.objective_scalar, result.runtime_seconds))
    runtime = time.perf_counter() - started
    return best.model_copy(
        update={
            "solver": "simulated_annealing_cpp_core_parallel",
            "configuration_id": f"{config.configuration_id}:native:parallel",
            "runtime_seconds": runtime,
            "evaluations": sum(result.evaluations for result in results),
            "diagnostics": [*dict.fromkeys(diagnostics + best.diagnostics)],
        }
    )


# ---------------------------------------------------------------------------
# Protocol v2 (documented at the top of cpp/sa_core.cpp)


def _num(value: float) -> str:
    """Shortest round-trip repr: strtod in the core parses back the same double."""
    return repr(float(value))


def _optional_num(value: float | None) -> str:
    return "none" if value is None else _num(value)


def _allowed_rotation_spec(item: ItemSpec) -> list[int]:
    """Declared rotation ids; the core applies unique_rotation_ids and the fit test."""
    if item.allowed_rotations == "fixed":
        return [0]
    if item.allowed_rotations == "all_orthogonal":
        return list(range(6))
    return [int(rotation_id) for rotation_id in item.allowed_rotations]


def _instance_rows(instance: ProblemInstance, mode: str, policy_id: str) -> list[str]:
    bin_spec = instance.bin
    constraints = instance.constraints
    index_by_id = {item.id: index for index, item in enumerate(instance.items)}
    rows = [
        PROTOCOL_VERSION,
        f"mode {mode}",
        f"sum_mode {_SUM_MODE}",
        # Unknown ids rank like min_z_y_x in the Python decoder too.
        f"policy {policy_id if policy_id in _KNOWN_POLICIES else 'min_z_y_x'}",
        f"bin {_num(bin_spec.length)} {_num(bin_spec.width)} {_num(bin_spec.height)} {_optional_num(bin_spec.max_weight)}",
        f"support {_optional_num(constraints.support_ratio_min)} {int(bool(constraints.stability_required))}",
        f"items {len(instance.items)}",
    ]
    for item in instance.items:
        rotations = _allowed_rotation_spec(item)
        rows.append(
            " ".join(
                [
                    _num(item.length),
                    _num(item.width),
                    _num(item.height),
                    _optional_num(item.weight),
                    str(len(rotations)),
                    *map(str, rotations),
                ]
            )
        )
    rows.append(f"min_distance {len(constraints.min_distance)}")
    for rule in constraints.min_distance:
        rows.append(f"{index_by_id[rule.item_id_a]} {index_by_id[rule.item_id_b]} {_num(rule.distance)}")
    return rows


def _build_decode_protocol(instance: ProblemInstance, candidate: CandidateSolution, policy_id: str) -> str:
    rows = _instance_rows(instance, "decode", policy_id)
    order = [int(index) for index in candidate.item_order]
    rotations = [int(rotation) for rotation in candidate.rotations]
    rows.append(" ".join(["candidate", str(len(order)), *map(str, order), str(len(rotations)), *map(str, rotations)]))
    rows.append("end")
    return "\n".join(rows) + "\n"


def _build_solve_protocol(
    instance: ProblemInstance,
    config: SolverConfig,
    seed: int,
    lower_bound: int,
    initial_pool: list[CandidateSolution],
) -> str:
    rows = _instance_rows(instance, "solve", config.placement_policy_id)
    operators: list[str] = []
    for name, probability in config.operator_probabilities.items():
        # propose_neighbor treats unknown operator names as swap.
        operators += [name if name in _KNOWN_OPERATORS else "swap", _num(probability)]
    rows += [
        f"lower_bound {lower_bound}",
        f"seed {seed % 2**64}",
        f"max_evaluations {config.max_evaluations}",
        f"time_limit_seconds {_num(config.time_limit_seconds)}",
        f"iterations_per_temperature {config.iterations_per_temperature}",
        f"stagnation_limit {config.stagnation_limit}",
        f"ruin_size {config.ruin_size}",
        f"initial_temperature {_optional_num(config.initial_temperature)}",
        f"target_initial_acceptance {_num(config.target_initial_acceptance)}",
        f"temperature_calibration_samples {config.temperature_calibration_samples}",
        f"cooling_rate {_num(config.cooling_rate)}",
        f"min_temperature {_num(config.min_temperature)}",
        f"min_temperature_ratio {_num(config.min_temperature_ratio)}",
        f"reheating_factor {_num(config.reheating_factor)}",
        f"multi_start_count {config.multi_start_count}",
        f"convergence_sample_rate {config.convergence_sample_rate}",
        " ".join(["operators", str(len(config.operator_probabilities)), *operators]),
    ]
    if initial_pool:
        rows.append(f"initial_pool {len(initial_pool)}")
        for candidate in initial_pool:
            rows.append(" ".join(map(str, candidate.item_order)))
            rows.append(" ".join(map(str, candidate.rotations)))
    rows.append("end")
    return "\n".join(rows) + "\n"
