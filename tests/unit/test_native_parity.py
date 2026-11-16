"""Parity between the native core (cpp/sa_core.cpp) and the Python reference.

The extreme-point decoder is deterministic, so the C++ decode of a candidate
must reproduce the Python decode exactly: same bins, same placements (bin id,
rotation id, coordinates) and the same energy. The SA uses another RNG stream,
so for it we check feasibility, bounds and budget accounting instead.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from adaptive_3dbp.benchmarks.mpv import REPORTED_CLASSES, mpv_instance
from adaptive_3dbp.constructors.pool import build_initial_pool, randomized_candidate, volume_desc_candidate
from adaptive_3dbp.decoders.extreme_points import ExtremePointDecoder
from adaptive_3dbp.domain.bounds import simple_lower_bound
from adaptive_3dbp.domain.models import (
    BinSpec,
    CandidateSolution,
    DecodedSolution,
    ItemSpec,
    MinDistanceConstraint,
    PackingConstraints,
    ProblemInstance,
    SearchBudget,
    SolverConfig,
)
from adaptive_3dbp.domain.validation import validate_solution
from adaptive_3dbp.experiments.control import SolveControl
from adaptive_3dbp.experiments.runner import solve_instance
from adaptive_3dbp.metaheuristics.simulated_annealing.solver import SimulatedAnnealingSolver
from adaptive_3dbp.native import cpp_bridge


# CI that must exercise the native core sets ADAPTIVE_3DBP_REQUIRE_NATIVE=1:
# a missing compiler then fails the suite instead of skipping it.
REQUIRE_NATIVE = os.environ.get("ADAPTIVE_3DBP_REQUIRE_NATIVE") == "1"
HAS_COMPILER = shutil.which("g++") is not None
pytestmark = pytest.mark.skipif(
    not HAS_COMPILER and not REQUIRE_NATIVE, reason="g++ is required to build the native core"
)

ROOT = Path(__file__).resolve().parents[2]
# Curated, versioned fixtures keep parity coverage in a fresh public checkout.
CAMPAIGN_DIR = ROOT / "tests/fixtures/native_parity"
CAMPAIGN_FILES = (
    "small-low-cubic-homogeneous-nodist-test-0-slot5.json",
    "small-low-mixed-heterogeneous-nodist-validation-0-slot8.json",
    "small-high-mixed-heterogeneous-nodist-train-0-slot3.json",
    "small-low-elongated-heterogeneous-nodist-test-0-slot2.json",
    "small-high-elongated-homogeneous-nodist-train-0-slot0.json",
    "medium-high-mixed-homogeneous-nodist-train-0-slot6.json",
    "medium-high-elongated-homogeneous-nodist-train-1-slot7.json",
)


# ---------------------------------------------------------------------------
# Instances


def _load(path: Path) -> ProblemInstance:
    return ProblemInstance.model_validate(json.loads(path.read_text()))


def _dims(rng: random.Random, low: float, high: float) -> tuple[float, float, float]:
    return tuple(round(rng.uniform(low, high), 2) for _ in range(3))  # type: ignore[return-value]


def _items(rng: random.Random, count: int, low: float, high: float, **fields) -> list[ItemSpec]:
    items = []
    for index in range(count):
        length, width, height = _dims(rng, low, high)
        extra = {key: value(index) if callable(value) else value for key, value in fields.items()}
        items.append(ItemSpec(id=f"it-{index:02d}", length=length, width=width, height=height, **extra))
    return items


def _synthetic_instances() -> dict[str, ProblemInstance]:
    cube = BinSpec(length=100, width=100, height=100)
    instances: dict[str, ProblemInstance] = {}

    rng = random.Random(101)
    instances["fixed-rotations"] = ProblemInstance(
        instance_id="syn-fixed",
        bin=cube,
        items=_items(rng, 18, 12, 60, allowed_rotations="fixed"),
    )

    rng = random.Random(102)
    rotation_lists = ([0, 1], [2, 4, 5], [5, 3], [1, 1, 0], [3], "all_orthogonal", "fixed")
    items = _items(rng, 16, 10, 55, allowed_rotations=lambda i: rotation_lists[i % len(rotation_lists)])
    # Equal sides: several rotation ids give the same box (unique_rotation_ids).
    items += [
        ItemSpec(id="sq-a", length=30, width=30, height=45, allowed_rotations=[4, 0, 2, 1]),
        ItemSpec(id="sq-b", length=40, width=20, height=20, allowed_rotations="all_orthogonal"),
    ]
    instances["rotation-lists"] = ProblemInstance(instance_id="syn-rotlists", bin=cube, items=items)

    rng = random.Random(103)
    instances["weight-limit"] = ProblemInstance(
        instance_id="syn-weight",
        bin=BinSpec(length=100, width=100, height=100, max_weight=95),
        items=_items(rng, 20, 15, 45, weight=lambda i: round(10 + (i * 7.3) % 25, 1)),
    )

    rng = random.Random(104)
    items = _items(rng, 16, 15, 40)
    rules = [
        MinDistanceConstraint(item_id_a=items[a].id, item_id_b=items[b].id, distance=distance)
        for a, b, distance in [(0, 1, 5.0), (2, 3, 12.5), (4, 9, 3.25), (5, 6, 20.0), (7, 8, 1.0), (10, 11, 8.0),
                               (12, 13, 15.0), (1, 4, 2.0), (3, 14, 6.0), (15, 0, 9.5),
                               (1, 0, 30.0)]  # duplicate pair: the first rule wins
    ]
    instances["min-distance"] = ProblemInstance(
        instance_id="syn-distance",
        bin=cube,
        items=items,
        constraints=PackingConstraints(min_distance=rules),
    )

    rng = random.Random(105)
    instances["support-ratio"] = ProblemInstance(
        instance_id="syn-support",
        bin=cube,
        items=_items(rng, 20, 15, 50),
        constraints=PackingConstraints(support_ratio_min=0.6),
    )

    rng = random.Random(106)
    instances["stability"] = ProblemInstance(
        instance_id="syn-stability",
        bin=cube,
        items=_items(rng, 18, 15, 45),
        constraints=PackingConstraints(stability_required=True),
    )

    rng = random.Random(107)
    items = _items(
        rng,
        22,
        10,
        50,
        weight=lambda i: 5.0 + (i % 6) * 4.5,
        allowed_rotations=lambda i: ("all_orthogonal", "fixed", [0, 2, 4])[i % 3],
    )
    rules = [
        MinDistanceConstraint(item_id_a=items[a].id, item_id_b=items[b].id, distance=distance)
        for a, b, distance in [(0, 5, 4.0), (3, 8, 10.0), (11, 17, 2.5), (19, 21, 6.0)]
    ]
    instances["all-constraints"] = ProblemInstance(
        instance_id="syn-mixed",
        bin=BinSpec(length=120, width=80, height=60, max_weight=110),
        items=items,
        constraints=PackingConstraints(support_ratio_min=0.4, min_distance=rules),
    )

    # Identical cubes: every choice is a tie, so this pins the tie-breaking;
    # the weight limit (14 cubes per bin) spreads them over several bins.
    instances["identical-cubes"] = ProblemInstance(
        instance_id="syn-cubes",
        bin=BinSpec(length=100, width=100, height=100, max_weight=50),
        items=[ItemSpec(id=f"cube-{i:02d}", length=25, width=25, height=25, weight=3.5) for i in range(30)],
    )

    # Decimal sides whose sums land within rounding error of the bin walls.
    sides = (33.33, 33.34, 16.67, 50.0, 25.1, 24.9, 66.66)
    rng = random.Random(109)
    instances["near-epsilon"] = ProblemInstance(
        instance_id="syn-epsilon",
        bin=cube,
        items=[
            ItemSpec(id=f"eps-{i:02d}", length=rng.choice(sides), width=rng.choice(sides), height=rng.choice(sides))
            for i in range(24)
        ],
        constraints=PackingConstraints(support_ratio_min=0.5),
    )
    return instances


SYNTHETIC = _synthetic_instances()
INSTANCE_IDS = ["example_20", *[name.removesuffix(".json") for name in CAMPAIGN_FILES], *SYNTHETIC]


def _instance(instance_id: str) -> ProblemInstance:
    if instance_id == "example_20":
        return _load(ROOT / "data/raw/example_20.json")
    if instance_id in SYNTHETIC:
        return SYNTHETIC[instance_id]
    path = CAMPAIGN_DIR / f"{instance_id}.json"
    return _load(path)


def _candidates(instance: ProblemInstance) -> list[CandidateSolution]:
    """Constructive orders plus random orders/rotations with fixed seeds."""
    n = len(instance.items)
    candidates = list(build_initial_pool(instance, seed=11, pool_size=3))
    for seed in (1, 2, 3):
        rng = random.Random(1000 + seed)
        order = list(range(n))
        rng.shuffle(order)
        # Any id 0..5, so preferred rotations that are not allowed/fitting occur.
        candidates.append(CandidateSolution(tuple(order), tuple(rng.randrange(6) for _ in range(n))))
    candidates.append(randomized_candidate(instance, random.Random(2024)))
    # Short rotation vector: the missing entries decode with preferred rotation 0.
    reversed_order = tuple(reversed(range(n)))
    candidates.append(CandidateSolution(reversed_order, tuple(5 for _ in range(n // 2))))
    return candidates


# ---------------------------------------------------------------------------
# Helpers


@pytest.fixture(scope="module", autouse=True)
def native_binary() -> Path:
    if not HAS_COMPILER:
        pytest.fail("ADAPTIVE_3DBP_REQUIRE_NATIVE=1 but g++ is not available")
    return cpp_bridge.ensure_native_core()


def _placement_row(placement) -> tuple:
    return (
        placement.item_index,
        placement.bin_id,
        placement.rotation_id,
        placement.x,
        placement.y,
        placement.z,
        placement.placed_length,
        placement.placed_width,
        placement.placed_height,
    )


def _assert_same_decoding(python: DecodedSolution, native: DecodedSolution) -> None:
    """Exact equality: the native decoder reproduces the Python floats bit for bit."""
    assert native.score[0] == python.score[0], "bin count differs"
    assert [_placement_row(p) for p in native.placements] == [_placement_row(p) for p in python.placements]
    assert native.objective_scalar == python.objective_scalar
    assert native.score == python.score
    assert native.diagnostics == python.diagnostics
    assert native.feasible == python.feasible


def _run_raw(protocol: str) -> dict:
    completed = subprocess.run(
        [str(cpp_bridge.ensure_native_core())], input=protocol, capture_output=True, text=True, check=False, timeout=60
    )
    return json.loads(completed.stdout)


# ---------------------------------------------------------------------------
# Decoder parity


@pytest.mark.parametrize("instance_id", INSTANCE_IDS)
def test_native_decoder_matches_python(instance_id: str) -> None:
    instance = _instance(instance_id)
    decoder = ExtremePointDecoder()
    candidates = _candidates(instance)
    cases = [(candidate, "min_z_y_x") for candidate in candidates]
    cases += [(candidates[0], "contact_then_z"), (candidates[3], "contact_then_z")]
    cases += [(candidates[0], "min_waste"), (candidates[4], "min_waste")]
    for candidate, policy in cases:
        python = decoder.decode(instance, candidate, policy)
        native = cpp_bridge.decode_with_native_core(instance, candidate, policy)
        _assert_same_decoding(python, native)
        assert python.feasible, python.diagnostics


def test_native_decoder_matches_python_on_unplaceable_and_overweight_items() -> None:
    """Skipped items and the empty bin left by an overweight item match too."""
    instance = ProblemInstance(
        instance_id="edge",
        bin=BinSpec(length=50, width=50, height=50, max_weight=20),
        items=[
            ItemSpec(id="heavy", length=10, width=10, height=10, weight=25),
            ItemSpec(id="too-long", length=60, width=10, height=10, allowed_rotations="fixed"),
            ItemSpec(id="ok-a", length=20, width=20, height=20, weight=5),
            ItemSpec(id="ok-b", length=30, width=20, height=20, weight=12),
            ItemSpec(id="ok-c", length=25, width=25, height=25, weight=4),
        ],
    )
    decoder = ExtremePointDecoder()
    for candidate in _candidates(instance):
        python = decoder.decode(instance, candidate)
        native = cpp_bridge.decode_with_native_core(instance, candidate)
        _assert_same_decoding(python, native)
        assert not native.feasible
        assert any("too-long has no fitting rotation" in message for message in native.diagnostics)


MPV_CASES = [(instance_class, rotations) for instance_class in REPORTED_CLASSES for rotations in ("fixed", "all_orthogonal")]


@pytest.mark.parametrize(("instance_class", "rotations"), MPV_CASES, ids=[f"class{c}-{r}" for c, r in MPV_CASES])
def test_native_decoder_matches_python_on_mpv_benchmark(instance_class: int, rotations: str) -> None:
    """Martello-Pisinger-Vigo instances, n=200 (v=1): larger and integer-sized."""
    instance = mpv_instance(instance_class, 200, 1, rotations=rotations)
    rng = random.Random(instance_class * 10 + (rotations == "fixed"))
    candidates = [volume_desc_candidate(instance)]
    for _ in range(2 if rotations == "fixed" else 1):
        order = list(range(len(instance.items)))
        rng.shuffle(order)
        candidates.append(CandidateSolution(tuple(order), tuple(rng.randrange(6) for _ in order)))
    decoder = ExtremePointDecoder()
    for candidate in candidates:
        _assert_same_decoding(decoder.decode(instance, candidate), cpp_bridge.decode_with_native_core(instance, candidate))


# ---------------------------------------------------------------------------
# Native SA


@pytest.mark.parametrize("instance_id", INSTANCE_IDS)
def test_native_sa_is_feasible_and_within_budget(instance_id: str) -> None:
    instance = _instance(instance_id)
    config = SolverConfig(
        max_evaluations=240,
        iterations_per_temperature=30,
        stagnation_limit=120,
        time_limit_seconds=30,
        use_native_core=True,
    )
    result = cpp_bridge.solve_with_native_core(instance, config, seed=5)
    lower_bound = simple_lower_bound(instance)
    assert result.solver == "simulated_annealing_cpp_core"
    assert result.status == "feasible", result.diagnostics
    assert result.diagnostics == []
    assert validate_solution(instance, result.placements) == []
    assert len(result.placements) == len(instance.items)
    assert result.lower_bound == lower_bound
    assert result.best_found_bins >= lower_bound
    assert 0 < result.evaluations <= config.max_evaluations
    assert result.convergence and result.convergence[-1].evaluation <= config.max_evaluations


def test_native_sa_lower_bound_stop_matches_python_accounting() -> None:
    """A start already at the lower bound costs exactly its initial decode.

    Independent of the RNG stream: no calibration walk, no main-loop step,
    and the remaining starts are skipped, in both engines.
    """
    instance = ProblemInstance(
        instance_id="trivial",
        bin=BinSpec(length=10, width=10, height=10),
        items=[ItemSpec(id=f"t{i}", length=2, width=3, height=4) for i in range(4)],
    )
    config = SolverConfig(max_evaluations=500, temperature_calibration_samples=24, time_limit_seconds=10)
    python = SimulatedAnnealingSolver().solve(
        instance, config, SearchBudget(max_evaluations=500, timeout_seconds=10), seed=3
    )
    native = cpp_bridge.solve_with_native_core(instance, config, seed=3)
    assert python.best_found_bins == native.best_found_bins == 1
    assert native.evaluations == python.evaluations == 1
    pool = build_initial_pool(instance, seed=3, pool_size=config.multi_start_count)
    payload = _run_raw(cpp_bridge._build_solve_protocol(instance, config, 3, simple_lower_bound(instance), pool))
    assert [(start["calibration_evaluations"], start["stop_reason"]) for start in payload["starts"]] == [
        (0, "lower_bound")
    ]


def test_native_sa_stop_temperature_is_relative_to_t0() -> None:
    """T0 below the old absolute floor (1e-3) still anneals the whole budget.

    The stop temperature is max(min_temperature, min_temperature_ratio * T0),
    so neither engine stops right after calibration. The weights force >= 5
    bins while the geometric bound is 1, so no lower-bound stop either.
    """
    instance = _instance("weight-limit")
    config = SolverConfig(
        max_evaluations=200,
        initial_temperature=5e-4,
        stagnation_limit=10**6,
        multi_start_count=1,
        time_limit_seconds=60,
    )
    assert config.initial_temperature < 1e-3
    python = SimulatedAnnealingSolver().solve(
        instance, config, SearchBudget(max_evaluations=200, timeout_seconds=60), seed=4
    )
    native = cpp_bridge.solve_with_native_core(instance, config, seed=4)
    assert native.status == python.status == "feasible"
    assert native.evaluations == python.evaluations == config.max_evaluations

    pool = build_initial_pool(instance, seed=4, pool_size=1)
    payload = _run_raw(cpp_bridge._build_solve_protocol(instance, config, 4, simple_lower_bound(instance), pool))
    start = payload["starts"][0]
    assert start["stop_temperature"] == pytest.approx(config.min_temperature_ratio * 5e-4)
    assert start["stop_reason"] == "budget"


def test_native_sa_honours_initial_temperature_and_calibration_budget() -> None:
    # Not at the lower bound (weights force >= 5 bins, bound is 1), so starts calibrate.
    instance = _instance("weight-limit")
    pool = build_initial_pool(instance, seed=1, pool_size=1)
    fixed = SolverConfig(max_evaluations=60, initial_temperature=0.05, multi_start_count=1, time_limit_seconds=10)
    payload = _run_raw(cpp_bridge._build_solve_protocol(instance, fixed, 1, simple_lower_bound(instance), pool[:1]))
    assert payload["status"] == "feasible"
    assert payload["starts"][0]["initial_temperature"] == 0.05
    assert payload["starts"][0]["calibration_evaluations"] == 0

    calibrated = fixed.model_copy(update={"initial_temperature": None, "temperature_calibration_samples": 50})
    roomy = calibrated.model_copy(update={"max_evaluations": 1000, "temperature_calibration_samples": 24})
    payload = _run_raw(cpp_bridge._build_solve_protocol(instance, roomy, 1, simple_lower_bound(instance), pool[:1]))
    start = payload["starts"][0]
    # The walk stops once 24 usable deltas are in, or after 4 x 24 steps.
    assert start["uphill_samples"] <= 24 and start["calibration_evaluations"] <= 24 * 4
    assert start["uphill_samples"] == 24 or start["calibration_evaluations"] == 24 * 4

    tight = calibrated.model_copy(update={"max_evaluations": 10})
    payload = _run_raw(cpp_bridge._build_solve_protocol(instance, tight, 1, simple_lower_bound(instance), pool[:1]))
    # Calibration samples are evaluations and are capped by the budget.
    assert payload["starts"][0]["calibration_evaluations"] == 9
    assert payload["evaluations"] == 10


def test_native_parallel_pool_is_feasible_and_within_budget() -> None:
    instance = _instance("weight-limit")
    config = SolverConfig(
        max_evaluations=300,
        iterations_per_temperature=25,
        time_limit_seconds=30,
        use_native_core=True,
        parallel_mode="process_pool",
        parallel_workers=2,
        multi_start_count=4,
    )
    result = cpp_bridge.solve_with_native_core_parallel(instance, config, seed=9)
    assert result.status == "feasible", result.diagnostics
    assert validate_solution(instance, result.placements) == []
    assert result.evaluations <= config.max_evaluations


def test_native_solve_reports_invalid_instance_like_python() -> None:
    instance = ProblemInstance(
        instance_id="invalid",
        bin=BinSpec(length=10, width=10, height=10),
        items=[ItemSpec(id="big", length=12, width=2, height=2, allowed_rotations="fixed")],
    )
    result = cpp_bridge.solve_with_native_core(instance, SolverConfig(use_native_core=True), seed=0)
    assert result.status == "infeasible"
    assert result.diagnostics and "does not fit" in result.diagnostics[0]


def test_native_solve_with_control_drains_large_output() -> None:
    """With a SolveControl the payload (~1 MB here) is read while the core runs.

    Reading stdout only after exit deadlocked once the pipe buffer filled,
    and the run fell back to Python after the timeout.
    """
    instance = _instance("weight-limit")
    config = SolverConfig(
        max_evaluations=3000,
        stagnation_limit=10**6,
        multi_start_count=1,
        convergence_sample_rate=1,
        time_limit_seconds=30,
        use_native_core=True,
    )
    started = time.perf_counter()
    result = cpp_bridge.solve_with_native_core(instance, config, seed=2, control=SolveControl())
    assert time.perf_counter() - started < config.time_limit_seconds
    assert result.solver == "simulated_annealing_cpp_core"
    assert result.status == "feasible", result.diagnostics
    assert result.evaluations == config.max_evaluations
    # One point per main-loop step (~300 bytes each): far above a 64 KiB pipe buffer.
    assert len(result.convergence) > 2000

    # Paused before the start and resumed later: SIGSTOP/SIGCONT, same result path.
    control = SolveControl()
    control.pause()
    resumer = threading.Timer(0.3, control.resume)
    resumer.start()
    try:
        paused_result = cpp_bridge.solve_with_native_core(instance, config, seed=2, control=control)
    finally:
        resumer.join()
    assert paused_result.solver == "simulated_annealing_cpp_core"
    assert paused_result.evaluations == config.max_evaluations


# ---------------------------------------------------------------------------
# Corrupted payloads must become NativeCoreError (so the runner falls back)

SOLVE_CONFIG_FOR_PAYLOADS = SolverConfig(max_evaluations=40, multi_start_count=1, time_limit_seconds=10)


def _set(path: tuple, value):
    def mutate(payload: dict) -> None:
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _delete(path: tuple):
    def mutate(payload: dict) -> None:
        target = payload
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return mutate


PAYLOAD_CORRUPTIONS = {
    "missing-placements": _delete(("placements",)),
    "placements-not-a-list": _set(("placements",), 7),
    "item-index-out-of-range": _set(("placements", 0, "item_index"), 10_000),
    "negative-item-index": _set(("placements", 0, "item_index"), -1),
    "negative-coordinate": _set(("placements", 0, "x"), -3.0),
    "non-numeric-rotation": _set(("placements", 0, "rotation_id"), "left"),
    "missing-convergence-field": _delete(("convergence", 0, "temperature")),
    "non-numeric-runtime": _set(("runtime_seconds",), "fast"),
    "unplaced-bad-index": _set(("unplaced",), [{"item_index": 99, "reason": "could_not_be_placed"}]),
    "evaluations-over-budget": _set(("evaluations",), SOLVE_CONFIG_FOR_PAYLOADS.max_evaluations + 1),
}


@pytest.fixture(scope="module")
def solve_payload() -> tuple[ProblemInstance, subprocess.CompletedProcess[str]]:
    instance = _instance("fixed-rotations")
    config = SOLVE_CONFIG_FOR_PAYLOADS
    protocol = cpp_bridge._build_solve_protocol(
        instance, config, 0, simple_lower_bound(instance), build_initial_pool(instance, 0, 1)
    )
    completed = cpp_bridge._run_native_process(cpp_bridge.ensure_native_core(), protocol, 30.0, None)
    assert json.loads(completed.stdout)["status"] == "feasible"
    return instance, completed


def _patch_output(monkeypatch: pytest.MonkeyPatch, completed: subprocess.CompletedProcess[str], stdout: str) -> None:
    corrupted = subprocess.CompletedProcess(completed.args, 0, stdout, "")
    monkeypatch.setattr(cpp_bridge, "_run_native_process", lambda *args, **kwargs: corrupted)


@pytest.mark.parametrize("corruption", sorted(PAYLOAD_CORRUPTIONS))
def test_corrupted_solve_payload_raises_native_core_error(solve_payload, monkeypatch, corruption: str) -> None:
    instance, completed = solve_payload
    payload = copy.deepcopy(json.loads(completed.stdout))
    PAYLOAD_CORRUPTIONS[corruption](payload)
    _patch_output(monkeypatch, completed, json.dumps(payload))
    with pytest.raises(cpp_bridge.NativeCoreError):
        cpp_bridge.solve_with_native_core(instance, SOLVE_CONFIG_FOR_PAYLOADS, seed=0)


def test_truncated_payload_falls_back_to_python(solve_payload, monkeypatch) -> None:
    instance, completed = solve_payload
    _patch_output(monkeypatch, completed, completed.stdout[: len(completed.stdout) // 2])
    with pytest.raises(cpp_bridge.NativeCoreError):
        cpp_bridge.solve_with_native_core(instance, SOLVE_CONFIG_FOR_PAYLOADS, seed=0)
    # Through the runner the error is absorbed by the Python fallback.
    tiny = ProblemInstance(
        instance_id="tiny",
        bin=BinSpec(length=10, width=10, height=10),
        items=[ItemSpec(id=f"t{i}", length=6, width=6, height=6) for i in range(3)],
    )
    config = SOLVE_CONFIG_FOR_PAYLOADS.model_copy(update={"use_native_core": True})
    result = solve_instance(tiny, config, seed=0)
    assert "cpp" not in result.solver
    assert result.status == "feasible"


@pytest.mark.parametrize("corruption", ["missing-fragmentation", "item-index-out-of-range"])
def test_corrupted_decode_payload_raises_native_core_error(monkeypatch, corruption: str) -> None:
    instance = _instance("fixed-rotations")
    candidate = volume_desc_candidate(instance)
    protocol = cpp_bridge._build_decode_protocol(instance, candidate, "min_z_y_x")
    completed = cpp_bridge._run_native_process(cpp_bridge.ensure_native_core(), protocol, 30.0, None)
    payload = json.loads(completed.stdout)
    if corruption == "missing-fragmentation":
        del payload["fragmentation"]
    else:
        payload["placements"][0]["item_index"] = len(instance.items)
    _patch_output(monkeypatch, completed, json.dumps(payload))
    with pytest.raises(cpp_bridge.NativeCoreError, match="malformed native payload"):
        cpp_bridge.decode_with_native_core(instance, candidate)


def test_native_protocol_errors_raise() -> None:
    completed = subprocess.run(
        [str(cpp_bridge.ensure_native_core())],
        input="10 10 10 1 0\n",  # unversioned (v1) protocol
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 1
    with pytest.raises(cpp_bridge.NativeCoreError, match="protocol version"):
        cpp_bridge._parse_payload(completed)


# ---------------------------------------------------------------------------
# Rebuild check


def _point_bridge_at(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "sa_core.cpp"
    binary = tmp_path / "bin" / "sa_core"
    monkeypatch.setattr(cpp_bridge, "CPP_SOURCE", source)
    monkeypatch.setattr(cpp_bridge, "NATIVE_BINARY", binary)
    monkeypatch.setattr(cpp_bridge, "NATIVE_HASH_FILE", binary.with_name("sa_core.sha256"))
    monkeypatch.setattr(cpp_bridge, "NATIVE_LOCK_FILE", binary.with_name("sa_core.lock"))
    return source, binary


def test_rebuild_is_keyed_on_source_hash_flags_and_compiler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, binary = _point_bridge_at(monkeypatch, tmp_path)
    source.write_text("int main() { return 0; }\n")
    stamp_file = binary.with_name("sa_core.sha256")

    compiles: list[tuple[list[str], bytes]] = []
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        if "-o" not in command:  # g++ --version
            return real_run(command, **kwargs)
        compiled_source = Path(command[command.index("-o") - 1])
        compiles.append((command, compiled_source.read_bytes()))
        Path(command[command.index("-o") + 1]).write_text("binary")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cpp_bridge.subprocess, "run", fake_run)

    assert cpp_bridge.ensure_native_core() == binary
    assert len(compiles) == 1
    command, compiled_bytes = compiles[0]
    assert "-ffp-contract=off" in command
    # A private copy of exactly the hashed bytes is compiled, not the live file.
    assert Path(command[command.index("-o") - 1]) != source
    assert compiled_bytes == source.read_bytes()
    stamp = cpp_bridge._parse_stamp(stamp_file.read_text())
    assert stamp["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert stamp["flags"] == " ".join(cpp_bridge.COMPILE_FLAGS)
    assert stamp["compiler"] == cpp_bridge._compiler_identity(shutil.which("g++"))

    fingerprint = cpp_bridge.native_core_fingerprint()
    assert fingerprint["up_to_date"] is True
    assert fingerprint["source_sha256"] == stamp["source_sha256"]
    assert fingerprint["stamp"] == stamp_file.read_text()
    assert fingerprint["binary"] == str(binary)

    cpp_bridge.ensure_native_core()  # unchanged content: no rebuild, whatever the mtimes
    assert len(compiles) == 1

    stamp_file.write_text(stamp_file.read_text().replace(stamp["compiler"], "some other compiler 1.0"))
    cpp_bridge.ensure_native_core()  # built by another compiler: rebuild
    assert len(compiles) == 2

    source.write_text("int main() { return 1; }\n")
    assert cpp_bridge.native_core_fingerprint()["up_to_date"] is False
    cpp_bridge.ensure_native_core()
    assert len(compiles) == 3
    assert not list(binary.parent.glob("*.tmp"))


def test_concurrent_processes_compile_once(tmp_path: Path) -> None:
    """The flock makes racing processes wait for one build instead of each compiling."""
    source = tmp_path / "sa_core.cpp"
    source.write_text("int main() { return 0; }\n")
    binary = tmp_path / "bin" / "sa_core"
    child = textwrap.dedent(
        f"""
        import subprocess
        from pathlib import Path
        from adaptive_3dbp.native import cpp_bridge

        cpp_bridge.CPP_SOURCE = Path({str(source)!r})
        cpp_bridge.NATIVE_BINARY = Path({str(binary)!r})
        cpp_bridge.NATIVE_HASH_FILE = cpp_bridge.NATIVE_BINARY.with_name("sa_core.sha256")
        cpp_bridge.NATIVE_LOCK_FILE = cpp_bridge.NATIVE_BINARY.with_name("sa_core.lock")
        real_run = subprocess.run
        compiles = 0

        def counting_run(command, **kwargs):
            global compiles
            compiles += "-o" in command
            return real_run(command, **kwargs)

        subprocess.run = counting_run
        cpp_bridge.ensure_native_core()
        print(compiles)
        """
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    processes = [
        subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE, text=True, env=env) for _ in range(6)
    ]
    counts = [int(process.communicate(timeout=120)[0].strip()) for process in processes]
    assert sum(counts) == 1, counts
    assert subprocess.run([str(binary)], check=False).returncode == 0
