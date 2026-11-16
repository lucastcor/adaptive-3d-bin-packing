from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from adaptive_3dbp.domain.models import (
    BinSpec,
    ItemSpec,
    MinDistanceConstraint,
    PackingConstraints,
    ProblemInstance,
    SolverConfig,
)
from adaptive_3dbp.tuning.objective import TuningObjective, _solve_with_hard_timeout, config_to_flat_values, values_to_config
from adaptive_3dbp.tuning.space_conversion import flat_parameter_specs, to_irace_parameters

SEARCH_SPACE = json.loads(
    (Path(__file__).resolve().parents[2] / "configs" / "search_spaces" / "sa_search_space.json").read_text(
        encoding="utf-8"
    )
)


def _tiny_instance() -> ProblemInstance:
    return ProblemInstance(
        instance_id="tuning-test",
        bin=BinSpec(length=10, width=10, height=10),
        items=[
            ItemSpec(id="a", length=5, width=5, height=5),
            ItemSpec(id="b", length=5, width=5, height=5),
            ItemSpec(id="c", length=4, width=4, height=4),
        ],
    )


def test_flat_specs_expand_operator_simplex() -> None:
    specs = flat_parameter_specs(SEARCH_SPACE)
    names = [spec["name"] for spec in specs]
    assert "op_swap" in names and "op_ruin_recreate" in names
    assert "operator_probabilities" not in names


def test_irace_parameters_render() -> None:
    text = to_irace_parameters(SEARCH_SPACE)
    assert "cooling_rate" in text
    assert "i,log" in text  # log-scaled integer parameters
    assert "op_swap" in text


def test_values_round_trip_renormalises_operators() -> None:
    values = {
        "cooling_rate": 0.93,
        "ruin_size": 5,
        "op_swap": 0.5,
        "op_relocate": 0.5,
        "op_reverse": 0.5,
        "op_rotate": 0.25,
        "op_ruin_recreate": 0.25,
    }
    config = values_to_config(values)
    assert abs(sum(config.operator_probabilities.values()) - 1.0) < 1e-9
    flat = config_to_flat_values(config, ["cooling_rate", "op_swap"])
    assert flat["cooling_rate"] == 0.93
    assert abs(flat["op_swap"] - 0.25) < 1e-9


def test_objective_evaluates_config() -> None:
    objective = TuningObjective(
        instances=[_tiny_instance()],
        seeds=[5],
        max_evaluations=60,
        time_limit_seconds=2.0,
    )
    loss = objective.evaluate_config(SolverConfig(multi_start_count=1))
    assert loss >= 0.0
    assert len(objective.history) == 1
    assert objective.solver_runs == 1
    assert objective.runs_per_trial == 1


def test_hard_timeout_drains_large_worker_result() -> None:
    widths = [
        30.97,
        32.53,
        28.25,
        35.04,
        36.63,
        37.64,
        34.95,
        38.4,
        38.55,
        38.08,
        31.98,
        36.29,
        34.66,
        30.57,
        36.34,
        29.3,
        32.48,
    ]
    heights = [
        30.43,
        31.25,
        29.26,
        28.48,
        29.66,
        30.21,
        26.41,
        30.04,
        30.08,
        29.14,
        31.57,
        30.14,
        26.5,
        33.71,
        27.06,
        30.76,
        33.9,
    ]
    items = [
        ItemSpec(id=f"item-{index:03d}", length=92.0, width=width, height=height)
        for index, (width, height) in enumerate(zip(widths, heights, strict=True))
    ]
    # The regression is about draining a LARGE result through the worker
    # queue, so the search must spend its whole budget. Pairwise distances
    # larger than the bin force one item per bin: the lower bound is never
    # reached and the lower-bound early stop cannot shorten the run.
    instance = ProblemInstance(
        instance_id="queue-regression",
        bin=BinSpec(length=100, width=100, height=100),
        items=items,
        constraints=PackingConstraints(
            min_distance=[
                MinDistanceConstraint(item_id_a=left.id, item_id_b=right.id, distance=500.0)
                for index, left in enumerate(items)
                for right in items[index + 1 :]
            ]
        ),
    )
    config = SolverConfig(
        max_evaluations=600,
        time_limit_seconds=20.0,
        temperature_calibration_samples=6,
        multi_start_count=1,
        use_native_core=False,
        parallel_mode="sequential",
        parallel_workers=1,
    )

    result = _solve_with_hard_timeout(instance, config, seed=10023, timeout_seconds=20.0)

    assert result.status == "feasible"
    assert result.evaluations == 600
    assert len(result.convergence) > 100


def test_validation_loss_uses_fresh_seeds_and_counts_runs() -> None:
    objective = TuningObjective(
        instances=[_tiny_instance()],
        seeds=[5, 6],
        max_evaluations=60,
        time_limit_seconds=2.0,
    )
    loss = objective.validation_loss(SolverConfig(multi_start_count=1))
    assert loss >= 0.0
    assert objective.solver_runs == 2  # uma execução por seed de validação


def _instance(suffix: str, item_count: int) -> ProblemInstance:
    return ProblemInstance(
        instance_id=f"par-{suffix}",
        bin=BinSpec(length=10, width=10, height=10),
        items=[ItemSpec(id=f"{suffix}-{index}", length=4, width=4, height=4) for index in range(item_count)],
    )


def _deterministic_fields(records: list[dict]) -> list[tuple]:
    # Saídas ALGORÍTMICAS do run: determinísticas dado (instância, config, seed)
    # e independentes de máquina/contenção. É sobre isto que a ciência repousa;
    # se serial e paralelo baterem aqui, a campanha é reproduzível. (runtime
    # fica de fora: depende de máquina e contenção; a perda padrão não o usa.)
    return [
        (
            r["instance_id"],
            r["seed"],
            r["bins_used"],
            r["lower_bound"],
            r["gap_to_lower_bound"],
            r["optimal_proven"],
            r["evaluations"],
            r["status"],
            r["objective_scalar"],
        )
        for r in records
    ]


def test_parallel_runs_reproduce_serial_exactly() -> None:
    instances = [_instance("a", 3), _instance("b", 4), _instance("c", 5)]
    seeds = [11, 23]
    config = SolverConfig(multi_start_count=1)
    # time_limit folgado: o orçamento que para a busca são as avaliações, então
    # cada run é determinístico mesmo sob contenção (modo "por avaliações").
    serial = TuningObjective(instances=instances, seeds=seeds, max_evaluations=80, time_limit_seconds=20.0, max_parallel_runs=1)
    parallel = TuningObjective(instances=instances, seeds=seeds, max_evaluations=80, time_limit_seconds=20.0, max_parallel_runs=4)

    loss_serial = serial.evaluate_config(config)
    loss_parallel = parallel.evaluate_config(config)

    assert serial.solver_runs == parallel.solver_runs == len(instances) * len(seeds)
    # Saídas algorítmicas IDÊNTICAS, na mesma ordem determinística de records.
    assert _deterministic_fields(serial.records) == _deterministic_fields(parallel.records)
    # Sem o termo de relógio (padrão), a perda também é idêntica.
    assert loss_serial == loss_parallel


def test_evaluate_configs_matches_serial_loop() -> None:
    instances = [_instance("a", 3), _instance("b", 4)]
    seeds = [11, 23]
    configs = [SolverConfig(multi_start_count=1, cooling_rate=0.90), SolverConfig(multi_start_count=1, cooling_rate=0.97)]

    serial = TuningObjective(instances=instances, seeds=seeds, max_evaluations=60, time_limit_seconds=20.0, max_parallel_runs=1)
    [serial.evaluate_config(config) for config in configs]

    parallel = TuningObjective(instances=instances, seeds=seeds, max_evaluations=60, time_limit_seconds=20.0, max_parallel_runs=4)
    parallel.evaluate_configs(configs)

    assert _deterministic_fields(serial.records) == _deterministic_fields(parallel.records)


def test_replicate_losses_grouped_matches_per_instance() -> None:
    instances = [_instance("a", 3), _instance("b", 4), _instance("c", 3)]
    seeds = [11, 23]
    config = SolverConfig(multi_start_count=1)

    serial = TuningObjective(instances=instances, seeds=seeds, max_evaluations=60, time_limit_seconds=20.0, max_parallel_runs=1)
    serial_groups = [serial.replicate_losses_on(instance, config) for instance in instances]

    parallel = TuningObjective(instances=instances, seeds=seeds, max_evaluations=60, time_limit_seconds=20.0, max_parallel_runs=4)
    parallel_groups = parallel.replicate_losses_grouped(config)

    # Mesma estrutura de agrupamento (uma lista de perdas por instância, em ordem).
    assert [len(group) for group in serial_groups] == [len(group) for group in parallel_groups]
    assert _deterministic_fields(serial.records) == _deterministic_fields(parallel.records)


def test_best_values_from_campaign_parses_csv(tmp_path: Path) -> None:
    from adaptive_3dbp.tuning.irace_runner import best_values_from_campaign
    from adaptive_3dbp.tuning.objective import values_to_config

    log = tmp_path / "evaluations.csv"
    log.write_text(
        "configuration_id;instance_id;seed;cost;values_json\n"
        '1;a;5;0.50;{"cooling_rate": 0.9, "ruin_size": 3}\n'
        '1;a;9;0.30;{"cooling_rate": 0.9, "ruin_size": 3}\n'
        '2;a;5;0.20;{"cooling_rate": 0.95, "ruin_size": 5}\n',
        encoding="utf-8",
    )
    values, stats = best_values_from_campaign(tmp_path)
    # Sem saída do irace (log/irace.Rdata) não há elite: o fallback compara só
    # as configurações mais corridas. A config 2, com UMA avaliação de sorte,
    # não pode vencer a que foi medida em mais experimentos.
    assert values == {"cooling_rate": 0.9, "ruin_size": 3}
    assert stats["selection"] == "fallback_most_raced"
    assert stats["evaluations"] == 3
    assert stats["distinct_configurations"] == 2
    config = values_to_config(values)
    assert config.cooling_rate == 0.9 and config.ruin_size == 3


def test_markdown_report_marks_best_and_skipped() -> None:
    from adaptive_3dbp.tuning.compare import render_markdown_report

    results = {
        "n_trials": 5,
        "budget_solver_runs": 10,
        "tuning_seeds": [1, 2],
        "validation_seeds": [10001, 10002],
        "tuners": {
            "random_search": {
                "status": "ok",
                "validation_loss": 0.5,
                "training_best_loss": 0.4,
                "tuning_seconds": 1.0,
                "solver_runs": 10,
            },
            "smac3": {"status": "skipped", "reason": "smac não instalado"},
        },
    }
    report = render_markdown_report(results)
    assert "**(melhor)**" in report
    assert "smac não instalado" in report
    assert "Perda (validação)" in report


def test_campaign_runs_sample_convergence_but_keep_improvements() -> None:
    # The sample rate only thins what is recorded; improvements are always
    # kept by both cores. Rate 1 made the MPV pilot write 27 GB of traces.
    objective = TuningObjective(instances=[], max_evaluations=3000)
    assert objective._fixed(SolverConfig()).convergence_sample_rate == 30
    assert TuningObjective(instances=[], max_evaluations=50)._fixed(SolverConfig()).convergence_sample_rate == 1


@pytest.mark.parametrize(
    "core",
    [
        "python_sa",
        "parallel_tempering",
        pytest.param(
            "native_sa",
            marks=pytest.mark.skipif(
                shutil.which("g++") is None and os.environ.get("ADAPTIVE_3DBP_REQUIRE_NATIVE") != "1",
                reason="g++ is required to build the native core",
            ),
        ),
    ],
)
def test_convergence_sample_rate_does_not_change_the_search(core: str) -> None:
    from adaptive_3dbp.benchmarks.mpv import mpv_instance
    from adaptive_3dbp.experiments.runner import solve_instance

    # Lower bound 7, best known ~9-10 bins: the search never stops early.
    instance = mpv_instance(8, 50, 1)
    base = SolverConfig(
        max_evaluations=150 if core != "native_sa" else 3000,
        temperature_calibration_samples=6,
        algorithm="parallel_tempering" if core == "parallel_tempering" else "simulated_annealing",
        use_native_core=core == "native_sa",
    )
    dense, sparse = (
        solve_instance(instance, base.model_copy(update={"convergence_sample_rate": rate}), seed=5)
        for rate in (1, 30)
    )
    if core == "native_sa":
        assert dense.solver == sparse.solver == "simulated_annealing_cpp_core"  # no silent Python fallback
    assert (dense.best_found_bins, dense.objective_scalar, dense.evaluations) == (
        sparse.best_found_bins,
        sparse.objective_scalar,
        sparse.evaluations,
    )
    assert [p.model_dump() for p in dense.placements] == [p.model_dump() for p in sparse.placements]
    assert len(sparse.convergence) < len(dense.convergence)
    # Every improvement is still recorded at the sparse rate.
    improvements = [(p.evaluation, p.best_objective_scalar) for p in dense.convergence if p.improved]
    assert improvements
    assert improvements == [(p.evaluation, p.best_objective_scalar) for p in sparse.convergence if p.improved]
