from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from adaptive_3dbp.domain.models import BinSpec, ItemSpec, ProblemInstance
from adaptive_3dbp.tuning.irace_runner import (
    best_values_from_campaign,
    generate_irace_campaign,
    irace_available,
    parse_irace_elites,
    run_irace,
)
from adaptive_3dbp.tuning.objective import TuningObjective
from adaptive_3dbp.tuning.smac_runner import (
    derive_solver_seeds,
    make_smac_instance_target,
    make_smac_target,
    run_smac,
    smac_available,
)

SEARCH_SPACE = json.loads(
    (Path(__file__).resolve().parents[2] / "configs" / "search_spaces" / "sa_search_space.json").read_text(
        encoding="utf-8"
    )
)


def _tiny_instance(suffix: str = "a") -> ProblemInstance:
    return ProblemInstance(
        instance_id=f"runner-{suffix}",
        bin=BinSpec(length=10, width=10, height=10),
        items=[ItemSpec(id=f"{suffix}-{index}", length=5, width=5, height=5) for index in range(3)],
    )


class _FakeObjective:
    """Duck-typed TuningObjective: records the solver seeds of every call."""

    def __init__(self, seeds: list[int], instances: int = 2) -> None:
        self.seeds = list(seeds)
        self.instances = [object()] * instances
        self.calls: list[tuple[dict, list[int]]] = []
        self.solver_runs = 0

    @property
    def runs_per_trial(self) -> int:
        return len(self.instances) * len(self.seeds)

    def evaluate_values(self, values: dict) -> float:
        self.calls.append((dict(values), list(self.seeds)))
        self.solver_runs += self.runs_per_trial
        # Custo depende da seed: replicar a mesma seed daria custo idêntico.
        return (float(values.get("cooling_rate", 0.9)) - 0.95) ** 2 + (sum(self.seeds) % 97) * 1e-4


# ---------------------------------------------------------------- SMAC3 seeds


def test_derive_solver_seeds_is_deterministic_and_avoids_reserved_seeds() -> None:
    base = [11, 23, 47]
    first = derive_solver_seeds(123, base)
    assert first == derive_solver_seeds(123, base)
    assert len(first) == len(base) and len(set(first)) == len(base)
    assert set(first).isdisjoint(derive_solver_seeds(124, base))
    # Nunca cai nas seeds canônicas de treino nem nas de validação/teste.
    reserved = {seed + offset for seed in base for offset in (0, 10_000, 20_000)}
    for trial_seed in range(200):
        derived = derive_solver_seeds(trial_seed, base)
        assert reserved.isdisjoint(derived)
        assert all(1_000_000 <= seed < 2**31 - 1 for seed in derived)


def test_smac_target_uses_smac_seed_and_restores_objective_seeds() -> None:
    objective = _FakeObjective([11, 23])
    target = make_smac_target(objective)

    loss_a = target({"cooling_rate": 0.93}, seed=1)
    loss_b = target({"cooling_rate": 0.93}, seed=2)
    target({"cooling_rate": 0.93}, seed=1)

    seeds_a, seeds_b, seeds_a_again = (call[1] for call in objective.calls)
    assert seeds_a == derive_solver_seeds(1, [11, 23])
    assert seeds_a != seeds_b  # nova seed do SMAC = nova replicação de verdade
    assert seeds_a == seeds_a_again  # mesma seed do SMAC = mesmos runs (pareamento)
    assert loss_a != loss_b
    assert objective.seeds == [11, 23]
    # Orçamento: cada chamada custa exatamente runs_per_trial runs do solver.
    assert objective.solver_runs == 3 * objective.runs_per_trial


def test_smac_target_restores_seeds_when_objective_fails() -> None:
    objective = _FakeObjective([5])

    def boom(values: dict) -> float:
        raise RuntimeError("solver crashed")

    objective.evaluate_values = boom  # type: ignore[method-assign]
    target = make_smac_target(objective)
    with pytest.raises(RuntimeError):
        target({}, seed=9)
    assert objective.seeds == [5]


def test_smac_instance_target_runs_one_solver_run_on_the_requested_instance() -> None:
    instances = [_tiny_instance("a"), _tiny_instance("b")]
    objective = TuningObjective(instances=instances, seeds=[5, 6], max_evaluations=40, time_limit_seconds=5.0)
    target = make_smac_instance_target(objective)
    loss = target({"cooling_rate": 0.93, "multi_start_count": 1}, instance="runner-b", seed=77)
    assert loss >= 0.0
    assert objective.solver_runs == 1
    assert [(record["instance_id"], record["seed"]) for record in objective.records] == [
        ("runner-b", derive_solver_seeds(77, [5])[0])
    ]
    assert objective.seeds == [5, 6]


def test_smac_target_with_real_objective_runs_derived_seeds() -> None:
    objective = TuningObjective(instances=[_tiny_instance()], seeds=[5, 6], max_evaluations=40, time_limit_seconds=5.0)
    target = make_smac_target(objective)
    loss = target({"cooling_rate": 0.93, "multi_start_count": 1}, seed=77)
    assert loss >= 0.0
    assert objective.solver_runs == objective.runs_per_trial == 2
    assert [record["seed"] for record in objective.records] == derive_solver_seeds(77, [5, 6])
    assert objective.seeds == [5, 6]


@pytest.mark.skipif(not smac_available(), reason="smac não instalado")
def test_run_smac_reevaluates_configs_on_distinct_seeds(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # SMAC grava smac3_output/ no diretório corrente
    objective = _FakeObjective([11, 23])
    payload = run_smac(objective, SEARCH_SPACE, n_trials=10, seed=0)

    assert len(objective.calls) == 10
    assert objective.solver_runs == 10 * objective.runs_per_trial
    assert objective.seeds == [11, 23]
    by_config: dict[str, list[tuple[int, ...]]] = {}
    for values, seeds in objective.calls:
        by_config.setdefault(json.dumps(values, sort_keys=True), []).append(tuple(seeds))
    repeated = [runs for runs in by_config.values() if len(runs) > 1]
    assert repeated, "intensificação do SMAC deveria reavaliar ao menos uma configuração"
    for runs in repeated:
        assert len(set(runs)) == len(runs)  # cada reavaliação usa seeds de solver novas
    assert payload["incumbent_loss"] == payload["incumbent_loss"]  # não-NaN


# ---------------------------------------------------------------- irace elites

# Trecho real do stdout do irace 4.4.3 (campanha gerada por
# generate_irace_campaign com o espaço sa_search_space.json, maxExperiments=400
# e um target-runner sintético). Corte entre a última iteração e o fim da
# execução: tabela de elites da iteração, pós-seleção e o resumo final.
_ELITE_HEADER = (
    "   target_initial_acceptance cooling_rate iterations_per_temperature stagnation_limit ruin_size "
    "multi_start_count parallel_workers op_swap op_relocate op_reverse op_rotate op_ruin_recreate"
)
_ROW_48 = (
    "48                    0.8883       0.9493                         76              200         6"
    "                 5                8  0.6113      0.8968     0.8001    0.5073           0.8256"
)
CAPTURED_ITERATION_TAIL = f"""\
|.|         10|          4|         48|   0.04727367237|         14|00:00:00|+1.00|1.00|0.0000|
+-+-----------+-----------+-----------+----------------+-----------+--------+-----+----+------+
Best-so-far configuration:          48    mean value:    0.04727367237
Description of the best-so-far configuration:
   .ID. target_initial_acceptance cooling_rate iterations_per_temperature stagnation_limit ruin_size multi_start_count parallel_workers op_swap op_relocate op_reverse op_rotate op_ruin_recreate .PARENT.
48   48                    0.8883       0.9493                         76              200         6                 5                8  0.6113      0.8968     0.8001    0.5073           0.8256       20

# 2026-10-21 00:10:13 -03: Elite configurations (first number is the configuration ID; listed from best to worst according to the sum of ranks):
{_ELITE_HEADER}
{_ROW_48}
44                    0.5582       0.9527                         56             1696         7                 7                8  0.3618      0.6460     0.7604    0.4431           0.7740
20                    0.8391       0.9534                         39              259         7                 6                7  0.5716      0.8435     0.7310    0.9408           0.7669
7                     0.7545       0.9575                         61             1139         5                 4                8  0.4682      0.4977     0.7827    0.6260           0.7496
# 2026-10-21 00:10:13 -03: Stopped because there is not enough budget left to race more than the minimum (5).
# You may either increase the budget or set 'minNbSurvival' to a lower value.
# Iteration: 8
# nbIterations: 8
# experimentsUsed: 379
# number of elites: 4
"""
CAPTURED_FINAL = f"""\
# Total CPU user time: 7.182, CPU sys time: 3.133, Wall-clock time: 11.77
# 2026-10-21 00:10:13 -03: Starting post-selection:
# Configurations selected: 48, 44, 20, 7, 1, 2.
# Pending instances: 1, 1, 1, 1, 7, 7.
+-+-----------+-----------+-----------+----------------+-----------+--------+-----+----+------+
|-|          6|          1|         48|   0.04727367237|         12|00:00:00|   NA|  NA|    NA|
+-+-----------+-----------+-----------+----------------+-----------+--------+-----+----+------+
Best-so-far configuration:          48    mean value:    0.04727367237

# 2026-10-21 00:10:13 -03: Elite configurations (first number is the configuration ID; listed from best to worst according to the sum of ranks):
{_ELITE_HEADER}
{_ROW_48}
# Total CPU user time: 7.488, CPU sys time: 3.238, Wall-clock time: 12.214
# Best configurations (first number is the configuration ID; listed from best to worst according to the sum of ranks):
{_ELITE_HEADER}
{_ROW_48}
# Best configurations as commandlines (first number is the configuration ID; listed from best to worst according to the sum of ranks):
48 --target_initial_acceptance 0.8883 --cooling_rate 0.9493 --iterations_per_temperature 76 --stagnation_limit 200 --ruin_size 6 --multi_start_count 5 --parallel_workers 8 --op_swap 0.6113 --op_relocate 0.8968 --op_reverse 0.8001 --op_rotate 0.5073 --op_ruin_recreate 0.8256


# Testing of elite configurations: 1
# Testing iteration configurations: FALSE
# 2026-10-21 00:10:13 -03: No test instances, skip testing
"""
CAPTURED_STDOUT = CAPTURED_ITERATION_TAIL + CAPTURED_FINAL

_VALUES_48 = {
    "target_initial_acceptance": 0.8883,
    "cooling_rate": 0.9493,
    "iterations_per_temperature": 76,
    "stagnation_limit": 200,
    "ruin_size": 6,
    "multi_start_count": 5,
    "parallel_workers": 8,
    "op_swap": 0.6113,
    "op_relocate": 0.8968,
    "op_reverse": 0.8001,
    "op_rotate": 0.5073,
    "op_ruin_recreate": 0.8256,
}
_VALUES_54 = {**_VALUES_48, "cooling_rate": 0.9488, "ruin_size": 6, "stagnation_limit": 1609}


def _write_evaluations(campaign_dir: Path, rows: list[tuple[str, int, int, float, dict]]) -> None:
    with (campaign_dir / "evaluations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["configuration_id", "instance_id", "seed", "cost", "values_json"])
        for configuration_id, instance_id, seed, cost, values in rows:
            writer.writerow([configuration_id, instance_id, seed, repr(cost), json.dumps(values, sort_keys=True)])


def _captured_campaign_rows() -> list[tuple[str, int, int, float, dict]]:
    # Mesma situação da execução real capturada: o elite 48 sobreviveu a 11
    # experimentos; o 54 foi descartado após 5 instâncias "fáceis" e tem média
    # MENOR só por ter sido medido num subconjunto diferente.
    elite = [0.166, 0.027, -0.0245, -0.0292, 0.0712, 0.0051, -0.0070, 0.1466, 0.0682, 0.1196, -0.0229]
    lucky = [0.166, 0.027, -0.0244, -0.0291, 0.0713]
    rows = [("48", 1 + index % 4, 1000 + index, cost, _VALUES_48) for index, cost in enumerate(elite)]
    rows += [("54", 1 + index % 4, 1000 + index, cost - 0.03, _VALUES_54) for index, cost in enumerate(lucky)]
    return rows


def test_parse_irace_elites_reads_final_commandlines() -> None:
    elites = parse_irace_elites(CAPTURED_STDOUT)
    assert [elite["configuration_id"] for elite in elites] == ["48"]
    assert elites[0]["values"] == _VALUES_48


def test_parse_irace_elites_multiple_final_elites_in_order() -> None:
    # configurations_print_command: sprintf("%-*d %s\n", largura, id, comando).
    text = (
        "# Best configurations as commandlines (first number is the configuration ID; listed from best to worst "
        "according to the sum of ranks):\n"
        "124 --cooling_rate 0.9535 --ruin_size 6\n"
        "59  --cooling_rate 0.9297 --ruin_size 6\n"
        "7   --cooling_rate=0.95 --ruin_size=5\n"
    )
    elites = parse_irace_elites(text)
    assert [elite["configuration_id"] for elite in elites] == ["124", "59", "7"]
    assert elites[1]["values"] == {"cooling_rate": 0.9297, "ruin_size": 6}
    assert elites[2]["values"] == {"cooling_rate": 0.95, "ruin_size": 5}


def test_parse_irace_elites_falls_back_to_last_iteration_elites() -> None:
    # Execução interrompida antes do resumo final (timeout): vale a tabela de
    # elites da última iteração, na ordem do irace.
    elites = parse_irace_elites(CAPTURED_ITERATION_TAIL)
    assert [elite["configuration_id"] for elite in elites] == ["48", "44", "20", "7"]
    assert all(elite["values"] is None for elite in elites)


def test_parse_irace_elites_handles_r_wrapped_table() -> None:
    # R quebra data frames largos em blocos de colunas, repetindo os IDs
    # (formato observado com print.data.frame e largura 80).
    text = (
        "# Best configurations (first number is the configuration ID; listed from best to worst according to the "
        "sum of ranks):\n"
        "   target_initial_acceptance cooling_rate iterations_per_temperature\n"
        "48                    0.8883       0.9493                         76\n"
        "44                    0.5582       0.9527                         56\n"
        "   stagnation_limit ruin_size multi_start_count parallel_workers op_swap\n"
        "48              200         6                 5                8  0.6113\n"
        "44             1696         7                 7                8  0.3618\n"
    )
    assert [elite["configuration_id"] for elite in parse_irace_elites(text)] == ["48", "44"]


def test_parse_irace_elites_best_so_far_and_empty() -> None:
    assert parse_irace_elites("Best-so-far configuration:           7    mean value:     0.13\n") == [
        {"configuration_id": "7", "values": None}
    ]
    assert parse_irace_elites("Error: == irace == Insufficient budget\n") == []


def test_best_values_uses_irace_elite_not_best_subset_mean(tmp_path: Path) -> None:
    _write_evaluations(tmp_path, _captured_campaign_rows())
    (tmp_path / "irace_stdout.log").write_text(CAPTURED_STDOUT, encoding="utf-8")

    values, stats = best_values_from_campaign(tmp_path)

    assert values == _VALUES_48
    assert stats["selection"] == "irace_elite"
    assert stats["elite_source"] == "stdout"
    assert stats["configuration_id"] == "48"
    assert stats["best_replications"] == 11
    assert stats["evaluations"] == 16 and stats["distinct_configurations"] == 2


def test_best_values_prefers_rdata_export_over_stdout(tmp_path: Path) -> None:
    _write_evaluations(tmp_path, _captured_campaign_rows())
    (tmp_path / "irace_stdout.log").write_text(CAPTURED_STDOUT, encoding="utf-8")
    # irace_elites.txt = iraceResults$allElites[[último]] exportado do irace.Rdata.
    (tmp_path / "irace_elites.txt").write_text("54\n48\n", encoding="utf-8")

    values, stats = best_values_from_campaign(tmp_path)

    assert values == _VALUES_54
    assert stats["elite_source"] == "irace.Rdata"
    assert stats["final_elites"] == ["54", "48"]


def test_best_values_uses_commandline_values_when_elite_missing_from_log(tmp_path: Path) -> None:
    _write_evaluations(tmp_path, [row for row in _captured_campaign_rows() if row[0] != "48"])
    (tmp_path / "irace_stdout.log").write_text(CAPTURED_STDOUT, encoding="utf-8")

    values, stats = best_values_from_campaign(tmp_path)

    assert values == _VALUES_48
    assert stats["best_mean_cost"] is None and stats["best_replications"] == 0


def test_best_values_fallback_compares_most_raced_on_shared_experiments(tmp_path: Path) -> None:
    rows = [
        # 1 e 2 correram nos mesmos 3 experimentos; 3 só em 1 (e com sorte).
        ("1", 1, 11, 0.40, {"cooling_rate": 0.90}),
        ("1", 2, 12, 0.50, {"cooling_rate": 0.90}),
        ("1", 3, 13, 0.30, {"cooling_rate": 0.90}),
        ("2", 1, 11, 0.35, {"cooling_rate": 0.95}),
        ("2", 2, 12, 0.45, {"cooling_rate": 0.95}),
        ("2", 3, 13, 0.32, {"cooling_rate": 0.95}),
        ("3", 1, 11, 0.01, {"cooling_rate": 0.99}),
    ]
    _write_evaluations(tmp_path, rows)

    values, stats = best_values_from_campaign(tmp_path)

    assert values == {"cooling_rate": 0.95}
    assert stats["selection"] == "fallback_most_raced"
    assert stats["shared_experiments"] == 3


def test_best_values_without_evaluations(tmp_path: Path) -> None:
    assert best_values_from_campaign(tmp_path) == (None, {"evaluations": 0})


def test_generate_irace_campaign_clears_stale_outputs_and_pins_interpreter(tmp_path: Path) -> None:
    import sys

    for stale in ("evaluations.csv", "irace_stdout.log", "irace_stderr.log", "irace.Rdata", "irace_elites.txt"):
        (tmp_path / stale).write_text("old run", encoding="utf-8")
    generate_irace_campaign(tmp_path, SEARCH_SPACE, [_tiny_instance()], src_path=Path("src"))
    for stale in ("evaluations.csv", "irace_stdout.log", "irace_stderr.log", "irace.Rdata", "irace_elites.txt"):
        assert not (tmp_path / stale).exists()
    shebang = (tmp_path / "target-runner.py").read_text(encoding="utf-8").splitlines()[0]
    if not any(ch.isspace() for ch in sys.executable):
        assert shebang == f"#!{sys.executable}"


_FAKE_TARGET_RUNNER = '''#!{python}
# Target-runner sintético (mesmo contrato de CLI e mesmo evaluations.csv do
# gerado por generate_irace_campaign), para exercitar o irace real em segundos.
import csv, json, random, sys
from pathlib import Path
configuration_id, instance_id, seed, _instance = sys.argv[1:5]
values, key = {{}}, None
for token in sys.argv[5:]:
    if token.startswith("--"):
        key = token[2:]
    elif key is not None:
        values[key] = float(token) if "." in token else int(token)
        key = None
rng = random.Random(int(seed))
cost = (values["cooling_rate"] - 0.95) ** 2 * 100 + abs(values["ruin_size"] - 6) * 0.05 + rng.gauss(0, 0.05)
log = Path(__file__).resolve().parent / "evaluations.csv"
header = not log.exists()
with log.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter=";")
    if header:
        writer.writerow(["configuration_id", "instance_id", "seed", "cost", "values_json"])
    writer.writerow([configuration_id, instance_id, seed, repr(cost), json.dumps(values, sort_keys=True)])
print(cost)
'''


@pytest.mark.skipif(not irace_available(), reason="irace (R) não instalado")
def test_real_irace_incumbent_is_irace_final_elite(tmp_path: Path) -> None:
    import sys

    space = {
        "cooling_rate": {"type": "float", "low": 0.88, "high": 0.995},
        "ruin_size": {"type": "int", "low": 1, "high": 12},
    }
    instances = [_tiny_instance(suffix) for suffix in "abc"]
    # 96 = mínimo que o irace aceita para 2 parâmetros.
    generate_irace_campaign(tmp_path, space, instances, src_path=Path("src"), max_experiments=96, seed=1)
    runner = tmp_path / "target-runner.py"
    runner.write_text(_FAKE_TARGET_RUNNER.format(python=sys.executable), encoding="utf-8")
    runner.chmod(0o755)

    result = run_irace(tmp_path, timeout_seconds=120)
    assert result["returncode"] == 0, result

    final = parse_irace_elites((tmp_path / "irace_stdout.log").read_text(encoding="utf-8"))
    values, stats = best_values_from_campaign(tmp_path)
    assert stats["selection"] == "irace_elite"
    assert stats["configuration_id"] == final[0]["configuration_id"]
    assert values == final[0]["values"]
    if result["elites_exported"]:
        assert stats["elite_source"] == "irace.Rdata"
