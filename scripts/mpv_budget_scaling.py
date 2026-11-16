"""SA padrão com orçamentos maiores nas 60 instâncias de teste MPV (n=50, v=1..10).

Uso: .venv/bin/python scripts/mpv_budget_scaling.py <saida.csv> 30000,300000
Com a 0.2.0 o resultado está em
artifacts/campaigns/mpv-pilot-001/budget_scaling_v0.2.0.csv (11 processos,
cerca de 9 minutos). Compare com scripts/mpv_compare_published.py.

Escreve um CSV no mesmo formato mínimo que mpv_compare_published.py lê
(label, instance_id, seed, bins_used, lower_bound, evaluations, runtime_seconds).
O schedule escala com o orçamento: iterations_per_temperature e stagnation_limit
crescem na mesma proporção, para o SA usar o orçamento em vez de parar pela
temperatura.
"""
from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor

from adaptive_3dbp.benchmarks.mpv import mpv_instance
from adaptive_3dbp.domain.models import SolverConfig
from adaptive_3dbp.experiments.runner import solve_instance

CLASSES = (1, 4, 5, 6, 7, 8)
SEEDS = (11, 23, 47)
BASE_BUDGET = 3000


def run(task: tuple[int, int, int, int]) -> dict:
    budget, cls, v, seed = task
    scale = budget / BASE_BUDGET
    config = SolverConfig(
        max_evaluations=budget,
        time_limit_seconds=1800.0,
        iterations_per_temperature=max(80, int(80 * scale)),
        stagnation_limit=max(600, int(600 * scale)),
        use_native_core=True,
    )
    instance = mpv_instance(cls, 50, v)
    result = solve_instance(instance, config, seed=seed)
    return {
        "label": f"sa_default_{budget}:test",
        "instance_id": instance.instance_id,
        "seed": seed,
        "bins_used": result.best_found_bins,
        "lower_bound": result.lower_bound,
        "evaluations": result.evaluations,
        "runtime_seconds": round(result.runtime_seconds, 3),
        "diagnostics": "; ".join(result.diagnostics),
    }


if __name__ == "__main__":
    out = sys.argv[1]
    budgets = [int(x) for x in sys.argv[2].split(",")]
    tasks = [(b, c, v, s) for b in budgets for c in CLASSES for v in range(1, 11) for s in SEEDS]
    with ProcessPoolExecutor(max_workers=11) as pool, open(out, "w", encoding="utf-8", newline="") as handle:
        writer = None
        for row in pool.map(run, tasks, chunksize=1):
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
    print(f"ok: {len(tasks)} runs -> {out}")
