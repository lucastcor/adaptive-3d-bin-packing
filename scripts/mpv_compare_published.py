"""Compara bins médios no teste MPV (v=1..10) com a literatura (n=50, orientação fixa).

Uso: .venv/bin/python scripts/mpv_compare_published.py <runs.csv> [<runs.csv> ...]
Cada CSV precisa das colunas label, instance_id, seed, bins_used, lower_bound.
Labels com sufixo ':test' são agrupadas pelo prefixo (o braço).
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from statistics import mean

# docs/benchmark_mpv.md, n=50 (média de 10 instâncias)
PUBLISHED = {
    "L2": {1: 12.5, 4: 28.7, 5: 7.3, 6: 8.7, 7: 6.3, 8: 8.0},
    "LB-B": {1: 12.9, 4: 29.0, 5: 7.6, 6: 9.4, 7: 6.8, 8: 8.7},
    "C-EP": {1: 13.7, 4: 29.4, 5: 8.4, 6: 10.1, 7: 7.5, 8: 9.4},
    "TS3": {1: 13.4, 4: 29.4, 5: 8.4, 6: 9.9, 7: 7.5, 8: 9.3},
    "GLS": {1: 13.4, 4: 29.4, 5: 8.3, 6: 9.8, 7: 7.4, 8: 9.2},
    "TS2P": {1: 13.4, 4: 29.4, 5: 8.3, 6: 9.8, 7: 7.4, 8: 9.2},
    "BR-aNB": {1: 13.4, 4: 29.4, 5: 8.3, 6: 9.8, 7: 7.4, 8: 9.2},
}
ID_RE = re.compile(r"mpv-c(\d+)-n(\d+)-v(\d+)")


def load(paths: list[str]) -> dict[str, dict[tuple[int, int], list[int]]]:
    """arm -> (class, v) -> bins por seed."""
    arms: dict[str, dict[tuple[int, int], list[int]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                label = row["label"]
                if not label.endswith(":test"):
                    continue
                match = ID_RE.fullmatch(row["instance_id"])
                if match is None or row["bins_used"] in ("", "None"):
                    continue
                cls, _n, v = (int(g) for g in match.groups())
                arms[label.removesuffix(":test")][(cls, v)].append(int(row["bins_used"]))
    return arms


def table(arms: dict[str, dict[tuple[int, int], list[int]]]) -> str:
    classes = sorted({cls for cells in arms.values() for cls, _ in cells})
    header = "| Método | " + " | ".join(f"Cl{c}" for c in classes) + " | Soma |"
    lines = [header, "|" + "---|" * (len(classes) + 2)]
    for name, values in PUBLISHED.items():
        cells = [values.get(c) for c in classes]
        total = sum(x for x in cells if x is not None)
        lines.append(f"| {name} (publicado) | " + " | ".join(f"{x:.1f}" for x in cells) + f" | {total:.1f} |")
    for arm, cells_by_key in sorted(arms.items()):
        for variant, reduce in (("média seeds", mean), ("melhor seed", min)):
            per_class = []
            for c in classes:
                per_instance = [reduce(b) for (cls, _v), b in cells_by_key.items() if cls == c]
                per_class.append(mean(per_instance) if len(per_instance) == 10 else float("nan"))
            lines.append(
                f"| {arm} ({variant}) | " + " | ".join(f"{x:.2f}" for x in per_class) + f" | {sum(per_class):.2f} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    print(table(load(sys.argv[1:])))
