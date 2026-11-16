#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Análises complementares pedidas no parecer técnico (resposta à revisão).

Calcula, a partir dos artefatos JÁ EXISTENTES de uma campanha (sem reexecutar
o solver):

  1. Força dos lower bounds (L0, L2, clique+residual, máximo combinado) sobre as
     instâncias do dataset (parecer §3.3).
  2. Decomposição de variância da perda em efeitos de instância, de braço
     (configuração) e de interação instância×configuração (parecer §3.5).
  3. Virtual Best Configuration (oráculo retrospectivo por instância) e o ganho
     potencial da personalização vs. o melhor braço global (parecer §3.6).
  4. Estabilidade do ranking por bootstrap das instâncias de teste e correlação
     validação↔teste (parecer §3.8).

Uso:
    PYTHONPATH=src python3 scripts/review_analysis.py \
        --campaign artifacts/campaigns/campaign-632591ae \
        --out artifacts/analysis/review_response
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load_runs(campaign: Path) -> list[dict]:
    rows = []
    with open(campaign / "runs.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def arm_of(label: str) -> tuple[str, str]:
    if ":" in label:
        arm, phase = label.rsplit(":", 1)
        return arm, phase
    return label, ""


def per_arm_instance_loss(rows, phase) -> dict[str, dict[str, float]]:
    """mean loss over seeds, by arm -> instance."""
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        arm, ph = arm_of(r["label"])
        if ph != phase:
            continue
        loss = r.get("loss")
        if loss in (None, ""):
            continue
        grouped[arm][r["instance_id"]].append(float(loss))
    return {a: {i: statistics.fmean(v) for i, v in inst.items()} for a, inst in grouped.items()}


# ---------------------------------------------------------------- 1. bounds
def bound_strength(campaign: Path) -> dict:
    from adaptive_3dbp.domain.bounds import (
        clique_volume_bound,
        mt_l2_volume_bound,
        strong_lower_bound,
        volume_lower_bound,
    )
    from adaptive_3dbp.domain.models import ProblemInstance

    inst_dir = campaign / "dataset" / "instances"
    files = sorted(inst_dir.glob("*.json"))
    rows = []
    for fp in files:
        inst = ProblemInstance.model_validate_json(fp.read_text(encoding="utf-8"))
        l0 = volume_lower_bound(inst)
        l2 = mt_l2_volume_bound(inst)
        cl = clique_volume_bound(inst)
        mx = strong_lower_bound(inst)
        rows.append({"instance": fp.stem, "L0": l0, "L2": l2, "clique_res": cl, "max": mx})
    n = len(rows)

    def mean(k):
        return statistics.fmean(r[k] for r in rows)

    improves_over_l0 = sum(1 for r in rows if r["max"] > r["L0"])
    l2_strict = sum(1 for r in rows if r["L2"] > r["L0"])
    clique_strict = sum(1 for r in rows if r["clique_res"] > r["L0"])
    clique_is_max = sum(1 for r in rows if r["clique_res"] == r["max"] and r["clique_res"] > r["L0"])
    mean_uplift = statistics.fmean(r["max"] - r["L0"] for r in rows)
    return {
        "n_instances": n,
        "mean_L0": mean("L0"),
        "mean_L2": mean("L2"),
        "mean_clique_residual": mean("clique_res"),
        "mean_max": mean("max"),
        "mean_absolute_uplift_over_L0": mean_uplift,
        "instances_where_max_gt_L0": improves_over_l0,
        "instances_where_L2_gt_L0": l2_strict,
        "instances_where_clique_gt_L0": clique_strict,
        "instances_where_clique_defines_max_above_L0": clique_is_max,
        "per_instance": rows,
    }


def per_arm_instance_seed_loss(rows, phase):
    """Full per-(arm, instance, seed) losses (no aggregation)."""
    out = defaultdict(lambda: defaultdict(dict))  # arm -> instance -> {seed: loss}
    for r in rows:
        arm, ph = arm_of(r["label"])
        if ph != phase:
            continue
        loss = r.get("loss")
        if loss in (None, ""):
            continue
        out[arm][r["instance_id"]][r["seed"]] = float(loss)
    return out


# ------------------------------------------------ 2. variance decomposition
def variance_decomposition(per_seed: dict) -> dict:
    """Balanced two-way ANOVA WITH replication (seeds) for the test losses.

    Model:  y_{aik} = mu + alpha_a + beta_i + (alpha*beta)_{ai} + eps_{aik}
      a = arm/method, i = instance, k = seed replication.

    Because seeds give within-cell replication, the interaction SS is separated
    from the residual (seed/measurement) SS — addressing the reviewer's point
    that interaction must not be conflated with residual. We report the fraction
    of total SS for each component. NOTE: 'arm' is a CONFIGURATION METHOD (a
    procedure that returns a configuration or, for the contextual arm, a policy),
    not a single SA hyperparameter vector; this decomposes the effect of the
    *method/strategy*, not of pure SA parameters.
    """
    arms = sorted(per_seed)
    instances = sorted({i for a in per_seed for i in per_seed[a]})
    # balanced complete design: keep cells present for ALL arms, common seeds
    seeds_per_cell = []
    cells = {}  # (a,i) -> list of (seed, loss)
    for a in arms:
        for i in instances:
            d = per_seed[a].get(i)
            if d:
                cells[(a, i)] = sorted(d.items())
                seeds_per_cell.append(len(d))
    if not cells:
        return {}
    r = min(seeds_per_cell)  # replications per cell (balanced -> all equal)
    # truncate each cell to r seeds (sorted) to keep design balanced
    obs = {}  # (a,i) -> [losses]
    for (a, i), sl in cells.items():
        obs[(a, i)] = [v for _, v in sl[:r]]
    # only keep instances/arms that form a complete grid
    full_arms = [a for a in arms if all((a, i) in obs for i in instances)]
    full_inst = [i for i in instances if all((a, i) in obs for a in full_arms)]
    vals = [v for a in full_arms for i in full_inst for v in obs[(a, i)]]
    n = len(vals)
    grand = statistics.fmean(vals)
    cell_mean = {(a, i): statistics.fmean(obs[(a, i)]) for a in full_arms for i in full_inst}
    arm_mean = {a: statistics.fmean([cell_mean[(a, i)] for i in full_inst]) for a in full_arms}
    inst_mean = {i: statistics.fmean([cell_mean[(a, i)] for a in full_arms]) for i in full_inst}
    na, ni = len(full_arms), len(full_inst)
    ss_total = sum((v - grand) ** 2 for v in vals)
    ss_arm = ni * r * sum((arm_mean[a] - grand) ** 2 for a in full_arms)
    ss_inst = na * r * sum((inst_mean[i] - grand) ** 2 for i in full_inst)
    ss_inter = r * sum(
        (cell_mean[(a, i)] - arm_mean[a] - inst_mean[i] + grand) ** 2
        for a in full_arms for i in full_inst
    )
    ss_resid = sum((v - cell_mean[(a, i)]) ** 2 for a in full_arms for i in full_inst for v in obs[(a, i)])
    return {
        "design": f"{na} arms x {ni} instances x {r} seeds = {n} obs (balanced)",
        "n_arms": na,
        "n_instances": ni,
        "n_seeds": r,
        "ss_total": ss_total,
        "frac_instance": ss_inst / ss_total if ss_total else 0.0,
        "frac_arm_method": ss_arm / ss_total if ss_total else 0.0,
        "frac_interaction": ss_inter / ss_total if ss_total else 0.0,
        "frac_residual_seed": ss_resid / ss_total if ss_total else 0.0,
    }


# --------------------------------------------------- 3. Virtual Best Arm oracle
def virtual_best_arm(loss_matrix: dict[str, dict[str, float]], main_arms: list[str]) -> dict:
    """Virtual Best ARM (a.k.a. Virtual Best Method/Selector), NOT a Virtual Best
    Configuration. For each instance it picks the best of the final ARMS' losses;
    arms are configuration METHODS (one returns a fixed config, the contextual arm
    returns a policy). So this is the oracle of per-instance METHOD selection,
    restricted to the observed portfolio of arms — an upper bound on what an
    instance-wise selector over THIS portfolio could achieve, not a global ceiling
    over all configurations Theta. It is also OPTIMISTIC: taking min over arm means
    selects favourable seed noise (winner's curse over methods), so the true
    expected loss of the chosen arm on fresh seeds tends to be slightly higher.
    """
    arms = [a for a in main_arms if a in loss_matrix]
    instances = sorted({i for a in arms for i in loss_matrix[a]})
    vba = {}
    for j in instances:
        cand = [loss_matrix[a][j] for a in arms if j in loss_matrix[a]]
        if cand:
            vba[j] = min(cand)
    vba_mean = statistics.fmean(vba.values()) if vba else float("nan")
    arm_mean = {a: statistics.fmean([loss_matrix[a][j] for j in instances if j in loss_matrix[a]]) for a in arms}
    best_arm = min(arm_mean, key=arm_mean.get)
    best_arm_mean = arm_mean[best_arm]
    return {
        "arms_considered": arms,
        "vba_mean_loss": vba_mean,
        "best_single_arm": best_arm,
        "best_single_arm_mean_loss": best_arm_mean,
        "headroom_portfolio": best_arm_mean - vba_mean,
        "headroom_relative": (best_arm_mean - vba_mean) / best_arm_mean if best_arm_mean else 0.0,
        "note": "Virtual Best ARM over observed portfolio; optimistic (winner's curse over methods).",
        "arm_mean_losses": arm_mean,
    }


# ----------------------------------------------- 4. ranking stability + corr
def ranking_stability(loss_matrix: dict[str, dict[str, float]], main_arms: list[str],
                      n_boot: int = 10000, seed: int = 0) -> dict:
    arms = [a for a in main_arms if a in loss_matrix]
    instances = sorted({i for a in arms for i in loss_matrix[a]})
    rng = random.Random(seed)
    best_count = defaultdict(int)
    for _ in range(n_boot):
        sample = [rng.choice(instances) for _ in instances]
        arm_mean = {}
        for a in arms:
            vals = [loss_matrix[a][j] for j in sample if j in loss_matrix[a]]
            if vals:
                arm_mean[a] = statistics.fmean(vals)
        if arm_mean:
            best_count[min(arm_mean, key=arm_mean.get)] += 1
    return {a: best_count[a] / n_boot for a in arms}


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n)))
    return num / den if den else float("nan")


def val_test_correlation(val: dict, test: dict, arms: list[str]) -> dict:
    common = [a for a in arms if a in val and a in test]
    vmean = {a: statistics.fmean(val[a].values()) for a in common}
    tmean = {a: statistics.fmean(test[a].values()) for a in common}
    xs = [vmean[a] for a in common]
    ys = [tmean[a] for a in common]
    return {
        "arms": common,
        "validation_mean_loss": vmean,
        "test_mean_loss": tmean,
        "spearman_val_test": spearman(xs, ys) if len(common) > 2 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--proposal", default="noisy_contextual_ego")
    args = ap.parse_args()
    campaign = Path(args.campaign)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_runs(campaign)
    test = per_arm_instance_loss(rows, "test")
    val = per_arm_instance_loss(rows, "validation")
    test_seeds = per_arm_instance_seed_loss(rows, "test")
    main_arms = ["fixed_default", "random_search", "classic_ego",
                 "noisy_contextual_ego", "smac3", "irace",
                 "per_class", "nearest_neighbor"]
    main_present = [a for a in main_arms if a in test]

    result = {
        "campaign": campaign.name,
        "lower_bound_strength": bound_strength(campaign),
        "variance_decomposition_test": variance_decomposition(test_seeds),
        "virtual_best_arm_test": virtual_best_arm(test, main_present),
        "ranking_stability_test": ranking_stability(test, main_present),
        "validation_test_correlation": val_test_correlation(val, test, main_present),
    }
    (out / "review_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # markdown resumido
    lb = result["lower_bound_strength"]
    vd = result["variance_decomposition_test"]
    vb = result["virtual_best_arm_test"]
    rs = result["ranking_stability_test"]
    md = []
    md.append(f"# Análises complementares — {campaign.name}\n")
    md.append("## 1. Força dos lower bounds\n")
    md.append("| Bound | Valor médio | Uplift vs L0 | Instâncias > L0 |")
    md.append("|---|---|---|---|")
    md.append(f"| L0 (volumétrico) | {lb['mean_L0']:.3f} | — | — |")
    md.append(f"| L2 (Martello-Toth 1D) | {lb['mean_L2']:.3f} | — | {lb['instances_where_L2_gt_L0']}/{lb['n_instances']} |")
    md.append(f"| Clique + residual | {lb['mean_clique_residual']:.3f} | — | {lb['instances_where_clique_gt_L0']}/{lb['n_instances']} |")
    md.append(f"| **Máximo combinado** | **{lb['mean_max']:.3f}** | +{lb['mean_absolute_uplift_over_L0']:.3f} | {lb['instances_where_max_gt_L0']}/{lb['n_instances']} |\n")
    md.append("## 2. Decomposição de variância da perda (ANOVA 2-way com replicação, teste)\n")
    md.append(f"- Desenho: {vd.get('design','')}")
    md.append(f"- Fração explicada pela INSTÂNCIA: {vd.get('frac_instance', 0):.1%}")
    md.append(f"- Fração explicada pelo MÉTODO/braço: {vd.get('frac_arm_method', 0):.1%}")
    md.append(f"- Fração de INTERAÇÃO (instância×método, isolada): {vd.get('frac_interaction', 0):.1%}")
    md.append(f"- Fração de RESÍDUO (seed/medição): {vd.get('frac_residual_seed', 0):.1%}\n")
    md.append("## 3. Virtual Best Arm (oráculo de seleção de método por instância)\n")
    md.append(f"- VBA (perda média do oráculo de método por instância): {vb['vba_mean_loss']:.4f}")
    md.append(f"- Melhor método global único: {vb['best_single_arm']} ({vb['best_single_arm_mean_loss']:.4f})")
    md.append(f"- Ganho retrospectivo no portfólio avaliado: {vb['headroom_portfolio']:.4f} "
              f"({vb['headroom_relative']:.1%} do melhor global) — otimista (winner's curse)\n")
    md.append("## 4. Estabilidade do ranking (bootstrap das instâncias de teste)\n")
    md.append("| Braço | P(ser o melhor) |")
    md.append("|---|---|")
    for a, p in sorted(rs.items(), key=lambda kv: kv[1], reverse=True):
        md.append(f"| {a} | {p:.1%} |")
    corr = result["validation_test_correlation"].get("spearman_val_test")
    md.append(f"\n- Correlação de Spearman validação↔teste (perda média por braço): "
              f"{corr:.3f}" if corr is not None else "\n- Correlação val↔teste: n/d")
    (out / "review_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md))
    print(f"\n[ok] artefatos em {out}")


if __name__ == "__main__":
    main()
