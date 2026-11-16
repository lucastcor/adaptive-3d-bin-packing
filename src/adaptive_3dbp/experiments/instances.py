"""Gerador determinístico de datasets estratificados para campanhas.

Estratos (o objetivo experimental é provar que o contexto da instância
importa, então o dataset precisa variar exatamente as dimensões que o GPR usa
como features): tamanho (nº de itens), heterogeneidade de tamanhos, taxa de
ocupação alvo, forma dos itens (cúbicos vs alongados) e presença de restrição
``min_distance``. Os splits train/validation/test são estratificados: cada
classe aparece em todos os splits.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable

from adaptive_3dbp.domain.models import (
    BinSpec,
    ItemSpec,
    MinDistanceConstraint,
    PackingConstraints,
    ProblemInstance,
)

BIN_SIDE = 100.0

SIZE_CLASSES = {"small": (10, 20), "medium": (30, 60), "large": (80, 140)}
# Ocupação-alvo em MÚLTIPLOS do volume de UMA caixa (volume total ≈ alvo × V_bin).
# Valores < 1 fazem toda instância caber em 1 caixa (LB=1, problema degenera em
# "cabe ou não cabe" e nenhum configurador se diferencia). Multi-caixa garante
# LB ≥ 2 e espaço real de otimização, onde bins/gap discriminam os braços.
OCCUPANCY_CLASSES = {"low": 1.6, "high": 3.0}
SHAPE_CLASSES = ("cubic", "elongated", "mixed")
HETEROGENEITY_CLASSES = ("homogeneous", "heterogeneous")


@dataclass(frozen=True, slots=True)
class InstanceClass:
    size: str
    occupancy: str
    shape: str
    heterogeneity: str
    min_distance: bool

    @property
    def label(self) -> str:
        distance = "dist" if self.min_distance else "nodist"
        return f"{self.size}-{self.occupancy}-{self.shape}-{self.heterogeneity}-{distance}"


CandidateEvaluator = Callable[[ProblemInstance, InstanceClass, str, int, int], dict[str, Any]]


def all_classes(include_min_distance: bool = True) -> list[InstanceClass]:
    distance_options = (False, True) if include_min_distance else (False,)
    return [
        InstanceClass(size, occupancy, shape, heterogeneity, distance)
        for size, occupancy, shape, heterogeneity, distance in product(
            SIZE_CLASSES, OCCUPANCY_CLASSES, SHAPE_CLASSES, HETEROGENEITY_CLASSES, distance_options
        )
    ]


def _item_dimensions(spec: InstanceClass, rng: random.Random, base_templates: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    if spec.heterogeneity == "homogeneous" and base_templates:
        length, width, height = base_templates[rng.randrange(len(base_templates))]
        jitter = lambda value: max(2.0, value * rng.uniform(0.95, 1.05))  # noqa: E731
        return jitter(length), jitter(width), jitter(height)
    base = rng.uniform(8.0, 30.0)
    if spec.shape == "cubic":
        ratio = lambda: rng.uniform(0.85, 1.2)  # noqa: E731
        return base, base * ratio(), base * ratio()
    if spec.shape == "elongated":
        return base * rng.uniform(2.5, 4.0), base * rng.uniform(0.7, 1.1), base * rng.uniform(0.7, 1.1)
    # mixed
    if rng.random() < 0.5:
        return base * rng.uniform(2.0, 3.2), base, base * rng.uniform(0.8, 1.2)
    return base, base * rng.uniform(0.85, 1.2), base * rng.uniform(0.85, 1.2)


def generate_instance(
    spec: InstanceClass,
    seed: int,
    instance_id: str | None = None,
    occupancy_scale: float = 1.0,
) -> ProblemInstance:
    rng = random.Random(seed)
    low, high = SIZE_CLASSES[spec.size]
    n_items = rng.randint(low, high)
    target_occupancy = OCCUPANCY_CLASSES[spec.occupancy] * max(0.25, float(occupancy_scale))

    templates: list[tuple[float, float, float]] = []
    if spec.heterogeneity == "homogeneous":
        template_count = rng.randint(2, 3)
        templates = [
            _item_dimensions(
                InstanceClass(spec.size, spec.occupancy, spec.shape, "heterogeneous", spec.min_distance),
                rng,
                [],
            )
            for _ in range(template_count)
        ]

    dims = [_item_dimensions(spec, rng, templates) for _ in range(n_items)]
    total_volume = sum(length * width * height for length, width, height in dims)
    bin_volume = BIN_SIDE**3
    # Escala isotrópica para atingir a ocupação alvo; itens que excederem a
    # caixa são encolhidos individualmente (em vez de limitar a escala global,
    # que derrubava a ocupação de classes alongadas para muito abaixo do alvo).
    scale = (target_occupancy * bin_volume / total_volume) ** (1.0 / 3.0)
    limit = BIN_SIDE * 0.92

    scaled_dims: list[tuple[float, float, float]] = []
    clamped_flags: list[bool] = []
    for length, width, height in dims:
        scaled = (length * scale, width * scale, height * scale)
        item_factor = min(1.0, limit / max(scaled))
        clamped_flags.append(item_factor < 1.0)
        scaled_dims.append((scaled[0] * item_factor, scaled[1] * item_factor, scaled[2] * item_factor))

    # Passada corretiva: o clamp por item (necessário para caber na caixa)
    # reduz o volume total abaixo do alvo — em classes alongadas/altas a perda
    # chega a ~40%, derrubando o LB e a dificuldade pretendida da classe.
    # Re-escala os itens NÃO clampados para recuperar o déficit (respeitando o
    # próprio limite de cada um).
    target_volume = target_occupancy * bin_volume
    current_volume = sum(l * w * h for l, w, h in scaled_dims)
    free_volume = sum(l * w * h for (l, w, h), clamped in zip(scaled_dims, clamped_flags) if not clamped)
    if free_volume > 1e-9 and current_volume < target_volume:
        boost = ((target_volume - (current_volume - free_volume)) / free_volume) ** (1.0 / 3.0)
        corrected: list[tuple[float, float, float]] = []
        for (l, w, h), clamped in zip(scaled_dims, clamped_flags):
            if clamped:
                corrected.append((l, w, h))
                continue
            factor = min(boost, limit / max(l, w, h))
            corrected.append((l * factor, w * factor, h * factor))
        scaled_dims = corrected

    # Segunda passada (classes com todos os itens clampados, ex. small-elongated):
    # engrossa os lados curtos dos itens clampados, preservando o caráter
    # alongado (razão de aspecto >= 2.2), para aproximar o volume-alvo.
    current_volume = sum(l * w * h for l, w, h in scaled_dims)
    if current_volume < target_volume * 0.95:
        clamped_volume = sum(l * w * h for (l, w, h), clamped in zip(scaled_dims, clamped_flags) if clamped)
        if clamped_volume > 1e-9:
            needed = (target_volume - (current_volume - clamped_volume)) / clamped_volume
            thicken_global = max(needed, 1.0) ** 0.5  # aplica nos 2 lados curtos: volume escala t²
            corrected2: list[tuple[float, float, float]] = []
            for (l, w, h), clamped in zip(scaled_dims, clamped_flags):
                if not clamped:
                    corrected2.append((l, w, h))
                    continue
                longest = max(l, w, h)
                short_cap = longest / 2.2 if spec.shape == "elongated" else limit
                bound = min(
                    thicken_global,
                    short_cap / max(min(l, w, h), 1e-9),
                    limit / max(sorted((l, w, h))[1], 1e-9),
                )
                t = max(1.0, bound)
                corrected2.append(
                    tuple(dim if dim == longest else dim * t for dim in (l, w, h))  # type: ignore[arg-type]
                )
            scaled_dims = corrected2

    items = []
    for index, (length, width, height) in enumerate(scaled_dims):
        items.append(
            ItemSpec(
                id=f"item-{index:03d}",
                length=round(max(length, 1.0), 2),
                width=round(max(width, 1.0), 2),
                height=round(max(height, 1.0), 2),
            )
        )

    min_distance: list[MinDistanceConstraint] = []
    if spec.min_distance and n_items >= 4:
        pair_count = max(1, n_items // 12)
        indices = rng.sample(range(n_items), min(2 * pair_count, n_items))
        for pair in range(pair_count):
            a, b = indices[2 * pair], indices[2 * pair + 1]
            min_distance.append(
                MinDistanceConstraint(
                    item_id_a=items[a].id,
                    item_id_b=items[b].id,
                    distance=round(rng.uniform(4.0, 12.0), 1),
                )
            )

    return ProblemInstance(
        instance_id=instance_id or f"{spec.label}-s{seed}",
        bin=BinSpec(length=BIN_SIDE, width=BIN_SIDE, height=BIN_SIDE),
        items=items,
        constraints=PackingConstraints(min_distance=min_distance),
    )


def generate_dataset(
    output_dir: Path,
    per_class: dict[str, int] | None = None,
    seed: int = 0,
    include_min_distance: bool = True,
    classes: list[InstanceClass] | None = None,
    replacement_classes: list[InstanceClass] | None = None,
    candidate_evaluator: CandidateEvaluator | None = None,
    candidate_attempts: int = 1,
    max_replacement_classes: int = 8,
    occupancy_scale: float = 1.0,
    curation_metadata: dict[str, Any] | None = None,
    require_accepted_candidate: bool = False,
    replace_rejected_classes: bool = False,
    curated_splits: tuple[str, ...] = ("train",),
) -> dict[str, Any]:
    """Gera instâncias estratificadas e o manifest com splits train/val/test.

    Com ``candidate_evaluator``, só os splits em ``curated_splits`` (padrão:
    apenas ``train``) passam pela curadoria. Validação e teste são gerados sem
    rodar o solver: nenhuma informação de desempenho sobre instâncias
    held-out influencia quais instâncias ou classes entram no benchmark. A
    curadoria anterior rodava o painel default nas instâncias de teste (com
    as seeds de teste) e descartava as "fáceis demais" para ele, o que
    enviesava o teste contra o braço fixed_default.
    """
    per_class = per_class or {"train": 2, "validation": 1, "test": 1}
    selected = classes or all_classes(include_min_distance)
    output_dir.mkdir(parents=True, exist_ok=True)
    instances_dir = output_dir / "instances"
    instances_dir.mkdir(exist_ok=True)

    manifest: dict[str, Any] = {
        "seed": seed,
        "per_class": per_class,
        "classes": [spec.label for spec in selected],
        "requested_classes": [spec.label for spec in selected],
        "occupancy_scale": max(0.25, float(occupancy_scale)),
        "splits": {split: [] for split in per_class},
    }
    if candidate_evaluator is not None:
        manifest["curation"] = {
            "enabled": True,
            "curated_splits": list(curated_splits),
            "candidate_attempts": max(1, int(candidate_attempts)),
            "replace_rejected_classes": bool(replace_rejected_classes),
            "max_replacement_classes": max(0, int(max_replacement_classes)),
            **(curation_metadata or {}),
        }
    replacement_pool: list[InstanceClass] = []
    if replace_rejected_classes:
        seen_replacements: set[InstanceClass] = set()
        for pool_spec in replacement_classes or selected:
            if pool_spec not in seen_replacements:
                replacement_pool.append(pool_spec)
                seen_replacements.add(pool_spec)

    def candidate_specs_for(
        requested: InstanceClass,
        used_actual: set[InstanceClass] | None = None,
    ) -> list[InstanceClass]:
        if not replace_rejected_classes or not replacement_pool:
            return [requested]
        alternates = [pool_spec for pool_spec in replacement_pool if pool_spec != requested]
        used_actual = used_actual or set()
        preferred = [pool_spec for pool_spec in alternates if pool_spec not in used_actual]
        fallback = [pool_spec for pool_spec in alternates if pool_spec in used_actual]
        return [requested, *(preferred + fallback)[: max(0, int(max_replacement_classes))]]

    def best_candidate_for_spec(
        spec: InstanceClass,
        instance_id: str,
        base_counter: int,
        split: str,
        replica: int,
    ) -> tuple[ProblemInstance, dict[str, Any] | None]:
        if candidate_evaluator is None:
            instance_seed = seed * 100_003 + base_counter
            return (
                generate_instance(
                    spec,
                    instance_seed,
                    instance_id=instance_id,
                    occupancy_scale=max(0.25, float(occupancy_scale)),
                ),
                None,
            )
        if split not in curated_splits:
            # Held-out split: same seed formula as attempt 0, no solver runs.
            return (
                generate_instance(
                    spec,
                    seed * 100_003 + base_counter * 997,
                    instance_id=instance_id,
                    occupancy_scale=max(0.25, float(occupancy_scale)),
                ),
                None,
            )
        best_rank = float("inf")
        best_instance: ProblemInstance | None = None
        best_curation: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        for attempt in range(max(1, int(candidate_attempts))):
            instance_seed = seed * 100_003 + base_counter * 997 + attempt
            candidate = generate_instance(
                spec,
                instance_seed,
                instance_id=instance_id,
                occupancy_scale=max(0.25, float(occupancy_scale)),
            )
            evaluation = candidate_evaluator(candidate, spec, split, replica, attempt)
            accepted = bool(evaluation.get("accepted", False))
            rank = float(evaluation.get("rank", 1_000_000.0))
            attempt_record = {
                "attempt": attempt,
                "seed": instance_seed,
                "accepted": accepted,
                **evaluation,
            }
            attempts.append(attempt_record)
            if rank < best_rank:
                best_rank = rank
                best_instance = candidate
                best_curation = attempt_record
            if accepted:
                best_instance = candidate
                best_curation = attempt_record
                break
        if best_instance is None or best_curation is None:  # pragma: no cover - defensive.
            raise RuntimeError("dataset curation did not produce any candidate instance")
        curation = dict(best_curation)
        curation["attempts_used"] = len(attempts)
        curation["fallback"] = not bool(best_curation.get("accepted", False))
        return best_instance, curation

    def compact_curation(curation: dict[str, Any] | None) -> dict[str, Any] | None:
        if curation is None:
            return None
        keep = {
            "attempt",
            "attempts_used",
            "seed",
            "accepted",
            "fallback",
            "hard_reject",
            "reason",
            "preflight_status",
            "preflight_loss",
            "preflight_bins_used",
            "preflight_lower_bound",
            "preflight_gap_to_lower_bound",
            "preflight_optimal_proven",
            "preflight_evaluations",
            "preflight_runtime_seconds",
            "preflight_projected_runtime_seconds",
            "preflight_too_easy",
            "preflight_too_slow",
            "preflight_quality_discriminative",
            "preflight_bins_spread",
            "preflight_gap_spread",
            "preflight_lower_bound_shortfall",
            "preflight_seed_count",
            "instances",
            "accepted_instances",
            "fallback_instances",
            "hard_reject_instances",
            "discriminative_instances",
            "mean_rank",
            "mean_preflight_lower_bound",
            "mean_preflight_bins_spread",
            "mean_preflight_gap_spread",
        }
        return {key: value for key, value in curation.items() if key in keep}

    slot_plans: list[list[dict[str, int | str]]] = []
    counter = 0
    for _slot_index, _requested_spec in enumerate(selected):
        planned_entries: list[dict[str, int | str]] = []
        for split, quantity in per_class.items():
            for replica in range(quantity):
                planned_entries.append({"split": split, "replica": replica, "base_counter": counter})
                counter += 1
        slot_plans.append(planned_entries)

    def annotate_curation(
        curation: dict[str, Any] | None,
        requested: InstanceClass,
        actual: InstanceClass,
        slot_index: int,
    ) -> dict[str, Any] | None:
        if curation is None:
            return None
        curation["requested_class"] = requested.label
        curation["actual_class"] = actual.label
        curation["class_replaced"] = actual != requested
        curation["class_slot"] = slot_index
        return curation

    def slot_curation_summary(curations: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not curations:
            return None

        def mean_value(key: str) -> float | None:
            values = [
                float(curation[key])
                for curation in curations
                if curation.get(key) is not None and isinstance(curation.get(key), (int, float))
            ]
            return sum(values) / len(values) if values else None

        return {
            "accepted": all(bool(curation.get("accepted", False)) for curation in curations),
            "fallback": any(bool(curation.get("fallback", False)) for curation in curations),
            "instances": len(curations),
            "accepted_instances": sum(1 for curation in curations if bool(curation.get("accepted", False))),
            "fallback_instances": sum(1 for curation in curations if bool(curation.get("fallback", False))),
            "hard_reject_instances": sum(1 for curation in curations if bool(curation.get("hard_reject", False))),
            "discriminative_instances": sum(
                1 for curation in curations if bool(curation.get("preflight_quality_discriminative", False))
            ),
            "mean_rank": mean_value("rank"),
            "mean_preflight_lower_bound": mean_value("preflight_lower_bound"),
            "mean_preflight_bins_spread": mean_value("preflight_bins_spread"),
            "mean_preflight_gap_spread": mean_value("preflight_gap_spread"),
        }

    def slot_score(summary: dict[str, Any] | None, class_replaced: bool) -> float:
        if summary is None:
            return 0.0
        rank = float(summary.get("mean_rank") or 0.0)
        return (
            rank
            + 1_000.0 * float(summary.get("hard_reject_instances") or 0)
            + 100.0 * float(summary.get("fallback_instances") or 0)
            + (0.01 if class_replaced else 0.0)
        )

    def build_slot_candidate(
        requested: InstanceClass,
        actual: InstanceClass,
        slot_index: int,
        planned_entries: list[dict[str, int | str]],
    ) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, str | None]:
        generated_entries: list[dict[str, Any]] = []
        curations: list[dict[str, Any]] = []
        for planned in planned_entries:
            split = str(planned["split"])
            replica = int(planned["replica"])
            base_counter = int(planned["base_counter"])
            instance_id = f"{actual.label}-{split}-{replica}-slot{slot_index}"
            instance, curation = best_candidate_for_spec(actual, instance_id, base_counter, split, replica)
            curation = annotate_curation(curation, requested, actual, slot_index)
            if curation is not None:
                curations.append(curation)
                if (
                    require_accepted_candidate
                    and not bool(curation.get("accepted", False))
                    and bool(curation.get("hard_reject", False))
                ):
                    reason = str(curation.get("reason") or curation.get("preflight_status") or "hard reject")
                    return (
                        None,
                        slot_curation_summary(curations),
                        f"{actual.label}/{split}/{replica}: {reason}",
                    )
            generated_entries.append(
                {
                    "split": split,
                    "replica": replica,
                    "base_counter": base_counter,
                    "instance": instance,
                    "curation": curation,
                }
            )
        return generated_entries, slot_curation_summary(curations), None

    def resolve_class_slot(
        requested: InstanceClass,
        slot_index: int,
        used_actual: set[InstanceClass],
        planned_entries: list[dict[str, int | str]],
    ) -> tuple[InstanceClass, dict[str, Any] | None, list[dict[str, Any]]]:
        best_fallback_score = float("inf")
        best_fallback: tuple[InstanceClass, dict[str, Any] | None, list[dict[str, Any]]] | None = None
        failures: list[str] = []
        for candidate_spec in candidate_specs_for(requested, used_actual):
            generated_entries, selection_summary, failure = build_slot_candidate(
                requested,
                candidate_spec,
                slot_index,
                planned_entries,
            )
            if generated_entries is None:
                if failure:
                    failures.append(failure)
                continue
            if selection_summary is None or not bool(selection_summary.get("fallback", False)):
                return candidate_spec, selection_summary, generated_entries
            if not replace_rejected_classes:
                return candidate_spec, selection_summary, generated_entries
            candidate_score = slot_score(selection_summary, candidate_spec != requested)
            if candidate_score < best_fallback_score:
                best_fallback_score = candidate_score
                best_fallback = (candidate_spec, selection_summary, generated_entries)

        if best_fallback is not None:
            return best_fallback

        details = "; ".join(failures[:3]) if failures else "nenhuma classe candidata gerou o slot completo"
        raise ValueError(
            f"curadoria não encontrou classe tratável para slot {slot_index} ({requested.label}) "
            f"em {max(1, int(candidate_attempts))} tentativa(s) por instância; {details}. "
            "Reduza classes, aumente candidate_attempts/max_replacement_classes ou use um manifest curado."
        )

    resolved_slots: list[dict[str, Any]] = []
    used_actual: set[InstanceClass] = set()
    for slot_index, requested_spec in enumerate(selected):
        actual_spec, selection_curation, generated_entries = resolve_class_slot(
            requested_spec,
            slot_index,
            used_actual,
            slot_plans[slot_index],
        )
        used_actual.add(actual_spec)
        resolved_slots.append(
            {
                "slot": slot_index,
                "requested_spec": requested_spec,
                "actual_spec": actual_spec,
                "requested_class": requested_spec.label,
                "actual_class": actual_spec.label,
                "class_replaced": actual_spec != requested_spec,
                "selection_curation": compact_curation(selection_curation),
                "generated_entries": generated_entries,
            }
        )

    manifest["class_slots"] = [
        {
            "slot": slot["slot"],
            "requested_class": slot["requested_class"],
            "actual_class": slot["actual_class"],
            "class_replaced": slot["class_replaced"],
            **(
                {"selection_curation": slot["selection_curation"]}
                if slot.get("selection_curation") is not None
                else {}
            ),
        }
        for slot in resolved_slots
    ]
    actual_classes: list[str] = []
    for slot in resolved_slots:
        label = str(slot["actual_class"])
        if label not in actual_classes:
            actual_classes.append(label)
    manifest["classes"] = actual_classes

    for slot in resolved_slots:
        requested_spec = slot["requested_spec"]
        actual_spec = slot["actual_spec"]
        slot_index = int(slot["slot"])
        for generated in slot["generated_entries"]:
            instance = generated["instance"]
            selected_curation = generated["curation"]
            path = instances_dir / f"{instance.instance_id}.json"
            path.write_text(instance.model_dump_json(indent=2), encoding="utf-8")
            entry = {
                "path": str(path),
                "instance_id": instance.instance_id,
                "class": actual_spec.label,
                "requested_class": requested_spec.label,
                "class_slot": slot_index,
            }
            if selected_curation is not None:
                entry["curation"] = selected_curation
            manifest["splits"][str(generated["split"])].append(entry)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
