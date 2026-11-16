from __future__ import annotations

import json
import csv
from pathlib import Path

from adaptive_3dbp.domain.models import SolverConfig
from adaptive_3dbp.domain.models import SolverResult
import adaptive_3dbp.experiments.campaign as campaign_module
from adaptive_3dbp.experiments.campaign import (
    ArmOutcome,
    _make_dataset_candidate_evaluator,
    _discrimination_warning,
    _select_campaign_classes,
    export_results_table_tex,
    render_campaign_report,
    run_campaign,
)
from adaptive_3dbp.experiments.instances import InstanceClass, all_classes, generate_dataset, generate_instance
from adaptive_3dbp.learning.suggest import suggest_configs
from adaptive_3dbp.persistence.sqlite import SQLiteExperimentRepository


def test_generate_instance_is_deterministic_and_hits_occupancy() -> None:
    spec = InstanceClass("small", "high", "elongated", "heterogeneous", False)
    a = generate_instance(spec, 42)
    b = generate_instance(spec, 42)
    assert a.model_dump() == b.model_dump()
    occupancy = sum(i.length * i.width * i.height for i in a.items) / a.bin.volume
    # Ocupação multi-caixa: alvo "high" = 3.0x o volume de uma caixa (LB >= 2,
    # senão toda instância cabe em 1 caixa e nenhum configurador discrimina).
    assert 2.0 <= occupancy <= 3.3


def test_generate_instance_occupancy_scale_makes_harder_instances() -> None:
    spec = InstanceClass("small", "high", "mixed", "heterogeneous", False)
    base = generate_instance(spec, 42)
    harder = generate_instance(spec, 42, occupancy_scale=1.6)

    base_occupancy = sum(item.length * item.width * item.height for item in base.items) / base.bin.volume
    harder_occupancy = sum(item.length * item.width * item.height for item in harder.items) / harder.bin.volume

    assert harder_occupancy > base_occupancy
    assert harder_occupancy >= 4.0


def test_generate_dataset_writes_manifest_and_splits(tmp_path: Path) -> None:
    classes = all_classes(include_min_distance=False)[:2]
    manifest = generate_dataset(tmp_path, per_class={"train": 1, "validation": 1, "test": 1}, seed=1, classes=classes)
    assert set(manifest["splits"]) == {"train", "validation", "test"}
    for split in manifest["splits"].values():
        assert len(split) == 2
        for entry in split:
            assert Path(entry["path"]).exists()


def test_generate_dataset_records_candidate_curation(tmp_path: Path) -> None:
    spec = InstanceClass("small", "low", "elongated", "homogeneous", False)

    def evaluator(*args: object) -> dict[str, object]:
        attempt = int(args[-1])
        return {
            "accepted": attempt == 1,
            "rank": 0.0 if attempt == 1 else 100.0,
            "preflight_status": "feasible",
            "preflight_too_easy": attempt == 0,
            "preflight_evaluations": 8 + attempt,
        }

    manifest = generate_dataset(
        tmp_path,
        per_class={"train": 1},
        seed=1,
        classes=[spec],
        candidate_evaluator=evaluator,
        candidate_attempts=2,
        curation_metadata={"prefer_nontrivial": True},
    )

    assert manifest["curation"]["enabled"] is True
    assert manifest["curation"]["prefer_nontrivial"] is True
    curation = manifest["splits"]["train"][0]["curation"]
    assert curation["accepted"] is True
    assert curation["attempt"] == 1
    assert curation["attempts_used"] == 2
    assert curation["fallback"] is False


def test_generate_dataset_replaces_rejected_class_by_stratified_slot(tmp_path: Path) -> None:
    rejected = InstanceClass("small", "low", "elongated", "homogeneous", False)
    accepted = InstanceClass("small", "high", "mixed", "heterogeneous", False)

    def evaluator(_instance: object, spec: InstanceClass, *_args: object) -> dict[str, object]:
        ok = spec == accepted
        return {
            "accepted": ok,
            "hard_reject": not ok,
            "rank": 0.0 if ok else 10_000.0,
            "preflight_status": "feasible" if ok else "timeout",
            "preflight_quality_discriminative": ok,
            "preflight_too_easy": False,
            "preflight_too_slow": not ok,
            "preflight_evaluations": 10,
        }

    manifest = generate_dataset(
        tmp_path,
        per_class={"train": 2, "validation": 1, "test": 1},
        seed=1,
        classes=[rejected],
        replacement_classes=[rejected, accepted],
        candidate_evaluator=evaluator,
        candidate_attempts=1,
        replace_rejected_classes=True,
        require_accepted_candidate=True,
    )

    assert manifest["classes"] == [accepted.label]
    assert manifest["requested_classes"] == [rejected.label]
    assert len(manifest["class_slots"]) == 1
    slot = manifest["class_slots"][0]
    assert slot["slot"] == 0
    assert slot["requested_class"] == rejected.label
    assert slot["actual_class"] == accepted.label
    assert slot["class_replaced"] is True
    assert slot["selection_curation"]["accepted"] is True
    all_entries = [entry for entries in manifest["splits"].values() for entry in entries]
    assert len(all_entries) == 4
    assert len({entry["instance_id"] for entry in all_entries}) == 4
    assert {entry["class"] for entry in all_entries} == {accepted.label}
    assert {entry["requested_class"] for entry in all_entries} == {rejected.label}
    assert {entry["class_slot"] for entry in all_entries} == {0}
    for entry in manifest["splits"]["train"]:
        assert entry["curation"]["class_replaced"] is True
        assert entry["curation"]["requested_class"] == rejected.label
        assert entry["curation"]["actual_class"] == accepted.label
    # Held-out instances are never curated (no solver run on them).
    for split in ("validation", "test"):
        assert all("curation" not in entry for entry in manifest["splits"][split])


def test_curation_never_inspects_held_out_splits(tmp_path: Path) -> None:
    # Regression for the selection bias in campaigns 632591ae/6f670fb1: the
    # default-config panel used to run on validation/test instances (with the
    # test seeds) and a class could be swapped because of how it behaved on
    # the test split. Only training instances may inform curation.
    requested = InstanceClass("small", "low", "mixed", "homogeneous", False)
    held_out_hostile = InstanceClass("medium", "high", "cubic", "homogeneous", False)
    seen_splits: list[str] = []

    def evaluator(_instance: object, spec: InstanceClass, split: str, *_args: object) -> dict[str, object]:
        seen_splits.append(split)
        ok = spec == held_out_hostile or split != "train"
        return {
            "accepted": ok,
            "hard_reject": not ok,
            "rank": 0.0 if ok else 10_000.0,
            "preflight_status": "feasible" if ok else "error",
            "preflight_quality_discriminative": ok,
            "preflight_too_easy": False,
            "preflight_too_slow": False,
            "preflight_evaluations": 10,
        }

    manifest = generate_dataset(
        tmp_path,
        per_class={"train": 1, "validation": 1, "test": 1},
        seed=1,
        classes=[requested],
        replacement_classes=[requested, held_out_hostile],
        candidate_evaluator=evaluator,
        candidate_attempts=1,
        replace_rejected_classes=True,
        require_accepted_candidate=True,
        max_replacement_classes=2,
    )

    assert set(seen_splits) == {"train"}
    assert manifest["curation"]["curated_splits"] == ["train"]
    assert manifest["class_slots"][0]["actual_class"] == held_out_hostile.label
    assert {entry["class"] for entries in manifest["splits"].values() for entry in entries} == {
        held_out_hostile.label
    }


def test_candidate_curation_prefers_discriminative_quality(monkeypatch) -> None:
    spec = InstanceClass("small", "high", "mixed", "heterogeneous", False)
    instance = generate_instance(spec, 7, occupancy_scale=1.6)

    def fake_solve(_instance, config: SolverConfig, seed: int, timeout_seconds: float) -> SolverResult:
        bins = 4 if config.configuration_id == "curation-exploratory" else 5
        lower_bound = 4
        return SolverResult(
            instance_id=_instance.instance_id,
            solver="simulated_annealing",
            solver_version="test",
            configuration_id=config.configuration_id,
            seed=seed,
            status="feasible",
            best_found_bins=bins,
            lower_bound=lower_bound,
            best_known_bins=None,
            optimal_proven=bins == lower_bound,
            gap_to_best_known=None,
            gap_to_lower_bound=(bins - lower_bound) / lower_bound,
            runtime_seconds=0.05,
            evaluations=config.max_evaluations,
            objective_scalar=float(bins),
            placements=[],
            bin_metrics=[],
            convergence=[],
            features={},
            diagnostics=[],
        )

    monkeypatch.setattr(campaign_module, "_solve_with_hard_timeout", fake_solve)
    evaluator, metadata = _make_dataset_candidate_evaluator(
        max_evaluations=600,
        time_limit_seconds=36.0,
        temperature_calibration_samples=6,
        generate_args={"prefer_discriminative": True, "curation_seed_count": 2},
        split_seeds={"train": [11, 23]},
    )

    result = evaluator(instance, spec, "train", 0, 0)

    assert metadata["prefer_discriminative"] is True
    assert result["accepted"] is True
    assert result["preflight_quality_discriminative"] is True
    assert result["preflight_bins_spread"] >= 1.0


def test_generate_dataset_fails_on_hard_rejected_candidates(tmp_path: Path) -> None:
    spec = InstanceClass("small", "low", "elongated", "homogeneous", False)

    def evaluator(*_args: object) -> dict[str, object]:
        return {
            "accepted": False,
            "hard_reject": True,
            "rank": 10_000.0,
            "preflight_status": "timeout",
            "preflight_too_easy": False,
            "preflight_too_slow": True,
            "preflight_evaluations": 0,
        }

    try:
        generate_dataset(
            tmp_path,
            per_class={"train": 1},
            seed=1,
            classes=[spec],
            candidate_evaluator=evaluator,
            candidate_attempts=2,
            require_accepted_candidate=True,
        )
    except ValueError as exc:
        assert "não encontrou classe tratável" in str(exc)
    else:  # pragma: no cover - assertion guard.
        raise AssertionError("expected hard-rejected dataset candidate to fail")


def test_campaign_class_subset_is_balanced_for_pilots() -> None:
    pilot = _select_campaign_classes(all_classes(include_min_distance=False), 3)
    assert len(pilot) == 3
    assert {spec.size for spec in pilot} == {"small"}
    assert {spec.occupancy for spec in pilot} == {"low", "high"}
    assert len({spec.shape for spec in pilot}) >= 2
    assert any(spec.shape == "elongated" for spec in pilot)

    selected = _select_campaign_classes(all_classes(include_min_distance=False), 6)
    assert len(selected) == 6
    assert {spec.occupancy for spec in selected} == {"low", "high"}
    assert len({spec.size for spec in selected}) >= 2
    assert len({spec.shape for spec in selected}) >= 2
    assert not all(spec.size == "medium" and spec.occupancy == "high" for spec in selected)


def test_suggest_fallback_and_contextual(tmp_path: Path) -> None:
    spec = InstanceClass("small", "low", "cubic", "homogeneous", False)
    instance = generate_instance(spec, 3)
    fallback = suggest_configs(instance, [], n_suggestions=2)
    assert fallback["mode"] == "random_fallback"
    assert len(fallback["suggestions"]) == 2

    # observações sintéticas: vetor x compatível (16 features + 17 config)
    from adaptive_3dbp.learning.online_gpr import observation_vector

    observations = []
    for index in range(6):
        config = SolverConfig(cooling_rate=0.9 + index * 0.01)
        x, _ = observation_vector(instance, config)
        observations.append(
            {
                "x_json": json.dumps(x),
                "target": 0.01 * (index + 1),
                "noise_variance": 1e-4,
                "config_json": config.model_dump_json(),
            }
        )
    contextual = suggest_configs(instance, observations, n_suggestions=2)
    assert contextual["mode"] == "contextual_ego"
    assert len(contextual["suggestions"]) == 2
    assert contextual["suggestions"][0]["expected_improvement"] is not None


def test_campaign_smoke_with_persistence(tmp_path: Path) -> None:
    repository = SQLiteExperimentRepository(tmp_path / "exp.db")
    spec = {
        "campaign_id": "test-campaign",
        "dataset": {
            "generate": {
                "per_class": {"train": 1, "validation": 1, "test": 1},
                "seed": 3,
                "include_min_distance": False,
                "classes_limit": 1,
                "curate": False,
            }
        },
        "seeds": [5],
        "trials": 1,
        "budget": {"max_evaluations": 40, "time_limit_seconds": 1.0},
        "tuners": ["fixed_default"],
        "output_dir": str(tmp_path / "campaign"),
        "persist": True,
    }
    summary = run_campaign(spec, repository=repository)
    assert summary["campaign_id"] == "test-campaign"
    arms = {arm["arm"]: arm for arm in summary["arms"]}
    assert arms["fixed_default"]["status"] == "ok"
    assert arms["fixed_default"]["validation_loss"] is not None
    assert Path(summary["artifacts"]["runs_csv"]).exists()
    assert Path(summary["artifacts"]["solutions_jsonl"]).exists()
    assert Path(summary["artifacts"]["metadata"]).exists()
    assert summary["dataset"]["source"] == "generated"
    assert summary["dataset"]["summary"] == {"train": 1, "validation": 1, "test": 1}
    assert Path(summary["artifacts"]["dataset_manifest"]).exists()
    # persistência com campaign_id
    import sqlite3

    connection = sqlite3.connect(tmp_path / "exp.db")
    count = connection.execute(
        "SELECT COUNT(*) FROM experiment_runs WHERE campaign_id = 'test-campaign'"
    ).fetchone()[0]
    assert count >= 2  # validation + test


def test_campaign_reports_progress_and_uses_fresh_eval_seeds(tmp_path: Path) -> None:
    events = []
    spec = {
        "campaign_id": "progress-campaign",
        "dataset": {
            "generate": {
                "per_class": {"train": 1, "validation": 1, "test": 1},
                "seed": 4,
                "include_min_distance": False,
                "classes_limit": 1,
                "curate": False,
            }
        },
        "seeds": [5],
        "trials": 1,
        "budget": {"max_evaluations": 25, "time_limit_seconds": 1.0},
        "tuners": ["fixed_default"],
        "output_dir": str(tmp_path / "campaign"),
        "persist": False,
    }

    summary = run_campaign(spec, progress_callback=events.append)

    solver_events = [event for event in events if event["stage"] == "solver"]
    assert solver_events
    assert summary["estimated_solver_runs"] == 2
    assert solver_events[-1]["completed_solver_runs"] == 2

    runs_csv = Path(summary["artifacts"]["runs_csv"])
    with runs_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["label"] for row in rows} == {"fixed_default:validation", "fixed_default:test"}
    assert {int(row["seed"]) for row in rows} == {10005, 20005}


def test_campaign_does_not_mark_timeout_validation_as_best(tmp_path: Path) -> None:
    outcome = ArmOutcome(
        arm="fixed_default",
        status="ok",
        validation_loss=11.0501,
        test_loss=8.8414,
        validation_metrics={
            "runs": 2,
            "feasible_runs": 0,
            "timeout_runs": 2,
            "error_runs": 0,
            "optimal_rate": 0.0,
            "mean_runtime_seconds": 149.0,
        },
        test_metrics={
            "runs": 2,
            "feasible_runs": 2,
            "timeout_runs": 0,
            "error_runs": 0,
            "optimal_rate": 0.0,
            "mean_bins_used": 4.0,
            "mean_gap_to_lower_bound": 0.0,
        },
    )
    table = export_results_table_tex(tmp_path, [outcome]).read_text(encoding="utf-8")
    assert "\\textbf" not in table
    assert "Campanha diagn\\'ostica" in table

    report = render_campaign_report(
        {
            "campaign_id": "timeout-campaign",
            "created_at": "2026-06-13T00:00:00-0300",
            "package_version": "test",
            "git_revision": None,
            "hardware": {},
            "seeds": [11],
            "validation_seeds": [10011],
            "test_seeds": [20011],
            "estimated_solver_runs": 4,
            "temperature_calibration_samples": 6,
            "dataset_summary": {"train": 1, "validation": 1, "test": 1},
        },
        [outcome],
    )
    assert "**(melhor)**" not in report
    assert "não foram viáveis" in report


def test_campaign_flags_non_discriminative_lower_bound_ties(tmp_path: Path) -> None:
    outcomes = [
        ArmOutcome(
            arm="fixed_default",
            status="ok",
            validation_loss=0.001,
            test_loss=0.001,
            validation_metrics={
                "runs": 2,
                "feasible_runs": 2,
                "timeout_runs": 0,
                "error_runs": 0,
                "optimal_rate": 1.0,
                "mean_bins_used": 4.0,
                "mean_gap_to_lower_bound": 0.0,
                "mean_evaluations": 8.0,
            },
            test_metrics={
                "runs": 2,
                "feasible_runs": 2,
                "timeout_runs": 0,
                "error_runs": 0,
                "optimal_rate": 1.0,
                "mean_bins_used": 4.0,
                "mean_gap_to_lower_bound": 0.0,
                "mean_evaluations": 8.0,
            },
        ),
        ArmOutcome(
            arm="random_search",
            status="ok",
            validation_loss=0.0011,
            test_loss=0.0011,
            validation_metrics={
                "runs": 2,
                "feasible_runs": 2,
                "timeout_runs": 0,
                "error_runs": 0,
                "optimal_rate": 1.0,
                "mean_bins_used": 4.0,
                "mean_gap_to_lower_bound": 0.0,
                "mean_evaluations": 8.0,
            },
            test_metrics={
                "runs": 2,
                "feasible_runs": 2,
                "timeout_runs": 0,
                "error_runs": 0,
                "optimal_rate": 1.0,
                "mean_bins_used": 4.0,
                "mean_gap_to_lower_bound": 0.0,
                "mean_evaluations": 8.0,
            },
        ),
    ]
    warning = _discrimination_warning(outcomes)
    assert warning is not None
    assert "pouco discriminativo" in warning
    table = export_results_table_tex(tmp_path, outcomes).read_text(encoding="utf-8")
    assert "\\textbf" not in table
    assert "dataset pouco discriminativo" in table


def test_campaign_flags_primary_quality_ties(tmp_path: Path) -> None:
    outcomes = [
        ArmOutcome(
            arm="fixed_default",
            status="ok",
            validation_loss=0.5065,
            test_loss=0.5104,
            validation_metrics={
                "runs": 15,
                "feasible_runs": 15,
                "timeout_runs": 0,
                "error_runs": 0,
                "optimal_rate": 0.0,
                "mean_bins_used": 3.8,
                "mean_gap_to_lower_bound": 0.4833333333,
                "mean_evaluations": 1200.0,
                "mean_runtime_seconds": 10.64,
            },
            test_metrics={
                "runs": 15,
                "feasible_runs": 15,
                "timeout_runs": 0,
                "error_runs": 0,
                "optimal_rate": 0.0,
                "mean_bins_used": 3.8,
                "mean_gap_to_lower_bound": 0.4833333333,
                "mean_evaluations": 1200.0,
                "mean_runtime_seconds": 15.48,
            },
        ),
        ArmOutcome(
            arm="noisy_contextual_ego",
            status="ok",
            validation_loss=0.5064,
            test_loss=0.5091,
            validation_metrics={
                "runs": 15,
                "feasible_runs": 15,
                "timeout_runs": 0,
                "error_runs": 0,
                "optimal_rate": 0.0,
                "mean_bins_used": 3.8,
                "mean_gap_to_lower_bound": 0.4833333333,
                "mean_evaluations": 1200.0,
                "mean_runtime_seconds": 10.41,
            },
            test_metrics={
                "runs": 15,
                "feasible_runs": 15,
                "timeout_runs": 0,
                "error_runs": 0,
                "optimal_rate": 0.0,
                "mean_bins_used": 3.8,
                "mean_gap_to_lower_bound": 0.4833333333,
                "mean_evaluations": 1200.0,
                "mean_runtime_seconds": 13.33,
            },
        ),
    ]

    warning = _discrimination_warning(outcomes)
    assert warning is not None
    assert "Qualidade primária não discriminou" in warning
    table = export_results_table_tex(tmp_path, outcomes).read_text(encoding="utf-8")
    assert "\\textbf" not in table
    assert "dataset pouco discriminativo" in table
