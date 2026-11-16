from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adaptive_3dbp.domain.models import ProblemInstance, SolverConfig
from adaptive_3dbp.domain.validation import validate_problem_instance
from adaptive_3dbp.experiments.runner import solve_instance
from adaptive_3dbp.io.spreadsheet import instance_from_spreadsheet, write_instance_template
from adaptive_3dbp.persistence.sqlite import SQLiteExperimentRepository


def _load_instance(path: Path) -> ProblemInstance:
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return instance_from_spreadsheet(path)
    return ProblemInstance.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _cmd_validate(args: argparse.Namespace) -> int:
    instance = _load_instance(Path(args.instance))
    diagnostics = validate_problem_instance(instance)
    print(json.dumps({"valid": not diagnostics, "diagnostics": diagnostics}, indent=2))
    return 0 if not diagnostics else 2


def _cmd_solve(args: argparse.Namespace) -> int:
    instance = _load_instance(Path(args.instance))
    config = SolverConfig(
        max_evaluations=args.max_evaluations,
        time_limit_seconds=args.time_limit_seconds,
        use_native_core=args.native,
        parallel_mode=args.parallel_mode,
        parallel_workers=args.parallel_workers,
        multi_start_count=args.multi_start_count,
        convergence_sample_rate=args.convergence_sample_rate,
    )
    result = solve_instance(instance, config, args.seed)
    if args.persist:
        repository = SQLiteExperimentRepository(args.database)
        run_id = repository.save_run(instance, config, result)
        learning = repository.observe_and_train(
            run_id,
            instance,
            config,
            result,
            Path(args.database).parent / "models",
        )
        result = result.model_copy(
            update={
                "artifacts": {**result.artifacts, "run_id": str(run_id)},
                "diagnostics": [*result.diagnostics, f"online GPR update: {learning}"],
            }
        )
    print(result.model_dump_json(indent=2))
    return 0 if result.status == "feasible" else 1


def _cmd_import_spreadsheet(args: argparse.Namespace) -> int:
    instance = instance_from_spreadsheet(Path(args.spreadsheet))
    diagnostics = validate_problem_instance(instance)
    payload = instance.model_dump_json(indent=2)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
        print(f"instance written to {args.out}")
    else:
        print(payload)
    if diagnostics:
        print(json.dumps({"diagnostics": diagnostics}, indent=2), file=sys.stderr)
        return 2
    return 0


def _cmd_spreadsheet_template(args: argparse.Namespace) -> int:
    path = write_instance_template(Path(args.out))
    print(f"template written to {path}")
    return 0


def _cmd_compare_tuners(args: argparse.Namespace) -> int:
    from adaptive_3dbp.tuning.compare import run_comparison
    from adaptive_3dbp.tuning.objective import TuningObjective

    instances = [_load_instance(Path(path)) for path in args.instance]
    seeds = [int(part) for part in str(args.seeds).replace(";", ",").split(",") if part.strip()]
    objective = TuningObjective(
        instances=instances,
        seeds=seeds,
        max_evaluations=args.max_evaluations,
        time_limit_seconds=args.time_limit_seconds,
    )
    output_dir = Path(args.out)
    results = run_comparison(
        objective,
        json.loads(Path(args.search_space).read_text(encoding="utf-8")),
        n_trials=args.trials,
        seed=args.seed,
        output_dir=output_dir,
        include_smac=not args.no_smac,
        include_irace=not args.no_irace,
    )
    print(results["report_markdown"])
    print(f"relatorio completo: {output_dir / 'comparison.md'} e {output_dir / 'comparison.json'}")
    return 0


def _cmd_generate_dataset(args: argparse.Namespace) -> int:
    from adaptive_3dbp.experiments.instances import generate_dataset

    per_class = {"train": args.train, "validation": args.validation, "test": args.test}
    manifest = generate_dataset(
        Path(args.out),
        per_class=per_class,
        seed=args.seed,
        include_min_distance=not args.no_min_distance,
    )
    counts = {split: len(entries) for split, entries in manifest["splits"].items()}
    print(json.dumps({"out": args.out, "classes": len(manifest["classes"]), "instances": counts}, indent=2))
    return 0


def _cmd_run_campaign(args: argparse.Namespace) -> int:
    from adaptive_3dbp.experiments.campaign import run_campaign
    from adaptive_3dbp.persistence.sqlite import SQLiteExperimentRepository

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    repository = SQLiteExperimentRepository(args.database) if spec.get("persist") else None

    def progress(update: dict) -> None:
        print(f"[{update['fraction']*100:5.1f}%] {update['stage']}: {update['detail']}")

    summary = run_campaign(spec, progress_callback=progress, repository=repository)
    print(summary["report_markdown"])
    print(f"artefatos: {summary['artifacts']['output_dir']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptive-3dbp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-instance")
    validate.add_argument("instance")
    validate.set_defaults(func=_cmd_validate)

    import_xlsx = subparsers.add_parser("import-spreadsheet", help="convert an .xlsx instance to JSON")
    import_xlsx.add_argument("spreadsheet")
    import_xlsx.add_argument("--out", default=None, help="output JSON path (default: stdout)")
    import_xlsx.set_defaults(func=_cmd_import_spreadsheet)

    template = subparsers.add_parser("spreadsheet-template", help="generate a template .xlsx instance")
    template.add_argument("--out", default="data/raw/instance_template.xlsx")
    template.set_defaults(func=_cmd_spreadsheet_template)

    compare = subparsers.add_parser(
        "compare-tuners",
        help="compare Random Search, EGOs, SMAC3 and irace on the same objective/budget",
    )
    compare.add_argument("--instance", action="append", required=True, help="instance path (.json/.xlsx); repeatable")
    compare.add_argument("--trials", type=int, default=20, help="trials per tuner (each = instances x seeds solver runs)")
    compare.add_argument("--seeds", default="11,23,47", help="tuning seeds, comma-separated")
    compare.add_argument("--seed", type=int, default=0, help="tuner RNG seed")
    compare.add_argument("--max-evaluations", type=int, default=1500)
    compare.add_argument("--time-limit-seconds", type=float, default=5.0)
    compare.add_argument("--search-space", default="configs/search_spaces/sa_search_space.json")
    compare.add_argument("--out", default="artifacts/tuning/comparison")
    compare.add_argument("--no-smac", action="store_true")
    compare.add_argument("--no-irace", action="store_true")
    compare.set_defaults(func=_cmd_compare_tuners)

    dataset = subparsers.add_parser("generate-dataset", help="generate stratified instance dataset with splits")
    dataset.add_argument("--out", default="data/datasets/default")
    dataset.add_argument("--seed", type=int, default=0)
    dataset.add_argument("--train", type=int, default=2, help="instances per class in train")
    dataset.add_argument("--validation", type=int, default=1)
    dataset.add_argument("--test", type=int, default=1)
    dataset.add_argument("--no-min-distance", action="store_true")
    dataset.set_defaults(func=_cmd_generate_dataset)

    campaign = subparsers.add_parser("run-campaign", help="run a reproducible experimental campaign from a spec JSON")
    campaign.add_argument("--spec", required=True, help="campaign spec JSON (see configs/campaigns/pilot.json)")
    campaign.add_argument("--database", default="artifacts/experiments.db")
    campaign.set_defaults(func=_cmd_run_campaign)

    solve = subparsers.add_parser("solve")
    solve.add_argument("--instance", required=True)
    solve.add_argument("--seed", type=int, default=123)
    solve.add_argument("--max-evaluations", type=int, default=2500)
    solve.add_argument("--time-limit-seconds", type=float, default=10.0)
    solve.add_argument("--native", action="store_true")
    solve.add_argument("--multi-start-count", type=int, default=3)
    solve.add_argument(
        "--parallel-mode",
        choices=["sequential", "process_pool", "cooperative_process_pool"],
        default="sequential",
    )
    solve.add_argument("--parallel-workers", type=int, default=1)
    solve.add_argument("--convergence-sample-rate", type=int, default=1)
    solve.add_argument("--persist", action="store_true")
    solve.add_argument("--database", default="artifacts/experiments.db")
    solve.set_defaults(func=_cmd_solve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
