"""irace baseline integration via generated scenario files.

irace (Lopez-Ibanez et al., Operations Research Perspectives, 2016) implements
iterated racing with F-race statistical elimination and runs in R. This module
generates everything irace needs (parameters.txt, scenario.txt, instance list
and a Python target-runner) and can launch it when the ``irace`` executable is
on PATH; otherwise the generated directory documents the exact protocol for an
external run.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from adaptive_3dbp.domain.models import ProblemInstance
from adaptive_3dbp.tuning.space_conversion import to_irace_parameters

_EVALUATIONS_LOG = "evaluations.csv"
_STDOUT_LOG = "irace_stdout.log"
_STDERR_LOG = "irace_stderr.log"
_IRACE_LOGFILE = "irace.Rdata"  # default scenario logFile, relative to execDir
_ELITES_FILE = "irace_elites.txt"

# Marcadores da saída final do irace. Formato assumido (irace 3.x/4.x),
# conferido contra uma execução real do irace 4.4.3 e contra o fonte
# (configurations_print_command imprime UMA linha "<id> <comando>" por
# configuração via cat, nunca quebrada):
#
#   # Best configurations (first number is the configuration ID; listed from best to worst according to the sum of ranks):
#      cooling_rate ruin_size ...
#   48       0.9493         6 ...
#   # Best configurations as commandlines (first number is the configuration ID; ...):
#   48 --cooling_rate 0.9493 --ruin_size 6 ...
#
# Ao fim de cada iteração há também "# <data>: Elite configurations (first
# number is the configuration ID; ...)" e "Best-so-far configuration: <id>".
_FINAL_COMMANDLINES_MARKER = "# Best configurations as commandlines"
_FINAL_TABLE_MARKER = "# Best configurations (first number is the configuration ID"
_ELITE_TABLE_MARKER = "Elite configurations (first number is the configuration ID"
_BEST_SO_FAR = re.compile(r"^Best-so-far configuration:\s+(\d+)\b")
_ROW_ID = re.compile(r"^\s*(\d+)(?:\s+(.*))?$")

# Lê os elites finais direto de iraceResults (o mesmo que
# irace::getFinalElites() usa), sem depender do layout do stdout.
_EXPORT_ELITES_R = (
    f"e <- new.env(); load('{_IRACE_LOGFILE}', envir = e); r <- e$iraceResults; "
    f"writeLines(as.character(r$allElites[[length(r$allElites)]]), '{_ELITES_FILE}')"
)

_TARGET_RUNNER = '''#!{python_executable}
"""irace target-runner: prints a single cost (mean performance loss).

Every evaluation is also appended to evaluations.csv so the harness can read
the campaign back and compare irace against the other tuners on the same
shared objective (see adaptive_3dbp.tuning.compare).
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "{src_path}")

from adaptive_3dbp.domain.models import ProblemInstance
from adaptive_3dbp.tuning.objective import TuningObjective, values_to_config

def main() -> int:
    # irace calls: target-runner <id.configuration> <id.instance> <seed> <instance> [config args]
    configuration_id, instance_id, seed, instance_path = sys.argv[1:5]
    raw_args = sys.argv[5:]
    values = {{}}
    key = None
    for token in raw_args:
        if token.startswith("--"):
            key = token[2:]
        elif key is not None:
            values[key] = token
            key = None

    instance = ProblemInstance.model_validate(json.loads(Path(instance_path).read_text(encoding="utf-8")))
    objective = TuningObjective(
        instances=[instance],
        seeds=[int(seed)],
        max_evaluations={max_evaluations},
        time_limit_seconds={time_limit_seconds},
        temperature_calibration_samples={temperature_calibration_samples},
        use_native_core={use_native_core},
    )
    typed = {{}}
    for name, value in values.items():
        try:
            typed[name] = int(value)
        except ValueError:
            try:
                typed[name] = float(value)
            except ValueError:
                typed[name] = value
    cost = objective.evaluate_values(typed)

    log_path = Path(__file__).resolve().parent / "evaluations.csv"
    write_header = not log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        if write_header:
            writer.writerow(["configuration_id", "instance_id", "seed", "cost", "values_json"])
        writer.writerow([configuration_id, instance_id, seed, repr(cost), json.dumps(typed, sort_keys=True)])

    print(cost)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_SCENARIO = """# Generated irace scenario (Lopez-Ibanez et al., 2016).
parameterFile = "parameters.txt"
trainInstancesDir = "instances"
trainInstancesFile = "instances.txt"
targetRunner = "./target-runner.py"
maxExperiments = {max_experiments}
deterministic = 0
seed = {seed}
"""


def irace_executable() -> str | None:
    executable = shutil.which("irace")
    if executable:
        return executable
    rscript = shutil.which("Rscript")
    if not rscript:
        return None
    try:
        completed = subprocess.run(
            [
                rscript,
                "-e",
                "cat(system.file('bin', 'irace', package='irace'))",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    candidate = completed.stdout.strip()
    if candidate and Path(candidate).exists():
        if os.access(candidate, os.X_OK):
            return candidate
        return f"{shlex.quote(rscript)} {shlex.quote(candidate)}"
    return None


def irace_available() -> bool:
    return irace_executable() is not None


def _runner_interpreter() -> str:
    # irace executa o target-runner pelo shebang. "/usr/bin/env python3" pega o
    # python3 do PATH, que em geral NÃO é o venv do projeto (sem pydantic etc.)
    # e todo experimento falharia; fixa o interpretador que gerou a campanha.
    executable = sys.executable
    if executable and not any(ch.isspace() for ch in executable):
        return executable
    return "/usr/bin/env python3"


def generate_irace_campaign(
    output_dir: Path,
    search_space_definition: dict[str, Any],
    instances: list[ProblemInstance],
    src_path: Path,
    max_experiments: int = 300,
    max_evaluations: int = 1500,
    time_limit_seconds: float = 5.0,
    seed: int = 0,
    temperature_calibration_samples: int = 24,
    use_native_core: bool = False,
) -> Path:
    """Write an irace scenario whose target runner evaluates configurations with
    exactly the same fixed solver settings (budget, calibration, core) as the
    other campaign arms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Saídas de uma execução anterior no mesmo diretório carregariam IDs de
    # configuração de OUTRA corrida do irace — e a seleção do elite casa
    # justamente esses IDs com evaluations.csv. Começa sempre limpo.
    for stale in (_EVALUATIONS_LOG, _STDOUT_LOG, _STDERR_LOG, _IRACE_LOGFILE, _ELITES_FILE):
        (output_dir / stale).unlink(missing_ok=True)
    (output_dir / "parameters.txt").write_text(to_irace_parameters(search_space_definition), encoding="utf-8")
    (output_dir / "scenario.txt").write_text(
        _SCENARIO.format(max_experiments=max_experiments, seed=seed), encoding="utf-8"
    )
    instances_dir = output_dir / "instances"
    instances_dir.mkdir(exist_ok=True)
    names = []
    for instance in instances:
        name = f"{instance.instance_id}.json"
        (instances_dir / name).write_text(instance.model_dump_json(), encoding="utf-8")
        names.append(name)
    (output_dir / "instances.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    runner = output_dir / "target-runner.py"
    runner.write_text(
        _TARGET_RUNNER.format(
            python_executable=_runner_interpreter(),
            src_path=str(src_path),
            max_evaluations=max_evaluations,
            time_limit_seconds=time_limit_seconds,
            temperature_calibration_samples=int(temperature_calibration_samples),
            use_native_core=bool(use_native_core),
        ),
        encoding="utf-8",
    )
    runner.chmod(0o755)
    return output_dir


def run_irace(campaign_dir: Path, timeout_seconds: float | None = None) -> dict[str, Any]:  # pragma: no cover - requires R irace
    executable = irace_executable()
    if executable is None:
        raise RuntimeError("irace executable not found on PATH; install R package 'irace'")
    command = [executable] if Path(executable).exists() else shlex.split(executable)
    completed = subprocess.run(
        [*command, "--scenario", "scenario.txt"],
        cwd=campaign_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    (campaign_dir / _STDOUT_LOG).write_text(completed.stdout, encoding="utf-8")
    (campaign_dir / _STDERR_LOG).write_text(completed.stderr, encoding="utf-8")
    elites_exported = completed.returncode == 0 and export_final_elites(campaign_dir)
    # irace aborta com "Error: == irace == ..." no stderr (ex.: "Insufficient
    # budget: ... maxExperiments of at least '300'"); devolve a mensagem para o
    # chamador reportar o braço como pulado em vez de "sem avaliações".
    error_lines = [line.strip() for line in completed.stderr.splitlines() if "Error" in line]
    return {
        "tuner": "irace",
        "returncode": completed.returncode,
        "error": (error_lines[-1] if error_lines else completed.stderr.strip()[-500:] or None)
        if completed.returncode != 0
        else None,
        "elites_exported": elites_exported,
        "stdout_tail": completed.stdout[-2000:],
        "campaign_dir": str(campaign_dir),
    }


def export_final_elites(campaign_dir: Path, timeout_seconds: float = 60.0) -> bool:  # pragma: no cover - requires R irace
    """Dump irace's final elite IDs (best first) from irace.Rdata.

    Writes ``irace_elites.txt`` with ``iraceResults$allElites[[last]]`` — the
    list irace::getFinalElites() reads — so the incumbent does not depend on
    the stdout layout. Best-effort: False when Rscript or the log is missing.
    """
    rscript = shutil.which("Rscript")
    if rscript is None or not (campaign_dir / _IRACE_LOGFILE).exists():
        return False
    try:
        completed = subprocess.run(
            [rscript, "-e", _EXPORT_ELITES_R],
            cwd=campaign_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and (campaign_dir / _ELITES_FILE).exists()


def load_campaign_evaluations(campaign_dir: Path) -> list[dict[str, Any]]:
    """Read back every (config, instance, seed, cost) evaluated by irace."""
    import csv

    log_path = campaign_dir / _EVALUATIONS_LOG
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle, delimiter=";"):
            try:
                rows.append(
                    {
                        "configuration_id": record["configuration_id"],
                        "instance_id": record["instance_id"],
                        "seed": int(record["seed"]),
                        "cost": float(record["cost"]),
                        "values": json.loads(record["values_json"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def _typed_value(value: str) -> Any:
    # Mesma coerção do target-runner (int -> float -> str).
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_commandline(text: str) -> dict[str, Any]:
    """``--name value`` (switch ``"--name "``) or ``--name=value`` tokens."""
    values: dict[str, Any] = {}
    key = None
    for token in text.split():
        if token.startswith("--"):
            name, sep, value = token[2:].partition("=")
            if sep:
                values[name] = _typed_value(value)
                key = None
            else:
                key = name
        elif key is not None:
            values[key] = _typed_value(token)
            key = None
    return values


def _last_block(lines: list[str], marker: str) -> list[str]:
    """Lines after the LAST line containing ``marker``, up to the next comment
    (``#``) or blank line."""
    start = None
    for index, line in enumerate(lines):
        if marker in line:
            start = index
    if start is None:
        return []
    block = []
    for line in lines[start + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            break
        block.append(line)
    return block


def _unique_ids(elites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Tabelas largas podem sair quebradas pelo R em blocos de colunas, cada
    # bloco repetindo os IDs: mantém a primeira ocorrência, na ordem.
    seen: set[str] = set()
    unique = []
    for elite in elites:
        if elite["configuration_id"] not in seen:
            seen.add(elite["configuration_id"])
            unique.append(elite)
    return unique


def parse_irace_elites(stdout_text: str) -> list[dict[str, Any]]:
    """irace's final elite configurations from its stdout, best first.

    Returns ``[{"configuration_id": "48", "values": {...} | None}, ...]``.
    Sources, in order (format documented at the module markers above): the
    final "Best configurations as commandlines" block (IDs + values); the
    final "Best configurations" table (IDs only — R may wrap wide tables); the
    last per-iteration "Elite configurations" table (run interrupted before
    the final summary); the last "Best-so-far configuration: <id>" line.
    """
    lines = stdout_text.splitlines()
    elites = []
    for line in _last_block(lines, _FINAL_COMMANDLINES_MARKER):
        match = _ROW_ID.match(line)
        if match:
            elites.append({"configuration_id": match.group(1), "values": _parse_commandline(match.group(2) or "")})
    if elites:
        return _unique_ids(elites)
    for marker in (_FINAL_TABLE_MARKER, _ELITE_TABLE_MARKER):
        ids = [match.group(1) for line in _last_block(lines, marker) if (match := _ROW_ID.match(line))]
        if ids:
            return _unique_ids([{"configuration_id": identifier, "values": None} for identifier in ids])
    for line in reversed(lines):
        match = _BEST_SO_FAR.match(line)
        if match:
            return [{"configuration_id": match.group(1), "values": None}]
    return []


def load_irace_elites(campaign_dir: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Final elites (best first) and where they came from.

    ``irace_elites.txt`` (exported from irace.Rdata by :func:`export_final_elites`)
    is authoritative; otherwise the saved stdout is parsed. Values parsed from
    the stdout commandlines are attached when available.
    """
    stdout_path = campaign_dir / _STDOUT_LOG
    parsed = (
        parse_irace_elites(stdout_path.read_text(encoding="utf-8", errors="replace")) if stdout_path.exists() else []
    )
    elites_path = campaign_dir / _ELITES_FILE
    if elites_path.exists():
        ids = [line.strip() for line in elites_path.read_text(encoding="utf-8").splitlines() if line.strip().isdigit()]
        if ids:
            values_by_id = {elite["configuration_id"]: elite["values"] for elite in parsed}
            return [{"configuration_id": identifier, "values": values_by_id.get(identifier)} for identifier in ids], "irace.Rdata"
    if parsed:
        return parsed, "stdout"
    return [], None


def best_values_from_campaign(campaign_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """irace's own incumbent: the best of its final elite configurations.

    Returns (values, stats). The elite IDs come from irace itself
    (:func:`load_irace_elites`) and are matched to evaluations.csv through
    the configuration ID the target-runner logs, so ``values`` are exactly
    what irace evaluated. Ranking the log by mean cost instead (the previous
    behaviour) compares configurations raced on DIFFERENT instance subsets: a
    newcomer eliminated after a few lucky instances beats the elite that
    survived the whole race (selection bias on noisy costs).

    Fallback when irace's output is missing/unparseable: among the most-raced
    configurations (elites accumulate the most experiments) take the best mean
    over the experiments they ALL share — flagged ``selection="fallback_most_raced"``.
    ``best_mean_cost`` is the mean over all of the chosen configuration's
    evaluations, or None when it has none in the log.
    """
    rows = load_campaign_evaluations(campaign_dir)
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(str(row["configuration_id"]), []).append(row)
    stats: dict[str, Any] = {"evaluations": len(rows), "distinct_configurations": len(by_id)}

    elites, source = load_irace_elites(campaign_dir)
    for elite in elites:
        elite_rows = by_id.get(elite["configuration_id"])
        if not elite_rows and not elite["values"]:
            continue
        costs = [row["cost"] for row in elite_rows or []]
        return (elite_rows[0]["values"] if elite_rows else elite["values"]), {
            **stats,
            "selection": "irace_elite",
            "elite_source": source,
            "configuration_id": elite["configuration_id"],
            "final_elites": [item["configuration_id"] for item in elites],
            "best_mean_cost": statistics.fmean(costs) if costs else None,
            "best_replications": len(costs),
        }

    if not rows:
        return None, {"evaluations": 0}
    experiments = {
        identifier: {(row["instance_id"], row["seed"]): row["cost"] for row in id_rows}
        for identifier, id_rows in by_id.items()
    }
    most_raced = max(len(costs) for costs in experiments.values())
    candidates = [identifier for identifier, costs in experiments.items() if len(costs) == most_raced]
    shared = set.intersection(*(set(experiments[identifier]) for identifier in candidates))

    def shared_mean(identifier: str) -> float:
        keys = shared or experiments[identifier].keys()
        return statistics.fmean(experiments[identifier][key] for key in keys)

    best = min(
        candidates,
        key=lambda identifier: (shared_mean(identifier), int(identifier) if identifier.isdigit() else float("inf")),
    )
    costs = [row["cost"] for row in by_id[best]]
    return by_id[best][0]["values"], {
        **stats,
        "selection": "fallback_most_raced",
        "configuration_id": best,
        "shared_experiments": len(shared),
        "best_mean_cost": statistics.fmean(costs),
        "best_replications": len(costs),
    }


def save_summary(campaign_dir: Path, summary: dict[str, Any]) -> None:
    (campaign_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
