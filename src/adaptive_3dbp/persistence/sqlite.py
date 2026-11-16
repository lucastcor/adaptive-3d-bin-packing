from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from adaptive_3dbp import __version__
from adaptive_3dbp.domain.models import ProblemInstance, SolverConfig, SolverResult
from adaptive_3dbp.learning.online_gpr import build_learning_observation, train_online_gpr


def _version_prefix() -> str:
    """'0.3.' for version 0.3.x: observations are comparable within a minor."""
    major, minor, *_rest = __version__.split(".")
    return f"{major}.{minor}."


class SQLiteExperimentRepository:
    def __init__(self, db_path: Path | str = "artifacts/experiments.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instance_id TEXT NOT NULL,
                    solver TEXT NOT NULL,
                    solver_version TEXT NOT NULL,
                    configuration_id TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    bins_used INTEGER NOT NULL,
                    lower_bound INTEGER NOT NULL,
                    best_known_bins INTEGER,
                    gap_to_lower_bound REAL NOT NULL,
                    gap_to_best_known REAL,
                    runtime_seconds REAL NOT NULL,
                    evaluations INTEGER NOT NULL,
                    objective_scalar REAL NOT NULL,
                    instance_json TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS placements (
                    placement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    item_id TEXT NOT NULL,
                    item_index INTEGER NOT NULL,
                    bin_id INTEGER NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    z REAL NOT NULL,
                    rotation_id INTEGER NOT NULL,
                    placed_length REAL NOT NULL,
                    placed_width REAL NOT NULL,
                    placed_height REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS learning_observations (
                    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL UNIQUE,
                    instance_id TEXT NOT NULL,
                    x_json TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    target REAL NOT NULL,
                    noise_variance REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS surrogate_models (
                    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    observation_count INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            # Migração leve: agrupar execuções por campanha/experimento
            # (ALTER TABLE idempotente — colunas só são criadas se ausentes).
            existing = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(experiment_runs)").fetchall()
            }
            for column in ("campaign_id", "experiment_id"):
                if column not in existing:
                    connection.execute(f"ALTER TABLE experiment_runs ADD COLUMN {column} TEXT")

    def save_run(
        self,
        instance: ProblemInstance,
        config: SolverConfig,
        result: SolverResult,
        campaign_id: str | None = None,
        experiment_id: str | None = None,
    ) -> int:
        record = result.as_experiment_record()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO experiment_runs (
                    instance_id, solver, solver_version, configuration_id, seed, status,
                    bins_used, lower_bound, best_known_bins, gap_to_lower_bound, gap_to_best_known,
                    runtime_seconds, evaluations, objective_scalar, instance_json, config_json, result_json,
                    campaign_id, experiment_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["instance_id"],
                    record["solver"],
                    record["solver_version"],
                    record["configuration_id"],
                    record["seed"],
                    record["status"],
                    record["best_found_bins"],
                    record["lower_bound"],
                    record["best_known_bins"],
                    record["gap_to_lower_bound"],
                    record["gap_to_best_known"],
                    record["runtime_seconds"],
                    record["evaluations"],
                    record["objective_scalar"],
                    instance.model_dump_json(),
                    config.model_dump_json(),
                    result.model_dump_json(),
                    campaign_id,
                    experiment_id,
                ),
            )
            run_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO placements (
                    run_id, item_id, item_index, bin_id, x, y, z, rotation_id,
                    placed_length, placed_width, placed_height
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        placement.item_id,
                        placement.item_index,
                        placement.bin_id,
                        placement.x,
                        placement.y,
                        placement.z,
                        placement.rotation_id,
                        placement.placed_length,
                        placement.placed_width,
                        placement.placed_height,
                    )
                    for placement in result.placements
                ],
            )
        return run_id

    def save_learning_observation(self, run_id: int, instance: ProblemInstance, config: SolverConfig, result: SolverResult) -> None:
        observation = build_learning_observation(run_id, instance, config, result)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO learning_observations (
                    run_id, instance_id, x_json, context_json, config_json, result_json,
                    target, noise_variance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.run_id,
                    observation.instance_id,
                    json.dumps(observation.x),
                    observation.context_json,
                    observation.config_json,
                    observation.result_json,
                    observation.target,
                    observation.noise_variance,
                ),
            )

    def train_online_surrogate(self, artifact_root: Path | str = "artifacts/models") -> dict[str, object]:
        observations = self.list_learning_observations(limit=5000)
        metadata = train_online_gpr(observations, Path(artifact_root))
        if metadata.get("trained"):
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO surrogate_models (
                        model_type, artifact_path, observation_count, metrics_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        str(metadata["model_type"]),
                        str(metadata["artifact_path"]),
                        int(metadata["observation_count"]),
                        json.dumps(metadata, sort_keys=True),
                    ),
                )
        return metadata

    def observe_and_train(
        self,
        run_id: int,
        instance: ProblemInstance,
        config: SolverConfig,
        result: SolverResult,
        artifact_root: Path | str = "artifacts/models",
    ) -> dict[str, object]:
        self.save_learning_observation(run_id, instance, config, result)
        return self.train_online_surrogate(artifact_root)

    def list_runs(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, instance_id, solver, configuration_id, seed, status,
                       bins_used, lower_bound, runtime_seconds, evaluations, created_at
                FROM experiment_runs
                ORDER BY run_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_learning_observations(self, limit: int = 5000, compatible_only: bool = True) -> list[dict[str, Any]]:
        """Learning observations, by default only those of the current solver
        major.minor version. 0.2 changed the energy, the T0 calibration and the
        loss (no runtime term): older targets are not comparable and would be
        mixed into the online GP."""
        version_filter = "WHERE json_extract(result_json, '$.solver_version') LIKE ?" if compatible_only else ""
        params: tuple[Any, ...] = (limit,)
        if compatible_only:
            params = (_version_prefix() + "%", limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT observation_id, run_id, instance_id, x_json, context_json, config_json,
                       result_json, target, noise_variance, created_at
                FROM learning_observations
                {version_filter}
                ORDER BY observation_id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_surrogate_model(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT model_id, model_type, artifact_path, observation_count, metrics_json, created_at
                FROM surrogate_models
                ORDER BY model_id DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["metrics"] = json.loads(data["metrics_json"])
        return data

    def get_result(self, run_id: int) -> SolverResult | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM experiment_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return SolverResult.model_validate(json.loads(row["result_json"]))

    def get_run_bundle(self, run_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT instance_json, config_json, result_json
                FROM experiment_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "instance": json.loads(row["instance_json"]),
            "config": json.loads(row["config_json"]),
            "result": json.loads(row["result_json"]),
        }
