from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adaptive_3dbp.api import main as api_main
from adaptive_3dbp.api.main import app
from adaptive_3dbp.persistence.sqlite import SQLiteExperimentRepository


ROOT = Path(__file__).resolve().parents[2]


def test_api_allows_local_vite_dev_ports() -> None:
    client = TestClient(app)
    origin = "http://127.0.0.1:5174"
    response = client.options(
        "/api/campaigns",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_api_solve_endpoint() -> None:
    client = TestClient(app)
    instance = json.loads((ROOT / "data/raw/example_20.json").read_text())
    response = client.post(
        "/api/solve",
        json={
            "instance": instance,
            "config": {
                "max_evaluations": 80,
                "iterations_per_temperature": 20,
                "time_limit_seconds": 5,
                "use_native_core": False,
            },
            "seed": 11,
            "persist": False,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["result"]["status"] == "feasible"


def test_api_solve_endpoint_with_min_distance_constraint() -> None:
    client = TestClient(app)
    instance = {
        "instance_id": "distance-api",
        "problem_type": "offline_3d_bpp",
        "bin": {"length": 20, "width": 10, "height": 10, "max_weight": None},
        "items": [
            {"id": "a", "length": 2, "width": 2, "height": 2, "weight": None, "allowed_rotations": "fixed"},
            {"id": "b", "length": 2, "width": 2, "height": 2, "weight": None, "allowed_rotations": "fixed"},
        ],
        "constraints": {
            "orthogonal_only": True,
            "stability_required": False,
            "support_ratio_min": None,
            "min_distance": [{"item_id_a": "a", "item_id_b": "b", "distance": 3}],
        },
    }
    response = client.post(
        "/api/solve",
        json={
            "instance": instance,
            "config": {
                "max_evaluations": 20,
                "iterations_per_temperature": 5,
                "time_limit_seconds": 5,
                "use_native_core": True,
            },
            "seed": 5,
            "persist": False,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["result"]["status"] == "feasible"
    assert payload["result"]["solver"] == "simulated_annealing_cpp_core"


def test_api_learning_status_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/api/learning/status")
    assert response.status_code == 200
    payload = response.json()
    assert "observation_count" in payload
    assert "latest_model" in payload


def test_api_result_spreadsheet_export_endpoint() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    client = TestClient(app)
    instance = {
        "instance_id": "solution-export-api",
        "problem_type": "offline_3d_bpp",
        "bin": {"length": 20, "width": 10, "height": 10, "max_weight": None},
        "items": [
            {"id": "a", "length": 2, "width": 2, "height": 2, "weight": None, "allowed_rotations": "fixed"},
            {"id": "b", "length": 3, "width": 2, "height": 2, "weight": None, "allowed_rotations": "fixed"},
        ],
        "constraints": {
            "orthogonal_only": True,
            "stability_required": False,
            "support_ratio_min": None,
            "min_distance": [],
        },
    }
    solve_response = client.post(
        "/api/solve",
        json={
            "instance": instance,
            "config": {
                "max_evaluations": 12,
                "iterations_per_temperature": 4,
                "time_limit_seconds": 5,
                "use_native_core": False,
            },
            "seed": 19,
            "persist": False,
        },
    )
    assert solve_response.status_code == 200, solve_response.text
    result = solve_response.json()["result"]

    export_response = client.post(
        "/api/results/export-spreadsheet",
        json={"instance": instance, "result": result, "run_id": None},
    )
    assert export_response.status_code == 200, export_response.text
    assert "spreadsheetml.sheet" in export_response.headers["content-type"]
    workbook = openpyxl.load_workbook(io.BytesIO(export_response.content), data_only=True)
    assert {"resumo", "placements", "bin_metrics", "convergence"}.issubset(set(workbook.sheetnames))
    assert workbook["placements"].max_row == len(instance["items"]) + 1
    assert workbook["placements"]["A2"].value in {"a", "b"}


def test_api_solve_job_lifecycle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api_main, "repository", SQLiteExperimentRepository(tmp_path / "experiments.db"))
    client = TestClient(api_main.app)
    instance = {
        "instance_id": "job-api",
        "problem_type": "offline_3d_bpp",
        "bin": {"length": 20, "width": 10, "height": 10, "max_weight": None},
        "items": [
            {"id": "a", "length": 2, "width": 2, "height": 2, "weight": None, "allowed_rotations": "fixed"},
            {"id": "b", "length": 3, "width": 2, "height": 2, "weight": None, "allowed_rotations": "fixed"},
        ],
        "constraints": {
            "orthogonal_only": True,
            "stability_required": False,
            "support_ratio_min": None,
            "min_distance": [],
        },
    }
    response = client.post(
        "/api/solve-jobs",
        json={
            "instance": instance,
            "config": {
                "max_evaluations": 40,
                "iterations_per_temperature": 10,
                "time_limit_seconds": 5,
                "use_native_core": False,
            },
            "seed": 23,
            "persist": True,
        },
    )
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["job_id"]
    assert job["pause_supported"] is True

    pause_response = client.post(f"/api/solve-jobs/{job['job_id']}/pause")
    assert pause_response.status_code == 200, pause_response.text
    if pause_response.json()["status"] == "paused":
        resume_response = client.post(f"/api/solve-jobs/{job['job_id']}/resume")
        assert resume_response.status_code == 200, resume_response.text

    for _ in range(30):
        status_response = client.get(f"/api/solve-jobs/{job['job_id']}")
        assert status_response.status_code == 200, status_response.text
        job = status_response.json()
        if job["status"] in {"completed", "error", "cancelled"}:
            break
        time.sleep(0.05)
    assert job["status"] == "completed", job
    assert job["result"]["status"] == "feasible"
    assert job["run_id"] is not None


def test_api_run_bundle_endpoint(tmp_path: Path, monkeypatch) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    monkeypatch.setattr(api_main, "repository", SQLiteExperimentRepository(tmp_path / "experiments.db"))
    client = TestClient(api_main.app)
    instance = json.loads((ROOT / "data/raw/example_20.json").read_text())
    response = client.post(
        "/api/solve",
        json={
            "instance": instance,
            "config": {
                "max_evaluations": 20,
                "iterations_per_temperature": 5,
                "time_limit_seconds": 5,
                "use_native_core": False,
            },
            "seed": 17,
            "persist": True,
        },
    )
    assert response.status_code == 200, response.text
    run_id = response.json()["run_id"]

    bundle_response = client.get(f"/api/runs/{run_id}/bundle")
    assert bundle_response.status_code == 200, bundle_response.text
    bundle = bundle_response.json()
    assert bundle["instance"]["instance_id"] == instance["instance_id"]
    assert bundle["result"]["status"] == "feasible"
    assert bundle["config"]["max_evaluations"] == 20

    export_response = client.get(f"/api/runs/{run_id}/result-spreadsheet")
    assert export_response.status_code == 200, export_response.text
    workbook = openpyxl.load_workbook(io.BytesIO(export_response.content), data_only=True)
    assert workbook["resumo"]["B2"].value == run_id
    assert workbook["placements"].max_row >= 2


# ------------------------------------------------- campaign path confinement


@pytest.fixture
def campaign_sandbox(tmp_path: Path, monkeypatch):
    """Point the API at a throwaway project root and capture submitted jobs
    instead of running campaigns."""
    root = tmp_path / "project"
    (root / "artifacts" / "campaigns").mkdir(parents=True)
    (root / "data" / "datasets").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.json").write_text('{"token": "do-not-leak"}', encoding="utf-8")
    monkeypatch.setattr(api_main, "ROOT_DIR", root)
    monkeypatch.setattr(api_main, "CAMPAIGN_JOBS_DIR", root / "artifacts" / "campaigns" / "_jobs")
    monkeypatch.setattr(api_main, "campaign_jobs", {})
    submitted: list = []

    class _RecordingExecutor:
        def submit(self, fn, *args):
            submitted.append(args[0])

    monkeypatch.setattr(api_main, "campaign_executor", _RecordingExecutor())
    return root, outside, submitted


def _write_manifest(path: Path, entry_paths: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    splits = {"train": [{"path": entry} for entry in entry_paths], "validation": [], "test": []}
    path.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "spec",
    [
        {"output_dir": "/tmp/adaptive-3dbp-escape"},
        {"output_dir": "artifacts/../../escape"},
        {"output_dir": "artifacts/campaigns/../../../escape"},
        {"output_dir": "artifacts"},  # a raiz em si não é diretório de campanha
        {"output_dir": "~/escape"},
        {"output_dir": ["artifacts/x"]},
        {"output_dir": "artifacts/x\x00y"},
        {"campaign_id": "../../escape"},
        {"campaign_id": "a/b"},
        {"dataset": {"manifest": "/etc/passwd"}},
        {"dataset": {"manifest": "data/../../escape/manifest.json"}},
        {"dataset": "manifest"},
    ],
)
def test_campaign_start_rejects_paths_outside_allowed_roots(campaign_sandbox, spec: dict) -> None:
    root, _outside, submitted = campaign_sandbox
    response = TestClient(api_main.app).post("/api/campaigns", json=spec)
    assert response.status_code == 400, response.text
    assert submitted == []
    assert not (root / "artifacts" / "campaigns" / "_jobs").exists()


def test_campaign_start_rejects_symlink_escapes(campaign_sandbox) -> None:
    root, outside, submitted = campaign_sandbox
    (root / "data" / "link").symlink_to(outside, target_is_directory=True)
    _write_manifest(outside / "manifest.json", [str(outside / "secret.json")])
    client = TestClient(api_main.app)

    escaped_output = client.post("/api/campaigns", json={"output_dir": "data/link/run"})
    escaped_manifest = client.post("/api/campaigns", json={"dataset": {"manifest": "data/link/manifest.json"}})

    assert escaped_output.status_code == 400, escaped_output.text
    assert escaped_manifest.status_code == 400, escaped_manifest.text
    assert submitted == []


def test_campaign_start_rejects_manifest_entries_outside_roots(campaign_sandbox) -> None:
    root, outside, submitted = campaign_sandbox
    inside_instance = root / "data" / "datasets" / "d1" / "a.json"
    inside_instance.parent.mkdir(parents=True)
    inside_instance.write_text("{}", encoding="utf-8")
    client = TestClient(api_main.app)
    escaping_entries = [
        str(outside / "secret.json"),
        str(root / "data" / "datasets" / ".." / ".." / ".." / "outside" / "secret.json"),
    ]
    for index, entry in enumerate(escaping_entries):
        manifest = _write_manifest(root / "data" / "datasets" / f"bad{index}" / "manifest.json", [str(inside_instance), entry])
        response = client.post("/api/campaigns", json={"dataset": {"manifest": str(manifest)}})
        assert response.status_code == 400, response.text
        assert "splits.train[1]" in response.json()["detail"]
    missing = client.post("/api/campaigns", json={"dataset": {"manifest": "data/datasets/none/manifest.json"}})
    assert missing.status_code == 400
    assert submitted == []


def test_campaign_start_accepts_and_pins_paths_inside_roots(campaign_sandbox) -> None:
    root, _outside, submitted = campaign_sandbox
    instance_path = root / "data" / "datasets" / "d1" / "a.json"
    instance_path.parent.mkdir(parents=True)
    instance_path.write_text("{}", encoding="utf-8")
    _write_manifest(root / "data" / "datasets" / "d1" / "manifest.json", [str(instance_path)])
    client = TestClient(api_main.app)

    response = client.post(
        "/api/campaigns",
        json={"output_dir": "artifacts/campaigns/custom", "dataset": {"manifest": "data/datasets/d1/manifest.json"}},
    )
    default = client.post("/api/campaigns", json={"campaign_id": "pilot-1"})

    assert response.status_code == 200, response.text
    assert default.status_code == 200, default.text
    custom_spec, default_spec = (job.spec for job in submitted)
    # Caminhos gravados resolvidos e absolutos: run_campaign abre exatamente
    # o que foi validado, independentemente do CWD do servidor.
    assert custom_spec["output_dir"] == str((root / "artifacts" / "campaigns" / "custom").resolve())
    assert custom_spec["dataset"]["manifest"] == str((root / "data" / "datasets" / "d1" / "manifest.json").resolve())
    assert default_spec["output_dir"] == str((root / "artifacts" / "campaigns" / "pilot-1").resolve())


def _register_campaign(summary: dict) -> str:
    job = api_main.CampaignJob(job_id="f" * 32, spec={}, status="completed", summary=summary)
    api_main.campaign_jobs[job.job_id] = job
    return job.job_id


def test_campaign_download_only_serves_files_inside_campaign_output_dir(campaign_sandbox) -> None:
    root, outside, _submitted = campaign_sandbox
    output_dir = root / "artifacts" / "campaigns" / "c1"
    output_dir.mkdir()
    (output_dir / "runs.csv").write_text("label;loss\n", encoding="utf-8")
    sibling = root / "artifacts" / "campaigns" / "c2"
    sibling.mkdir()
    (sibling / "metadata.json").write_text("{}", encoding="utf-8")
    (output_dir / "summary.json").symlink_to(outside / "secret.json")
    dataset_manifest = _write_manifest(root / "data" / "datasets" / "d1" / "manifest.json", [])
    job_id = _register_campaign(
        {
            "campaign_id": "c1",
            "artifacts": {
                "output_dir": str(output_dir),
                "runs_csv": str(output_dir / "runs.csv"),
                "report": str(outside / "secret.json"),
                "metadata": str(output_dir / ".." / "c2" / "metadata.json"),
                "summary": str(output_dir / "summary.json"),
                "dataset_manifest": str(dataset_manifest),
            },
        }
    )
    client = TestClient(api_main.app)

    runs = client.get(f"/api/campaigns/{job_id}/files/runs.csv")
    assert runs.status_code == 200, runs.text
    assert runs.text == "label;loss\n"
    assert runs.headers["content-disposition"] == 'attachment; filename="c1_runs.csv"'
    # Manifest de entrada fora do output_dir, mas dentro das raízes: permitido.
    assert client.get(f"/api/campaigns/{job_id}/files/dataset_manifest.json").status_code == 200
    for artifact in ("report.md", "metadata.json", "summary.json"):  # absoluto, "..", symlink
        response = client.get(f"/api/campaigns/{job_id}/files/{artifact}")
        assert response.status_code == 400, (artifact, response.text)
        assert "do-not-leak" not in response.text


def test_campaign_download_rejects_output_dir_outside_roots(campaign_sandbox) -> None:
    _root, outside, _submitted = campaign_sandbox
    (outside / "runs.csv").write_text("secret", encoding="utf-8")
    job_id = _register_campaign(
        {
            "campaign_id": "evil",
            "artifacts": {"output_dir": str(outside), "dataset_manifest": str(outside / "secret.json")},
        }
    )
    client = TestClient(api_main.app)
    for artifact in ("runs.csv", "dataset_manifest.json"):
        response = client.get(f"/api/campaigns/{job_id}/files/{artifact}")
        assert response.status_code == 400, (artifact, response.text)


def test_campaign_status_from_disk_rejects_unsafe_job_ids(campaign_sandbox) -> None:
    root, _outside, _submitted = campaign_sandbox
    api_main.CAMPAIGN_JOBS_DIR.mkdir(parents=True)
    (root / "artifacts" / "campaigns" / "escape.json").write_text('{"status": "completed"}', encoding="utf-8")
    assert api_main._load_campaign_status_from_disk("../escape") is None
    assert api_main._load_campaign_status_from_disk("..") is None
