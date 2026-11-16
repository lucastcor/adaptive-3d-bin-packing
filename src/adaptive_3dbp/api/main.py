from __future__ import annotations

import json
import re
import signal
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from adaptive_3dbp import __version__
from adaptive_3dbp.api.schemas import ResultExportRequest, SolveRequest
from adaptive_3dbp.domain.models import ProblemInstance, SolverConfig, SolverResult
from adaptive_3dbp.domain.validation import validate_problem_instance
from adaptive_3dbp.experiments.control import SolveCancelled, SolveControl
from adaptive_3dbp.experiments.runner import solve_instance
from adaptive_3dbp.features.instance_features import extract_instance_features
from adaptive_3dbp.io.spreadsheet import (
    instance_from_spreadsheet,
    instance_to_spreadsheet_bytes,
    result_to_spreadsheet_bytes,
    template_bytes,
)
from adaptive_3dbp.experiments.campaign import run_campaign
from adaptive_3dbp.experiments.instances import generate_dataset
from adaptive_3dbp.learning.suggest import suggest_configs
from adaptive_3dbp.native.cpp_bridge import native_core_available
from adaptive_3dbp.persistence.sqlite import SQLiteExperimentRepository
from adaptive_3dbp.system.profile import system_info, system_usage

ROOT_DIR = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = ROOT_DIR / "data" / "raw" / "example_20.json"

app = FastAPI(
    title="Adaptive 3D Bin Packing API",
    version=__version__,
    description="Offline 3D-BPP solver with Extreme Points, Simulated Annealing and optional C++ core.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repository = SQLiteExperimentRepository(ROOT_DIR / "artifacts" / "experiments.db")
job_executor = ThreadPoolExecutor(max_workers=2)
campaign_executor = ThreadPoolExecutor(max_workers=1)


@dataclass
class CampaignJob:
    job_id: str
    spec: dict
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    progress: dict = field(default_factory=lambda: {"stage": "queued", "detail": "", "fraction": 0.0})
    summary: dict | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


campaign_jobs: dict[str, CampaignJob] = {}
campaign_jobs_lock = threading.Lock()

# Estado durável dos jobs de campanha: além do dicionário em memória, cada
# job grava artifacts/campaigns/_jobs/{job_id}.json. Assim o acompanhamento
# sobrevive a restarts do servidor (ex.: uvicorn --reload disparado por
# escrita de arquivos) e o navegador pode reabrir uma campanha em andamento.
CAMPAIGN_JOBS_DIR = ROOT_DIR / "artifacts" / "campaigns" / "_jobs"

# Caminhos que chegam pela API (output_dir da campanha, manifest do dataset e
# as entradas desse manifest) só podem apontar para dentro destas raízes do
# projeto. Tudo passa por Path.resolve() — que colapsa ".." e segue symlinks —
# antes da checagem, então nem "../" nem um symlink para fora escapam delas.
ALLOWED_PATH_ROOT_NAMES = ("artifacts", "data")
# campaign_id vira nome de diretório (artifacts/campaigns/{id}) e de arquivo
# no download; job_id vira nome de arquivo em _jobs/. Sem "/", "." nem "..".
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def _allowed_path_roots() -> tuple[Path, ...]:
    # Lido de ROOT_DIR a cada chamada (e não congelado no import) para os
    # testes poderem apontar o projeto para um diretório temporário.
    return tuple((ROOT_DIR / name).resolve() for name in ALLOWED_PATH_ROOT_NAMES)


def _resolve_allowed_path(raw: object, field_name: str, base: Path | None = None, expand_user: bool = True) -> Path:
    """Resolve ``raw`` and require it to lie strictly inside an allowed root.

    Relative paths are anchored at ``base`` (default: ROOT_DIR). Raises HTTP
    400 for empty/non-string values, unresolvable paths and anything whose
    resolved location (after ``..`` and symlinks) is outside the roots.
    """
    if not isinstance(raw, (str, Path)) or not str(raw):
        raise HTTPException(status_code=400, detail=f"{field_name}: caminho inválido")
    try:
        candidate = Path(raw)
        if expand_user:
            candidate = candidate.expanduser()
        if not candidate.is_absolute():
            candidate = (base or ROOT_DIR) / candidate
        resolved = candidate.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name}: caminho inválido ({exc})") from exc
    for root in _allowed_path_roots():
        if resolved != root and resolved.is_relative_to(root):
            return resolved
    allowed = ", ".join(f"{name}/" for name in ALLOWED_PATH_ROOT_NAMES)
    raise HTTPException(status_code=400, detail=f"{field_name} fora das raízes permitidas ({allowed}): {raw}")


def _resolve_inside(raw: object, parent: Path, field_name: str) -> Path:
    """Resolve ``raw`` and require it inside ``parent`` (already resolved)."""
    path = _resolve_allowed_path(raw, field_name)
    if not path.is_relative_to(parent) or path == parent:
        raise HTTPException(status_code=400, detail=f"{field_name} fora do diretório da campanha")
    return path


def _validate_manifest(raw: object) -> Path:
    """Validate the manifest path AND every instance path listed in it.

    run_campaign reads each ``splits[*][*].path`` with a plain
    ``Path(entry["path"]).read_text()`` (relative to the process CWD), so a
    manifest inside the roots could still point at arbitrary files; the
    entries are checked exactly the way they will be opened.
    """
    path = _resolve_allowed_path(raw, "dataset.manifest")
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"dataset.manifest não encontrado: {raw}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"dataset.manifest ilegível: {exc}") from exc
    splits = manifest.get("splits") if isinstance(manifest, dict) else None
    if not isinstance(splits, dict):
        raise HTTPException(status_code=400, detail="dataset.manifest sem 'splits' válido")
    for split, entries in splits.items():
        for index, entry in enumerate(entries if isinstance(entries, list) else [None]):
            entry_path = entry.get("path") if isinstance(entry, dict) else None
            _resolve_allowed_path(
                entry_path, f"dataset.manifest splits.{split}[{index}].path", base=Path.cwd(), expand_user=False
            )
    return path


def _validate_campaign_spec(spec: dict, job_id: str) -> None:
    """Reject request-controlled paths outside the allowed roots (HTTP 400).

    Validated paths are written back resolved and absolute, so run_campaign
    opens exactly the path that was checked (no CWD-relative reinterpretation).
    """
    campaign_id = spec.get("campaign_id") or f"campaign-{job_id[:8]}"
    if not isinstance(campaign_id, str) or not _SAFE_ID.fullmatch(campaign_id):
        raise HTTPException(status_code=400, detail="campaign_id inválido (use apenas letras, dígitos, '-' e '_')")
    spec["campaign_id"] = campaign_id
    output_dir = spec.get("output_dir") or ROOT_DIR / "artifacts" / "campaigns" / campaign_id
    spec["output_dir"] = str(_resolve_allowed_path(output_dir, "output_dir"))
    dataset = spec.get("dataset")
    if dataset is None:
        return
    if not isinstance(dataset, dict):
        raise HTTPException(status_code=400, detail="dataset deve ser um objeto")
    if "manifest" in dataset:
        dataset["manifest"] = str(_validate_manifest(dataset["manifest"]))


def _persist_campaign_status(job: CampaignJob) -> None:
    try:
        CAMPAIGN_JOBS_DIR.mkdir(parents=True, exist_ok=True)
        payload = _campaign_payload(job)
        (CAMPAIGN_JOBS_DIR / f"{job.job_id}.json").write_text(
            json.dumps(payload, default=str), encoding="utf-8"
        )
    except OSError:  # pragma: no cover - persistência de status é best-effort
        pass


def _load_campaign_status_from_disk(job_id: str) -> dict | None:
    if not _SAFE_ID.fullmatch(job_id):
        return None
    path = CAMPAIGN_JOBS_DIR / f"{job_id}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    # Job interrompido por restart: estava queued/running mas não há mais
    # executor rodando — sinalizar como interrompido para o frontend.
    if payload.get("status") in ("queued", "running") and job_id not in campaign_jobs:
        payload["status"] = "error"
        payload["error"] = (
            "o servidor reiniciou durante a campanha (job interrompido); os artefatos parciais "
            f"estão em {payload.get('summary', {}).get('artifacts', {}).get('output_dir') or 'artifacts/campaigns'}"
        )
    return payload


def _run_campaign_job(job: CampaignJob) -> None:
    last_persist = 0.0

    def on_progress(progress: dict) -> None:
        nonlocal last_persist
        with job.lock:
            job.progress = progress
        now = time.time()
        if now - last_persist >= 1.0:
            last_persist = now
            _persist_campaign_status(job)

    with job.lock:
        job.status = "running"
    _persist_campaign_status(job)
    try:
        summary = run_campaign(job.spec, progress_callback=on_progress, repository=repository, project_root=ROOT_DIR)
        with job.lock:
            job.summary = summary
            job.status = "completed"
    except Exception as exc:  # pragma: no cover - fronteira defensiva da API.
        with job.lock:
            job.status = "error"
            job.error = str(exc)
    _persist_campaign_status(job)


def _campaign_payload(job: CampaignJob) -> dict[str, object]:
    with job.lock:
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "summary": job.summary,
            "error": job.error,
        }


@dataclass
class SolveJob:
    job_id: str
    instance: ProblemInstance
    config: SolverConfig
    seed: int
    persist: bool
    control: SolveControl
    pause_supported: bool
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    latest_point: object | None = None
    result: SolverResult | None = None
    run_id: int | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


solve_jobs: dict[str, SolveJob] = {}
solve_jobs_lock = threading.Lock()


def _pause_supported(config: SolverConfig) -> bool:
    """Pause capability matrix.

    - Python sequential: cooperative checkpoint in the SA loop -> supported.
    - Python/native cooperative pool: pause flag travels via central memory
      and workers honour it at sync points -> supported. (Native+cooperative
      always falls back to the Python cooperative core.)
    - Native sequential / native parallel: SIGSTOP/SIGCONT on the subprocess
      -> supported on POSIX only (Windows has no SIGSTOP).
    - Python independent process_pool: workers have no communication channel
      -> NOT supported (cancel discards the result at the end).
    """
    if config.parallel_mode == "cooperative_process_pool":
        return True
    if config.use_native_core:
        return hasattr(signal, "SIGSTOP")
    return config.parallel_mode == "sequential"


def _job_payload(job: SolveJob) -> dict[str, object]:
    with job.lock:
        now = time.time()
        elapsed = 0.0
        if job.started_at is not None:
            elapsed = (job.finished_at or now) - job.started_at
        visible_status = "paused" if job.status == "running" and job.control.is_paused else job.status
        latest = job.latest_point.model_dump() if job.latest_point is not None else None
        latest_evaluation = int(latest["evaluation"]) if latest is not None else 0
        evaluation_fraction = latest_evaluation / max(job.config.max_evaluations, 1)
        time_fraction = elapsed / max(job.config.time_limit_seconds, 0.001)
        percent = 100.0 if visible_status == "completed" else min(99.0, max(evaluation_fraction, time_fraction) * 100.0)
        return {
            "job_id": job.job_id,
            "status": visible_status,
            "pause_supported": job.pause_supported,
            "run_id": job.run_id,
            "error": job.error,
            "progress": {
                "percent": percent,
                "elapsed_seconds": elapsed,
                "evaluation": latest_evaluation,
                "max_evaluations": job.config.max_evaluations,
                "time_limit_seconds": job.config.time_limit_seconds,
                "latest_point": latest,
            },
            "result": job.result.model_dump() if job.result is not None else None,
        }


def _run_solve_job(job: SolveJob) -> None:
    def on_progress(point: object) -> None:
        with job.lock:
            job.latest_point = point

    with job.lock:
        job.status = "running"
        job.started_at = time.time()

    try:
        result = solve_instance(
            job.instance,
            job.config,
            job.seed,
            control=job.control if job.pause_supported else None,
            progress_callback=on_progress,
        )
        if job.control.is_cancelled:
            raise SolveCancelled("solve cancelled")
        run_id = repository.save_run(job.instance, job.config, result) if job.persist else None
        if run_id is not None:
            learning = repository.observe_and_train(
                run_id,
                job.instance,
                job.config,
                result,
                ROOT_DIR / "artifacts" / "models",
            )
            result = result.model_copy(
                update={
                    "diagnostics": [
                        *result.diagnostics,
                        f"online GPR update: {learning}",
                    ]
                }
            )
        with job.lock:
            job.result = result
            job.run_id = run_id
            job.status = "completed"
            job.finished_at = time.time()
    except SolveCancelled:
        with job.lock:
            job.status = "cancelled"
            job.error = "execucao interrompida pelo usuario"
            job.finished_at = time.time()
    except Exception as exc:  # pragma: no cover - defensive API boundary.
        with job.lock:
            job.status = "error"
            job.error = str(exc)
            job.finished_at = time.time()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": __version__,
        "native_core_available": native_core_available(),
    }


@app.get("/api/system/info")
def get_system_info() -> dict[str, object]:
    info = system_info()
    info["native_core_available"] = native_core_available()
    return info


@app.get("/api/system/usage")
def get_system_usage() -> dict[str, object]:
    return system_usage()


@app.get("/api/examples/default-instance")
def default_instance() -> ProblemInstance:
    return ProblemInstance.model_validate(json.loads(EXAMPLE_PATH.read_text(encoding="utf-8")))


@app.get("/api/examples/spreadsheet-template")
def spreadsheet_template() -> Response:
    return Response(
        content=template_bytes(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="instance_template.xlsx"'},
    )


@app.post("/api/instances/import-spreadsheet")
async def import_spreadsheet(request: Request) -> dict[str, object]:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=422, detail="request body must contain the .xlsx bytes")
    try:
        instance = instance_from_spreadsheet(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not parse spreadsheet: {exc}") from exc
    diagnostics = validate_problem_instance(instance)
    return {"valid": not diagnostics, "diagnostics": diagnostics, "instance": instance.model_dump()}


@app.post("/api/instances/export-spreadsheet")
def export_spreadsheet(instance: ProblemInstance) -> Response:
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in instance.instance_id) or "instance"
    return Response(
        content=instance_to_spreadsheet_bytes(instance),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.xlsx"'},
    )


@app.post("/api/results/export-spreadsheet")
def export_result_spreadsheet(payload: ResultExportRequest) -> Response:
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in payload.instance.instance_id) or "solution"
    return Response(
        content=result_to_spreadsheet_bytes(payload.instance, payload.result, payload.run_id),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_solution.xlsx"'},
    )


@app.post("/api/validate")
def validate_instance(instance: ProblemInstance) -> dict[str, object]:
    diagnostics = validate_problem_instance(instance)
    return {"valid": not diagnostics, "diagnostics": diagnostics}


@app.post("/api/features")
def instance_features(instance: ProblemInstance) -> dict[str, float]:
    diagnostics = validate_problem_instance(instance)
    if diagnostics:
        raise HTTPException(status_code=422, detail=diagnostics)
    return extract_instance_features(instance)


@app.post("/api/solve")
def solve(request: SolveRequest) -> dict[str, object]:
    diagnostics = validate_problem_instance(request.instance)
    if diagnostics:
        raise HTTPException(status_code=422, detail=diagnostics)
    result = solve_instance(request.instance, request.config, request.seed)
    run_id = repository.save_run(request.instance, request.config, result) if request.persist else None
    learning: dict[str, object] | None = None
    if run_id is not None:
        learning = repository.observe_and_train(
            run_id,
            request.instance,
            request.config,
            result,
            ROOT_DIR / "artifacts" / "models",
        )
        result = result.model_copy(
            update={
                "diagnostics": [
                    *result.diagnostics,
                    f"online GPR update: {learning}",
                ]
            }
        )
    return {"run_id": run_id, "result": result.model_dump()}


@app.post("/api/solve-jobs")
def start_solve_job(request: SolveRequest) -> dict[str, object]:
    diagnostics = validate_problem_instance(request.instance)
    if diagnostics:
        raise HTTPException(status_code=422, detail=diagnostics)
    job_id = uuid.uuid4().hex
    job = SolveJob(
        job_id=job_id,
        instance=request.instance,
        config=request.config,
        seed=request.seed,
        persist=request.persist,
        control=SolveControl(),
        pause_supported=_pause_supported(request.config),
    )
    with solve_jobs_lock:
        solve_jobs[job_id] = job
    job_executor.submit(_run_solve_job, job)
    return _job_payload(job)


@app.get("/api/solve-jobs/{job_id}")
def get_solve_job(job_id: str) -> dict[str, object]:
    with solve_jobs_lock:
        job = solve_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _job_payload(job)


@app.post("/api/solve-jobs/{job_id}/pause")
def pause_solve_job(job_id: str) -> dict[str, object]:
    with solve_jobs_lock:
        job = solve_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if not job.pause_supported:
        raise HTTPException(status_code=409, detail="pause is not supported for this core/mode; cancel instead")
    with job.lock:
        should_pause = job.status == "running"
    if should_pause:
        job.control.pause()
    return _job_payload(job)


@app.post("/api/solve-jobs/{job_id}/resume")
def resume_solve_job(job_id: str) -> dict[str, object]:
    with solve_jobs_lock:
        job = solve_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    job.control.resume()
    return _job_payload(job)


@app.post("/api/solve-jobs/{job_id}/cancel")
def cancel_solve_job(job_id: str) -> dict[str, object]:
    with solve_jobs_lock:
        job = solve_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    job.control.cancel()
    with job.lock:
        if job.status in {"queued", "running"}:
            job.status = "cancelled"
            job.error = "execucao interrompida pelo usuario"
            job.finished_at = time.time()
    return _job_payload(job)


@app.get("/api/config/default")
def default_config() -> SolverConfig:
    return SolverConfig()


@app.post("/api/config/suggest")
def config_suggest(payload: dict) -> dict[str, object]:
    try:
        instance = ProblemInstance.model_validate(payload.get("instance"))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"instância inválida: {exc}") from exc
    diagnostics = validate_problem_instance(instance)
    if diagnostics:
        raise HTTPException(status_code=422, detail=diagnostics)
    observations = repository.list_learning_observations(limit=5000)
    return suggest_configs(
        instance,
        observations,
        n_suggestions=int(payload.get("n_suggestions", 3)),
        seed=int(payload.get("seed", 0)),
    )


@app.post("/api/datasets/generate")
def datasets_generate(payload: dict) -> dict[str, object]:
    name = str(payload.get("name") or f"dataset-{uuid.uuid4().hex[:6]}")
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name)
    manifest = generate_dataset(
        ROOT_DIR / "data" / "datasets" / safe_name,
        per_class=payload.get("per_class"),
        seed=int(payload.get("seed", 0)),
        include_min_distance=bool(payload.get("include_min_distance", True)),
    )
    return {"name": safe_name, "manifest_path": str(ROOT_DIR / "data" / "datasets" / safe_name / "manifest.json"), "manifest": manifest}


@app.post("/api/campaigns")
def start_campaign(spec: dict) -> dict[str, object]:
    job_id = uuid.uuid4().hex
    # campaign_id e output_dir definidos (e validados) já na largada: o
    # diretório de artefatos fica conhecido antes de qualquer execução,
    # permitindo fallback em disco, e nenhum caminho do corpo da requisição
    # lê/escreve fora de artifacts/ ou data/.
    _validate_campaign_spec(spec, job_id)
    job = CampaignJob(job_id=job_id, spec=spec)
    with campaign_jobs_lock:
        campaign_jobs[job_id] = job
    _persist_campaign_status(job)
    campaign_executor.submit(_run_campaign_job, job)
    return _campaign_payload(job)


@app.get("/api/campaigns/{job_id}")
def get_campaign(job_id: str) -> dict[str, object]:
    with campaign_jobs_lock:
        job = campaign_jobs.get(job_id)
    if job is not None:
        return _campaign_payload(job)
    payload = _load_campaign_status_from_disk(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="campanha não encontrada")
    return payload


def _campaign_summary_for(job_id: str) -> dict | None:
    with campaign_jobs_lock:
        job = campaign_jobs.get(job_id)
    if job is not None:
        with job.lock:
            return job.summary
    payload = _load_campaign_status_from_disk(job_id)
    return payload.get("summary") if payload else None


@app.get("/api/campaigns/{job_id}/files/{artifact}")
def get_campaign_file(job_id: str, artifact: str) -> Response:
    summary = _campaign_summary_for(job_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="campanha não encontrada ou ainda em execução")
    allowed = {
        "runs.csv": "text/csv",
        "solutions.jsonl": "application/x-ndjson",
        "report.md": "text/markdown",
        "metadata.json": "application/json",
        "dataset_manifest.json": "application/json",
        "summary.json": "application/json",
        "comparison.png": "image/png",
        "results_table.tex": "application/x-tex",
    }
    if artifact not in allowed:
        raise HTTPException(status_code=404, detail="artefato não disponível")
    artifacts = summary.get("artifacts") or {}
    campaign_id = str(summary.get("campaign_id") or "")
    # Os caminhos vêm do summary (memória ou _jobs/*.json em disco), não da
    # requisição — mas mesmo assim só servimos arquivos do output_dir da
    # campanha, e o próprio output_dir precisa estar dentro das raízes.
    output_dir = _resolve_allowed_path(
        artifacts.get("output_dir") or ROOT_DIR / "artifacts" / "campaigns" / campaign_id, "output_dir"
    )
    if artifact == "dataset_manifest.json":
        path_value = artifacts.get("dataset_manifest")
        # Exceção única: numa campanha com dataset.manifest o manifest de
        # ENTRADA fica fora do output_dir (ex.: data/datasets/x/manifest.json);
        # ele já foi validado nas raízes na largada e é revalidado aqui.
        path = _resolve_allowed_path(path_value or output_dir / "dataset" / "manifest.json", "dataset_manifest")
    else:
        key = {
            "runs.csv": "runs_csv",
            "solutions.jsonl": "solutions_jsonl",
            "report.md": "report",
            "metadata.json": "metadata",
            "summary.json": "summary",
            "comparison.png": "comparison_png",
            "results_table.tex": "results_table_tex",
        }[artifact]
        path = _resolve_inside(artifacts.get(key) or output_dir / artifact, output_dir, artifact)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="arquivo não gerado")
    disposition = "inline" if artifact == "comparison.png" else "attachment"
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in campaign_id) or "campaign"
    return Response(
        content=path.read_bytes(),
        media_type=allowed[artifact],
        headers={"Content-Disposition": f'{disposition}; filename="{safe_name}_{artifact}"'},
    )


@app.get("/api/runs")
def list_runs(limit: int = 25) -> list[dict[str, object]]:
    return repository.list_runs(limit=limit)


@app.get("/api/learning/status")
def learning_status() -> dict[str, object]:
    observations = repository.list_learning_observations(limit=5000)
    latest_model = repository.latest_surrogate_model()
    return {
        "observation_count": len(observations),
        "latest_model": latest_model,
    }


@app.get("/api/runs/{run_id}/bundle")
def get_run_bundle(run_id: int) -> dict[str, object]:
    bundle = repository.get_run_bundle(run_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="run not found")
    return bundle


@app.get("/api/runs/{run_id}/result-spreadsheet")
def export_run_result_spreadsheet(run_id: int) -> Response:
    bundle = repository.get_run_bundle(run_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="run not found")
    instance = ProblemInstance.model_validate(bundle["instance"])
    result = SolverResult.model_validate(bundle["result"])
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in instance.instance_id) or "solution"
    return Response(
        content=result_to_spreadsheet_bytes(instance, result, run_id),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_run_{run_id}_solution.xlsx"'},
    )


@app.get("/api/runs/{run_id}")
def get_run(run_id: int) -> dict[str, object]:
    result = repository.get_result(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="run not found")
    return result.model_dump()
