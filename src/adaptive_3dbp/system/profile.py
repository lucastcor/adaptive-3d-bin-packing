"""Hardware/OS profiling and live resource usage.

Used by the frontend to (a) recommend sensible worker counts for this machine
(parallel SA is CPU-bound: more workers than physical cores brings contention,
not speedup) and (b) show live CPU/memory consumption during runs.

psutil is used when available; everything degrades gracefully without it.
The solver does not use the GPU — GPUs are reported for transparency only.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]


def _detect_gpus() -> list[str]:
    gpus: list[str] = []
    # NVIDIA
    if shutil.which("nvidia-smi"):
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            gpus.extend(line.strip() for line in completed.stdout.splitlines() if line.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    # Apple Silicon: GPU integrada no SoC.
    if platform.system() == "Darwin" and platform.machine() == "arm64" and not gpus:
        gpus.append("Apple Silicon (GPU integrada, Metal)")
    return gpus


def system_info() -> dict[str, object]:
    logical = os.cpu_count() or 1
    physical: int | None = None
    total_memory_bytes: int | None = None
    if psutil is not None:
        physical = psutil.cpu_count(logical=False)
        total_memory_bytes = int(psutil.virtual_memory().total)

    effective_physical = physical or logical
    # CPU-bound workload: keep one core free for the OS/backend; never exceed
    # the physical core count (hyperthreads do not help the decoder hot path).
    recommended = max(1, min(effective_physical - 1, logical - 1)) if effective_physical > 1 else 1
    maximum = max(1, effective_physical)

    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python_version": sys.version.split()[0],
        "cpu_logical": logical,
        "cpu_physical": physical,
        "total_memory_bytes": total_memory_bytes,
        "gpus": _detect_gpus(),
        "gpu_used_by_solver": False,
        "psutil_available": psutil is not None,
        "pause_signals_available": hasattr(__import__("signal"), "SIGSTOP"),
        "workers": {
            "recommended": recommended,
            "maximum": maximum,
            "note": (
                "Solver e CPU-bound: workers alem dos nucleos fisicos geram contencao. "
                "Deixe 1 nucleo livre para o sistema."
            ),
        },
    }


def system_usage() -> dict[str, object]:
    if psutil is None:
        return {
            "available": False,
            "reason": "psutil nao instalado (pip install psutil)",
        }
    virtual = psutil.virtual_memory()
    process = psutil.Process()
    with process.oneshot():
        process_memory = process.memory_info().rss
        process_cpu = process.cpu_percent(interval=0.0)
    load_average: list[float] | None = None
    if hasattr(os, "getloadavg"):
        try:
            load_average = [round(value, 2) for value in os.getloadavg()]
        except OSError:  # pragma: no cover
            load_average = None
    return {
        "available": True,
        "cpu_percent": psutil.cpu_percent(interval=0.0),
        "cpu_per_core": psutil.cpu_percent(interval=0.0, percpu=True),
        "memory_total_bytes": int(virtual.total),
        "memory_used_bytes": int(virtual.used),
        "memory_percent": virtual.percent,
        "process_memory_bytes": int(process_memory),
        "process_cpu_percent": process_cpu,
        "load_average": load_average,
    }
