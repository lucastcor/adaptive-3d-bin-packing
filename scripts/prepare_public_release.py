#!/usr/bin/env python3
"""Build a source snapshot without private files or Git history; never publish."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = ("README.md", "CONTRIBUTING.md", "ACKNOWLEDGMENTS.md", ".gitignore", "pyproject.toml", "LICENSE")
PUBLIC_DIRS = ("src", "cpp", "frontend", "configs", "data/raw", "docker", "docs", "tests", "scripts", ".github")
SKIP_DIRS = {"node_modules", "dist", "__pycache__", ".pytest_cache", ".vite", "build", ".git", ".private"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".tsbuildinfo", ".pdf", ".docx", ".db", ".sqlite", ".npz", ".pem", ".key"}
SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "openai_key": re.compile(rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}\b"),
    "slack_token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
}


def public_paths() -> list[Path]:
    paths = [ROOT / name for name in PUBLIC_FILES if (ROOT / name).is_file()]
    for name in PUBLIC_DIRS:
        base = ROOT / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            rel = path.relative_to(ROOT)
            if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts):
                continue
            if rel.as_posix().startswith(("docs/articles/", "docs/artigo_latex/", "docs/guia_didatico/", "docs/research/", "docs/historico/")):
                continue
            if path.name.startswith(".env") and path.name != ".env.example":
                continue
            if path.name == ".DS_Store" or path.suffix.lower() in SKIP_SUFFIXES:
                continue
            if path.is_symlink():
                raise ValueError(f"Symlink not allowed in source export: {rel}")
            if path.is_file():
                paths.append(path)
    return sorted(set(paths))


def scan_bytes(data: bytes) -> list[str]:
    return [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(data)]


def audit_history() -> dict:
    """Scan reachable text blobs by format, reporting IDs/paths, never values."""
    rows = subprocess.check_output(["git", "rev-list", "--objects", "--all"], cwd=ROOT).splitlines()
    objects = {}
    for row in rows:
        oid, _, name = row.partition(b" ")
        objects[oid] = name.decode(errors="replace")
    proc = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    findings, scanned, private_paths = [], 0, set()
    try:
        for oid, name in objects.items():
            if name.startswith(("docs/articles/", "docs/artigo_latex/", "Vault Adaptive 3DBP/", "docs/research/")):
                private_paths.add(name)
            proc.stdin.write(oid + b"\n")
            proc.stdin.flush()
            header = proc.stdout.readline().split()
            if len(header) != 3:
                raise RuntimeError("Unexpected git cat-file response")
            kind, size = header[1], int(header[2])
            # Stream large objects instead of keeping them in memory.
            data = bytearray()
            remaining = size
            while remaining:
                chunk = proc.stdout.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise RuntimeError("Unexpected end of git object")
                if kind == b"blob" and size <= 2 * 1024 * 1024:
                    data.extend(chunk)
                remaining -= len(chunk)
            proc.stdout.read(1)
            if kind == b"blob" and data and b"\0" not in data:
                scanned += 1
                matches = scan_bytes(bytes(data))
                if matches:
                    findings.append({"object": oid.decode(), "path": name, "formats": matches})
    finally:
        proc.stdin.close()
        proc.wait()
    return {"text_blobs_scanned": scanned, "format_findings": findings,
            "historical_private_paths": len(private_paths),
            "note": "Limited credential-format scan. Does not establish that all secrets or private information are absent."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-history", action="store_true")
    args = parser.parse_args()
    output = ROOT / ".private/public-release"
    output.mkdir(parents=True, exist_ok=True)
    paths = public_paths()
    manifest, findings = [], []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        if len(data) > 95 * 1024 * 1024:
            raise ValueError(f"File exceeds publication limit: {rel}")
        if b"\0" not in data[:8192]:
            matches = scan_bytes(data)
            if matches:
                findings.append({"path": rel, "formats": matches})
        manifest.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    if findings:
        raise ValueError(f"Potential credentials in export (values withheld): {findings}")
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    # Replace only this tool's generated snapshot, never the private research archive.
    source = output / "source"
    if source.exists():
        shutil.rmtree(source)
    source.mkdir()
    for path in paths:
        destination = source / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    archive = output / f"adaptive-3dbp-{version}-public.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            rel = path.relative_to(ROOT)
            bundle.write(source / rel, f"adaptive-3dbp/{rel.as_posix()}")
    report = {"version": version, "file_count": len(paths), "source_bytes": sum(x["bytes"] for x in manifest),
              "zip_bytes": archive.stat().st_size, "zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "license_selected": (ROOT / "LICENSE").exists(), "format_findings": findings,
              "contains_git_history": False, "files": manifest}
    if args.audit_history:
        report["history"] = audit_history()
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False, indent=2))
    print(f"Review snapshot: {source}\nZIP: {archive}")


if __name__ == "__main__":
    main()
