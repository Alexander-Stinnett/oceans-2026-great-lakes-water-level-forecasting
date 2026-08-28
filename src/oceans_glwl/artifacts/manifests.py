from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from oceans_glwl.artifacts.io import repository_root


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(
    path: Path,
    *,
    root: Path,
    category: str,
    schema_version: str,
    parents: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "category": category,
        "schema_version": schema_version,
        "parents": parents or [],
        "provenance": provenance or {},
    }
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        record["rows"] = len(frame)
        record["columns"] = list(frame.columns)
    elif path.suffix == ".csv":
        record["rows"] = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
    return record


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else repository_root() / "artifacts/manifests/artifacts.json"
    return json.loads(target.read_text(encoding="utf-8"))


def verify_manifest(path: str | Path | None = None) -> list[dict[str, Any]]:
    root = repository_root()
    manifest = load_manifest(path)
    failures: list[dict[str, Any]] = []
    for record in manifest["artifacts"]:
        target = root / record["path"]
        observed = None if not target.is_file() else sha256_file(target)
        if observed != record["sha256"] or (target.is_file() and target.stat().st_size != record["bytes"]):
            failures.append({"path": record["path"], "expected": record["sha256"], "observed": observed})
    return failures

