"""Build immutable, analysis-ready exports from finalized OCEANS attempts."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

EXPORT_STEMS = (
    "condition_results",
    "test_horizon_metrics",
    "test_predictions",
    "validation_trials",
)
EXPECTED_HOST_CONDITIONS = {"sky": 170, "lavvy": 75, "mac": 91}
FINAL_TARGET_N_TRIALS = 50


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            flatten(f"{prefix}{key}.", item, out)
    elif isinstance(value, (list, tuple)):
        out[prefix[:-1]] = json.dumps(value, sort_keys=True, default=str)
    else:
        out[prefix[:-1]] = value


def _with_identity(
    frame: pd.DataFrame, *, condition_id: str, source_host: str, source: Path
) -> pd.DataFrame:
    frame = frame.copy()
    if "condition_id" in frame.columns:
        actual = set(frame["condition_id"].dropna().astype(str).unique())
        if actual != {condition_id}:
            raise RuntimeError(
                f"Unexpected condition_id values in {source}: {sorted(actual)}"
            )
    else:
        frame.insert(0, "condition_id", condition_id)

    if "source_host" in frame.columns:
        actual_hosts = set(frame["source_host"].dropna().astype(str).unique())
        if actual_hosts != {source_host}:
            raise RuntimeError(
                f"Unexpected source_host values in {source}: {sorted(actual_hosts)}"
            )
    else:
        frame.insert(1, "source_host", source_host)
    return frame


def _write_parquet_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _concat(frames: list[pd.DataFrame], label: str) -> pd.DataFrame:
    if not frames:
        raise RuntimeError(f"No {label} rows were found")
    return pd.concat(frames, ignore_index=True)


def aggregate_host(
    *, results_root: Path, output_root: Path, host_label: str, expected: int
) -> dict[str, int]:
    """Export one host's final-50 attempts without modifying source artifacts."""

    host_label = host_label.lower()
    conditions_root = results_root / "conditions"
    if not conditions_root.is_dir():
        raise FileNotFoundError(f"Conditions directory does not exist: {conditions_root}")
    if expected <= 0:
        raise ValueError("expected must be positive")

    summary_rows: list[dict[str, Any]] = []
    horizon_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    trial_frames: list[pd.DataFrame] = []
    seen_conditions: set[str] = set()

    for status_path in sorted(conditions_root.glob("*/attempts/*/status.json")):
        status = read_json(status_path)
        if (
            status.get("status") != "satisfied_attempt"
            or status.get("complete_trials") != FINAL_TARGET_N_TRIALS
            or status.get("target_n_trials") != FINAL_TARGET_N_TRIALS
        ):
            continue

        run_dir = status_path.parent
        condition_id = str(status["condition_id"])
        if condition_id in seen_conditions:
            raise RuntimeError(f"Duplicate final-50 condition: {condition_id}")
        if str(status.get("worker_id", "")).lower() != host_label:
            raise RuntimeError(
                f"Final attempt {run_dir} belongs to worker {status.get('worker_id')!r}, "
                f"not host {host_label!r}"
            )

        condition = read_json(run_dir / "condition.json")
        best = read_json(run_dir / "best_validation_config.json")
        validation_metrics = read_json(run_dir / "validation_metrics.json")
        test_metrics = read_json(run_dir / "test_metrics_summary.json")
        if str(condition.get("condition_id")) != condition_id:
            raise RuntimeError(f"Condition identity mismatch in {run_dir}")
        if str(condition.get("study_name")) != str(status.get("study_name")):
            raise RuntimeError(f"Study identity mismatch in {run_dir}")

        row: dict[str, Any] = {
            "condition_id": condition_id,
            "source_host": host_label,
            "source_attempt_path": str(run_dir),
        }
        flatten("condition.", condition, row)
        flatten("status.", status, row)
        flatten("best.", best, row)
        flatten("validation.", validation_metrics, row)
        flatten("test.", test_metrics, row)
        summary_rows.append(row)

        horizon_path = run_dir / "test_horizon_metrics.csv"
        prediction_path = run_dir / "test_predictions.csv"
        trials_path = run_dir / "validation_trials.parquet"
        horizon_frames.append(
            _with_identity(
                pd.read_csv(horizon_path),
                condition_id=condition_id,
                source_host=host_label,
                source=horizon_path,
            )
        )
        prediction_frames.append(
            _with_identity(
                pd.read_csv(prediction_path),
                condition_id=condition_id,
                source_host=host_label,
                source=prediction_path,
            )
        )
        trials = _with_identity(
            pd.read_parquet(trials_path),
            condition_id=condition_id,
            source_host=host_label,
            source=trials_path,
        )
        complete_trials = int((trials["state"] == "COMPLETE").sum())
        if complete_trials != FINAL_TARGET_N_TRIALS:
            raise RuntimeError(
                f"Expected {FINAL_TARGET_N_TRIALS} COMPLETE trials for {condition_id}, "
                f"found {complete_trials}"
            )
        trial_frames.append(trials)
        seen_conditions.add(condition_id)

    if len(seen_conditions) != expected:
        raise RuntimeError(
            f"Expected {expected} conditions for {host_label}, "
            f"found {len(seen_conditions)}"
        )

    exports = {
        "condition_results": pd.DataFrame(summary_rows),
        "test_horizon_metrics": _concat(horizon_frames, "horizon metric"),
        "test_predictions": _concat(prediction_frames, "prediction"),
        "validation_trials": _concat(trial_frames, "validation trial"),
    }
    for stem, frame in exports.items():
        _write_parquet_atomic(frame, output_root / f"{stem}_{host_label}.parquet")

    return {
        "conditions": len(seen_conditions),
        "horizon_rows": len(exports["test_horizon_metrics"]),
        "prediction_rows": len(exports["test_predictions"]),
        "trial_rows": len(exports["validation_trials"]),
    }


def merge_hosts(
    *,
    export_root: Path,
    hosts: Iterable[str] = ("sky", "lavvy", "mac"),
    expected_by_host: Mapping[str, int] = EXPECTED_HOST_CONDITIONS,
) -> dict[str, int]:
    """Merge validated host exports into four canonical ``*_ALL`` tables."""

    normalized_hosts = tuple(host.lower() for host in hosts)
    host_conditions: dict[str, set[str]] = {}
    loaded: dict[str, dict[str, pd.DataFrame]] = {}

    for host in normalized_hosts:
        host_frames = {
            stem: pd.read_parquet(export_root / f"{stem}_{host}.parquet")
            for stem in EXPORT_STEMS
        }
        summary = host_frames["condition_results"]
        conditions = set(summary["condition_id"].astype(str))
        expected = expected_by_host[host]
        if len(summary) != expected or len(conditions) != expected:
            raise RuntimeError(
                f"Expected {expected} unique condition rows for {host}, "
                f"found {len(summary)} rows and {len(conditions)} IDs"
            )
        if set(summary["source_host"].astype(str)) != {host}:
            raise RuntimeError(f"Invalid source_host values in {host} summary export")
        for stem, frame in host_frames.items():
            frame_conditions = set(frame["condition_id"].astype(str))
            if frame_conditions != conditions:
                raise RuntimeError(f"Condition coverage mismatch in {stem}_{host}.parquet")
        host_conditions[host] = conditions
        loaded[host] = host_frames

    for index, host in enumerate(normalized_hosts):
        for other in normalized_hosts[index + 1 :]:
            overlap = host_conditions[host] & host_conditions[other]
            if overlap:
                raise RuntimeError(
                    f"Host exports {host} and {other} overlap on {len(overlap)} conditions"
                )

    expected_total = sum(expected_by_host[host] for host in normalized_hosts)
    row_counts: dict[str, int] = {}
    for stem in EXPORT_STEMS:
        merged = pd.concat([loaded[host][stem] for host in normalized_hosts], ignore_index=True)
        if merged["condition_id"].nunique() != expected_total:
            raise RuntimeError(f"Merged {stem} does not cover {expected_total} conditions")
        _write_parquet_atomic(merged, export_root / f"{stem}_ALL.parquet")
        row_counts[stem] = len(merged)
    return row_counts
