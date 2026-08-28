from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from oceans_glwl.analysis.contracts import validate_final_search_exports
from oceans_glwl.analysis.goldens import assert_manuscript_goldens
from oceans_glwl.analysis.rq1 import analyze_rq1
from oceans_glwl.analysis.rq2 import analyze_rq2
from oceans_glwl.analysis.rq3 import analyze_rq3
from oceans_glwl.analysis.tables import manuscript_rq3_table, write_tables
from oceans_glwl.artifacts.io import (
    canonical_data_path,
    final_search_root,
    read_json,
    repository_root,
    rq3_raw_root,
)
from oceans_glwl.artifacts.manifests import verify_manifest
from oceans_glwl.data.splits import build_chronological_splits
from oceans_glwl.data.validate import load_frame
from oceans_glwl.plotting.rq1_predictions import run as generate_rq1_figure


def reproduce_paper(*, root: Path | None = None, write_outputs: bool = True) -> dict[str, Any]:
    repo = root or repository_root()
    failures = verify_manifest(repo / "artifacts/manifests/artifacts.json")
    if failures:
        raise RuntimeError(f"Frozen artifact verification failed: {failures}")
    final_root = final_search_root(repo)
    conditions = pd.read_parquet(final_root / "condition_results_ALL.parquet")
    horizons = pd.read_parquet(final_root / "test_horizon_metrics_ALL.parquet")
    predictions = pd.read_parquet(final_root / "test_predictions_ALL.parquet")
    trials = pd.read_parquet(final_root / "validation_trials_ALL.parquet")
    contract_summary = validate_final_search_exports(conditions, horizons, predictions, trials)

    frame = load_frame(canonical_data_path(repo))
    splits = build_chronological_splits(frame)
    split_summary = splits.to_metadata()
    rq1 = analyze_rq1(conditions, predictions)
    rq2, rq2_summary = analyze_rq2(conditions, predictions)

    raw_root = rq3_raw_root(repo)
    rq3 = analyze_rq3(
        frame,
        read_json(raw_root / "frozen_condition_index.json"),
        read_json(raw_root / "analysis_policy.json"),
        read_json(raw_root / "seed_refit_plan.json"),
        pd.read_parquet(raw_root / "rq3_refit_status.parquet"),
        pd.read_parquet(raw_root / "rq3_refit_predictions.parquet"),
    )
    rq3_table = manuscript_rq3_table(rq3["regime_rmse_aggregate"], rq3["persistence_regime_rmse"])
    golden_summary = assert_manuscript_goldens(
        rq1, rq2, rq2_summary, rq3["regime_rmse_aggregate"], rq3["persistence_regime_rmse"]
    )
    outputs: dict[str, object] = {}
    if write_outputs:
        table_paths = write_tables(repo / "paper/generated_tables", rq1, rq2, rq3_table)
        figure = generate_rq1_figure(final_root, repo / "paper/figures/generated")
        manuscript_analysis = repo / "artifacts/frozen/rq3/manuscript_analysis"
        manuscript_analysis.mkdir(parents=True, exist_ok=True)
        for name, product in rq3.items():
            if name != "manuscript_predictions":
                product.to_parquet(manuscript_analysis / f"{name}.parquet", index=False)
        outputs = {"tables": [str(path) for path in table_paths], "figure": figure}
    return {
        "status": "success",
        "artifact_verification": "passed",
        "final_search_contracts": contract_summary,
        "splits": split_summary,
        "rq1_rows": len(rq1),
        "rq2_rows": len(rq2),
        "rq2_mean_relative_difference_percent": rq2_summary["mean_relative_difference_percent"],
        "rq3_rows": len(rq3_table),
        "rq3_forecast_origins": 2_926,
        "golden_checks": golden_summary,
        "outputs": outputs,
    }


def main() -> None:
    print(json.dumps(reproduce_paper(), indent=2, default=str))

