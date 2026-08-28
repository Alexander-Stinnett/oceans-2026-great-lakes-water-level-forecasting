from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from oceans_glwl.analysis.rq1 import analyze_rq1
from oceans_glwl.analysis.rq2 import analyze_rq2
from oceans_glwl.analysis.rq3 import analyze_rq3
from oceans_glwl.data.validate import load_frame


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def final_exports(repo_root: Path) -> dict[str, pd.DataFrame]:
    root = repo_root / "artifacts/frozen/final_search"
    return {
        "conditions": pd.read_parquet(root / "condition_results_ALL.parquet"),
        "horizons": pd.read_parquet(root / "test_horizon_metrics_ALL.parquet"),
        "predictions": pd.read_parquet(root / "test_predictions_ALL.parquet"),
        "trials": pd.read_parquet(root / "validation_trials_ALL.parquet"),
    }


@pytest.fixture(scope="session")
def canonical_frame(repo_root: Path) -> pd.DataFrame:
    return load_frame(repo_root / "data/canonical/combined_full.csv")


@pytest.fixture(scope="session")
def rq1_result(final_exports: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return analyze_rq1(final_exports["conditions"], final_exports["predictions"])


@pytest.fixture(scope="session")
def rq2_result(final_exports: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, object]]:
    return analyze_rq2(final_exports["conditions"], final_exports["predictions"])


@pytest.fixture(scope="session")
def rq3_result(repo_root: Path, canonical_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    root = repo_root / "artifacts/frozen/rq3/raw"

    def load(name: str) -> dict[str, Any]:
        return json.loads((root / name).read_text(encoding="utf-8"))

    return analyze_rq3(
        canonical_frame,
        load("frozen_condition_index.json"),
        load("analysis_policy.json"),
        load("seed_refit_plan.json"),
        pd.read_parquet(root / "rq3_refit_status.parquet"),
        pd.read_parquet(root / "rq3_refit_predictions.parquet"),
    )
