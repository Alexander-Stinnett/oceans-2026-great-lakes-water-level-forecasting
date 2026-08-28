from __future__ import annotations

import numpy as np
import pandas as pd

from oceans_glwl.analysis.common_origins import manuscript_surface
from oceans_glwl.analysis.selection import architecture_winners


def _rmse_cm(frame: pd.DataFrame) -> float:
    error = frame["actual"].to_numpy(dtype=np.float64) - frame["prediction"].to_numpy(dtype=np.float64)
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("Cannot compute RMSE from empty or non-finite predictions")
    return float(np.sqrt(np.mean(np.square(error))) * 100.0)


def analyze_rq1(condition_results: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    selected = architecture_winners(condition_results)
    rows: list[dict[str, object]] = []
    for _, winner in selected.iterrows():
        condition_id = str(winner["condition_id"])
        condition_predictions = manuscript_surface(
            predictions.loc[
                predictions["condition_id"].eq(condition_id)
                & predictions["lead_days"].eq(180)
            ]
        )
        rows.append(
            {
                "architecture": winner["condition.model_name"],
                "condition_id": condition_id,
                "input_variant": winner["condition.input_variant"],
                "context_days": int(winner["condition.context_days"]),
                "best_trial_number": int(winner["best.trial_number"]),
                "validation_rmse_cm": float(winner["best.validation_value"]) * 100.0,
                "test_rmse_cm": _rmse_cm(condition_predictions),
                "forecast_origins": condition_predictions["forecast_origin"].nunique(),
            }
        )
    return pd.DataFrame(rows).sort_values("architecture").reset_index(drop=True)

